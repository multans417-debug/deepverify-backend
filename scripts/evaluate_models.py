#!/usr/bin/env python
"""Evaluate trained models on the HELD-OUT test split (threshold/temperature come from validation, saved in metadata).
Also scores a capped train subsample and the validation split to expose overfitting, and checks split leakage."""
import _bootstrap  # noqa: F401
import argparse, json, os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from deepverify.data.manifest import class_counts, read_manifest
from deepverify.evaluation.metrics import compute_metrics, sigmoid
from deepverify.models.loader import load_bundle
from deepverify.training.common import atomic_write_json, hardware_info, pick_device, subsample, utc_now
from deepverify.training.datasets import ImageManifestDataset, VideoFramesDataset
from deepverify.training.engine import predict_logits


def score(bundle, rows, kind, device, batch, workers):
    ds = (ImageManifestDataset(rows, bundle.preprocess, False) if kind == "image"
          else VideoFramesDataset(rows, bundle.preprocess, bundle.video.num_frames, False))
    z, y = predict_logits(bundle.model.to(device), DataLoader(ds, batch_size=batch, num_workers=workers), device)
    return y, sigmoid(z / bundle.temperature)


def leakage(path: Path) -> dict:
    rows = read_manifest(path)
    g: dict = {}
    for r in rows:
        g.setdefault(r["group"], set()).add(r["split"])
    bad = sum(1 for s in g.values() if len(s) > 1)
    return {"groups": len(g), "groups_in_multiple_splits": bad, "passed": bad == 0}


def evaluate(kind, manifest, art, device, args):
    bundle = load_bundle(kind, art, allow_smoke=args.allow_smoke)
    mpath = Path(manifest)
    test = read_manifest(mpath, "test")
    if not test:
        return {"kind": kind, "error": "no test split in manifest"}
    y, p = score(bundle, test, kind, device, args.batch_size, args.num_workers)
    res = {"kind": kind, "model_version": bundle.version, "smoke_test_model": bundle.smoke_test, "manifest": str(mpath),
           "decision_threshold": bundle.threshold, "threshold_source": bundle.metadata["threshold_method"],
           "test": compute_metrics(y, p, bundle.threshold), "test_class_distribution": class_counts(test),
           "leakage_check": leakage(mpath), "evaluated_utc": utc_now(), "hardware": hardware_info(device), "flags": []}
    yv, pv = score(bundle, read_manifest(mpath, "val"), kind, device, args.batch_size, args.num_workers)
    res["validation"] = compute_metrics(yv, pv, bundle.threshold)
    ytr, ptr = score(bundle, subsample(read_manifest(mpath, "train"), args.train_sample, 0), kind, device, args.batch_size, args.num_workers)
    res["train_subsample_no_augmentation"] = compute_metrics(ytr, ptr, bundle.threshold)
    f, t = res["flags"], res["test"]
    if t["n"] < 100: f.append(f"Only {t['n']} test samples: confidence intervals are wide; treat metrics as indicative only.")
    if min(t["class_counts"].values()) < 20: f.append("Fewer than 20 test samples in one class: per-class metrics unreliable.")
    if t.get("roc_auc") is not None and t["roc_auc"] > 0.995: f.append("Suspiciously high test ROC-AUC: investigate leakage / trivially separable data.")
    if res["train_subsample_no_augmentation"]["accuracy"] - t["accuracy"] > 0.10: f.append("Train accuracy exceeds test accuracy by >10 points: overfitting.")
    if not res["leakage_check"]["passed"]: f.append("LEAKAGE: some source groups appear in more than one split.")
    if bundle.smoke_test: f.append("SMOKE-TEST MODEL: numbers are not meaningful.")
    atomic_write_json(art / f"{kind}_test_metrics.json", res)
    return res


def report(results: list, art: Path, out: Path):
    L = ["# DeepVerify training & evaluation report", f"_Generated {utc_now()} by scripts/evaluate_models.py. Numbers below are read from "
         "artifacts/*_test_metrics.json and *_training_metrics.json; nothing is hand-written._", ""]
    for r in results:
        if "error" in r: L += [f"## {r['kind']}", f"Not evaluated: {r['error']}", ""]; continue
        k, t = r["kind"], r["test"]
        tm = json.loads((art / f"{k}_training_metrics.json").read_text())
        L += [f"## {k.capitalize()} model (version {r['model_version']})", "",
              f"- Hardware: {json.dumps(r['hardware'])}; training time: {tm['train_seconds']} s",
              f"- Dataset sizes: {json.dumps(tm['sizes'])}; test class distribution: {json.dumps(r['test_class_distribution'])}",
              f"- Decision threshold {r['decision_threshold']:.4f} ({r['threshold_source']}); selected on validation only",
              f"- Leakage check (source groups across splits): {json.dumps(r['leakage_check'])}", "",
              "| split | n | accuracy | precision(FAKE) | recall(FAKE) | F1(FAKE) | ROC-AUC | ECE |", "|---|---|---|---|---|---|---|---|"]
        for name, m in (("train (subsample)", r["train_subsample_no_augmentation"]), ("validation", r["validation"]), ("**test**", t)):
            auc = "n/a" if m.get("roc_auc") is None else f"{m['roc_auc']:.4f}"
            L.append(f"| {name} | {m['n']} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {auc} | {m['ece_10_bins']:.4f} |")
        cm = t["confusion_matrix"]["matrix"]
        L += ["", f"Test confusion matrix (rows=true REAL,FAKE; cols=pred REAL,FAKE): `{cm}`", "",
              f"Per-class (test): `{json.dumps(t['per_class'])}`", ""]
        L += ["**Flags:** " + ("; ".join(r["flags"]) if r["flags"] else "none raised"), ""]
        if tm.get("warnings"): L += ["**Training warnings:** " + "; ".join(tm["warnings"]), ""]
    L += ["## Interpretation limits", "- Metrics describe the held-out split of the *training dataset's* distribution only; real-world uploads (other generators, "
          "compression, lighting) will perform worse and false positives/negatives will occur.",
          "- Scores are temperature-scaled on validation data; ECE above shows calibration on the test split. Do not present them as certainties.", ""]
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-manifest"); ap.add_argument("--video-manifest")
    ap.add_argument("--artifact-dir", default=os.getenv("ARTIFACT_DIR", "artifacts"))
    ap.add_argument("--report", default="docs/training_report.md")
    ap.add_argument("--batch-size", type=int, default=16); ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--train-sample", type=int, default=2000); ap.add_argument("--device", default="auto")
    ap.add_argument("--allow-smoke", action="store_true")
    a = ap.parse_args()
    art, device, results = Path(a.artifact_dir), pick_device(a.device), []
    for kind, man in (("image", a.image_manifest), ("video", a.video_manifest)):
        if man: results.append(evaluate(kind, man, art, device, a))
    if not results: raise SystemExit("Provide --image-manifest and/or --video-manifest")
    report(results, art, Path(a.report))
    print(json.dumps([{k: r.get(k) for k in ("kind", "flags")} | {"test": {m: r["test"].get(m) for m in ("n", "accuracy", "f1", "roc_auc")} if "test" in r else r} for r in results], indent=2))


if __name__ == "__main__":
    main()
