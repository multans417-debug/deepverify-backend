#!/usr/bin/env python
"""Train the video-level REAL/FAKE classifier (ResNet18 frame encoder + GRU) from video_manifest.csv."""
import _bootstrap  # noqa: F401
import argparse, json, logging, os
from pathlib import Path

from deepverify.data.manifest import class_counts, read_manifest
from deepverify.models.video_model import ARCH, VideoDeepfakeNet
from deepverify.preprocessing.image import PreprocessConfig
from deepverify.preprocessing.video import VideoConfig
from deepverify.training.artifacts import calibrate_and_threshold, save_artifacts
from deepverify.training.common import hardware_info, pick_device, seed_everything, subsample
from deepverify.training.datasets import VideoFramesDataset
from deepverify.training.engine import train_model
from deepverify.training.run import training_warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="video_manifest.csv (video_prep_config.json must sit next to it)")
    ap.add_argument("--artifact-dir", default=os.getenv("ARTIFACT_DIR", "artifacts"))
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--freeze", choices=["none", "early", "backbone"], default="early")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--max-train", type=int); ap.add_argument("--max-val", type=int)
    ap.add_argument("--num-workers", type=int, default=2); ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--device", default="auto")
    ap.add_argument("--no-pretrained", action="store_true"); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    seed_everything(a.seed); device = pick_device(a.device)
    mpath = Path(a.manifest)
    vcfg = VideoConfig.from_dict(json.loads((mpath.parent / "video_prep_config.json").read_text())["video_config"])
    # serving limits are part of the saved config so the API enforces what training assumed
    vcfg = VideoConfig(**{**vcfg.to_dict(), "max_seconds": 60.0, "timeout_seconds": 45.0})
    train_rows = subsample(read_manifest(mpath, "train"), a.max_train, a.seed)
    val_rows = subsample(read_manifest(mpath, "val"), a.max_val, a.seed)
    if not train_rows or not val_rows:
        raise SystemExit("ERROR: empty train or validation split.")
    tc = class_counts(train_rows)
    if min(tc.values()) == 0:
        raise SystemExit("ERROR: training split contains a single class.")
    pre = PreprocessConfig(input_size=vcfg.input_size, use_face_crop=True, face_margin=vcfg.face_margin, face_detector=vcfg.face_detector)
    model = VideoDeepfakeNet(hidden=a.hidden, pretrained=not a.no_pretrained); model.freeze(a.freeze)
    art = Path(a.artifact_dir)
    res = train_model(model, VideoFramesDataset(train_rows, pre, vcfg.num_frames, True, a.seed),
                      VideoFramesDataset(val_rows, pre, vcfg.num_frames, False), device=device, epochs=a.epochs,
                      batch_size=a.batch_size, lr=a.lr, weight_decay=a.weight_decay, num_workers=a.num_workers,
                      pos_weight=tc["REAL"] / tc["FAKE"], ckpt_path=art / "checkpoints" / "video_last.pt", resume=a.resume,
                      patience=a.patience, seed=a.seed)
    cal = calibrate_and_threshold(res["val_logits"], res["val_labels"])
    sizes = {"train": len(train_rows), "val": len(val_rows), "train_classes": tc, "val_classes": class_counts(val_rows)}
    warns = training_warnings(res["history"], cal["val_metrics_at_threshold"], sizes, vcfg.face_detector)
    info = {"manifest": str(mpath), "seed": a.seed, "epochs_requested": a.epochs, "best_epoch": res["best_epoch"], "freeze": a.freeze,
            "pretrained_init": not a.no_pretrained, "lr": a.lr, "batch_size": a.batch_size, "pos_weight": tc["REAL"] / tc["FAKE"],
            "sizes": sizes, "train_seconds": res["train_seconds"], "hardware": hardware_info(device),
            "resumed_from_epoch": res["resumed_from_epoch"], "stopped_early": res["stopped_early"],
            "split_policy": "split by source group (identity / original video) before frame extraction"}
    metrics = {"history": res["history"], "validation": cal["val_metrics_at_threshold"],
               "validation_uncalibrated": cal["val_metrics_uncalibrated_at_0.5"], "warnings": warns, **{k: info[k] for k in ("sizes", "train_seconds", "hardware")}}
    save_artifacts("video", art, state_dict=res["best_state"], architecture=ARCH, model_config={"hidden": a.hidden, "dropout": 0.3},
                   preprocess=pre.to_dict(), video=vcfg.to_dict(), cal=cal, training_info=info, metrics=metrics,
                   smoke=a.smoke or vcfg.face_detector == "center", notes=warns)
    print(json.dumps({"best_epoch": res["best_epoch"], "val": {k: cal["val_metrics_at_threshold"].get(k) for k in ("accuracy", "f1", "roc_auc")},
                      "threshold": cal["threshold"], "temperature": cal["temperature"], "warnings": warns}, indent=2))


if __name__ == "__main__":
    main()
