from __future__ import annotations

import os
from pathlib import Path

import torch

from deepverify.evaluation.metrics import compute_metrics, fit_temperature, select_threshold, sigmoid
from deepverify.training.common import atomic_write_json, dependency_versions, utc_now
from deepverify.utils.artifacts import ARTIFACT_FILES, sha256_file


def calibrate_and_threshold(val_logits, val_labels) -> dict:
    """Temperature + threshold are fit on VALIDATION data only; the test split is never touched here."""
    T = fit_temperature(val_logits, val_labels)
    probs = sigmoid(val_logits / T)
    thr, method = select_threshold(val_labels, probs)
    return {"temperature": T, "threshold": thr, "threshold_method": method,
            "val_metrics_at_threshold": compute_metrics(val_labels, probs, thr),
            "val_metrics_uncalibrated_at_0.5": compute_metrics(val_labels, sigmoid(val_logits), 0.5)}


def save_artifacts(kind: str, artifact_dir: Path, *, state_dict, architecture: str, model_config: dict, preprocess: dict,
                   video: dict | None, cal: dict, training_info: dict, metrics: dict, smoke: bool, notes: list) -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    wname, mname = ARTIFACT_FILES[kind]
    tmp = artifact_dir / (wname + ".tmp")
    torch.save(state_dict, tmp)
    os.replace(tmp, artifact_dir / wname)
    version = f"1.0.{utc_now().replace('-', '').replace(':', '').replace('T', '').replace('Z', '')}"
    meta = {
        "kind": kind, "model_version": version, "architecture": architecture, "model_config": model_config,
        "class_mapping": {"REAL": 0, "FAKE": 1},
        "output": "single logit z; score = sigmoid(z / temperature) = estimated P(FAKE)",
        "preprocess": preprocess, "video": video,
        "threshold": cal["threshold"], "threshold_method": cal["threshold_method"],
        "calibration": {"method": "temperature_scaling", "temperature": cal["temperature"], "fit_on": "validation split",
                        "caveat": "Calibrated only for data resembling the training domain; not validated on other sources."},
        "aggregation": ("GRU over time-ordered face crops; last hidden state -> one logit" if kind == "video"
                        else "single image -> one logit"),
        "weights_file": wname, "weights_sha256": sha256_file(artifact_dir / wname),
        "training": training_info, "dependency_versions": dependency_versions(), "created_utc": utc_now(),
        "smoke_test": smoke, "notes": notes,
    }
    atomic_write_json(artifact_dir / mname, meta)
    atomic_write_json(artifact_dir / f"{kind}_training_metrics.json", metrics)
    return meta
