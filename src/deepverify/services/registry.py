from __future__ import annotations

import logging

from deepverify.config import Settings
from deepverify.utils.artifacts import ensure_artifacts
from deepverify.utils.errors import ModelUnavailable

log = logging.getLogger("deepverify.registry")
KINDS = ("image", "video")


class ModelRegistry:
    """Loads trained models once at start-up. Missing/invalid artifacts are reported, never replaced by fake output."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.bundles: dict = {}
        self.errors: dict = {}

    def load_all(self) -> None:
        import torch
        from deepverify.models.loader import load_bundle
        if self.settings.torch_threads > 0:
            torch.set_num_threads(self.settings.torch_threads)
        ensure_artifacts(self.settings.artifact_dir)
        for kind in KINDS:
            try:
                self.bundles[kind] = load_bundle(kind, self.settings.artifact_dir, self.settings.allow_smoke_models)
                self.errors.pop(kind, None)
                log.info("Loaded %s model %s", kind, self.bundles[kind].version)
            except Exception as exc:   # report per model; the service stays up and answers 503 for that model
                self.errors[kind] = f"{type(exc).__name__}: {exc}"
                log.warning("Could not load %s model: %s", kind, self.errors[kind])

    def get(self, kind: str):
        b = self.bundles.get(kind)
        if b is None:
            raise ModelUnavailable(f"The {kind} detection model is not available on this server "
                                   f"({self.errors.get(kind, 'trained artifacts not found')}).")
        return b

    def ready(self) -> bool:
        return all(k in self.bundles for k in self.settings.required_models)

    def describe(self) -> dict:
        out = {}
        for kind in KINDS:
            b = self.bundles.get(kind)
            if b is None:
                out[kind] = {"loaded": False, "training_status": "no trained artifacts loaded", "error": self.errors.get(kind)}
                continue
            m = b.metadata
            tr = m.get("training", {})
            out[kind] = {"loaded": True, "model_version": b.version, "architecture": m.get("architecture"),
                         "training_status": "smoke-test only" if b.smoke_test else "trained",
                         "decision_threshold": b.threshold, "score_is_calibrated": b.calibrated,
                         "calibration": m.get("calibration"), "class_mapping": m.get("class_mapping"),
                         "input_size": b.preprocess.input_size, "frames_per_video": b.video.num_frames if b.video else None,
                         "trained_on_sizes": tr.get("sizes"), "best_epoch": tr.get("best_epoch"), "created_utc": m.get("created_utc"),
                         "notes": m.get("notes", [])}
        return out
