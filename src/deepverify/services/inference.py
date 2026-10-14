"""Real inference only. Uses exactly the preprocessing functions that produced the training data."""
from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np
import torch

from deepverify.config import Settings
from deepverify.preprocessing.face import crop_square, make_detector
from deepverify.preprocessing.image import load_rgb, preprocess_rgb
from deepverify.preprocessing.video import extract_face_crops
from deepverify.utils.errors import NoFaceFound


def _result(media_type: str, logit: float, bundle, started: float, details: dict) -> dict:
    p = float(torch.sigmoid(torch.tensor(logit / bundle.temperature)))
    fake = p >= bundle.threshold
    return {"success": True, "media_type": media_type, "prediction": "FAKE" if fake else "REAL",
            "fake_probability": round(p, 6), "confidence": round(p if fake else 1.0 - p, 6),
            "score_is_calibrated": bundle.calibrated, "decision_threshold": bundle.threshold,
            "model_version": bundle.version, "processing_time_seconds": round(time.perf_counter() - started, 3),
            "details": details}


def detect_image(path: Path, bundle, settings: Settings) -> dict:
    started = time.perf_counter()
    rgb = load_rgb(path, settings.max_image_pixels)
    pre, details = bundle.preprocess, {}
    if pre.use_face_crop:
        box = make_detector(pre.face_detector).detect_largest(rgb)
        if box is None:
            raise NoFaceFound("No face was detected in the image; this model analyses face crops.")
        rgb = crop_square(rgb, box, pre.face_margin, pre.input_size)
        details["face_detected"] = True
    x = torch.from_numpy(preprocess_rgb(rgb, pre)).unsqueeze(0)
    with torch.inference_mode():
        logit = float(bundle.model(x)[0])
    return _result("image", logit, bundle, started, details)


def _aggregate_frame_scores(probs: np.ndarray, agg_mode: str = "conf") -> float:
    sz = len(probs)
    if sz == 0:
        return 0.5
    if agg_mode == "mean":
        return float(np.mean(probs))
    if agg_mode == "vote":
        return float(np.sum(probs > 0.5) / sz)
    # 'conf' mode (official HuggingFace DeepGuard video pipeline aggregation)
    t_fake, t_real = 0.8, 0.2
    fakes = probs[probs > t_fake]
    if len(fakes) > (sz / 2.5):
        return float(np.mean(fakes))
    reals = probs[probs < t_real]
    if len(reals) > 0.9 * sz:
        return float(np.mean(reals))
    return float(np.mean(probs))


def detect_video(path: Path, bundle, settings: Settings) -> dict:
    started = time.perf_counter()
    cfg = dataclasses.replace(bundle.video, max_seconds=float(settings.max_video_seconds),
                              timeout_seconds=float(settings.extract_timeout_seconds))
    crops, stats = extract_face_crops(path, cfg)
    arch = bundle.metadata.get("architecture", "")
    if arch in ("ms_eff_gcvit_b0", "ms_eff_gcvit"):
        x = torch.from_numpy(np.stack([preprocess_rgb(c, bundle.preprocess) for c in crops]))
        with torch.inference_mode():
            frame_logits = bundle.model(x)
            probs = torch.sigmoid(frame_logits / bundle.temperature).cpu().numpy()
        p_agg = _aggregate_frame_scores(probs, agg_mode="conf")
        p_clamped = max(1e-6, min(1.0 - 1e-6, p_agg))
        logit = float(np.log(p_clamped / (1.0 - p_clamped)) * bundle.temperature)
        agg_desc = f"MS-EffGCViT frame-level confidence aggregation over {len(crops)} face crops"
    else:
        x = torch.from_numpy(np.stack([preprocess_rgb(c, bundle.preprocess) for c in crops])).unsqueeze(0)
        with torch.inference_mode():
            logit = float(bundle.model(x)[0])
        agg_desc = "GRU over time-ordered face crops (video-level logit)"

    return _result("video", logit, bundle, started, {**stats, "aggregation": agg_desc})

