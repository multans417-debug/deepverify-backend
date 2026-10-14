"""Training-only augmentation. Class-preserving: flip, mild crop, brightness/contrast, JPEG recompression.
For video, parameters are drawn ONCE per clip so all frames receive the same transform."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AugParams:
    flip: bool
    crop_frac: float      # keep this fraction of height/width
    crop_x: float         # 0..1 position of the crop window
    crop_y: float
    brightness: float     # additive, in [-0.12, 0.12] of full range
    contrast: float       # multiplicative around the mean
    jpeg_quality: int     # 0 = no recompression


def draw_params(rng: np.random.Generator) -> AugParams:
    return AugParams(
        flip=bool(rng.random() < 0.5),
        crop_frac=float(rng.uniform(0.85, 1.0)),
        crop_x=float(rng.random()), crop_y=float(rng.random()),
        brightness=float(rng.uniform(-0.12, 0.12)),
        contrast=float(rng.uniform(0.85, 1.15)),
        jpeg_quality=int(rng.integers(70, 96)) if rng.random() < 0.3 else 0,
    )


def apply_params(rgb: np.ndarray, p: AugParams) -> np.ndarray:
    h, w = rgb.shape[:2]
    ch, cw = int(h * p.crop_frac), int(w * p.crop_frac)
    y0, x0 = int((h - ch) * p.crop_y), int((w - cw) * p.crop_x)
    out = rgb[y0:y0 + ch, x0:x0 + cw]
    out = cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)
    if p.flip:
        out = out[:, ::-1]
    f = out.astype(np.float32)
    f = (f - f.mean()) * p.contrast + f.mean() + p.brightness * 255.0
    out = np.clip(f, 0, 255).astype(np.uint8)
    if p.jpeg_quality:
        ok, enc = cv2.imencode(".jpg", out[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, p.jpeg_quality])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1]
    return np.ascontiguousarray(out)
