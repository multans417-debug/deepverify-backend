"""Face detection. OpenCV Haar cascade (ships inside opencv, no model download, CPU only).
It is weaker than modern CNN detectors: it misses profile/occluded/small faces. Documented limitation."""
from __future__ import annotations

import cv2
import numpy as np


class HaarFaceDetector:
    name = "haar"

    def __init__(self, min_size: int = 48, scale_factor: float = 1.1, min_neighbors: int = 5, max_side: int = 640):
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise RuntimeError("Could not load OpenCV Haar cascade for faces.")
        self.min_size, self.scale_factor, self.min_neighbors, self.max_side = min_size, scale_factor, min_neighbors, max_side

    def config(self) -> dict:
        return {"name": self.name, "min_size": self.min_size, "scale_factor": self.scale_factor,
                "min_neighbors": self.min_neighbors, "max_side": self.max_side}

    def detect_largest(self, rgb: np.ndarray):
        """Return (x, y, w, h) in original-image pixels for the largest face, or None."""
        h, w = rgb.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        small = cv2.resize(rgb, (int(w * scale), int(h * scale))) if scale < 1.0 else rgb
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        faces = self._cascade.detectMultiScale(gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors,
                                               minSize=(self.min_size, self.min_size))
        if len(faces) == 0:
            return None
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        return int(x / scale), int(y / scale), int(fw / scale), int(fh / scale)


class CenterCropDetector:
    """SMOKE-TEST ONLY: pretends the centre of the frame is a face. Never use for real detection."""
    name = "center"

    def config(self) -> dict:
        return {"name": self.name, "warning": "not a face detector"}

    def detect_largest(self, rgb: np.ndarray):
        h, w = rgb.shape[:2]
        s = int(min(h, w) * 0.7)
        return (w - s) // 2, (h - s) // 2, s, s


def make_detector(name: str):
    if name == "haar":
        return HaarFaceDetector()
    if name == "center":
        return CenterCropDetector()
    raise ValueError(f"Unknown face detector {name!r}")


def crop_square(rgb: np.ndarray, box, margin: float, size: int) -> np.ndarray:
    """Expand the box by `margin`, make it square, clamp to the image, resize to size x size."""
    x, y, w, h = box
    side = int(max(w, h) * (1.0 + margin))
    cx, cy = x + w / 2.0, y + h / 2.0
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    H, W = rgb.shape[:2]
    x0, y0 = max(0, min(x0, W - 1)), max(0, min(y0, H - 1))
    x1, y1 = min(W, x0 + side), min(H, y0 + side)
    crop = rgb[y0:y1, x0:x1]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
