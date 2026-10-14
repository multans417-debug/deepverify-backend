"""Image loading + preprocessing. The SAME functions are used for training and for serving."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from deepverify.utils.errors import InvalidMedia

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PreprocessConfig:
    input_size: int = 224
    mean: tuple = IMAGENET_MEAN
    std: tuple = IMAGENET_STD
    use_face_crop: bool = False      # crop the largest detected face before resizing
    face_margin: float = 0.3
    face_detector: str = "haar"      # "haar" | "center" (center is for smoke tests only)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessConfig":
        fields = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        for k in ("mean", "std"):
            if k in fields:
                fields[k] = tuple(fields[k])
        return cls(**fields)


def load_rgb(path, max_pixels: int = 40_000_000) -> np.ndarray:
    """Decode an image file to an RGB uint8 array (EXIF-orientation applied). Raises InvalidMedia."""
    try:
        with Image.open(path) as im:
            if im.width * im.height > max_pixels:
                raise InvalidMedia("Image dimensions exceed the allowed pixel limit.")
            im = ImageOps.exif_transpose(im)
            im.load()
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
    except InvalidMedia:
        raise
    except Exception as exc:
        raise InvalidMedia("File could not be decoded as an image.") from exc


def preprocess_rgb(rgb: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """RGB uint8 HxWx3 -> float32 CxHxW, resized + normalised exactly as in training."""
    s = cfg.input_size
    if rgb.shape[0] != s or rgb.shape[1] != s:
        rgb = cv2.resize(rgb, (s, s), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32) / 255.0
    x = (x - np.asarray(cfg.mean, np.float32)) / np.asarray(cfg.std, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))
