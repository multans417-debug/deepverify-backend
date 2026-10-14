from __future__ import annotations

import json
import os
import platform
import random
import time
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np

PACKAGES = ["torch", "torchvision", "numpy", "opencv-python-headless", "opencv-python", "pillow", "scikit-learn", "scipy"]


def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def pick_device(arg: str = "auto"):
    import torch
    if arg == "auto":
        arg = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(arg)


def dependency_versions() -> dict:
    out = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            out[p] = importlib_metadata.version(p)
        except importlib_metadata.PackageNotFoundError:
            pass
    return out


def hardware_info(device) -> dict:
    import torch
    info = {"platform": platform.platform(), "cpu_count": os.cpu_count(), "device": str(device)}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    return info


def subsample(rows: list, max_n: int | None, seed: int) -> list:
    """Class-balanced-preserving seeded cap (keeps the original class ratio)."""
    if not max_n or len(rows) <= max_n:
        return rows
    rng = random.Random(seed)
    out = []
    for lab in (0, 1):
        sub = [r for r in rows if r["label"] == lab]
        rng.shuffle(sub)
        out += sub[: max(1, round(max_n * len(sub) / len(rows)))]
    return out


def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
