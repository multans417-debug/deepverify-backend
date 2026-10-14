from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from deepverify.preprocessing.augment import apply_params, draw_params
from deepverify.preprocessing.image import PreprocessConfig, load_rgb, preprocess_rgb


class ImageManifestDataset(Dataset):
    """Rows: {path,label,...}. Augmentation only when train=True. Deterministic per (seed, epoch, index)."""

    def __init__(self, rows: list, cfg: PreprocessConfig, train: bool, seed: int = 42):
        self.rows, self.cfg, self.train, self.seed, self.epoch = rows, cfg, train, seed, 0

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        rgb = load_rgb(r["path"])
        if self.train:
            rgb = apply_params(rgb, draw_params(np.random.default_rng([self.seed, self.epoch, i])))
        return torch.from_numpy(preprocess_rgb(rgb, self.cfg)), torch.tensor(float(r["label"]))


class VideoFramesDataset(Dataset):
    """Rows from video_manifest.csv. Loads the cached face-crop PNGs (T per video) as a (T,3,H,W) clip.
    Augmentation parameters are drawn once per clip so all frames get the same transform."""

    def __init__(self, rows: list, cfg: PreprocessConfig, num_frames: int, train: bool, seed: int = 42):
        self.rows, self.cfg, self.T, self.train, self.seed, self.epoch = rows, cfg, num_frames, train, seed, 0

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        files = sorted(Path(r["frames_dir"]).glob("*.png"))
        if len(files) != self.T:
            raise RuntimeError(f"{r['frames_dir']}: expected {self.T} frames, found {len(files)} - re-run prepare_dataset.py")
        params = draw_params(np.random.default_rng([self.seed, self.epoch, i])) if self.train else None
        frames = []
        for f in files:
            rgb = load_rgb(f)
            if params is not None:
                rgb = apply_params(rgb, params)
            frames.append(preprocess_rgb(rgb, self.cfg))
        return torch.from_numpy(np.stack(frames)), torch.tensor(float(r["label"]))
