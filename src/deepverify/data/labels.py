"""Label discovery from VERIFIED sources only: documented metadata or unambiguous directory names.
Anything ambiguous is excluded and reported - labels are never guessed."""
from __future__ import annotations

import json
import os
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

REAL_PARTS = {"real", "reals", "original", "originals", "original_sequences", "pristine", "celeb-real",
              "youtube-real", "youtube_real", "authentic"}
FAKE_PARTS = {"fake", "fakes", "manipulated", "manipulated_sequences", "deepfakes", "face2face", "faceswap",
              "neuraltextures", "faceshifter", "celeb-synthesis", "synthetic", "deepfake"}
SPLIT_PARTS = {"train": "train", "training": "train", "valid": "val", "val": "val", "validation": "val",
               "test": "test", "testing": "test"}


def label_from_path(path: Path, root: Path) -> str | None:
    """'REAL' / 'FAKE' from directory names relative to root; None when absent or contradictory."""
    parts = {p.lower() for p in path.relative_to(root).parts[:-1]}
    is_real, is_fake = bool(parts & REAL_PARTS), bool(parts & FAKE_PARTS)
    if is_real == is_fake:
        return None
    return "REAL" if is_real else "FAKE"


def split_from_path(path: Path, root: Path) -> str | None:
    for p in path.relative_to(root).parts[:-1]:
        if p.lower() in SPLIT_PARTS:
            return SPLIT_PARTS[p.lower()]
    return None


def load_dfdc_metadata(root: Path) -> dict:
    """DFDC-style metadata.json: {"x.mp4": {"label": "FAKE", "original": "y.mp4", ...}}. Merges all found."""
    meta: dict = {}
    for mp in root.rglob("metadata.json"):
        try:
            data = json.loads(mp.read_text())
        except Exception:
            continue
        if isinstance(data, dict):
            for name, rec in data.items():
                if isinstance(rec, dict) and str(rec.get("label", "")).upper() in {"REAL", "FAKE"}:
                    meta[(mp.parent, name)] = rec
    return meta


def discover(root: Path, exts: set, use_metadata: bool = True) -> tuple:
    """Return (items, exclusions). item: dict(path, label, stem, original, preset_split)."""
    root = Path(root)
    meta = load_dfdc_metadata(root) if use_metadata else {}
    items, excluded = [], []
    for dirpath, _, files in os.walk(root):
        for fn in sorted(files):
            p = Path(dirpath) / fn
            if p.suffix.lower() not in exts:
                continue
            rec = meta.get((p.parent, fn))
            if rec is not None:
                label, original = str(rec["label"]).upper(), rec.get("original")
                source = "metadata.json"
            else:
                label, original, source = label_from_path(p, root), None, "directory"
            if label is None:
                excluded.append({"path": str(p), "reason": "ambiguous_or_missing_label"})
                continue
            items.append({"path": str(p), "label": label, "stem": p.stem, "original": original,
                          "preset_split": split_from_path(p, root), "label_source": source})
    return items, excluded
