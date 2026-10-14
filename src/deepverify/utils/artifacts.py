"""Artifact file names, hashing and optional verified download at service start-up."""
from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from pathlib import Path

log = logging.getLogger("deepverify.artifacts")

ARTIFACT_FILES = {
    "image": ("image_model.pth", "image_model_metadata.json"),
    "video": ("video_model.pth", "video_model_metadata.json"),
}
# file name -> (URL env var, SHA256 env var)
_ENV = {
    "image_model.pth": ("IMAGE_MODEL_URL", "IMAGE_MODEL_SHA256"),
    "image_model_metadata.json": ("IMAGE_METADATA_URL", "IMAGE_METADATA_SHA256"),
    "video_model.pth": ("VIDEO_MODEL_URL", "VIDEO_MODEL_SHA256"),
    "video_model_metadata.json": ("VIDEO_METADATA_URL", "VIDEO_METADATA_SHA256"),
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def auto_create_safetensors_metadata(kind: str, wpath: Path, mpath: Path) -> None:
    import json
    sha = sha256_file(wpath)
    meta = {
        "kind": kind,
        "architecture": "ms_eff_gcvit_b0",
        "model_version": "1.0.0-kodf",
        "weights_sha256": sha,
        "smoke_test": False,
        "threshold": 0.5,
        "calibration": {
            "temperature": 1.0,
            "threshold": 0.5,
            "method": "none"
        },
        "preprocess": {
            "input_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "use_face_crop": True,
            "face_margin": 0.2,
            "face_detector": "haar"
        },
        "video": {
            "num_frames": 16,
            "min_valid_frames": 4,
            "candidate_multiplier": 2,
            "input_size": 224,
            "face_margin": 0.2,
            "face_detector": "haar"
        } if kind == "video" else None,
        "model_config": {
            "variant": "b0",
            "dataset": "kodf"
        },
        "notes": [f"Loaded from HuggingFace KoreaPeter/ms-eff-gcvit-deepfake-b0-kodf safetensors for {kind} inference"]
    }
    if kind != "video":
        meta.pop("video", None)
    mpath.write_text(json.dumps(meta, indent=2))


def resolve_artifact_paths(kind: str, artifact_dir: Path) -> tuple[Path, Path]:
    artifact_dir = Path(artifact_dir)
    if kind == "image":
        for w_candidate in ("model.safetensors", "image_model.safetensors", "image_model.pth"):
            wpath = artifact_dir / w_candidate
            if wpath.exists():
                mpath = artifact_dir / "image_model_metadata.json"
                if w_candidate.endswith(".safetensors") and not mpath.exists():
                    auto_create_safetensors_metadata("image", wpath, mpath)
                return wpath, mpath
    elif kind == "video":
        for w_candidate in ("video_model.pth", "video_model.safetensors"):
            wpath = artifact_dir / w_candidate
            if wpath.exists():
                mpath = artifact_dir / "video_model_metadata.json"
                if w_candidate.endswith(".safetensors") and not mpath.exists():
                    auto_create_safetensors_metadata("video", wpath, mpath)
                return wpath, mpath
        # Fallback for video inference using model.safetensors checkpoint
        for w_candidate in ("model.safetensors", "image_model.safetensors"):
            wpath = artifact_dir / w_candidate
            if wpath.exists():
                mpath = artifact_dir / "video_model_metadata.json"
                if not mpath.exists():
                    auto_create_safetensors_metadata("video", wpath, mpath)
                return wpath, mpath
    weights_name, meta_name = ARTIFACT_FILES[kind]
    return artifact_dir / weights_name, artifact_dir / meta_name



def ensure_artifacts(artifact_dir: Path) -> list:
    """Download any missing artifact whose URL env var is set. SHA256 is mandatory. Returns messages."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    notes: list = []
    for name, (url_var, sha_var) in _ENV.items():
        dest = artifact_dir / name
        url, expected = os.getenv(url_var, "").strip(), os.getenv(sha_var, "").strip().lower()
        if dest.exists() or not url:
            continue
        if not url.startswith("https://"):
            notes.append(f"{name}: refusing non-https URL")
            continue
        if not expected:
            notes.append(f"{name}: {sha_var} is required to download; skipped")
            continue
        part = dest.with_suffix(dest.suffix + ".part")
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, open(part, "wb") as out:  # https enforced above
                while block := resp.read(1 << 20):
                    out.write(block)
            if sha256_file(part) != expected:
                part.unlink(missing_ok=True)
                notes.append(f"{name}: SHA256 mismatch, download discarded")
                continue
            os.replace(part, dest)
            notes.append(f"{name}: downloaded and verified")
        except Exception as exc:  # network errors must not crash start-up
            part.unlink(missing_ok=True)
            notes.append(f"{name}: download failed ({type(exc).__name__})")
    for n in notes:
        log.warning(n)
    return notes

