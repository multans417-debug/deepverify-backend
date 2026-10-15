"""Load a trained model + its metadata for inference. Weights are state_dicts loaded with weights_only=True."""
from __future__ import annotations

import gc
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch

from deepverify.models import gcvit_model, image_model, video_model
from deepverify.preprocessing.image import PreprocessConfig
from deepverify.preprocessing.video import VideoConfig
from deepverify.utils.artifacts import ARTIFACT_FILES, resolve_artifact_paths, sha256_file

log = logging.getLogger("deepverify.loader")


@dataclass
class ModelBundle:
    kind: str
    model: torch.nn.Module
    metadata: dict
    preprocess: PreprocessConfig
    video: VideoConfig | None
    threshold: float
    temperature: float
    calibrated: bool
    smoke_test: bool
    version: str = ""


def build_from_metadata(meta: dict) -> torch.nn.Module:
    cfg, arch = meta.get("model_config", {}), meta["architecture"]
    if arch == image_model.ARCH:
        return image_model.ImageDeepfakeNet(pretrained=False, dropout=cfg.get("dropout", 0.3))
    if arch == video_model.ARCH:
        return video_model.VideoDeepfakeNet(hidden=cfg.get("hidden", 128), dropout=cfg.get("dropout", 0.3), pretrained=False)
    if arch in (gcvit_model.ARCH, "ms_eff_gcvit", "ms_eff_gcvit_b0"):
        return gcvit_model.GCViTDeepfakeNet(variant=cfg.get("variant", "b0"), dataset=cfg.get("dataset", "kodf"))
    raise ValueError(f"Unknown architecture {arch!r}")


def load_bundle(
    kind: str,
    artifact_dir: Path,
    allow_smoke: bool = False,
    model_cache: dict[tuple[str, str], torch.nn.Module] | None = None,
) -> ModelBundle:
    wpath, mpath = resolve_artifact_paths(kind, artifact_dir)
    if not wpath.exists() or not mpath.exists():
        weights_name, meta_name = ARTIFACT_FILES[kind]
        raise FileNotFoundError(f"Missing trained artifacts for the {kind} model ({weights_name}, {meta_name}).")
    meta = json.loads(mpath.read_text())
    if meta.get("kind") != kind:
        raise ValueError(f"{mpath.name} describes a {meta.get('kind')!r} model, expected {kind!r}.")
    if meta.get("smoke_test") and not allow_smoke:
        raise ValueError(f"The {kind} model was produced by a smoke-test run and is refused (set ALLOW_SMOKE_MODELS=true to override in tests only).")
    sha = sha256_file(wpath)
    if sha != meta.get("weights_sha256"):
        raise ValueError(f"{wpath.name} does not match the SHA256 recorded in {mpath.name}; refusing to load.")

    arch = meta.get("architecture", "")
    cache_key = (arch, sha)
    if model_cache is not None and cache_key in model_cache:
        model = model_cache[cache_key]
        log.info("Reusing in-memory %s model for %s (sha256=%s...)", arch, kind, sha[:8])
    else:
        model = build_from_metadata(meta)
        if wpath.suffix == ".safetensors":
            from safetensors.torch import load_file
            raw_sd = load_file(str(wpath))
            clean_sd = {k.removeprefix("model."): v for k, v in raw_sd.items() if k.startswith("model.") or not k.startswith("model.")}
            if hasattr(model, "net"):
                model.net.load_state_dict(clean_sd)
            else:
                model.load_state_dict(clean_sd)
            del raw_sd, clean_sd
        else:
            model.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True))
        model.eval()
        model.requires_grad_(False)
        if model_cache is not None:
            model_cache[cache_key] = model
        gc.collect()
    cal = meta.get("calibration", {})
    return ModelBundle(
        kind=kind, model=model, metadata=meta, preprocess=PreprocessConfig.from_dict(meta["preprocess"]),
        video=VideoConfig.from_dict(meta["video"]) if kind == "video" else None,
        threshold=float(meta["threshold"]), temperature=float(cal.get("temperature", 1.0)),
        calibrated=cal.get("method") == "temperature_scaling", smoke_test=bool(meta.get("smoke_test")),
        version=str(meta.get("model_version", "")),
    )

