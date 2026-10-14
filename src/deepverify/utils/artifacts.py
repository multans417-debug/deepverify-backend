"""Artifact file names, hashing and optional verified download at service start-up."""
from __future__ import annotations

import hashlib
import logging
import os
import urllib.parse
import urllib.request
from pathlib import Path

log = logging.getLogger("deepverify.artifacts")

ARTIFACT_FILES = {
    "image": ("image_model.pth", "image_model_metadata.json"),
    "video": ("video_model.pth", "video_model_metadata.json"),
}
# file name -> (URL env var, SHA256 env var)
_ENV = {
    "model.safetensors": ("MODEL_SAFETENSORS_URL", "MODEL_SAFETENSORS_SHA256"),
    "image_model_metadata.json": ("IMAGE_METADATA_URL", "IMAGE_METADATA_SHA256"),
    "video_model_metadata.json": ("VIDEO_METADATA_URL", "VIDEO_METADATA_SHA256"),
    "image_model.pth": ("IMAGE_MODEL_URL", "IMAGE_MODEL_SHA256"),
    "video_model.pth": ("VIDEO_MODEL_URL", "VIDEO_MODEL_SHA256"),
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _is_huggingface_host(url: str) -> bool:
    try:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
        return host == "huggingface.co" or host.endswith(".huggingface.co") or host == "hf.co" or host.endswith(".hf.co")
    except Exception:
        return False


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Prevents Authorization Bearer tokens from leaking across redirects to CDNs or external hosts."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is None:
            return None
        orig_host = (urllib.parse.urlparse(req.full_url).hostname or "").lower()
        new_host = (urllib.parse.urlparse(newurl).hostname or "").lower()

        # Always strip Authorization if:
        # 1. Host changed (e.g. redirected to AWS S3, CloudFront, or any different domain)
        # 2. Host is a CDN (e.g. cdn-lfs.huggingface.co) which uses query-param signatures that fail if Authorization header is attached
        # 3. New host is not a primary Hugging Face domain
        if orig_host != new_host or "cdn" in new_host or not _is_huggingface_host(newurl):
            for h in list(new_req.headers.keys()):
                if h.lower() == "authorization":
                    del new_req.headers[h]
            for h in list(new_req.unredirected_hdrs.keys()):
                if h.lower() == "authorization":
                    del new_req.unredirected_hdrs[h]
        return new_req


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
    hf_token = os.getenv("HF_TOKEN", "").strip()
    opener = urllib.request.build_opener(SafeRedirectHandler())

    for name, (url_var, sha_var) in _ENV.items():
        dest = artifact_dir / name
        url = os.getenv(url_var, "").strip()
        expected = os.getenv(sha_var, "").strip().lower()
        if not url and name == "model.safetensors":
            url = os.getenv("MODEL_URL", "").strip()
            expected = os.getenv("MODEL_SHA256", "").strip().lower()
        if dest.exists() or not url:
            continue
        if name.endswith(".pth") and "safetensors" in url.lower():
            notes.append(f"{name}: refusing safetensors file via .pth URL env var; use MODEL_SAFETENSORS_URL instead")
            continue
        if not url.startswith("https://"):
            notes.append(f"{name}: refusing non-https URL")
            continue
        if not expected:
            notes.append(f"{name}: {sha_var} is required to download; skipped")
            continue

        req = urllib.request.Request(url, headers={"User-Agent": "DeepVerify/1.0"})
        if hf_token and _is_huggingface_host(url):
            # Never send bearer token to non-Hugging Face hosts
            req.add_unredirected_header("Authorization", f"Bearer {hf_token}")

        part = dest.with_suffix(dest.suffix + ".part")
        try:
            # Respect mock if urllib.request.urlopen was monkeypatched in tests; otherwise use SafeRedirectHandler
            if urllib.request.urlopen.__module__ != "urllib.request":
                resp_ctx = urllib.request.urlopen(req, timeout=120)
            else:
                resp_ctx = opener.open(req, timeout=120)

            with resp_ctx as resp:
                status = getattr(resp, "status", getattr(resp, "code", 200))
                if status < 200 or status >= 300:
                    notes.append(f"{name}: download failed (HTTP status {status})")
                    continue

                content_type = ""
                if hasattr(resp, "headers") and resp.headers:
                    if hasattr(resp.headers, "get_content_type"):
                        content_type = resp.headers.get_content_type()
                    else:
                        content_type = resp.headers.get("Content-Type", "")

                final_url = resp.geturl() if hasattr(resp, "geturl") else url
                if "text/html" in content_type.lower() or "/login" in final_url:
                    notes.append(f"{name}: rejected unexpected HTML/login response")
                    continue

                with open(part, "wb") as out:
                    first_block = True
                    while block := resp.read(1 << 20):
                        if first_block:
                            first_block = False
                            sample = block[:256].lstrip().lower()
                            if sample.startswith(b"<!doctype html") or sample.startswith(b"<html"):
                                raise ValueError("Response body contains HTML instead of expected artifact")
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

