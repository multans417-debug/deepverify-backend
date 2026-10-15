"""Runtime configuration read from environment variables. No secrets live in code."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _list(name: str, default: str) -> tuple[str, ...]:
    raw = os.getenv(name, default)
    return tuple(p.strip().rstrip("/") for p in raw.split(",") if p.strip())


@dataclass(frozen=True)
class Settings:
    artifact_dir: Path
    cors_origins: tuple[str, ...]
    required_models: tuple[str, ...]
    max_image_bytes: int
    max_video_bytes: int
    max_image_pixels: int
    max_video_seconds: int
    extract_timeout_seconds: int
    max_concurrent_inferences: int
    queue_timeout_seconds: int
    rate_limit_per_minute: int
    trust_proxy_headers: bool
    torch_threads: int
    allow_smoke_models: bool
    torch_interop_threads: int = 1
    tmp_dir: Path | None = None
    hf_token: str | None = None
    app_name: str = "DeepVerify"
    api_version: str = "1.0.0"

    @classmethod
    def from_env(cls) -> "Settings":
        tmp = os.getenv("UPLOAD_TMP_DIR")
        return cls(
            artifact_dir=Path(os.getenv("ARTIFACT_DIR", "artifacts")),
            cors_origins=_list(
                "CORS_ORIGINS",
                "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,https://deepverify-frontend-zu2d.onrender.com",
            ),
            required_models=_list("REQUIRED_MODELS", "image,video"),
            max_image_bytes=_int("MAX_IMAGE_BYTES", 10 * 1024 * 1024),
            max_video_bytes=_int("MAX_VIDEO_BYTES", 100 * 1024 * 1024),
            max_image_pixels=_int("MAX_IMAGE_PIXELS", 40_000_000),
            max_video_seconds=_int("MAX_VIDEO_SECONDS", 60),
            extract_timeout_seconds=_int("EXTRACT_TIMEOUT_SECONDS", 45),
            max_concurrent_inferences=max(1, _int("MAX_CONCURRENT_INFERENCES", 1)),
            queue_timeout_seconds=_int("QUEUE_TIMEOUT_SECONDS", 20),
            rate_limit_per_minute=_int("RATE_LIMIT_PER_MINUTE", 20),
            trust_proxy_headers=_bool("TRUST_PROXY_HEADERS", False),
            torch_threads=_int("TORCH_THREADS", 1),
            allow_smoke_models=_bool("ALLOW_SMOKE_MODELS", False),
            torch_interop_threads=_int("TORCH_INTEROP_THREADS", 1),
            tmp_dir=Path(tmp) if tmp else None,
            hf_token=os.getenv("HF_TOKEN"),
        )
