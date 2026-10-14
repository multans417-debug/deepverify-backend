from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from deepverify.schemas.responses import DetectionResponse, HealthResponse, ModelsResponse, ReadyResponse
from deepverify.services import inference
from deepverify.utils.errors import RateLimited, ServerBusy
from deepverify.utils.sniff import sniff_image_format, sniff_video_container
from deepverify.utils.uploads import save_upload

router = APIRouter()


def _client_ip(request: Request) -> str:
    if request.app.state.settings.trust_proxy_headers:
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def rate_limit(request: Request) -> None:
    if not request.app.state.limiter.allow(_client_ip(request)):
        raise RateLimited("Too many requests; please slow down and retry shortly.")


async def _run_exclusive(state, fn, *args):
    """At most `max_concurrent_inferences` heavy jobs at once; others wait briefly, then get 503 (never unbounded queues)."""
    try:
        await asyncio.wait_for(state.semaphore.acquire(), timeout=state.settings.queue_timeout_seconds)
    except asyncio.TimeoutError:
        raise ServerBusy("The server is busy with other analyses; please retry shortly.")
    try:
        return await run_in_threadpool(fn, *args)
    finally:
        state.semaphore.release()


async def _detect(request: Request, file: UploadFile, media: str) -> DetectionResponse:
    state = request.app.state
    s = state.settings
    bundle = state.registry.get(media)                       # 503 before any upload bytes are processed
    limit, sniff = (s.max_image_bytes, sniff_image_format) if media == "image" else (s.max_video_bytes, sniff_video_container)
    fn = inference.detect_image if media == "image" else inference.detect_video
    with tempfile.TemporaryDirectory(prefix="dv_", dir=s.tmp_dir) as tmp:   # removed even on error/timeouts
        path, _fmt = await save_upload(file, Path(tmp), limit, sniff, media)
        result = await _run_exclusive(state, fn, path, bundle, s)
    return DetectionResponse(**result)


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health(request: Request):
    s = request.app.state.settings
    return HealthResponse(status="ok", service=s.app_name, version=s.api_version)


@router.get("/ready", response_model=ReadyResponse, tags=["system"])
async def ready(request: Request):
    reg = request.app.state.registry
    body = ReadyResponse(ready=reg.ready(), required_models=list(reg.settings.required_models), models=reg.describe())
    return JSONResponse(status_code=200 if body.ready else 503, content=body.model_dump())


@router.get("/api/v1/models", response_model=ModelsResponse, tags=["models"])
async def models(request: Request):
    return ModelsResponse(models=request.app.state.registry.describe())


@router.post("/api/v1/detect/image", response_model=DetectionResponse, tags=["detection"], dependencies=[Depends(rate_limit)])
async def detect_image(request: Request, file: UploadFile = File(...)):
    return await _detect(request, file, "image")


@router.post("/api/v1/detect/video", response_model=DetectionResponse, tags=["detection"], dependencies=[Depends(rate_limit)])
async def detect_video(request: Request, file: UploadFile = File(...)):
    return await _detect(request, file, "video")
