from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from deepverify.api.middleware import BodySizeLimitMiddleware
from deepverify.api.routes import router
from deepverify.config import Settings
from deepverify.services.registry import ModelRegistry
from deepverify.utils.errors import DeepVerifyError
from deepverify.utils.ratelimit import SlidingWindowLimiter

log = logging.getLogger("deepverify")


def _err(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"success": False, "error": {"code": code, "message": message}})


def create_app(settings: Settings | None = None, registry: ModelRegistry | None = None, load_models: bool = True) -> FastAPI:
    settings = settings or Settings.from_env()
    registry = registry or ModelRegistry(settings)
    if settings.tmp_dir:
        settings.tmp_dir.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_models:
            await run_in_threadpool(registry.load_all)
        yield

    app = FastAPI(title="DeepVerify API", version=settings.api_version, lifespan=lifespan,
                  description="Deepfake detection for images and videos. Results are probabilistic, not proof.")
    app.state.settings, app.state.registry = settings, registry
    app.state.semaphore = asyncio.Semaphore(settings.max_concurrent_inferences)
    app.state.limiter = SlidingWindowLimiter(settings.rate_limit_per_minute)

    app.add_middleware(BodySizeLimitMiddleware, limits={"/api/v1/detect/image": settings.max_image_bytes,
                                                        "/api/v1/detect/video": settings.max_video_bytes})
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["GET", "POST", "OPTIONS"],
                       allow_headers=["*"], allow_credentials=False, max_age=600)

    @app.exception_handler(DeepVerifyError)
    async def _domain(_: Request, exc: DeepVerifyError):
        return _err(exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        return _err(422, "invalid_request", "Invalid request: a multipart form field named 'file' is required.")

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        log.exception("Unhandled error (%s)", type(exc).__name__)   # no request bodies / file names are logged
        return _err(500, "internal_error", "An unexpected error occurred while processing the request.")

    app.include_router(router)
    return app


app = create_app()
