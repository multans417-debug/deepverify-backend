"""Domain errors. Each maps to an HTTP status in the API layer."""
from __future__ import annotations


class DeepVerifyError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ModelUnavailable(DeepVerifyError):
    status_code, code = 503, "model_unavailable"


class ServerBusy(DeepVerifyError):
    status_code, code = 503, "server_busy"


class PayloadTooLarge(DeepVerifyError):
    status_code, code = 413, "payload_too_large"


class UnsupportedMedia(DeepVerifyError):
    status_code, code = 415, "unsupported_media_type"


class InvalidMedia(DeepVerifyError):
    status_code, code = 422, "invalid_or_corrupt_media"


class NoFaceFound(DeepVerifyError):
    status_code, code = 422, "no_face_detected"


class VideoTooLong(DeepVerifyError):
    status_code, code = 422, "video_too_long"


class ProcessingTimeout(DeepVerifyError):
    status_code, code = 504, "processing_timeout"


class RateLimited(DeepVerifyError):
    status_code, code = 429, "rate_limited"
