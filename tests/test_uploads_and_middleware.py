"""Pure tests (no fastapi/torch): streaming upload validation and the body-size ASGI middleware."""
import asyncio
import json

import pytest

from deepverify.api.middleware import BodySizeLimitMiddleware
from deepverify.utils.errors import InvalidMedia, PayloadTooLarge, UnsupportedMedia
from deepverify.utils.sniff import sniff_image_format
from deepverify.utils.uploads import save_upload

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


class FakeUpload:
    def __init__(self, data: bytes, chunk=1 << 20):
        self._d, self._c = data, chunk

    async def read(self, n=-1):
        out, self._d = self._d[: min(n, self._c)], self._d[min(n, self._c):]
        return out


def run(coro):
    return asyncio.run(coro)


def test_save_upload_accepts_valid_and_ignores_client_name(tmp_path):
    path, fmt = run(save_upload(FakeUpload(PNG), tmp_path, 10_000, sniff_image_format, "image"))
    assert fmt == "png" and path.name == "upload.bin" and path.read_bytes() == PNG   # fixed server-side file name


def test_save_upload_rejects_wrong_type_even_if_named_png(tmp_path):
    with pytest.raises(UnsupportedMedia):
        run(save_upload(FakeUpload(b"MZ executable pretending to be a .png" + b"\0" * 40), tmp_path, 10_000, sniff_image_format, "image"))


def test_save_upload_enforces_size_while_streaming(tmp_path):
    big = PNG + b"\x00" * 5000
    with pytest.raises(PayloadTooLarge):
        run(save_upload(FakeUpload(big, chunk=1000), tmp_path, 3000, sniff_image_format, "image"))


def test_save_upload_rejects_empty(tmp_path):
    with pytest.raises(InvalidMedia):
        run(save_upload(FakeUpload(b""), tmp_path, 1000, sniff_image_format, "image"))


async def _inner(scope, receive, send):
    while True:
        m = await receive()
        if m["type"] == "http.disconnect":
            raise RuntimeError("disconnected")
        if not m.get("more_body"):
            break
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


async def _drive(headers, chunks, path="/api/v1/detect/image"):
    sent, queue = [], list(chunks)

    async def receive():
        if not queue:
            return {"type": "http.disconnect"}
        b = queue.pop(0)
        return {"type": "http.request", "body": b, "more_body": bool(queue)}

    async def send(m):
        sent.append(m)

    app = BodySizeLimitMiddleware(_inner, {"/api/v1/detect/image": 1000})
    await app({"type": "http", "method": "POST", "path": path, "headers": headers}, receive, send)
    return sent


def test_middleware_allows_small_and_unlisted_paths():
    assert run(_drive([(b"content-length", b"500")], [b"x" * 500]))[0]["status"] == 200
    assert run(_drive([], [b"x" * 900_000], path="/health"))[0]["status"] == 200


def test_middleware_rejects_declared_oversize():
    sent = run(_drive([(b"content-length", b"999999")], [b"x" * 10]))
    assert sent[0]["status"] == 413 and json.loads(sent[1]["body"])["error"]["code"] == "payload_too_large"


def test_middleware_cuts_off_chunked_oversize():
    assert run(_drive([], [b"x" * 200_000, b"x" * 200_000]))[0]["status"] == 413
