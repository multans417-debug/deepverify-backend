"""Pure-ASGI request body limiter: rejects oversized uploads while they stream (honest Content-Length is checked up front;
chunked/lying clients are cut off as soon as the byte count is exceeded), before the body is fully buffered."""
from __future__ import annotations

import json

from deepverify.utils.errors import PayloadTooLarge

_OVERHEAD = 256 * 1024   # multipart framing allowance


class BodySizeLimitMiddleware:
    def __init__(self, app, limits: dict):
        self.app, self.limits = app, limits   # {path: max file bytes}

    async def __call__(self, scope, receive, send):
        limit = self.limits.get(scope.get("path")) if scope["type"] == "http" and scope.get("method") == "POST" else None
        if limit is None:
            return await self.app(scope, receive, send)
        limit += _OVERHEAD
        state = {"seen": 0, "exceeded": False, "forwarded": False}

        async def reject():
            body = json.dumps({"success": False, "error": {"code": PayloadTooLarge.code,
                                                           "message": "Request body exceeds the allowed upload size."}}).encode()
            await send({"type": "http.response.start", "status": 413,
                        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
                                    (b"connection", b"close")]})
            await send({"type": "http.response.body", "body": body})

        declared = dict(scope["headers"]).get(b"content-length")
        if declared and declared.isdigit() and int(declared) > limit:
            return await reject()

        async def counted_receive():
            if state["exceeded"]:
                return {"type": "http.disconnect"}
            msg = await receive()
            if msg["type"] == "http.request":
                state["seen"] += len(msg.get("body", b""))
                if state["seen"] > limit:
                    state["exceeded"] = True          # stop the parser; whatever the app answers next is discarded
                    return {"type": "http.disconnect"}
            return msg

        async def guarded_send(msg):
            if state["exceeded"]:
                return
            state["forwarded"] = state["forwarded"] or msg["type"] == "http.response.start"
            await send(msg)

        try:
            await self.app(scope, counted_receive, guarded_send)
        except Exception:
            if not state["exceeded"]:
                raise
        if state["exceeded"] and not state["forwarded"]:
            await reject()
