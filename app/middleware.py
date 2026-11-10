"""Pure-ASGI middleware: request id, access log with redacted paths, request-size limit.

Implemented without BaseHTTPMiddleware so streaming responses and disconnect detection are untouched.
"""

from __future__ import annotations

import logging
import re
import time
import uuid

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.errors import OpenAIError
from app.log import redact, request_id_var

access_log = logging.getLogger("app.access")

_REQUEST_ID = re.compile(r"[A-Za-z0-9._\-]{1,64}")


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        incoming = headers.get("x-request-id", "")
        request_id = incoming if _REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status = 0

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                MutableHeaders(scope=message).append("X-Request-Id", request_id)
            await send(message)

        try:
            length = headers.get("content-length", "")
            if length.isdigit() and int(length) > self.max_body_bytes:
                response = OpenAIError(413, f"Request body exceeds {self.max_body_bytes} bytes.",
                                       code="request_too_large").response()
                await response(scope, receive, send_wrapper)
                return
            await self.app(scope, receive, send_wrapper)
        finally:
            access_log.info(
                "%s %s %s", scope.get("method"), redact(scope.get("path", "")), status,
                extra={"duration_ms": round((time.perf_counter() - started) * 1000, 1)},
            )
            request_id_var.reset(token)
