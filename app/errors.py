"""OpenAI-style error envelope (`{"error": {...}}`) and the FastAPI exception handlers."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger(__name__)


class OpenAIError(Exception):
    """An error rendered to clients in the OpenAI error format."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        type_: str = "invalid_request_error",
        code: str | None = None,
        param: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.type = type_
        self.code = code
        self.param = param
        self.headers = headers or {}

    def body(self) -> dict[str, Any]:
        return {"error": {"message": self.message, "type": self.type, "param": self.param, "code": self.code}}

    def response(self) -> JSONResponse:
        return JSONResponse(self.body(), status_code=self.status_code, headers=self.headers)

    def __repr__(self) -> str:
        return f"OpenAIError({self.status_code}, {self.code!r}, {self.message!r})"


def invalid_request(message: str, *, code: str | None = None, param: str | None = None) -> OpenAIError:
    return OpenAIError(400, message, code=code, param=param)


def authentication_error(message: str = "Incorrect API key provided.") -> OpenAIError:
    return OpenAIError(401, message, code="invalid_api_key", headers={"WWW-Authenticate": "Bearer"})


def not_found(message: str, *, code: str | None = None, param: str | None = None) -> OpenAIError:
    return OpenAIError(404, message, code=code, param=param)


def rate_limited(
    message: str, *, code: str = "rate_limit_exceeded", retry_after: float | None = None
) -> OpenAIError:
    headers = {"Retry-After": str(max(1, int(round(retry_after))))} if retry_after else None
    return OpenAIError(429, message, type_="rate_limit_error", code=code, headers=headers)


def insufficient_quota(message: str) -> OpenAIError:
    return OpenAIError(402, message, type_="insufficient_quota", code="insufficient_quota")


def upstream_error(message: str, *, status: int = 502, code: str = "upstream_error") -> OpenAIError:
    return OpenAIError(status, message, type_="api_error", code=code)


def _type_for_status(status: int) -> str:
    if status == 401:
        return "authentication_error"
    if status == 403:
        return "permission_error"
    if status == 429:
        return "rate_limit_error"
    if status >= 500:
        return "api_error"
    return "invalid_request_error"


async def _openai_error_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, OpenAIError)
    return exc.response()


async def _validation_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    errors = exc.errors()
    param = None
    message = "Invalid request body."
    if errors:
        first = errors[0]
        loc = [str(part) for part in first.get("loc", ()) if part != "body"]
        param = ".".join(loc) or None
        message = f"{first.get('msg', 'Invalid value')}" + (f" (at '{param}')" if param else "")
    return OpenAIError(400, message, param=param).response()


async def _http_error_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    message = exc.detail if isinstance(exc.detail, str) else "Request failed."
    if exc.status_code == 404:
        message = "Unknown request URL. This proxy implements /v1/chat/completions and /v1/models."
    err = OpenAIError(exc.status_code, message, type_=_type_for_status(exc.status_code))
    return JSONResponse(err.body(), status_code=exc.status_code, headers=getattr(exc, "headers", None))


async def _unhandled_handler(_: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error", exc_info=exc)
    return OpenAIError(500, "Internal server error.", type_="api_error", code="internal_error").response()


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(OpenAIError, _openai_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_handler)
    app.add_exception_handler(StarletteHTTPException, _http_error_handler)
    app.add_exception_handler(Exception, _unhandled_handler)
