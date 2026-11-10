"""Logging: JSON or text output, request-id propagation and redaction of secrets."""

from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
import time
from typing import Any

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"sk-hop-[A-Za-z0-9]{4,}"), "sk-hop-***"),
    (re.compile(r"hop_[A-Za-z0-9_\-]{6,}"), "hop_***"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=\-]{8,}"), r"\1***"),
]

_STANDARD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime", "request_id"}
_MARKER = "_hoplite_proxy_handler"


def redact(text: str) -> str:
    """Mask Hoplite keys, proxy keys and bearer tokens in free text."""
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _clean(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    return value


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    return {
        key: _clean(value)
        for key, value in record.__dict__.items()
        if key not in _STANDARD_ATTRS and not key.startswith("_")
    }


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "msg": redact(record.getMessage()),
        }
        payload.update(_extras(record))
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)s [%(request_id)s] %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        extras = _extras(record)
        if extras:
            line += " " + " ".join(f"{k}={json.dumps(v, ensure_ascii=False, default=str)}" for k, v in extras.items())
        return redact(line)


def setup_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Install (or re-install) the proxy's stdout handler on the root logger."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _MARKER, False):
            root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    setattr(handler, _MARKER, True)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())

    for noisy in ("httpx", "httpcore", "aiosqlite"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Route uvicorn's own logs through our handler; the access log is replaced by our middleware.
    for name in ("uvicorn", "uvicorn.error"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers.clear()
        uv_logger.propagate = True
    access = logging.getLogger("uvicorn.access")
    access.handlers.clear()
    access.disabled = True
