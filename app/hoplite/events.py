"""Consumer for Hoplite's `GET /api/events/stream` (server-sent events).

Two kinds of frames arrive on a thread-filtered stream:
* durable events (`run.completed`, `message.persisted`, ...): the SSE `id:` is a replay cursor
  that is sent back as `Last-Event-ID` when reconnecting;
* draft frames: full snapshots of the assistant's in-progress reply (`messageId`, `revision`),
  without a cursor or replay guarantee; `draft.removed` clears one.

Observed on the live API (2026-11):

    event: draft
    data: {"messageId":"msg_…","runId":"run_…","streamId":"assistant:s0","revision":4,"content":"<full text so far>"}

    event: draft.removed
    data: {"messageId":"msg_…"}

    event: run.completed
    data: {"id":"…","type":"run.completed","runId":"run_…","data":{"id":"run_…","status":"completed"},"cursor":"…",…}
    id: <cursor>

Revisions may skip numbers, and no `message.persisted` event is sent for the assistant reply before
`run.completed`, so the final text is read from the message list once the run settles. Hoplite does
not document the draft layout, so `classify` also accepts a few other shapes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx
from httpx_sse import SSEError, aconnect_sse

from app.hoplite.client import RETRYABLE_STATUS, HopliteClient, error_from_response

log = logging.getLogger(__name__)

TEXT_KEYS = ("content", "text", "markdown", "body", "delta")
HEARTBEAT_TYPES = frozenset({"heartbeat", "ping", "keepalive", "keep-alive"})


class CursorExpired(Exception):
    """The replay cursor is older than Hoplite's retention window (HTTP 410)."""


class StreamUnavailable(Exception):
    """The event stream cannot be used; callers should fall back to polling."""


class _TransientStreamError(Exception):
    def __init__(self, retry_after: float | None) -> None:
        super().__init__("transient event stream error")
        self.retry_after = retry_after


@dataclass
class StreamEvent:
    kind: str  # "event" | "draft" | "draft_removed" | "heartbeat"
    type: str
    cursor: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    status: str | None = None
    message_id: str | None = None
    role: str | None = None
    text: str | None = None
    revision: int | None = None
    stream_id: str | None = None


def _containers(payload: dict[str, Any]) -> list[dict[str, Any]]:
    found = [payload]
    for key in ("data", "draft", "message", "snapshot"):
        inner = payload.get(key)
        if isinstance(inner, dict):
            found.append(inner)
    return found


def _lookup(payload: dict[str, Any], *keys: str) -> Any:
    containers = _containers(payload)
    for key in keys:
        for container in containers:
            value = container.get(key)
            if value is not None and not isinstance(value, (dict, list)):
                return value
    return None


def draft_text(payload: dict[str, Any]) -> str | None:
    for container in _containers(payload):
        for key in TEXT_KEYS:
            value = container.get(key)
            if isinstance(value, str):
                return value
        parts = container.get("parts")
        if isinstance(parts, list):
            texts = [p["text"] for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str)]
            if texts:
                return "".join(texts)
    return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def classify(event_name: str | None, event_id: str | None, raw: str) -> StreamEvent:
    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        payload = {"raw": raw}
    if not isinstance(payload, dict):
        payload = {"value": payload}

    name = (event_name or "").strip()
    if not name or name == "message":
        name = str(payload.get("type") or "message")
    lname = name.lower()
    cursor = event_id or (str(payload["cursor"]) if payload.get("cursor") else None)

    if lname in HEARTBEAT_TYPES:
        return StreamEvent("heartbeat", name, cursor, payload)
    if lname.endswith("draft.removed"):
        return StreamEvent(
            "draft_removed", name, None, payload,
            message_id=_as_str(_lookup(payload, "messageId", "id")),
            run_id=_as_str(_lookup(payload, "runId")),
        )
    if lname.startswith("draft") or ".draft" in lname:
        return StreamEvent(
            "draft", name, None, payload,
            message_id=_as_str(_lookup(payload, "messageId", "id")),
            run_id=_as_str(_lookup(payload, "runId")),
            role=_as_str(_lookup(payload, "role")),
            text=draft_text(payload),
            revision=_as_int(_lookup(payload, "revision")),
            stream_id=_as_str(_lookup(payload, "streamId")),
        )

    event_type = str(payload.get("type") or name)
    inner = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    run_id = _as_str(payload.get("runId"))
    status = None
    message_id = None
    if event_type.startswith("run."):
        run_id = _as_str(inner.get("id")) or run_id
        status = _as_str(inner.get("status")) or event_type.split(".", 1)[1]
    elif event_type.startswith("message."):
        message_id = _as_str(inner.get("id"))
    return StreamEvent(
        "event", event_type, cursor, payload,
        run_id=run_id, status=status, message_id=message_id, role=_as_str(inner.get("role")),
    )


def _as_str(value: Any) -> str | None:
    return str(value) if value is not None else None


async def iter_thread_events(
    client: HopliteClient,
    thread_id: str,
    cursor: str | None,
    *,
    read_timeout: float = 45.0,
    max_failures: int = 8,
) -> AsyncIterator[StreamEvent]:
    """Yield events for one thread forever, reconnecting with `Last-Event-ID` when the stream drops.

    Raises CursorExpired on HTTP 410, StreamUnavailable when the endpoint is missing or keeps failing,
    and HopliteError for non-retryable HTTP errors (e.g. 401/403).
    """
    failures = 0
    last_cursor = cursor
    while True:
        retry_after: float | None = None
        headers = {**client.auth_headers, "Accept": "text/event-stream", "Cache-Control": "no-cache"}
        params = {"threadId": thread_id}
        if last_cursor:
            headers["Last-Event-ID"] = last_cursor
            params["cursor"] = last_cursor
        try:
            async with aconnect_sse(
                client.http, "GET", "/api/events/stream",
                params=params, headers=headers, timeout=httpx.Timeout(10.0, read=read_timeout),
            ) as source:
                resp = source.response
                if resp.status_code != 200:
                    await resp.aread()
                    if resp.status_code == 410:
                        raise CursorExpired("event cursor expired")
                    if resp.status_code in (404, 405, 501):
                        raise StreamUnavailable(f"event stream returned HTTP {resp.status_code}")
                    error = error_from_response(resp)
                    if resp.status_code not in RETRYABLE_STATUS:
                        raise error
                    raise _TransientStreamError(error.retry_after)
                failures = 0
                async for sse in source.aiter_sse():
                    event = classify(sse.event, sse.id or None, sse.data)
                    if event.kind in ("event", "heartbeat") and event.cursor:
                        last_cursor = event.cursor
                    yield event
        except _TransientStreamError as exc:
            failures += 1
            retry_after = exc.retry_after
        except (httpx.TransportError, SSEError) as exc:
            failures += 1
            log.info("event stream interrupted; reconnecting",
                     extra={"thread_id": thread_id, "error": exc.__class__.__name__, "failures": failures})
        if failures >= max_failures:
            raise StreamUnavailable(f"event stream failed {failures} times in a row")
        delay = retry_after if retry_after is not None else min(10.0, 0.5 * 2**failures)
        await asyncio.sleep(delay)
