"""Async client for the Hoplite REST API (only the endpoints this proxy needs).

Writes always carry an `Idempotency-Key` (required for service credentials), so retrying a write
after a transport error or a 429/5xx is safe: Hoplite returns the original result. The legacy body
fields `clientOperationId` / `clientMessageId` are not sent: when present they must equal the header
(otherwise Hoplite answers 400 `idempotency_key_mismatch`).
"""

from __future__ import annotations

import asyncio
import email.utils
import logging
import random
import time
from typing import Any
from urllib.parse import quote

import httpx

from app.errors import OpenAIError, insufficient_quota, rate_limited, upstream_error

log = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class HopliteError(Exception):
    def __init__(
        self,
        status: int,
        code: str | None,
        message: str,
        *,
        request_id: str | None = None,
        retry_after: float | None = None,
        body: Any = None,
    ) -> None:
        super().__init__(f"Hoplite API error {status}{f' {code}' if code else ''}: {message}")
        self.status = status
        self.code = code
        self.message = message
        self.request_id = request_id
        self.retry_after = retry_after
        self.body = body


def _seg(value: str) -> str:
    return quote(value, safe="")


def parse_retry_after(headers: httpx.Headers) -> float | None:
    raw = headers.get("retry-after") or headers.get("x-retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        try:
            return max(0.0, email.utils.parsedate_to_datetime(raw).timestamp() - time.time())
        except (TypeError, ValueError):
            return None


def error_from_response(resp: httpx.Response) -> HopliteError:
    try:
        body = resp.json()
    except ValueError:
        body = None
    code: str | None = None
    message: str | None = None
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            code = err.get("code") or err.get("type")
            message = err.get("message")
        elif isinstance(err, str):
            code = err
        message = message or body.get("message") or body.get("detail")
    if not message:
        message = code or resp.reason_phrase or f"HTTP {resp.status_code}"
    return HopliteError(
        resp.status_code,
        str(code) if code else None,
        str(message),
        request_id=resp.headers.get("x-request-id"),
        retry_after=parse_retry_after(resp.headers),
        body=body,
    )


def to_openai_error(exc: HopliteError) -> OpenAIError:
    """Translate an upstream Hoplite HTTP error into the error the proxy returns to its client."""
    status, detail = exc.status, exc.message
    if status == 400:
        return OpenAIError(400, f"Hoplite rejected the request: {detail}", code=exc.code or "upstream_invalid_request")
    if status in (401, 403):
        log.error(
            "hoplite rejected the upstream credential",
            extra={"status": status, "code": exc.code, "hoplite_request_id": exc.request_id},
        )
        return upstream_error(
            f"Hoplite rejected the proxy's upstream credential (HTTP {status}). "
            "Ask the proxy administrator to check the Hoplite key bound to this API key.",
            code="upstream_auth_error",
        )
    if status == 402:
        return insufficient_quota(f"Hoplite: {detail}")
    if status == 404:
        return OpenAIError(404, f"Hoplite resource not found: {detail}", code=exc.code or "not_found")
    if status == 409:
        return OpenAIError(409, f"Hoplite conflict: {detail}", code=exc.code or "conflict")
    if status == 413:
        return OpenAIError(413, f"The request is too large for Hoplite: {detail}", code="request_too_large")
    if status == 429:
        return rate_limited(f"Hoplite rate limit exceeded: {detail}", retry_after=exc.retry_after)
    if status in (501, 503):
        err = upstream_error(f"Hoplite is unavailable: {detail}", status=503, code=exc.code or "upstream_unavailable")
        if exc.retry_after:
            err.headers["Retry-After"] = str(max(1, int(exc.retry_after)))
        return err
    return upstream_error(f"Hoplite API error (HTTP {status}): {detail}")


class HopliteClient:
    """Thin typed wrapper bound to one Hoplite API key; shares the app-wide httpx client."""

    def __init__(
        self,
        http: httpx.AsyncClient,
        api_key: str,
        *,
        max_attempts: int = 4,
        max_backoff: float = 20.0,
    ) -> None:
        self._http = http
        self._api_key = api_key
        self._max_attempts = max_attempts
        self._max_backoff = max_backoff

    @property
    def http(self) -> httpx.AsyncClient:
        return self._http

    @property
    def auth_headers(self) -> dict[str, str]:
        return {"X-Api-Key": self._api_key}

    def _backoff(self, attempt: int) -> float:
        base = min(self._max_backoff, 0.5 * 2 ** (attempt - 1))
        return base * (0.5 + random.random() / 2)

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
        idempotency_key: str | None = None,
        expected: tuple[int, ...] = (200, 201, 202),
    ) -> dict[str, Any]:
        headers = {**self.auth_headers, "Accept": "application/json"}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        clean_params = {k: v for k, v in (params or {}).items() if v is not None} or None
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = await self._http.request(method, path, params=clean_params, json=json, headers=headers)
            except httpx.TransportError as exc:
                if attempt >= self._max_attempts:
                    raise HopliteError(
                        503, "upstream_unreachable", f"Hoplite API is unreachable ({exc.__class__.__name__})"
                    ) from exc
                delay = self._backoff(attempt)
                log.warning(
                    "hoplite request failed; retrying",
                    extra={"method": method, "path": path, "attempt": attempt, "error": exc.__class__.__name__},
                )
                await asyncio.sleep(delay)
                continue

            if resp.status_code in expected:
                if not resp.content:
                    return {}
                try:
                    data = resp.json()
                except ValueError as exc:
                    raise HopliteError(502, "invalid_json", "Hoplite returned a non-JSON response") from exc
                return data if isinstance(data, dict) else {"data": data}

            error = error_from_response(resp)
            retryable = resp.status_code in RETRYABLE_STATUS or error.code == "operation_outcome_pending"
            if retryable and attempt < self._max_attempts:
                delay = error.retry_after if error.retry_after is not None else self._backoff(attempt)
                if error.code == "operation_outcome_pending":
                    delay = max(delay, 1.0)
                delay = min(delay, self._max_backoff)
                log.warning(
                    "hoplite request rejected; retrying",
                    extra={"method": method, "path": path, "status": resp.status_code, "code": error.code,
                           "attempt": attempt, "delay_s": round(delay, 2)},
                )
                await asyncio.sleep(delay)
                continue
            raise error

    # --- catalog and projects -------------------------------------------------------------------

    async def list_models(self) -> dict[str, Any]:
        return await self.request("GET", "/api/model-providers")

    async def list_projects(self, *, page_size: int = 100, max_pages: int = 10) -> list[dict[str, Any]]:
        projects: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(max_pages):
            data = await self.request("GET", "/api/projects", params={"limit": page_size, "cursor": cursor})
            projects.extend(p for p in data.get("projects") or [] if isinstance(p, dict))
            cursor = data.get("nextCursor")
            if not cursor:
                break
        return projects

    async def get_project(self, project_id: str) -> dict[str, Any]:
        return (await self.request("GET", f"/api/projects/{_seg(project_id)}")).get("project") or {}

    # --- threads, messages and runs -----------------------------------------------------------

    async def create_thread(self, body: dict[str, Any], *, idempotency_key: str) -> dict[str, Any]:
        return await self.request("POST", "/api/threads", json=body, idempotency_key=idempotency_key)

    async def get_thread(self, thread_id: str) -> dict[str, Any]:
        return (await self.request("GET", f"/api/threads/{_seg(thread_id)}")).get("thread") or {}

    async def append_message(self, thread_id: str, body: dict[str, Any], *, idempotency_key: str) -> dict[str, Any]:
        return await self.request(
            "POST", f"/api/threads/{_seg(thread_id)}/messages", json=body, idempotency_key=idempotency_key
        )

    async def list_messages(
        self, thread_id: str, *, limit: int = 100, activity_limit: int = 0, cursor: str | None = None
    ) -> dict[str, Any]:
        return await self.request(
            "GET",
            f"/api/threads/{_seg(thread_id)}/messages",
            params={"limit": limit, "activityLimit": activity_limit, "cursor": cursor},
        )

    async def run_state(self, thread_id: str, run_id: str | None = None) -> dict[str, Any] | None:
        """Authoritative state of a run (the latest run when run_id is None); None if no run exists yet."""
        try:
            data = await self.request("GET", f"/api/threads/{_seg(thread_id)}/run-state", params={"runId": run_id})
        except HopliteError as exc:
            if exc.status == 404 and exc.code in (None, "run_not_found", "not_found"):
                return None
            raise
        run = data.get("run")
        return run if isinstance(run, dict) else None

    async def active_run(self, thread_id: str) -> dict[str, Any] | None:
        data = await self.request("GET", f"/api/threads/{_seg(thread_id)}/active-run")
        active = data.get("activeRun")
        return active if isinstance(active, dict) else None

    async def stop_run(self, thread_id: str, run_id: str, *, idempotency_key: str) -> dict[str, Any]:
        return await self.request(
            "POST", f"/api/threads/{_seg(thread_id)}/stop", json={"runId": run_id}, idempotency_key=idempotency_key
        )

    async def list_questions(self, thread_id: str) -> list[dict[str, Any]]:
        data = await self.request("GET", f"/api/threads/{_seg(thread_id)}/questions")
        return [q for q in data.get("questions") or [] if isinstance(q, dict)]

    async def answer_question(
        self, thread_id: str, question_id: str, body: dict[str, Any], *, idempotency_key: str
    ) -> dict[str, Any]:
        return await self.request(
            "POST",
            f"/api/threads/{_seg(thread_id)}/questions/{_seg(question_id)}",
            json=body,
            idempotency_key=idempotency_key,
        )

    async def thread_usage(self, thread_id: str) -> dict[str, Any]:
        return (await self.request("GET", f"/api/threads/{_seg(thread_id)}/usage")).get("usage") or {}

    async def run_failure(self, thread_id: str, run_id: str) -> dict[str, Any]:
        data = await self.request("GET", f"/api/threads/{_seg(thread_id)}/runs/{_seg(run_id)}/failure")
        return data.get("detail") or {}

    # --- events ---------------------------------------------------------------------------------

    async def event_head(self, *, thread_id: str | None = None, project_id: str | None = None) -> str | None:
        data = await self.request("GET", "/api/events/head", params={"threadId": thread_id, "projectId": project_id})
        cursor = data.get("cursor")
        return str(cursor) if cursor else None
