"""Runs in flight, decoupled from the HTTP requests that started them.

A run executes in its own task and publishes updates to any number of subscribers (HTTP
responses). This lets a client retry (e.g. the OpenAI SDK retrying after its 600 s timeout)
attach to the run that is already going instead of starting a second agent, and lets the proxy
wait a grace period after the last client disconnects before stopping the run.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.errors import OpenAIError

log = logging.getLogger(__name__)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_micros: float | None = None

    def as_openai(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


@dataclass
class Started:
    thread_id: str
    run_id: str | None
    model: str


@dataclass
class Delta:
    content: str = ""
    reasoning: str = ""


@dataclass
class Finished:
    finish_reason: str
    content: str
    reasoning: str
    usage: Usage
    thread_id: str
    run_id: str | None


@dataclass
class Failed:
    error: OpenAIError


Update = Started | Delta | Finished | Failed


@dataclass(eq=False)
class RunHandle:
    fingerprint: str
    key_id: str
    completion_id: str
    created: int
    idempotency_key: str | None = None
    grace_seconds: float = 60.0
    history: list[Update] = field(default_factory=list)
    started: Started | None = None
    outcome: Finished | Failed | None = None
    task: asyncio.Task[None] | None = None
    finished_at: float | None = None
    _subscribers: set[asyncio.Queue[Update]] = field(default_factory=set)
    _first: asyncio.Event = field(default_factory=asyncio.Event)
    _done: asyncio.Event = field(default_factory=asyncio.Event)
    _grace_task: asyncio.Task[None] | None = None
    _ever_subscribed: bool = False

    # --- publishing (runner side) ---------------------------------------------------------------

    def emit(self, update: Update) -> None:
        self.history.append(update)
        if isinstance(update, Started):
            self.started = update
            self._first.set()
        elif isinstance(update, (Finished, Failed)):
            self.outcome = update
            self._first.set()
        for queue in list(self._subscribers):
            queue.put_nowait(update)

    def mark_done(self) -> None:
        self.finished_at = time.monotonic()
        self._done.set()
        self._first.set()
        if self._grace_task is not None:
            self._grace_task.cancel()
            self._grace_task = None

    @property
    def done(self) -> bool:
        return self._done.is_set()

    # --- subscribing (HTTP side) ----------------------------------------------------------------

    def subscribe(self) -> asyncio.Queue[Update]:
        queue: asyncio.Queue[Update] = asyncio.Queue()
        for update in self.history:
            queue.put_nowait(update)
        self._subscribers.add(queue)
        self._ever_subscribed = True
        if self._grace_task is not None:
            self._grace_task.cancel()
            self._grace_task = None
        return queue

    def unsubscribe(self, queue: asyncio.Queue[Update]) -> None:
        self._subscribers.discard(queue)
        if self._subscribers or self.done or self.task is None or self.task.done():
            return
        if self._grace_task is None:
            self._grace_task = asyncio.get_running_loop().create_task(self._abandon_after_grace())

    @property
    def orphaned(self) -> bool:
        """True while a run is going but every client that asked for it has disconnected."""
        return self._ever_subscribed and not self._subscribers and not self.done

    async def _abandon_after_grace(self) -> None:
        try:
            await asyncio.sleep(self.grace_seconds)
        except asyncio.CancelledError:
            return
        self._grace_task = None
        if not self._subscribers and not self.done and self.task is not None:
            log.info("all clients disconnected; abandoning run", extra={"completion_id": self.completion_id})
            self.task.cancel()

    async def wait_started(self) -> Started:
        """Wait until the run has started; raises the error if it failed before starting."""
        await self._first.wait()
        if self.started is not None:
            return self.started
        if isinstance(self.outcome, Failed):
            raise self.outcome.error
        raise OpenAIError(500, "The run ended before it started.", type_="api_error", code="internal_error")


class InflightRegistry:
    def __init__(self, settings: Settings) -> None:
        self._handles: dict[str, RunHandle] = {}
        self._tasks: set[asyncio.Task[None]] = set()  # strong references: the loop only keeps weak ones
        self._cache_seconds = settings.result_cache_seconds
        self._grace_seconds = settings.disconnect_grace_seconds

    def start(self, handle: RunHandle, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        task = asyncio.get_running_loop().create_task(coro, name=handle.completion_id)
        handle.task = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def lookup(self, fingerprint: str, *, is_retry: bool, idempotency_key: str | None) -> RunHandle | None:
        """Return a run this request should attach to instead of starting a new one, if any."""
        self.prune()
        handle = self._handles.get(fingerprint)
        if handle is None:
            return None
        same_key = idempotency_key is not None and idempotency_key == handle.idempotency_key
        if handle.done:
            # Replay a completed answer only to explicit retries; never replay failures.
            return handle if isinstance(handle.outcome, Finished) and (is_retry or same_key) else None
        return handle if (is_retry or same_key or handle.orphaned) else None

    def create(
        self, fingerprint: str, key_id: str, *, completion_id: str, created: int, idempotency_key: str | None
    ) -> RunHandle:
        handle = RunHandle(
            fingerprint=fingerprint,
            key_id=key_id,
            completion_id=completion_id,
            created=created,
            idempotency_key=idempotency_key,
            grace_seconds=self._grace_seconds,
        )
        self._handles[fingerprint] = handle
        return handle

    def prune(self) -> None:
        now = time.monotonic()
        for fingerprint, handle in list(self._handles.items()):
            if handle.done and handle.finished_at is not None and now - handle.finished_at > self._cache_seconds:
                del self._handles[fingerprint]

    def running(self) -> list[asyncio.Task[None]]:
        return [task for task in self._tasks if not task.done()]

    async def shutdown(self, wait_seconds: float = 15.0) -> None:
        tasks = self.running()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=wait_seconds)


class KeySlots:
    """Per-key cap on concurrently running Hoplite runs (in-process)."""

    def __init__(self) -> None:
        self._active: dict[str, int] = {}

    def try_acquire(self, key_id: str, limit: int) -> bool:
        current = self._active.get(key_id, 0)
        if current >= limit:
            return False
        self._active[key_id] = current + 1
        return True

    def release(self, key_id: str) -> None:
        remaining = self._active.get(key_id, 0) - 1
        if remaining > 0:
            self._active[key_id] = remaining
        else:
            self._active.pop(key_id, None)

    def active(self, key_id: str) -> int:
        return self._active.get(key_id, 0)
