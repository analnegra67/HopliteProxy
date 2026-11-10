"""Drives one Hoplite run for a chat completion: start -> watch -> finalize.

Watching follows Hoplite's documented recipe: capture the event head, read a run-state snapshot,
then consume the thread-filtered SSE stream from that cursor. Draft frames become streamed text
deltas; persisted messages are the source of truth for the final text; `run.<status>` events (and
a periodic safety check) detect completion. If the event stream is unavailable the runner polls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import Settings
from app.context import KeyContext
from app.errors import OpenAIError, insufficient_quota, rate_limited, upstream_error
from app.hoplite.client import HopliteClient, HopliteError, to_openai_error
from app.hoplite.events import CursorExpired, StreamEvent, StreamUnavailable, iter_thread_events
from app.openai_api.schemas import ChatMessage
from app.service.conversation import ConversationMapper, ThreadPlan
from app.service.inflight import Delta, Failed, Finished, KeySlots, RunHandle, Started, Usage
from app.service.models import ResolvedModel

log = logging.getLogger(__name__)

TERMINAL = frozenset({"completed", "failed", "cancelled"})
SETTLED = TERMINAL | {"waiting"}
SAFETY_CHECK_SECONDS = 30.0
MAX_AUTO_ANSWERS = 3
CONTINUE_FALLBACK_STATUSES = frozenset({404, 409, 410, 422})


def _str(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


def _settled(run: dict[str, Any] | None) -> bool:
    return run is not None and run.get("status") in SETTLED


class DraftTracker:
    """Turns full-snapshot drafts and final messages into append-only text deltas.

    `sent` is exactly the text the client has received, which is what gets fingerprinted.
    """

    SEPARATOR = "\n\n"

    def __init__(self) -> None:
        self.sent = ""
        self._emitted: dict[str, str] = {}
        self._order: list[str] = []
        self._closed: set[str] = set()  # finalized: nothing more is sent for these
        self._drafts_closed: set[str] = set()  # `draft.removed` seen: only the persisted text may follow
        self._diverged: set[str] = set()

    def update(self, message_id: str, text: str) -> str:
        """Apply a draft snapshot."""
        if message_id in self._drafts_closed:
            return ""
        return self._apply(message_id, text)

    def _apply(self, message_id: str, text: str) -> str:
        if not text or message_id in self._closed:
            return ""
        previous = self._emitted.get(message_id)
        if previous is None:
            out = (self.SEPARATOR if self.sent else "") + text
            self._order.append(message_id)
        else:
            if not text.startswith(previous):
                self._diverged.add(message_id)  # cannot retract text that was already streamed
                return ""
            out = text[len(previous):]
            if not out:
                return ""
        self._emitted[message_id] = text
        self.sent += out
        return out

    def finalize(self, message_id: str, text: str) -> str:
        """Apply the persisted (authoritative) text of a message; sends whatever the drafts missed."""
        if message_id in self._closed:
            return ""
        target = message_id
        if message_id not in self._emitted:
            # A persisted message may carry a different id than its drafts: match an open draft by prefix.
            for candidate in self._order:
                if candidate not in self._closed and text.startswith(self._emitted[candidate]):
                    target = candidate
                    break
        out = self._apply(target, text)
        if target in self._diverged:
            log.warning("streamed draft diverged from the final message", extra={"message_id": message_id})
        self._closed.update({target, message_id})
        return out

    def discard(self, message_id: str | None) -> None:
        """`draft.removed`: stop applying drafts; the persisted message (if any) still completes the text."""
        if message_id:
            self._drafts_closed.add(message_id)

    def append_note(self, note: str) -> str:
        out = (self.SEPARATOR if self.sent else "") + note
        self.sent += out
        return out


@dataclass
class RunJob:
    ctx: KeyContext
    messages: list[ChatMessage]
    model: ResolvedModel
    reasoning: dict[str, str] | None
    plan: ThreadPlan
    completion_id: str
    request_id: str


@dataclass
class _RunState:
    thread_id: str | None = None
    run_id: str | None = None
    user_message_id: str | None = None
    cursor: str | None = None
    accepted: bool = False
    new_thread: bool = False
    model: str = "hoplite-default"
    usage_before: dict[str, Any] = field(default_factory=dict)
    tracker: DraftTracker = field(default_factory=DraftTracker)
    reasoning: list[str] = field(default_factory=list)
    seen_messages: set[str] = field(default_factory=set)
    last_sync: float = 0.0
    warned_draft_shape: bool = False

    def reset_thread(self) -> None:
        self.thread_id = self.run_id = self.user_message_id = self.cursor = None
        self.usage_before = {}


def format_question(question: dict[str, Any]) -> str:
    items = (question.get("request") or {}).get("questions") or []
    lines = ["**The agent needs your input:**"]
    for item in items:
        if not isinstance(item, dict):
            continue
        header = str(item.get("header") or "").strip()
        text = str(item.get("question") or "").strip()
        lines += ["", f"**{header}:** {text}" if header else text]
        for number, option in enumerate(item.get("options") or [], 1):
            if isinstance(option, dict):
                label = option.get("label") or option.get("value") or option.get("text") or json.dumps(option)
                description = option.get("description")
            else:
                label, description = option, None
            lines.append(f"{number}. {label}" + (f" — {description}" if description else ""))
        if item.get("multiSelect"):
            lines.append("_(several options may be chosen)_")
    lines += ["", "_Reply in your next message to answer._"]
    return "\n".join(lines)


def answer_body(question: dict[str, Any], text: str) -> dict[str, Any]:
    items = [q for q in (question.get("request") or {}).get("questions") or [] if isinstance(q, dict)]
    if len(items) <= 1:
        return {"answer": text}
    return {"answers": {str(item["question"]): text for item in items if item.get("question")}}


def classify_failure(detail: dict[str, Any]) -> tuple[str, OpenAIError]:
    message = str(detail.get("message") or "The Hoplite run failed.")
    text = f"{message} {detail.get('code') or ''}".lower()
    if "spend limit" in text or "budget has been reached" in text:
        return "spend_limit", insufficient_quota(
            f"Hoplite: {message}. Raise the key's spend_limit_micros or start a new conversation."
        )
    if any(s in text for s in ("out of credits", "subscription is inactive", "usage limit", "add credits")):
        return "quota", insufficient_quota(f"Hoplite: {message}")
    if "context limit" in text:
        return "context", OpenAIError(400, f"Hoplite: {message}", code="context_length_exceeded", param="messages")
    if "rate limit" in text or "at capacity" in text:
        return "rate_limit", rate_limited(f"Hoplite: {message}")
    return "other", upstream_error(f"The Hoplite run failed: {message}", code="run_failed")


class Runner:
    def __init__(
        self, *, settings: Settings, http: httpx.AsyncClient, mapper: ConversationMapper, slots: KeySlots
    ) -> None:
        self._settings = settings
        self._http = http
        self._mapper = mapper
        self._slots = slots

    def client_for(self, hoplite_api_key: str) -> HopliteClient:
        return HopliteClient(
            self._http,
            hoplite_api_key,
            max_attempts=self._settings.hoplite_max_attempts,
            max_backoff=self._settings.hoplite_max_backoff_seconds,
        )

    # --- lifecycle ------------------------------------------------------------------------------

    async def execute(self, handle: RunHandle, job: RunJob) -> None:
        ctx = job.ctx
        client = self.client_for(ctx.hoplite_api_key)
        st = _RunState(model=job.model.label)
        if not self._slots.try_acquire(ctx.key_id, ctx.max_concurrency):
            await self._mapper.release(job.plan)
            handle.emit(Failed(rate_limited(
                f"Too many concurrent requests for this API key (limit {ctx.max_concurrency}); "
                "retry when a running request finishes.",
                code="concurrency_limit_exceeded", retry_after=5,
            )))
            handle.mark_done()
            return
        try:
            await self._start(client, job, st)
            assert st.thread_id is not None
            handle.emit(Started(st.thread_id, st.run_id, st.model))
            try:
                async with asyncio.timeout(self._settings.run_timeout_seconds):
                    run = await self._watch_until_settled(client, job, st, handle)
            except TimeoutError:
                await self._stop_quietly(client, job, st, reason="timeout")
                raise OpenAIError(
                    504,
                    f"The Hoplite run did not finish within {self._settings.run_timeout_seconds:.0f} s "
                    f"and was stopped (thread {st.thread_id}).",
                    type_="api_error", code="run_timeout",
                ) from None
            handle.emit(await self._finalize(client, job, st, handle, run))
        except OpenAIError as exc:
            await self._fail(job, st, handle, exc)
        except HopliteError as exc:
            await self._fail(job, st, handle, to_openai_error(exc))
        except asyncio.CancelledError:
            if self._settings.stop_on_disconnect:
                await self._stop_quietly(client, job, st, reason="abandoned")
            if handle.outcome is None:
                handle.emit(Failed(OpenAIError(499, "The request was abandoned.", type_="api_error", code="abandoned")))
            raise
        except Exception:
            log.exception("unexpected error while running the hoplite agent", extra={"thread_id": st.thread_id})
            await self._fail(job, st, handle, OpenAIError(
                500, "Internal error while running the Hoplite agent.", type_="api_error", code="internal_error"
            ))
        finally:
            self._slots.release(ctx.key_id)
            handle.mark_done()

    async def _fail(self, job: RunJob, st: _RunState, handle: RunHandle, error: OpenAIError) -> None:
        if not st.accepted:
            await self._mapper.release(job.plan)
        log.warning("chat completion failed",
                    extra={"status": error.status_code, "code": error.code, "thread_id": st.thread_id})
        handle.emit(Failed(error))

    # --- starting a run -------------------------------------------------------------------------

    async def _start(self, client: HopliteClient, job: RunJob, st: _RunState) -> None:
        plan = job.plan
        if plan.mode != "new" and plan.thread_id:
            try:
                await self._continue_thread(client, job, st)
                return
            except HopliteError as exc:
                if st.accepted or plan.reason != "fingerprint" or exc.status not in CONTINUE_FALLBACK_STATUSES:
                    raise
                log.info("cannot continue the hoplite thread; starting a new one",
                         extra={"thread_id": plan.thread_id, "status": exc.status, "code": exc.code})
                st.reset_thread()
        await self._create_thread(client, job, st)

    async def _create_thread(self, client: HopliteClient, job: RunJob, st: _RunState) -> None:
        ctx = job.ctx
        body: dict[str, Any] = {
            "projectId": ctx.project_id,
            "prompt": job.plan.prompt,
            "autoMerge": False,
            "metadata": {"source": "openai-proxy", "requestId": job.request_id},
        }
        if job.model.hoplite_id:
            body["model"] = job.model.hoplite_id
        if job.reasoning:
            body["reasoning"] = job.reasoning
        if ctx.spend_limit_micros:
            body["spendLimitMicros"] = ctx.spend_limit_micros
        if ctx.repos:
            body["repos"] = ctx.repos
        data = await client.create_thread(body, idempotency_key=f"{job.completion_id}:thread")
        thread = data.get("thread") or {}
        thread_id = _str(thread.get("id"))
        if thread_id is None:
            raise HopliteError(502, "invalid_response", "Hoplite did not return a thread id")
        st.thread_id, st.accepted, st.new_thread = thread_id, True, True
        st.run_id = _str((data.get("run") or {}).get("id")) or _str(thread.get("latestRunId"))
        st.user_message_id = _str((data.get("message") or {}).get("id"))
        st.model = _str(thread.get("modelId")) or job.model.label
        st.cursor = await client.event_head(thread_id=thread_id)
        log.info("hoplite thread created", extra={"thread_id": thread_id, "run_id": st.run_id, "model": st.model})

    async def _continue_thread(self, client: HopliteClient, job: RunJob, st: _RunState) -> None:
        plan = job.plan
        thread_id = plan.thread_id
        assert thread_id is not None
        st.cursor = await client.event_head(thread_id=thread_id)  # before the write, so no event is missed
        st.thread_id = thread_id
        st.usage_before = await self._usage_quiet(client, thread_id)
        if plan.mode == "answer" and plan.question_id:
            pending = {str(q.get("questionId")): q for q in await client.list_questions(thread_id)}
            question = pending.get(plan.question_id)
            if question is not None:
                question_run = _str(question.get("runId"))
                if question_run:  # the run resumes after the answer: skip what was already delivered
                    delivered = await self._run_messages(client, thread_id, question_run)
                    st.seen_messages.update(str(m["id"]) for m in delivered)
                data = await client.answer_question(
                    thread_id, plan.question_id, answer_body(question, plan.content or ""),
                    idempotency_key=f"{job.completion_id}:answer",
                )
                st.accepted = True
                st.run_id = _str((data.get("result") or {}).get("runId")) or _str(question.get("runId"))
                st.model = job.model.hoplite_id or await self._thread_model(client, thread_id) or job.model.label
                log.info("answered the agent's question", extra={"thread_id": thread_id, "run_id": st.run_id})
                return
            log.info("the agent's question is no longer pending; sending the reply as a message",
                     extra={"thread_id": thread_id})
        body: dict[str, Any] = {"content": plan.content or ""}
        if job.model.hoplite_id:
            body["model"] = job.model.hoplite_id
        data = await client.append_message(thread_id, body, idempotency_key=f"{job.completion_id}:message")
        st.accepted = True
        st.user_message_id = _str((data.get("message") or {}).get("id"))
        st.run_id = _str((data.get("run") or {}).get("id"))
        st.model = job.model.hoplite_id or await self._thread_model(client, thread_id) or job.model.label
        log.info("message appended to hoplite thread",
                 extra={"thread_id": thread_id, "run_id": st.run_id, "queued": data.get("queued"), "via": plan.reason})

    # --- watching a run -------------------------------------------------------------------------

    async def _watch_until_settled(
        self, client: HopliteClient, job: RunJob, st: _RunState, handle: RunHandle
    ) -> dict[str, Any]:
        answered = 0
        while True:
            run = await self._watch(client, st, handle)
            if (run.get("status") == "waiting" and self._settings.question_policy == "auto"
                    and answered < MAX_AUTO_ANSWERS and await self._auto_answer(client, job, st, answered)):
                answered += 1
                continue
            return run

    async def _watch(self, client: HopliteClient, st: _RunState, handle: RunHandle) -> dict[str, Any]:
        run = await self._snapshot(client, st)
        if run is not None and _settled(run):
            return run
        queue: asyncio.Queue[object] = asyncio.Queue()
        reader = self._start_reader(client, st, queue)
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=SAFETY_CHECK_SECONDS)
                except TimeoutError:
                    run = await self._snapshot(client, st)
                    if run is not None and _settled(run):
                        return run
                    continue
                if isinstance(item, CursorExpired):
                    st.cursor = await client.event_head(thread_id=st.thread_id)
                    run = await self._snapshot(client, st)
                    if run is not None and _settled(run):
                        return run
                    reader = self._start_reader(client, st, queue)
                    continue
                if isinstance(item, StreamUnavailable):
                    log.warning("hoplite event stream unavailable; polling instead", extra={"reason": str(item)})
                    return await self._poll(client, st, handle)
                if isinstance(item, BaseException):
                    raise item
                assert isinstance(item, StreamEvent)
                run = await self._on_event(client, st, handle, item)
                if run is not None:
                    return run
        finally:
            reader.cancel()
            await asyncio.wait([reader])

    def _start_reader(self, client: HopliteClient, st: _RunState, queue: asyncio.Queue[object]) -> asyncio.Task[None]:
        assert st.thread_id is not None
        thread_id, cursor = st.thread_id, st.cursor

        async def read() -> None:
            try:
                async for event in iter_thread_events(
                    client, thread_id, cursor, read_timeout=self._settings.sse_read_timeout_seconds
                ):
                    if event.kind in ("event", "heartbeat") and event.cursor:
                        st.cursor = event.cursor
                    queue.put_nowait(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                queue.put_nowait(exc)

        return asyncio.create_task(read(), name=f"hoplite-events-{thread_id}")

    async def _on_event(
        self, client: HopliteClient, st: _RunState, handle: RunHandle, ev: StreamEvent
    ) -> dict[str, Any] | None:
        if ev.kind == "heartbeat":
            return None
        if ev.kind == "draft":
            if (ev.run_id and st.run_id and ev.run_id != st.run_id) or (ev.role and ev.role != "assistant"):
                return None
            if ev.stream_id and not ev.stream_id.startswith("assistant"):
                return None  # e.g. a reasoning stream: only persisted thinking messages are exposed
            if ev.text is None:
                if not st.warned_draft_shape:
                    st.warned_draft_shape = True
                    log.warning("draft frame without recognizable text",
                                extra={"event": ev.type, "keys": sorted(ev.data)})
                return None
            delta = st.tracker.update(ev.message_id or "draft", ev.text)
            if delta:
                handle.emit(Delta(content=delta))
            return None
        if ev.kind == "draft_removed":
            st.tracker.discard(ev.message_id)
            return None
        if ev.type.startswith("run."):
            if st.run_id is None and ev.run_id:
                await self._adopt_run(client, st, ev.run_id)
            if st.run_id is None or ev.run_id != st.run_id:
                return None
            if ev.status in SETTLED:
                run = await client.run_state(st.thread_id or "", st.run_id)
                if run is not None and _settled(run):
                    return run
            return None
        if ev.type == "message.persisted":
            if st.run_id and ev.role in (None, "assistant"):
                await self._sync_messages(client, st, handle)
            return None
        if ev.type == "thread.deleted":
            raise OpenAIError(502, "The Hoplite thread was deleted while the run was in progress.",
                              type_="api_error", code="thread_deleted")
        return None

    def _owns(self, st: _RunState, run: dict[str, Any]) -> bool:
        owner = _str((run.get("userMessage") or {}).get("id"))
        if st.user_message_id:
            return owner == st.user_message_id
        return st.new_thread

    async def _snapshot(self, client: HopliteClient, st: _RunState) -> dict[str, Any] | None:
        assert st.thread_id is not None
        if st.run_id is not None:
            return await client.run_state(st.thread_id, st.run_id)
        run = await client.run_state(st.thread_id)
        if run is None or not self._owns(st, run):
            return None
        st.run_id = _str(run.get("runId"))
        return run

    async def _adopt_run(self, client: HopliteClient, st: _RunState, run_id: str) -> None:
        run = await client.run_state(st.thread_id or "", run_id)
        if run is not None and self._owns(st, run):
            st.run_id = run_id

    async def _sync_messages(
        self, client: HopliteClient, st: _RunState, handle: RunHandle, *, force: bool = False
    ) -> None:
        """Fetch newly persisted assistant messages of this run and stream what the drafts missed."""
        if not st.run_id or not st.thread_id:
            return
        wait = self._settings.message_sync_interval_seconds - (time.monotonic() - st.last_sync)
        if wait > 0 and not force:
            await asyncio.sleep(wait)
        st.last_sync = time.monotonic()
        fresh = [m for m in await self._run_messages(client, st.thread_id, st.run_id)
                 if str(m["id"]) not in st.seen_messages]
        for msg in fresh:
            msg_id = str(msg["id"])
            st.seen_messages.add(msg_id)
            text = str(msg.get("content") or "")
            if (msg.get("kind") or "chat") == "thinking":
                if self._settings.include_reasoning and text.strip():
                    piece = ("\n\n" if st.reasoning else "") + text
                    st.reasoning.append(piece)
                    handle.emit(Delta(reasoning=piece))
                continue
            delta = st.tracker.finalize(msg_id, text)
            if delta:
                handle.emit(Delta(content=delta))

    async def _run_messages(self, client: HopliteClient, thread_id: str, run_id: str) -> list[dict[str, Any]]:
        """Assistant chat/thinking messages persisted for one run, oldest first."""
        collected: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(5):
            page = await client.list_messages(thread_id, limit=100, activity_limit=0, cursor=cursor)
            messages = [m for m in page.get("messages") or [] if isinstance(m, dict)]
            collected.extend(messages)
            cursor = _str(page.get("nextCursor"))
            if not page.get("hasMore") or not cursor or any(_str(m.get("runId")) != run_id for m in messages):
                break
        ours = [
            m for m in collected
            if _str(m.get("runId")) == run_id and m.get("role") == "assistant" and _str(m.get("id"))
            and (m.get("kind") or "chat") in ("chat", "thinking")
        ]
        ours.sort(key=lambda m: (str(m.get("createdAt") or ""), str(m.get("id"))))
        return ours

    async def _poll(self, client: HopliteClient, st: _RunState, handle: RunHandle) -> dict[str, Any]:
        interval = 2.0
        while True:
            run = await self._snapshot(client, st)
            await self._sync_messages(client, st, handle, force=True)
            if run is not None and _settled(run):
                return run
            await asyncio.sleep(interval)
            interval = min(5.0, interval * 1.5)

    async def _auto_answer(self, client: HopliteClient, job: RunJob, st: _RunState, round_: int) -> bool:
        assert st.thread_id is not None
        questions = [q for q in await client.list_questions(st.thread_id) if _str(q.get("runId")) in (None, st.run_id)]
        if not questions:
            return False
        st.cursor = await client.event_head(thread_id=st.thread_id)
        for question in questions:
            question_id = str(question.get("questionId"))
            data = await client.answer_question(
                st.thread_id, question_id, answer_body(question, self._settings.question_auto_answer),
                idempotency_key=f"{job.completion_id}:auto:{round_}:{question_id}"[:128],
            )
            st.run_id = _str((data.get("result") or {}).get("runId")) or st.run_id
        log.info("auto-answered the agent's question", extra={"thread_id": st.thread_id, "count": len(questions)})
        return True

    # --- finishing a run ------------------------------------------------------------------------

    async def _finalize(
        self, client: HopliteClient, job: RunJob, st: _RunState, handle: RunHandle, run: dict[str, Any]
    ) -> Finished:
        assert st.thread_id is not None
        status = str(run.get("status"))
        await self._sync_messages(client, st, handle, force=True)
        if not st.tracker.sent:
            final = run.get("assistantMessage")
            if isinstance(final, dict) and (final.get("kind") or "chat") == "chat" and final.get("content"):
                delta = st.tracker.finalize(str(final.get("id") or "final"), str(final["content"]))
                if delta:
                    handle.emit(Delta(content=delta))

        finish_reason = "stop"
        pending_question: str | None = None
        if status == "waiting":
            note, pending_question = await self._waiting_note(client, st)
            handle.emit(Delta(content=st.tracker.append_note(note)))
        elif status == "cancelled":
            log.warning("hoplite run was cancelled", extra={"thread_id": st.thread_id, "run_id": st.run_id})
            if not st.tracker.sent:
                raise OpenAIError(502, f"The Hoplite run was cancelled before it produced an answer "
                                       f"(thread {st.thread_id}).", type_="api_error", code="run_cancelled")
        elif status == "failed":
            detail = await self._failure_quiet(client, st)
            kind, error = classify_failure(detail)
            log.warning("hoplite run failed", extra={"thread_id": st.thread_id, "run_id": st.run_id,
                                                     "failure_code": detail.get("code"),
                                                     "failure": detail.get("message")})
            if kind == "spend_limit" and st.tracker.sent:
                finish_reason = "length"
            else:
                raise error

        usage = await self._usage_delta(client, st)
        content = st.tracker.sent
        try:
            await self._mapper.remember(job.ctx, job.messages, content, thread_id=st.thread_id,
                                        run_id=st.run_id, pending_question_id=pending_question)
        except Exception:
            log.exception("failed to store the conversation fingerprint")
        log.info("chat completion finished", extra={
            "thread_id": st.thread_id, "run_id": st.run_id, "status": status, "finish_reason": finish_reason,
            "chars": len(content), "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
        })
        return Finished(finish_reason, content, "".join(st.reasoning), usage, st.thread_id, st.run_id)

    async def _waiting_note(self, client: HopliteClient, st: _RunState) -> tuple[str, str | None]:
        assert st.thread_id is not None
        try:
            questions = await client.list_questions(st.thread_id)
        except HopliteError:
            questions = []
        mine = [q for q in questions if _str(q.get("runId")) in (None, st.run_id)]
        if mine:
            return format_question(mine[0]), _str(mine[0].get("questionId"))
        return (f"_The Hoplite agent is waiting for input (thread {st.thread_id}). "
                "Send a follow-up message to continue._"), None

    async def _usage_quiet(self, client: HopliteClient, thread_id: str) -> dict[str, Any]:
        try:
            return await client.thread_usage(thread_id)
        except HopliteError as exc:
            log.info("thread usage unavailable", extra={"status": exc.status, "code": exc.code})
            return {}

    async def _usage_delta(self, client: HopliteClient, st: _RunState) -> Usage:
        assert st.thread_id is not None
        after = await self._usage_quiet(client, st.thread_id)
        before = st.usage_before

        def diff(key: str) -> float:
            return max(0.0, float(after.get(key) or 0) - float(before.get(key) or 0))

        cost = diff("totalCostMicros") if "totalCostMicros" in after else None
        return Usage(int(diff("inputTokens")), int(diff("outputTokens")), cost)

    async def _failure_quiet(self, client: HopliteClient, st: _RunState) -> dict[str, Any]:
        if not st.thread_id or not st.run_id:
            return {}
        try:
            return await client.run_failure(st.thread_id, st.run_id)
        except HopliteError:
            return {}

    async def _thread_model(self, client: HopliteClient, thread_id: str) -> str | None:
        try:
            return _str((await client.get_thread(thread_id)).get("modelId"))
        except HopliteError:
            return None

    async def _stop_quietly(self, client: HopliteClient, job: RunJob, st: _RunState, *, reason: str) -> None:
        if not st.accepted or not st.thread_id:
            return
        run_id = st.run_id
        try:
            async with asyncio.timeout(10):
                if run_id is None:
                    run_id = _str((await client.active_run(st.thread_id) or {}).get("runId"))
                if run_id is None:
                    return
                await client.stop_run(st.thread_id, run_id, idempotency_key=f"{job.completion_id}:stop")
            log.info("hoplite run stopped", extra={"reason": reason, "thread_id": st.thread_id, "run_id": run_id})
        except Exception as exc:
            log.warning("could not stop the hoplite run",
                        extra={"reason": reason, "thread_id": st.thread_id, "error": repr(exc)})
