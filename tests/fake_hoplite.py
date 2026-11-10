"""In-memory fake of the Hoplite API, used by the tests and runnable as a local dev server:

    python -m tests.fake_hoplite --port 9100        (accepts the API key hop_dev_key)

Agent behaviour is scripted: a behaviour function maps the prompt / message text to steps
(Say, Think, Tool, Ask, Fail, Hang). A run plays its steps while a client watches the thread's
event stream (or, with the stream disabled, whenever run-state is polled).

The stream mimics what the live API sent in a probe (2026-11): `draft` frames
`{"messageId","runId","streamId":"assistant:s0","revision","content"}` without a cursor, then
`draft.removed`, and no `message.persisted` for the assistant reply before `run.completed`.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# --- scripted agent steps -------------------------------------------------------------------------


@dataclass
class Say:
    text: str
    chunks: int = 3
    draft_coverage: float = 1.0  # share of the text the drafts reach before the message is persisted


@dataclass
class Think:
    text: str


@dataclass
class Tool:
    name: str = "bash"
    output: str = "ok"


@dataclass
class Ask:
    question: str
    header: str = "Question"
    options: list[str] = field(default_factory=list)


@dataclass
class Fail:
    message: str = "Sorry, an error occurred"
    code: str = "agent_error"


@dataclass
class Hang:
    """Pause the run (it stays `running`) until `FakeHoplite.open_gate(gate)` is called, or forever."""

    gate: str | None = None


Step = Say | Think | Tool | Ask | Fail | Hang
Behaviour = Callable[[str], list[Step]]


def echo(text: str) -> list[Step]:
    return [Say(f"You said: {text}")]


# --- state ------------------------------------------------------------------------------------------


@dataclass
class FakeRun:
    id: str
    thread_id: str
    user_message_id: str
    steps: list[Step]
    status: str = "queued"
    position: int = 0
    finished_at: str | None = None
    failure: dict[str, Any] | None = None
    assistant_message: dict[str, Any] | None = None


@dataclass
class FakeThread:
    id: str
    project_id: str
    model_id: str
    spend_limit: int | None
    body: dict[str, Any]
    messages: list[dict[str, Any]] = field(default_factory=list)
    runs: list[FakeRun] = field(default_factory=list)
    questions: dict[str, dict[str, Any]] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=lambda: {"inputTokens": 0, "outputTokens": 0, "totalCostMicros": 0})
    archived: bool = False


class FakeError(Exception):
    def __init__(self, status: int, code: str, message: str = "", headers: dict[str, str] | None = None) -> None:
        super().__init__(code)
        self.status, self.code, self.message, self.headers = status, code, message or code, headers


TERMINAL = {"completed", "failed", "cancelled"}


class FakeHoplite:
    def __init__(
        self,
        *,
        api_keys: set[str] | None = None,
        projects: list[dict[str, Any]] | None = None,
        behaviour: Behaviour = echo,
        draft_event: str = "draft",
        frame_delay: float = 0.0,
        stream_enabled: bool = True,
        announce_assistant_messages: bool = False,
    ) -> None:
        self.api_keys = api_keys or {"hop_test_key"}
        self.projects = projects if projects is not None else [{"id": "proj_1", "name": "Test project", "repos": []}]
        self.models = [
            {"id": "gpt-6-sol", "name": "GPT-6 Sol", "provider": "openai"},
            {"id": "claude-sonnet-5", "name": "Claude Sonnet 5", "provider": "anthropic"},
        ]
        self.default_model = "gpt-6-sol"
        self.behaviour = behaviour
        self.draft_event = draft_event
        self.frame_delay = frame_delay
        self.stream_enabled = stream_enabled
        self.announce_assistant_messages = announce_assistant_messages
        self.threads: dict[str, FakeThread] = {}
        self.events: list[dict[str, Any]] = []
        self.calls: list[tuple[str, str, Any]] = []
        self.stops: list[tuple[str, str]] = []
        self.write_keys: list[tuple[str, str | None]] = []
        self.answers: list[Any] = []
        self.expired_cursors: set[str] = set()
        self.open_gates: set[str] = set()
        self.stream_connections = 0
        self._idempotency: dict[str, tuple[str, int, Any]] = {}
        self._injected: dict[tuple[str, str], list[tuple[int, Any, dict[str, str] | None]]] = {}
        self._ids = itertools.count(1)
        self.app = self._build_app()

    # --- helpers for tests ------------------------------------------------------------------------

    def inject(self, method: str, route: str, status: int, body: Any = None,
               headers: dict[str, str] | None = None, times: int = 1) -> None:
        """Make the next `times` calls to `method route` (a path template) return this response."""
        self._injected.setdefault((method, route), []).extend([(status, body, headers)] * times)

    def open_gate(self, gate: str) -> None:
        self.open_gates.add(gate)

    def calls_to(self, method: str, path_prefix: str) -> list[tuple[str, str, Any]]:
        return [c for c in self.calls if c[0] == method and c[1].startswith(path_prefix)]

    def thread_creations(self) -> list[dict[str, Any]]:
        return [c[2] for c in self.calls if c[0] == "POST" and c[1] == "/api/threads"]

    # --- internals ----------------------------------------------------------------------------------

    def _id(self, prefix: str) -> str:
        return f"{prefix}_{next(self._ids):04d}"

    def _thread(self, thread_id: str) -> FakeThread:
        thread = self.threads.get(thread_id)
        if thread is None:
            raise FakeError(404, "thread_not_found", f"Thread {thread_id} not found")
        return thread

    async def _enter(self, request: Request, route: str) -> Any:
        body = None
        if request.method in ("POST", "PATCH", "PUT"):
            raw = await request.body()
            body = json.loads(raw) if raw else None
        self.calls.append((request.method, request.url.path, body))
        if request.method != "GET":
            self.write_keys.append((request.url.path, request.headers.get("idempotency-key")))
        key = request.headers.get("x-api-key")
        auth = request.headers.get("authorization", "")
        if not key and auth.lower().startswith("bearer "):
            key = auth[7:]
        if key not in self.api_keys:
            raise FakeError(401, "unauthorized", "Invalid API key")
        queue = self._injected.get((request.method, route))
        if queue:
            status, payload, headers = queue.pop(0)
            raise FakeError(status, (payload or {}).get("error", "injected"), (payload or {}).get("message", ""),
                            headers)
        return body

    def _idempotent(self, request: Request, body: Any, status: int, produce: Callable[[], Any]) -> JSONResponse:
        key = request.headers.get("idempotency-key")
        for legacy in ("clientOperationId", "clientMessageId"):  # must match the header when present
            if isinstance(body, dict) and body.get(legacy) is not None and body[legacy] != key:
                raise FakeError(400, "idempotency_key_mismatch")
        digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        if key and key in self._idempotency:
            seen_digest, seen_status, seen_payload = self._idempotency[key]
            if seen_digest != digest:
                raise FakeError(409, "client_operation_conflict", "Idempotency key reused with a different payload")
            return JSONResponse({**seen_payload, "idempotent": True}, status_code=seen_status)
        payload = produce()
        if key:
            self._idempotency[key] = (digest, status, payload)
        return JSONResponse(payload, status_code=status)

    def _persist(self, thread: FakeThread, run: FakeRun | None, *, role: str, kind: str, content: str,
                 msg_id: str | None = None, **extra: Any) -> dict[str, Any]:
        msg = {
            "id": msg_id or self._id("msg"), "threadId": thread.id, "runId": run.id if run else None,
            "role": role, "kind": kind, "content": content, "createdAt": _now(), "metadata": None, **extra,
        }
        thread.messages.append(msg)
        return msg

    def _new_run(self, thread: FakeThread, user_msg: dict[str, Any], text: str) -> FakeRun:
        run = FakeRun(self._id("run"), thread.id, user_msg["id"], list(self.behaviour(text)))
        thread.runs.append(run)
        user_msg["runId"] = run.id
        return run

    def _durable(
        self, thread: FakeThread, run: FakeRun, event_type: str, data: dict[str, Any]
    ) -> tuple[str, dict, bool]:
        sequence = len(self.events) + 1
        event = {
            "id": f"evt_{sequence}", "orgId": "org_test", "projectId": thread.project_id, "threadId": thread.id,
            "runId": run.id, "sequence": str(sequence), "type": event_type, "data": data,
            "occurredAt": _now(), "publishedAt": _now(), "cursor": f"c{sequence}",
        }
        self.events.append(event)
        return event_type, event, True

    def draft_frame(self, thread: FakeThread, run: FakeRun, msg_id: str, revision: int, text: str) -> dict:
        return {"messageId": msg_id, "runId": run.id, "streamId": "assistant:s0", "revision": revision,
                "content": text}

    def _persisted(self, thread: FakeThread, run: FakeRun, msg: dict[str, Any]) -> Iterator[tuple[str, dict, bool]]:
        if msg["role"] != "assistant" or self.announce_assistant_messages:
            yield self._durable(thread, run, "message.persisted", {"id": msg["id"], "role": msg["role"]})

    def _advance(self, thread: FakeThread, run: FakeRun) -> Iterator[tuple[str, dict, bool]]:
        """Play the run's remaining steps, mutating state; yields (event name, payload, durable)."""
        if run.status in TERMINAL or run.status == "waiting":
            return
        if run.status == "queued":
            run.status = "running"
            thread.usage["inputTokens"] += 100
            thread.usage["totalCostMicros"] += 2500
            user = next((m for m in thread.messages if m["id"] == run.user_message_id), None)
            if user is not None:
                yield from self._persisted(thread, run, user)
            yield self._durable(thread, run, "run.running", {"id": run.id, "status": "running"})
        while run.position < len(run.steps):
            step = run.steps[run.position]
            if isinstance(step, Hang) and not (step.gate and step.gate in self.open_gates):
                return
            run.position += 1
            if isinstance(step, Say):
                msg_id = self._id("msg")
                count = max(1, step.chunks)
                reach = math.ceil(len(step.text) * step.draft_coverage)
                for revision in range(1, count + 1):
                    snapshot = step.text[: math.ceil(reach * revision / count)]
                    if snapshot:
                        yield self.draft_event, self.draft_frame(thread, run, msg_id, revision, snapshot), False
                yield "draft.removed", {"messageId": msg_id}, False
                run.assistant_message = self._persist(thread, run, role="assistant", kind="chat",
                                                      content=step.text, msg_id=msg_id)
                thread.usage["outputTokens"] += len(step.text.split())
                yield from self._persisted(thread, run, run.assistant_message)
            elif isinstance(step, Think):
                yield from self._persisted(
                    thread, run, self._persist(thread, run, role="assistant", kind="thinking", content=step.text))
            elif isinstance(step, Tool):
                yield from self._persisted(thread, run, self._persist(
                    thread, run, role="tool", kind="tool", content=step.output, toolName=step.name))
            elif isinstance(step, Ask):
                question_id = self._id("q")
                thread.questions[question_id] = {
                    "questionId": question_id, "runId": run.id,
                    "request": {"questions": [{"header": step.header, "question": step.question,
                                               "multiSelect": False, "options": [{"label": o} for o in step.options]}]},
                }
                run.status = "waiting"
                yield self._durable(thread, run, "run.waiting", {"id": run.id, "status": "waiting"})
                return
            elif isinstance(step, Fail):
                run.status, run.finished_at = "failed", _now()
                run.failure = {"code": step.code, "message": step.message}
                yield self._durable(thread, run, "run.failed", {"id": run.id, "status": "failed"})
                return
        run.status, run.finished_at = "completed", _now()
        yield self._durable(thread, run, "run.completed", {"id": run.id, "status": "completed"})

    def _thread_view(self, thread: FakeThread) -> dict[str, Any]:
        latest = thread.runs[-1] if thread.runs else None
        return {
            "id": thread.id, "projectId": thread.project_id, "modelId": thread.model_id, "title": None,
            "status": "archived" if thread.archived else (
                "running" if latest and latest.status == "running" else "ready"),
            "waitingOn": [], "blockedOn": [], "pendingWakeups": 0, "tasks": [], "pullRequests": [],
            "latestRunId": latest.id if latest else None, "runStatus": latest.status if latest else None,
            "spendLimitMicros": thread.spend_limit, "createdAt": _now(), "updatedAt": _now(),
        }

    def _run_view(self, thread: FakeThread, run: FakeRun) -> dict[str, Any]:
        user = next((m for m in thread.messages if m["id"] == run.user_message_id), None)
        return {"runId": run.id, "status": run.status, "finishedAt": run.finished_at,
                "assistantMessage": run.assistant_message, "userMessage": user}

    def _check_model(self, model: str | None) -> None:
        if model and model not in {m["id"] for m in self.models}:
            raise FakeError(400, "model_unavailable", f"The selected model {model} is unavailable")

    # --- app ----------------------------------------------------------------------------------------

    def _build_app(self) -> FastAPI:  # noqa: C901 - one closure per endpoint keeps the fake readable
        app = FastAPI()

        @app.exception_handler(FakeError)
        async def _fake_error(_: Request, exc: Exception) -> JSONResponse:
            assert isinstance(exc, FakeError)
            return JSONResponse({"ok": False, "error": exc.code, "message": exc.message},
                                status_code=exc.status, headers=exc.headers)

        @app.get("/api/model-providers")
        async def models(request: Request) -> dict:
            await self._enter(request, "/api/model-providers")
            return {"ok": True, "modelConfigurationRevision": "1",
                    "providers": [{"provider": "openai", "label": "OpenAI", "defaultModelId": "gpt-6-sol"}],
                    "runConfig": {"defaultModelId": self.default_model, "defaultReasoning": "medium",
                                  "defaultSpeed": "standard", "defaultsVersion": 1, "modelFallbackIds": [],
                                  "providerMappingsVersion": 1, "reasoningOptions": ["low", "medium", "high"],
                                  "speedOptions": ["standard"]},
                    "models": self.models}

        @app.get("/api/projects")
        async def projects(request: Request) -> dict:
            await self._enter(request, "/api/projects")
            return {"ok": True, "projects": self.projects, "nextCursor": None}

        @app.get("/api/projects/{project_id}")
        async def project(request: Request, project_id: str) -> dict:
            await self._enter(request, "/api/projects/{id}")
            for item in self.projects:
                if item["id"] == project_id:
                    return {"ok": True, "project": item}
            raise FakeError(404, "project_not_found", "Project not found")

        @app.post("/api/threads")
        async def create_thread(request: Request) -> JSONResponse:
            body = await self._enter(request, "/api/threads") or {}
            if body.get("projectId") not in {p["id"] for p in self.projects}:
                raise FakeError(400, "invalid_project", "projectId is not a project of this organization")
            if not str(body.get("prompt") or "").strip():
                raise FakeError(400, "invalid_request", "prompt is required")
            self._check_model(body.get("model"))

            def produce() -> dict:
                thread = FakeThread(self._id("thr"), body["projectId"], body.get("model") or self.default_model,
                                    body.get("spendLimitMicros"), body)
                self.threads[thread.id] = thread
                user = self._persist(thread, None, role="user", kind="chat", content=body["prompt"])
                run = self._new_run(thread, user, body["prompt"])
                return {"ok": True, "thread": self._thread_view(thread), "message": user,
                        "run": {"id": run.id, "status": run.status}}

            return self._idempotent(request, body, 201, produce)

        @app.get("/api/threads/{thread_id}")
        async def get_thread(request: Request, thread_id: str) -> dict:
            await self._enter(request, "/api/threads/{id}")
            return {"ok": True, "thread": self._thread_view(self._thread(thread_id))}

        @app.post("/api/threads/{thread_id}/messages")
        async def append(request: Request, thread_id: str) -> JSONResponse:
            body = await self._enter(request, "/api/threads/{id}/messages") or {}
            thread = self._thread(thread_id)
            if thread.archived:
                raise FakeError(409, "thread_archived", "The thread is archived")
            if not str(body.get("content") or "").strip():
                raise FakeError(400, "invalid_request", "content is required")
            self._check_model(body.get("model"))

            def produce() -> dict:
                if body.get("model"):
                    thread.model_id = body["model"]
                user = self._persist(thread, None, role="user", kind="chat", content=body["content"])
                run = self._new_run(thread, user, body["content"])
                return {"ok": True, "message": {"id": user["id"]}, "queued": False,
                        "run": {"id": run.id, "status": run.status}}

            return self._idempotent(request, body, 201, produce)

        @app.get("/api/threads/{thread_id}/messages")
        async def messages(request: Request, thread_id: str, limit: int = 500, activityLimit: int = 5000) -> dict:
            await self._enter(request, "/api/threads/{id}/messages")
            thread = self._thread(thread_id)
            items = [m for m in thread.messages if activityLimit > 0 or m["kind"] not in ("tool", "status")]
            return {"ok": True, "messages": items[-limit:], "hasMore": len(items) > limit, "nextCursor": None}

        @app.get("/api/threads/{thread_id}/run-state")
        async def run_state(request: Request, thread_id: str, runId: str | None = None) -> dict:
            await self._enter(request, "/api/threads/{id}/run-state")
            thread = self._thread(thread_id)
            if not self.stream_enabled:  # polling mode: progress happens when the client polls
                for item in thread.runs:
                    for _ in self._advance(thread, item):
                        pass
            run = next((r for r in thread.runs if r.id == runId), None) if runId else (
                thread.runs[-1] if thread.runs else None)
            if run is None:
                raise FakeError(404, "run_not_found", "Run not found")
            return {"ok": True, "run": self._run_view(thread, run)}

        @app.get("/api/threads/{thread_id}/active-run")
        async def active_run(request: Request, thread_id: str) -> dict:
            await self._enter(request, "/api/threads/{id}/active-run")
            thread = self._thread(thread_id)
            active = next((r for r in reversed(thread.runs) if r.status not in TERMINAL), None)
            return {"ok": True, "activeRun": {"runId": active.id, "status": active.status} if active else None}

        @app.post("/api/threads/{thread_id}/stop")
        async def stop(request: Request, thread_id: str) -> JSONResponse:
            body = await self._enter(request, "/api/threads/{id}/stop") or {}
            thread = self._thread(thread_id)
            run = next((r for r in thread.runs if r.id == body.get("runId")), None)
            if run is None:
                raise FakeError(409, "stop_run_id_required", "runId is required")
            self.stops.append((thread_id, run.id))
            if run.status not in TERMINAL:
                run.status, run.finished_at = "cancelled", _now()
                self._durable(thread, run, "run.cancelled", {"id": run.id, "status": "cancelled"})
            return JSONResponse({"ok": True, "result": {"threadId": thread_id, "runId": run.id, "status": "stopping"}},
                                status_code=202)

        @app.get("/api/threads/{thread_id}/questions")
        async def questions(request: Request, thread_id: str) -> dict:
            await self._enter(request, "/api/threads/{id}/questions")
            return {"ok": True, "questions": list(self._thread(thread_id).questions.values())}

        @app.post("/api/threads/{thread_id}/questions/{question_id}")
        async def answer(request: Request, thread_id: str, question_id: str) -> JSONResponse:
            body = await self._enter(request, "/api/threads/{id}/questions/{questionId}") or {}
            thread = self._thread(thread_id)
            question = thread.questions.pop(question_id, None)
            if question is None:
                raise FakeError(404, "question_not_found", "Question not found")
            text = body.get("answer") if "answer" in body else json.dumps(body.get("answers"))
            self.answers.append(text)
            run = next(r for r in thread.runs if r.id == question["runId"])
            if run.position >= len(run.steps):
                run.steps.append(Say(f"Thanks, noted: {text}"))
            run.status = "running"
            return JSONResponse({"ok": True, "result": {"threadId": thread_id, "runId": run.id, "status": "running"}},
                                status_code=202)

        @app.get("/api/threads/{thread_id}/usage")
        async def usage(request: Request, thread_id: str) -> dict:
            await self._enter(request, "/api/threads/{id}/usage")
            thread = self._thread(thread_id)
            return {"ok": True, "usage": {**thread.usage, "runCount": len(thread.runs),
                                          "spendLimitMicros": thread.spend_limit}}

        @app.get("/api/threads/{thread_id}/runs/{run_id}/failure")
        async def failure(request: Request, thread_id: str, run_id: str) -> dict:
            await self._enter(request, "/api/threads/{id}/runs/{runId}/failure")
            thread = self._thread(thread_id)
            run = next((r for r in thread.runs if r.id == run_id), None)
            if run is None or run.failure is None:
                raise FakeError(404, "failure_not_found", "No failure recorded")
            return {"ok": True, "detail": run.failure}

        @app.get("/api/events/head")
        async def head(request: Request) -> dict:
            await self._enter(request, "/api/events/head")
            return {"ok": True, "cursor": f"c{len(self.events)}"}

        @app.get("/api/events/stream")
        async def stream(request: Request, threadId: str | None = None, cursor: str | None = None) -> Any:
            await self._enter(request, "/api/events/stream")
            if not self.stream_enabled:
                raise FakeError(501, "not_implemented", "Events are not configured")
            thread = self._thread(threadId or "")
            start = request.headers.get("last-event-id") or cursor
            if start in self.expired_cursors:
                raise FakeError(410, "cursor_expired", "Cursor expired")
            try:
                offset = int(start[1:]) if start else len(self.events)
            except ValueError:
                offset = len(self.events)
            self.stream_connections += 1

            async def frames() -> Any:
                for event in self.events[offset:]:
                    if event["threadId"] == thread.id:
                        yield _sse(event["type"], event, durable=True)
                run = next((r for r in thread.runs if r.status not in TERMINAL and r.status != "waiting"), None)
                if run is not None:
                    for name, payload, durable in self._advance(thread, run):
                        if self.frame_delay:
                            await asyncio.sleep(self.frame_delay)
                        yield _sse(name, payload, durable=durable)
                position = f"c{len(self.events)}"
                yield _sse("heartbeat", {"type": "heartbeat", "cursor": position}, durable=True, cursor=position)

            return StreamingResponse(frames(), media_type="text/event-stream")

        return app


def _sse(name: str, payload: dict[str, Any], *, durable: bool, cursor: str | None = None) -> str:
    lines = []
    if durable:
        lines.append(f"id: {cursor or payload.get('cursor')}")
    lines.append(f"event: {name}")
    lines.append("data: " + json.dumps(payload, separators=(",", ":")))
    return "\n".join(lines) + "\n\n"


def _word_by_word(text: str) -> list[Step]:
    reply = f"You said: {text}"
    return [Think("Reading the request."), Tool("bash", "ls"), Say(reply, chunks=max(1, len(reply.split())))]


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the fake Hoplite API locally")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--api-key", default="hop_dev_key")
    parser.add_argument("--delay", type=float, default=0.15, help="seconds between streamed frames")
    args = parser.parse_args()
    fake = FakeHoplite(api_keys={args.api_key}, behaviour=_word_by_word, frame_delay=args.delay)
    uvicorn.run(fake.app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
