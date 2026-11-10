"""OpenAI-compatible endpoints: /v1/chat/completions, /v1/models."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from app.auth import get_services, require_client_key
from app.context import KeyContext
from app.errors import not_found
from app.hoplite.client import HopliteError, to_openai_error
from app.log import request_id_var
from app.openai_api.schemas import ChatCompletionRequest
from app.openai_api.sse import DONE, KEEPALIVE, chunk, completion, sse_data, tool_calls_delta, usage_chunk
from app.service.conversation import fingerprint
from app.service.inflight import Delta, Failed, Finished, RunHandle, Update
from app.service.models import DEFAULT_MODEL_ID, Catalog, ResolvedModel, reasoning_for
from app.service.prompt import check_tools, omitted_media, unsupported_fields
from app.service.runner import RunJob
from app.service.toolbridge import ToolMode, tool_mode
from app.services import Services

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai"])

DISCONNECT_CHECK_SECONDS = 1.0


# --- models ------------------------------------------------------------------------------------


async def _catalog(services: Services, ctx: KeyContext) -> Catalog:
    try:
        return await services.models.catalog(ctx, services.hoplite(ctx.hoplite_api_key))
    except HopliteError as exc:
        raise to_openai_error(exc) from exc


def _model_entries(services: Services, catalog: Catalog) -> list[dict[str, Any]]:
    created = int(services.started_at)
    entries = [{"id": DEFAULT_MODEL_ID, "object": "model", "created": created, "owned_by": "hoplite"}]
    seen = {DEFAULT_MODEL_ID}
    for model in catalog.models:
        if model["id"] not in seen:
            seen.add(model["id"])
            entries.append({"id": model["id"], "object": "model", "created": created,
                            "owned_by": model.get("provider") or "hoplite"})
    for alias in services.settings.model_aliases:
        if alias not in seen:
            seen.add(alias)
            entries.append({"id": alias, "object": "model", "created": created, "owned_by": "proxy-alias"})
    return entries


@router.get("/models")
async def list_models(
    ctx: KeyContext = Depends(require_client_key), services: Services = Depends(get_services)
) -> dict[str, Any]:
    return {"object": "list", "data": _model_entries(services, await _catalog(services, ctx))}


@router.get("/models/{model_id:path}")
async def get_model(
    model_id: str, ctx: KeyContext = Depends(require_client_key), services: Services = Depends(get_services)
) -> dict[str, Any]:
    for entry in _model_entries(services, await _catalog(services, ctx)):
        if entry["id"] == model_id:
            return entry
    raise not_found(f"The model `{model_id}` does not exist.", code="model_not_found", param="model")


# --- chat completions -------------------------------------------------------------------------


def _retry_count(request: Request) -> int:
    try:
        return int(request.headers.get("x-stainless-retry-count", "0"))
    except ValueError:
        return 0


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def request_fingerprint(
    ctx: KeyContext, model: ResolvedModel, body: ChatCompletionRequest, tools: ToolMode | None
) -> str:
    scope = ("request", ctx.key_id, ctx.project_id, model.hoplite_id or "", tools.digest() if tools else "")
    return fingerprint(*scope, messages=body.messages)


@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    ctx: KeyContext = Depends(require_client_key),
    services: Services = Depends(get_services),
) -> Response:
    settings = services.settings
    tools = tool_mode(body)
    if tools is not None and settings.tools_mode == "reject":
        check_tools(body)  # raises 400 tools_not_supported
    ignored = unsupported_fields(body)
    if ignored:
        log.info("ignoring unsupported request fields", extra={"ignored_fields": ignored})
    omitted = omitted_media(body.messages)
    if omitted:
        log.info("dropping unsupported content parts", extra={"omitted": omitted})
    if settings.log_prompts:
        log.info("request messages", extra={"messages": [m.model_dump(exclude_none=True) for m in body.messages]})

    model = await services.models.resolve(body.model, ctx, services.hoplite(ctx.hoplite_api_key))
    request_fp = request_fingerprint(ctx, model, body, tools)
    idempotency_key = request.headers.get("idempotency-key")
    handle = services.registry.lookup(request_fp, is_retry=_retry_count(request) > 0, idempotency_key=idempotency_key)
    if handle is not None:
        log.info("attaching to an in-flight run", extra={"completion_id": handle.completion_id})
    else:
        plan = await services.mapper.plan(
            ctx,
            body.messages,
            explicit_thread_id=request.headers.get("x-hoplite-thread-id") or None,
            force_new=_truthy(request.headers.get("x-hoplite-new-thread")),
            tools=tools,
        )
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        handle = services.registry.create(
            request_fp, ctx.key_id, completion_id=completion_id, created=int(time.time()),
            idempotency_key=idempotency_key,
        )
        job = RunJob(
            ctx=ctx,
            messages=list(body.messages),
            model=model,
            reasoning=reasoning_for(body.reasoning_effort),
            plan=plan,
            completion_id=completion_id,
            request_id=request_id_var.get(),
            tools=tools,
        )
        services.registry.start(handle, services.runner.execute(handle, job))
        log.info("chat completion accepted", extra={
            "completion_id": completion_id, "plan": plan.mode, "via": plan.reason, "thread_id": plan.thread_id,
            "model": model.label, "stream": body.stream, "client_tools": len(tools.tools) if tools else 0,
        })

    queue = handle.subscribe()
    streaming = False
    try:
        started = await handle.wait_started()
        headers = {"X-Hoplite-Thread-Id": started.thread_id}
        if started.run_id:
            headers["X-Hoplite-Run-Id"] = started.run_id
        if body.stream:
            include_usage = bool(body.stream_options and body.stream_options.include_usage)
            streaming = True
            return StreamingResponse(
                _stream(request, handle, queue, started.model, include_usage, settings.keepalive_seconds),
                media_type="text/event-stream",
                headers={**headers, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        outcome = await _wait_outcome(request, queue)
        if outcome is None:  # the client went away; the run gets a grace period for a retry
            return Response(status_code=499)
        if isinstance(outcome, Failed):
            raise outcome.error
        return JSONResponse(
            completion(handle.completion_id, handle.created, started.model, outcome,
                       include_reasoning=settings.include_reasoning),
            headers=headers,
        )
    finally:
        if not streaming:
            handle.unsubscribe(queue)


async def _wait_outcome(request: Request, queue: asyncio.Queue[Update]) -> Finished | Failed | None:
    while True:
        try:
            update = await asyncio.wait_for(queue.get(), timeout=DISCONNECT_CHECK_SECONDS)
        except TimeoutError:
            if await request.is_disconnected():
                return None
            continue
        if isinstance(update, (Finished, Failed)):
            return update


async def _stream(
    request: Request,
    handle: RunHandle,
    queue: asyncio.Queue[Update],
    model: str,
    include_usage: bool,
    keepalive_seconds: float,
) -> AsyncIterator[str]:
    cid, created = handle.completion_id, handle.created
    try:
        yield sse_data(chunk(cid, created, model, {"role": "assistant", "content": ""}, include_usage=include_usage))
        last_write = time.monotonic()
        while True:
            try:
                update = await asyncio.wait_for(queue.get(), timeout=min(DISCONNECT_CHECK_SECONDS, keepalive_seconds))
            except TimeoutError:
                if await request.is_disconnected():
                    return
                if time.monotonic() - last_write >= keepalive_seconds:
                    yield KEEPALIVE
                    last_write = time.monotonic()
                continue
            if isinstance(update, Delta):
                delta: dict[str, Any] = {}
                if update.content:
                    delta["content"] = update.content
                if update.reasoning:
                    delta["reasoning_content"] = update.reasoning
                if delta:
                    yield sse_data(chunk(cid, created, model, delta, include_usage=include_usage))
                    last_write = time.monotonic()
            elif isinstance(update, Finished):
                if update.tool_calls:
                    yield sse_data(chunk(cid, created, model, tool_calls_delta(update.tool_calls),
                                         include_usage=include_usage))
                yield sse_data(chunk(cid, created, model, {}, update.finish_reason, include_usage=include_usage))
                if include_usage:
                    yield sse_data(usage_chunk(cid, created, model, update))
                yield DONE
                return
            elif isinstance(update, Failed):
                yield sse_data(update.error.body())
                yield DONE
                return
    finally:
        handle.unsubscribe(queue)
