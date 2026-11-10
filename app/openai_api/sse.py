"""Builders for OpenAI `chat.completion` / `chat.completion.chunk` payloads and SSE frames."""

from __future__ import annotations

import json
from typing import Any

from app.service.inflight import Finished

DONE = "data: [DONE]\n\n"
KEEPALIVE = ": keep-alive\n\n"


def sse_data(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def chunk(
    completion_id: str,
    created: int,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
    *,
    include_usage: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": None,
        "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}],
    }
    if include_usage:
        payload["usage"] = None
    return payload


def usage_chunk(completion_id: str, created: int, model: str, finished: Finished) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": None,
        "choices": [],
        "usage": finished.usage.as_openai(),
    }


def completion(
    completion_id: str, created: int, model: str, finished: Finished, *, include_reasoning: bool
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": finished.content, "refusal": None}
    if include_reasoning and finished.reasoning:
        message["reasoning_content"] = finished.reasoning
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": message, "logprobs": None, "finish_reason": finished.finish_reason}],
        "usage": finished.usage.as_openai(),
        "system_fingerprint": None,
        "hoplite": {
            "thread_id": finished.thread_id,
            "run_id": finished.run_id,
            "cost_micros": finished.usage.cost_micros,
        },
    }
