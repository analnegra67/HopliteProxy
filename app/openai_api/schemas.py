"""Request models for the OpenAI Chat Completions API (only fields the proxy interprets).

Every other field is accepted (`extra="allow"`) so that clients are never rejected for sending
parameters Hoplite has no equivalent for; the router logs their names and ignores them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "developer", "user", "assistant", "tool", "function"]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Role
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="allow")

    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    reasoning_effort: str | None = None
    user: str | None = None
    metadata: dict[str, Any] | None = None
    # Client-side tools are served by the tool bridge (app.service.toolbridge).
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    parallel_tool_calls: bool | None = None
    functions: list[dict[str, Any]] | None = None
    function_call: Any = None
