"""Translate OpenAI chat messages into Hoplite prompts and follow-up messages."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.errors import OpenAIError, invalid_request
from app.openai_api.schemas import ChatCompletionRequest, ChatMessage

SYSTEM_ROLES = frozenset({"system", "developer"})

TOOLS_NOT_SUPPORTED = (
    "Client-side tool calling is not supported: this endpoint is backed by Hoplite cloud coding agents, "
    "which run their own tools inside a sandbox and never return OpenAI `tool_calls`. "
    "Remove `tools`/`functions` from the request (or set `tool_choice` to \"none\")."
)

_MEDIA_LABELS = {
    "image_url": "image",
    "input_image": "image",
    "input_audio": "audio",
    "file": "file",
    "input_file": "file",
}


def content_text(content: str | list[dict[str, Any]] | None) -> tuple[str, list[str]]:
    """Return the text of a message's content and the kinds of parts that had to be dropped."""
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    texts: list[str] = []
    omitted: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind in ("text", "input_text", "output_text") and isinstance(part.get("text"), str):
            texts.append(part["text"])
        elif kind == "refusal" and isinstance(part.get("refusal"), str):
            texts.append(part["refusal"])
        else:
            label = _MEDIA_LABELS.get(str(kind), str(kind or "unknown"))
            omitted.append(label)
            texts.append(f"[{label} omitted]")
    return "\n".join(texts), omitted


def message_text(msg: ChatMessage) -> str:
    return content_text(msg.content)[0]


def omitted_media(messages: Sequence[ChatMessage]) -> list[str]:
    kinds: set[str] = set()
    for msg in messages:
        kinds.update(content_text(msg.content)[1])
    return sorted(kinds)


def _tool_call_lines(msg: ChatMessage) -> list[str]:
    lines = []
    for call in msg.tool_calls or []:
        fn = call.get("function") or {}
        lines.append(f"[called tool {fn.get('name', '?')} with arguments {fn.get('arguments', '')}]")
    return lines


def render_message(msg: ChatMessage) -> str:
    text = message_text(msg).strip()
    if msg.role == "user":
        label = f"User ({msg.name})" if msg.name else "User"
    elif msg.role == "assistant":
        label = "Assistant"
    elif msg.role in ("tool", "function"):
        label = f"Tool result ({msg.tool_call_id or msg.name or 'unknown'})"
    else:
        label = "Instructions"
    body = "\n".join(line for line in [text, *_tool_call_lines(msg)] if line)
    return f"{label}: {body or '(empty)'}"


def build_thread_prompt(messages: Sequence[ChatMessage]) -> str:
    """Flatten a whole OpenAI conversation into the initial prompt of a new Hoplite thread."""
    instructions = [t for t in (message_text(m).strip() for m in messages if m.role in SYSTEM_ROLES) if t]
    conversation = [m for m in messages if m.role not in SYSTEM_ROLES]
    if not conversation:
        raise invalid_request("`messages` must contain at least one user message.", param="messages")
    current, history = conversation[-1], conversation[:-1]
    current_text = (message_text(current).strip() if current.role == "user" else render_message(current))
    current_text = current_text or "(empty message)"
    if not instructions and not history:
        return current_text
    sections = []
    if instructions:
        sections.append("<instructions>\n" + "\n\n".join(instructions) + "\n</instructions>")
    if history:
        sections.append(
            "<conversation_history>\n" + "\n\n".join(render_message(m) for m in history) + "\n</conversation_history>"
        )
        sections.append("Respond to the latest message:\n\n" + current_text)
    else:
        sections.append(current_text)
    return "\n\n".join(sections)


def build_followup_content(tail: Sequence[ChatMessage]) -> str:
    """Render the messages that follow the last assistant reply as one message for an existing thread."""
    if not tail:
        raise invalid_request("Nothing to send: the conversation ends with an assistant message.", param="messages")
    *before, last = tail
    rendered = []
    for msg in before:
        if msg.role in SYSTEM_ROLES:
            rendered.append("Updated instructions:\n" + message_text(msg).strip())
        else:
            rendered.append(render_message(msg))
    if last.role == "user":
        rendered.append(message_text(last).strip() or "(empty message)")
    elif last.role in SYSTEM_ROLES:
        rendered.append("Updated instructions:\n" + message_text(last).strip())
    else:
        rendered.append(render_message(last))
    return "\n\n".join(part for part in rendered if part)


def check_tools(req: ChatCompletionRequest) -> None:
    has_tools = bool(req.tools) or bool(req.functions)
    declined = req.tool_choice == "none" or req.function_call == "none"
    if has_tools and not declined:
        raise OpenAIError(400, TOOLS_NOT_SUPPORTED, code="tools_not_supported",
                          param="tools" if req.tools else "functions")


def unsupported_fields(req: ChatCompletionRequest) -> list[str]:
    """Names of request fields that were sent but have no Hoplite equivalent (they are ignored)."""
    return sorted(name for name, value in (req.model_extra or {}).items() if value is not None)
