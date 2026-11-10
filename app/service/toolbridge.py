"""Client-side tool calling on top of Hoplite agents (the "tool bridge").

Agentic clients such as opencode run their own tools (read, edit, shell, ...) on the user's machine and
expect the model to request them through OpenAI `tool_calls`. A Hoplite agent cannot call those tools
directly, so the proxy describes them in the prompt together with a small text protocol:

    <tool_calls>
    [{"name": "read", "arguments": {"filePath": "src/app.ts"}}]
    </tool_calls>

When the agent's reply contains that block, the proxy returns it to the client as `tool_calls`
(finish_reason "tool_calls"). The client runs the tools and sends the results back as `tool` messages,
which the proxy appends to the same Hoplite thread as <tool_result> blocks.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.errors import invalid_request
from app.openai_api.schemas import ChatCompletionRequest, ChatMessage
from app.service.prompt import SYSTEM_ROLES, message_text

log = logging.getLogger(__name__)

OPEN = "<tool_calls>"
CLOSE = "</tool_calls>"

PROTOCOL = """<protocol>
You are the reasoning engine of a coding agent ("the client") that runs on the user's own computer. The
instructions and tools above belong to the client. These rules override anything else:

1. Your own built-in tools (Read, Glob, Grep, Bash, Edit, Write, web search, ...) run in a cloud sandbox that does
   NOT contain the user's files. Never use them in this conversation: their results would be about the wrong machine.
2. The client tools listed in <client_tools> are NOT functions in your function-calling interface; calling them
   that way fails with "unavailable tool". The only way to use them is to WRITE a block like this as plain text at
   the end of your reply:
<tool_calls>
[{{"name": "read", "arguments": {{"filePath": "README.md"}}}}]
</tool_calls>
   Each entry is {{"name": "<client tool name>", "arguments": {{...matching its JSON schema...}}}}. {parallel}
3. After the block, stop. The client runs the calls on the user's computer and replies with <tool_result> blocks.
4. A short note before the block is shown to the user. When the task is complete, or you need the user's input,
   reply with plain text and no <tool_calls> block.
</protocol>"""

REMINDER = ("(Reminder: the user's files are reachable only through client tools. Request them with a <tool_calls> "
            "block written as text; do not use your built-in tools.)")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]

    def manifest(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    @property
    def required(self) -> list[str]:
        required = self.parameters.get("required") if isinstance(self.parameters, dict) else None
        return [str(r) for r in required] if isinstance(required, list) else []


@dataclass(frozen=True)
class ToolMode:
    tools: tuple[ToolSpec, ...]
    choice: str = "auto"  # "auto", "required" or the name of the tool that must be called
    parallel: bool = True

    @property
    def names(self) -> list[str]:
        return [t.name for t in self.tools]

    def spec(self, name: str) -> ToolSpec | None:
        return next((t for t in self.tools if t.name == name), None)

    def digest(self) -> str:
        canonical = json.dumps([t.manifest() for t in self.tools], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]

    def choice_instruction(self) -> str:
        if self.choice == "required":
            return "For this reply you must call at least one client tool."
        if self.choice != "auto":
            return f"For this reply you must call the client tool `{self.choice}`."
        return ""


def tool_mode(req: ChatCompletionRequest) -> ToolMode | None:
    """The client tools offered by a request, or None when it offers none (or declines them)."""
    specs: list[ToolSpec] = []
    for item in req.tools or []:
        if item.get("type", "function") != "function":
            log.info("ignoring a non-function tool", extra={"tool_type": item.get("type")})
            continue
        specs.append(_spec(item.get("function") or {}))
    for fn in req.functions or []:  # legacy `functions`
        specs.append(_spec(fn))
    if not specs:
        return None
    choice: Any = req.tool_choice if req.tool_choice is not None else req.function_call
    if choice == "none":
        return None
    selected = "auto"
    if choice == "required":
        selected = "required"
    elif isinstance(choice, dict):
        selected = str((choice.get("function") or {}).get("name") or choice.get("name") or "auto")
    names = {s.name for s in specs}
    if selected not in ("auto", "required") and selected not in names:
        raise invalid_request(f"tool_choice refers to an unknown tool {selected!r}.", param="tool_choice")
    return ToolMode(tuple(specs), selected, req.parallel_tool_calls is not False)


def _spec(fn: dict[str, Any]) -> ToolSpec:
    name = fn.get("name")
    if not isinstance(name, str) or not name.strip():
        raise invalid_request("Every tool needs a function name.", param="tools")
    parameters = fn.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {"type": "object", "properties": {}}
    return ToolSpec(name.strip(), str(fn.get("description") or ""), parameters)


# --- prompts ---------------------------------------------------------------------------------------


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[... truncated by the proxy: {len(text) - limit} more characters ...]"


def _arguments(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            return raw
    return raw if raw is not None else {}


def calls_block(tool_calls: Sequence[dict[str, Any]]) -> str:
    calls = []
    for call in tool_calls:
        fn = call.get("function") or {}
        calls.append({"name": fn.get("name"), "arguments": _arguments(fn.get("arguments"))})
    return f"{OPEN}\n{json.dumps(calls, ensure_ascii=False)}\n{CLOSE}"


def _names_by_id(messages: Sequence[ChatMessage]) -> dict[str, str]:
    names: dict[str, str] = {}
    for msg in messages:
        for call in msg.tool_calls or []:
            if call.get("id"):
                names[str(call["id"])] = str((call.get("function") or {}).get("name") or "")
    return names


def _result_block(msg: ChatMessage, names: dict[str, str], limit: int) -> str:
    call_id = msg.tool_call_id or ""
    name = names.get(call_id) or msg.name or ""
    return f'<tool_result id="{call_id}" name="{name}">\n{_clip(message_text(msg), limit)}\n</tool_result>'


def _render(messages: Sequence[ChatMessage], names: dict[str, str], limit: int, *, labels: bool) -> str:
    parts: list[str] = []
    for msg in messages:
        text = message_text(msg).strip()
        if msg.role == "tool" or msg.role == "function":
            parts.append(_result_block(msg, names, limit))
        elif msg.role == "assistant":
            body = "\n".join(p for p in [text, calls_block(msg.tool_calls) if msg.tool_calls else ""] if p)
            parts.append(f"Assistant: {body or '(empty)'}")
        elif msg.role in SYSTEM_ROLES:
            parts.append(f"Updated instructions:\n{text}")
        else:
            parts.append(f"User: {text}" if labels else text)
    return "\n\n".join(parts)


def split_turn(messages: Sequence[ChatMessage]) -> tuple[list[ChatMessage], list[ChatMessage]]:
    """(history, current turn): the current turn is everything after the last assistant message."""
    conversation = [m for m in messages if m.role not in SYSTEM_ROLES]
    last = max((i for i, m in enumerate(conversation) if m.role == "assistant"), default=-1)
    return conversation[: last + 1], conversation[last + 1:]


def build_bridge_prompt(messages: Sequence[ChatMessage], mode: ToolMode, *, result_limit: int) -> str:
    instructions = [t for t in (message_text(m).strip() for m in messages if m.role in SYSTEM_ROLES) if t]
    history, current = split_turn(messages)
    if not history and not current:
        raise invalid_request("`messages` must contain at least one user message.", param="messages")
    names = _names_by_id(messages)
    parallel = ("You may put several independent calls in the array." if mode.parallel
                else "Put exactly one call in the array.")
    sections = []
    if instructions:
        sections.append(
            "<client_instructions>\nThe client agent's own instructions. Wherever they tell you to use a tool, that "
            "means requesting the client tool with a <tool_calls> block (see <protocol>).\n\n"
            + "\n\n".join(instructions) + "\n</client_instructions>"
        )
    sections.append("<client_tools>\n" + json.dumps([t.manifest() for t in mode.tools], ensure_ascii=False, indent=1)
                    + "\n</client_tools>")
    sections.append(PROTOCOL.format(parallel=parallel))
    if history:
        sections.append("<conversation_history>\n" + _render(history, names, result_limit, labels=True)
                        + "\n</conversation_history>")
    turn = _render(current, names, result_limit, labels=False) if current else "(continue)"
    sections.append(("Latest input:\n\n" if history else "User request:\n\n") + turn)
    sections.append(" ".join(p for p in (mode.choice_instruction(), REMINDER) if p))
    return "\n\n".join(sections)


def build_bridge_followup(messages: Sequence[ChatMessage], tail: Sequence[ChatMessage], mode: ToolMode, *,
                          result_limit: int) -> str:
    """The message appended to an existing thread: tool results and/or the user's next message."""
    if not tail:
        raise invalid_request("Nothing to send: the conversation ends with an assistant message.", param="messages")
    text = _render(tail, _names_by_id(messages), result_limit, labels=False)
    return text + "\n\n" + " ".join(p for p in (mode.choice_instruction(), REMINDER) if p)


def repair_message(problem: str, mode: ToolMode) -> str:
    return (f"Your previous reply could not be used: {problem}. Reply again following the <protocol>: write one "
            f"{OPEN} block as plain text with a JSON array of {{\"name\", \"arguments\"}} objects using these client "
            f"tools: {', '.join(mode.names)}; or reply with plain text and no block if you are done. "
            "Do not use your built-in tools, and do not mention this correction in your reply.")


# Hoplite's own housekeeping steps that appear as tool activity but are not the agent's choice.
_HARNESS_TOOLS = frozenset({"workspace_setup", "workspace-setup"})


def native_tool_misuse(activity: Sequence[dict[str, Any]], mode: ToolMode) -> str | None:
    """Describe the agent's use of its own sandbox tools during a bridged run, if any."""
    attempted: list[str] = []
    used: list[str] = []
    for item in activity:
        name = str(item.get("toolName") or "")
        if not name or name in _HARNESS_TOOLS:
            continue
        if "unavailable tool" in str(item.get("content") or "") and name in mode.names:
            attempted.append(name)
        elif name not in used:
            used.append(name)
    if attempted:
        return (f"you tried to call the client tool `{attempted[0]}` as a built-in function, which cannot work; "
                "client tools must be requested in a <tool_calls> block written as text")
    if used:
        return (f"you used your built-in tools ({', '.join(used[:5])}), which run in a cloud sandbox without the "
                "user's files, so their results are not about the user's project")
    return None


# --- parsing replies ---------------------------------------------------------------------------------


def visible_prefix(text: str) -> str:
    """The part of a (possibly partial) reply that can be shown: everything before the tool-call block,
    holding back a trailing fragment that might be the start of `<tool_calls>`."""
    start = text.find(OPEN)
    if start >= 0:
        return text[:start]
    for size in range(min(len(OPEN) - 1, len(text)), 0, -1):
        if OPEN.startswith(text[-size:]):
            return text[:-size]
    return text


def split_reply(text: str) -> tuple[str, list[str]]:
    """Split a final reply into the visible text before the first block and the raw contents of all blocks."""
    start = text.find(OPEN)
    if start < 0:
        return text, []
    blocks: list[str] = []
    position = start
    while position >= 0:
        body_start = position + len(OPEN)
        end = text.find(CLOSE, body_start)
        if end < 0:
            blocks.append(text[body_start:])
            break
        blocks.append(text[body_start:end])
        position = text.find(OPEN, end + len(CLOSE))
    return text[:start], blocks


_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
_ARGUMENT_KEYS_MISSING = object()


def _load_json(raw: str) -> Any:
    body = _FENCE.sub("", raw.strip())
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    # Tolerate prose before the JSON or junk after it, but only decode from the outermost bracket:
    # picking an inner object out of broken JSON would turn a syntax error into a wrong call.
    starts = [i for i in (body.find("["), body.find("{")) if i >= 0]
    if starts:
        try:
            return json.JSONDecoder().raw_decode(body[min(starts):])[0]
        except json.JSONDecodeError:
            pass
    raise ValueError("the content of the <tool_calls> block is not valid JSON")


def parse_calls(blocks: Sequence[str], mode: ToolMode) -> tuple[list[dict[str, Any]], str | None]:
    """Validate the agent's tool calls. Returns (calls, None) or ([], problem description)."""
    calls: list[dict[str, Any]] = []
    for raw in blocks:
        try:
            data = _load_json(raw)
        except ValueError as exc:
            return [], str(exc)
        if isinstance(data, dict):
            data = data["tool_calls"] if isinstance(data.get("tool_calls"), list) else [data]
        if not isinstance(data, list):
            return [], "the <tool_calls> block must contain a JSON array of calls"
        for item in data:
            if not isinstance(item, dict):
                return [], "every tool call must be a JSON object with `name` and `arguments`"
            name = item.get("name") or (item.get("function") or {}).get("name")
            arguments = _ARGUMENT_KEYS_MISSING
            for key in ("arguments", "args", "input", "parameters"):
                if key in item:
                    arguments = item[key]
                    break
            if isinstance(item.get("function"), dict) and "arguments" in item["function"]:
                arguments = item["function"]["arguments"]
            if arguments is _ARGUMENT_KEYS_MISSING:
                # A common slip: arguments written next to "name" instead of nested under "arguments".
                arguments = {k: v for k, v in item.items() if k not in ("name", "type", "id", "function")}
            arguments = _arguments(arguments)
            if not isinstance(name, str) or mode.spec(name) is None:
                return [], f"unknown tool {name!r}; the client tools are: {', '.join(mode.names)}"
            if not isinstance(arguments, dict):
                return [], f"the arguments of `{name}` must be a JSON object"
            missing = [r for r in mode.spec(name).required if r not in arguments]  # type: ignore[union-attr]
            if missing:
                return [], f"`{name}` is missing required arguments: {', '.join(missing)}"
            calls.append({"name": name, "arguments": arguments})
    if not calls:
        return [], "the <tool_calls> block is empty"
    if mode.choice not in ("auto", "required") and not any(c["name"] == mode.choice for c in calls):
        return [], f"this reply must call `{mode.choice}`"
    if not mode.parallel and len(calls) > 1:
        log.info("dropping extra tool calls (parallel_tool_calls is false)", extra={"dropped": len(calls) - 1})
        calls = calls[:1]
    return calls, None


def to_openai(calls: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"call_{secrets.token_hex(12)}",
            "type": "function",
            "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)},
        }
        for c in calls
    ]


# --- conversation identity ------------------------------------------------------------------------


def canonical_arguments(raw: Any) -> Any:
    parsed = _arguments(raw)
    return parsed.strip() if isinstance(parsed, str) else parsed


def canonical_bridge(messages: Sequence[ChatMessage]) -> str:
    """Conversation identity for tool-using clients: ignores system prompts (agents regenerate them every
    request) and the text that accompanies tool calls; tool call arguments are compared as parsed JSON."""
    items: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role in SYSTEM_ROLES:
            continue
        item: dict[str, Any] = {"role": msg.role}
        if msg.role == "assistant" and msg.tool_calls:
            item["tool_calls"] = [
                {"name": (c.get("function") or {}).get("name"),
                 "arguments": canonical_arguments((c.get("function") or {}).get("arguments"))}
                for c in msg.tool_calls
            ]
        else:
            item["content"] = message_text(msg).strip()
        items.append(item)
    return json.dumps(items, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
