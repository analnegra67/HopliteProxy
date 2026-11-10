"""Map stateless OpenAI conversations onto stateful Hoplite threads.

After every answer the proxy stores a fingerprint of "history + the exact reply it returned"
pointing at the Hoplite thread. When the next request carries that same history plus new
messages, those new messages are appended to the same thread (same workspace, no re-upload of
history). Each fingerprint can be used once: regenerating a reply, editing history or forking a
conversation therefore starts a new thread, which keeps every thread linear.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from app.config import Settings
from app.context import KeyContext
from app.openai_api.schemas import ChatMessage
from app.service.prompt import SYSTEM_ROLES, build_followup_content, build_thread_prompt, message_text
from app.service.toolbridge import ToolMode, build_bridge_followup, build_bridge_prompt, canonical_bridge
from app.storage.sqlite import ConversationRecord, Store

log = logging.getLogger(__name__)


def canonical_messages(messages: Sequence[ChatMessage]) -> str:
    items = []
    for msg in messages:
        item: dict[str, object] = {
            "role": "system" if msg.role in SYSTEM_ROLES else msg.role,
            "content": message_text(msg).strip(),
        }
        if msg.name:
            item["name"] = msg.name
        if msg.tool_call_id:
            item["tool_call_id"] = msg.tool_call_id
        if msg.tool_calls:
            item["tool_calls"] = [
                {"name": (c.get("function") or {}).get("name"), "arguments": (c.get("function") or {}).get("arguments")}
                for c in msg.tool_calls
            ]
        items.append(item)
    return json.dumps(items, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(*scope: str, messages: Sequence[ChatMessage], tools: ToolMode | None = None) -> str:
    digest = hashlib.sha256()
    for part in scope:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    if tools is not None:  # tool-using clients: identity ignores regenerated system prompts
        digest.update(f"tools:{tools.digest()}\x00".encode())
        digest.update(canonical_bridge(messages).encode("utf-8"))
    else:
        digest.update(canonical_messages(messages).encode("utf-8"))
    return digest.hexdigest()


def last_assistant_index(messages: Sequence[ChatMessage]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "assistant":
            return index
    return None


@dataclass
class ThreadPlan:
    mode: Literal["new", "append", "answer"]
    prompt: str  # full flattened history: used for new threads and as a fallback
    content: str | None = None  # follow-up text for append / answer
    thread_id: str | None = None
    question_id: str | None = None
    consumed_fingerprint: str | None = None
    reason: str = "new"


class ConversationMapper:
    def __init__(self, store: Store, settings: Settings) -> None:
        self._store = store
        self._settings = settings

    @staticmethod
    def conversation_key(ctx: KeyContext, messages: Sequence[ChatMessage], tools: ToolMode | None = None) -> str:
        return fingerprint(ctx.key_id, ctx.project_id, messages=messages, tools=tools)

    async def plan(
        self,
        ctx: KeyContext,
        messages: Sequence[ChatMessage],
        *,
        explicit_thread_id: str | None = None,
        force_new: bool = False,
        tools: ToolMode | None = None,
    ) -> ThreadPlan:
        followup: Callable[[Sequence[ChatMessage]], str]
        if tools is not None:
            limit = self._settings.tool_result_max_chars
            prompt = build_bridge_prompt(messages, tools, result_limit=limit)

            def followup(tail: Sequence[ChatMessage]) -> str:
                return build_bridge_followup(messages, tail, tools, result_limit=limit)
        else:
            prompt = build_thread_prompt(messages)
            followup = build_followup_content
        index = last_assistant_index(messages)
        tail = list(messages[index + 1:]) if index is not None else []

        if explicit_thread_id:
            content = followup(tail or list(messages[-1:]))
            return ThreadPlan("append", prompt, content, thread_id=explicit_thread_id, reason="header")
        if force_new or self._settings.thread_reuse == "off" or index is None or not tail:
            return ThreadPlan("new", prompt, reason="forced" if force_new else "new")

        key = self.conversation_key(ctx, messages[: index + 1], tools)
        record = await self._store.consume_conversation(key, ctx.key_id)
        if record is None:
            return ThreadPlan("new", prompt, reason="new")
        content = followup(tail)
        if record.pending_question_id:
            return ThreadPlan("answer", prompt, content, thread_id=record.thread_id,
                              question_id=record.pending_question_id, consumed_fingerprint=key, reason="fingerprint")
        return ThreadPlan("append", prompt, content, thread_id=record.thread_id,
                          consumed_fingerprint=key, reason="fingerprint")

    async def remember(
        self,
        ctx: KeyContext,
        messages: Sequence[ChatMessage],
        reply: str,
        *,
        thread_id: str,
        run_id: str | None,
        pending_question_id: str | None,
        tools: ToolMode | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        reply_message = ChatMessage(role="assistant", content=reply or None, tool_calls=tool_calls)
        history = [*messages, reply_message]
        await self._store.save_conversation(
            ConversationRecord(
                fingerprint=self.conversation_key(ctx, history, tools),
                key_id=ctx.key_id,
                thread_id=thread_id,
                run_id=run_id,
                pending_question_id=pending_question_id,
            )
        )

    async def release(self, plan: ThreadPlan) -> None:
        """Give back a consumed continuation when the follow-up never reached Hoplite."""
        if plan.consumed_fingerprint:
            try:
                await self._store.release_conversation(plan.consumed_fingerprint)
            except Exception:
                log.exception("failed to release conversation fingerprint")
