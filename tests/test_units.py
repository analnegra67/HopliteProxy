"""Unit tests: keys, encryption, redaction, prompt building, draft tracking and SSE frame parsing."""

from __future__ import annotations

import time

import pytest

from app.errors import OpenAIError
from app.hoplite.events import classify
from app.log import redact
from app.openai_api.schemas import ChatCompletionRequest, ChatMessage
from app.security.crypto import SecretBox
from app.security.keys import (
    KEY_PREFIX,
    VerificationCache,
    check_secret,
    generate_key,
    hash_secret,
    hoplite_key_hint,
    parse_key,
    token_digest,
)
from app.service.conversation import canonical_messages, fingerprint
from app.service.models import reasoning_for
from app.service.prompt import (
    build_followup_content,
    build_thread_prompt,
    check_tools,
    omitted_media,
    unsupported_fields,
)
from app.service.runner import DraftTracker, answer_body, classify_failure, format_question


def msg(role: str, content=None, **extra) -> ChatMessage:
    return ChatMessage(role=role, content=content, **extra)


# --- keys and crypto ----------------------------------------------------------------------------


def test_generated_key_roundtrip() -> None:
    key = generate_key()
    assert key.token.startswith(KEY_PREFIX) and len(key.token) == len(KEY_PREFIX) + 52
    assert parse_key(key.token) == (key.key_id, key.secret)


@pytest.mark.parametrize("token", ["", "sk-hop-", "sk-hop-short", "sk-xyz-" + "a" * 52, "sk-hop-" + "a" * 51 + "!"])
def test_malformed_keys_are_rejected(token: str) -> None:
    assert parse_key(token) is None


def test_bcrypt_secret_check() -> None:
    hashed = hash_secret("s3cret", rounds=4)
    assert check_secret("s3cret", hashed)
    assert not check_secret("other", hashed)
    assert not check_secret("s3cret", "not-a-hash")


def test_verification_cache_expiry_and_invalidation() -> None:
    cache = VerificationCache(ttl_seconds=60)
    digest = token_digest("sk-hop-x")
    assert not cache.hit("id1", digest)
    cache.add("id1", digest)
    assert cache.hit("id1", digest)
    assert not cache.hit("id1", token_digest("sk-hop-y"))
    cache.invalidate("id1")
    assert not cache.hit("id1", digest)
    short = VerificationCache(ttl_seconds=0.01)
    short.add("id2", digest)
    time.sleep(0.02)
    assert not short.hit("id2", digest)


def test_secret_box_rotation() -> None:
    old, new = SecretBox.generate_key(), SecretBox.generate_key()
    token = SecretBox(old).encrypt("hop_abc")
    both = SecretBox(f"{new},{old}")
    assert both.decrypt(token) == "hop_abc"
    rotated = both.rotate(token)
    assert SecretBox(new).decrypt(rotated) == "hop_abc"
    with pytest.raises(ValueError):
        SecretBox("")
    with pytest.raises(ValueError):
        SecretBox("not-a-fernet-key")


def test_hint_and_redaction() -> None:
    assert hoplite_key_hint("hop_svc_abcdefgh1234") == "hop_svc_…1234"
    text = "key hop_live_Secret123 and sk-hop-AbCdEf123456 with Authorization: Bearer abcdefghijkl"
    cleaned = redact(text)
    assert "Secret123" not in cleaned and "AbCdEf123456" not in cleaned and "abcdefghijkl" not in cleaned


# --- prompt building ------------------------------------------------------------------------------


def test_single_user_message_is_sent_verbatim() -> None:
    assert build_thread_prompt([msg("user", "Fix the login bug")]) == "Fix the login bug"


def test_history_and_instructions_are_flattened() -> None:
    prompt = build_thread_prompt([
        msg("system", "Be terse."),
        msg("user", "Hi"),
        msg("assistant", "Hello!"),
        msg("user", [{"type": "text", "text": "Now fix it"}, {"type": "image_url", "image_url": {"url": "x"}}]),
    ])
    assert prompt.startswith("<instructions>\nBe terse.\n</instructions>")
    assert "User: Hi" in prompt and "Assistant: Hello!" in prompt
    assert prompt.endswith("Respond to the latest message:\n\nNow fix it\n[image omitted]")


def test_tool_calls_and_results_are_rendered() -> None:
    prompt = build_thread_prompt([
        msg("user", "What is the weather?"),
        msg("assistant", None, tool_calls=[{"id": "c1", "type": "function",
                                            "function": {"name": "weather", "arguments": "{\"city\":\"Kyiv\"}"}}]),
        msg("tool", "sunny", tool_call_id="c1"),
        msg("user", "Thanks"),
    ])
    assert "[called tool weather with arguments {\"city\":\"Kyiv\"}]" in prompt
    assert "Tool result (c1): sunny" in prompt


def test_requires_a_user_message() -> None:
    with pytest.raises(OpenAIError) as exc:
        build_thread_prompt([msg("system", "only instructions")])
    assert exc.value.status_code == 400


def test_followup_content() -> None:
    assert build_followup_content([msg("user", "next step")]) == "next step"
    mixed = build_followup_content([msg("system", "Use Python"), msg("user", "go")])
    assert mixed == "Updated instructions:\nUse Python\n\ngo"


def test_omitted_media_and_unsupported_fields() -> None:
    assert omitted_media([msg("user", [{"type": "input_audio", "input_audio": {}}])]) == ["audio"]
    req = ChatCompletionRequest.model_validate({
        "model": "m", "messages": [{"role": "user", "content": "x"}], "temperature": 0.2, "max_tokens": 10,
        "top_p": None,
    })
    assert unsupported_fields(req) == ["max_tokens", "temperature"]


def test_tools_are_rejected_unless_declined() -> None:
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    with pytest.raises(OpenAIError) as exc:
        check_tools(ChatCompletionRequest.model_validate({"messages": [{"role": "user", "content": "x"}],
                                                          "tools": tools}))
    assert exc.value.code == "tools_not_supported"
    check_tools(ChatCompletionRequest.model_validate({"messages": [{"role": "user", "content": "x"}],
                                                      "tools": tools, "tool_choice": "none"}))


def test_reasoning_mapping() -> None:
    assert reasoning_for("HIGH") == {"mode": "high"}
    assert reasoning_for("turbo") is None
    assert reasoning_for(None) is None


def test_fingerprint_ignores_whitespace_and_part_layout() -> None:
    a = [msg("user", "hello  "), msg("assistant", "hi")]
    b = [msg("user", [{"type": "text", "text": "hello"}]), msg("assistant", "hi\n")]
    assert canonical_messages(a) == canonical_messages(b)
    assert fingerprint("k", "p", messages=a) == fingerprint("k", "p", messages=b)
    assert fingerprint("k", "p", messages=a) != fingerprint("other", "p", messages=a)


# --- streaming helpers ----------------------------------------------------------------------------


def test_draft_tracker_emits_append_only_deltas() -> None:
    tracker = DraftTracker()
    assert tracker.update("m1", "Hel") == "Hel"
    assert tracker.update("m1", "Hello") == "lo"
    assert tracker.update("m1", "Hello") == ""
    assert tracker.finalize("m1", "Hello world") == " world"
    assert tracker.update("m1", "Hello world!!") == ""  # closed after finalize
    assert tracker.update("m2", "Second") == "\n\nSecond"
    assert tracker.sent == "Hello world\n\nSecond"


def test_draft_removed_still_lets_the_persisted_text_complete_the_message() -> None:
    tracker = DraftTracker()
    assert tracker.update("m1", "Partial") == "Partial"
    tracker.discard("m1")  # draft.removed
    assert tracker.update("m1", "Partial draft") == ""  # drafts are ignored after removal
    assert tracker.finalize("m1", "Partial, then the rest.") == ", then the rest."
    assert tracker.sent == "Partial, then the rest."


def test_draft_tracker_handles_rewrites_and_id_mismatch() -> None:
    tracker = DraftTracker()
    tracker.update("d1", "Draft text")
    assert tracker.update("d1", "Rewritten") == ""  # cannot retract what was streamed
    assert tracker.finalize("persisted-1", "Draft text, completed") == ", completed"
    assert tracker.finalize("persisted-2", "Another message") == "\n\nAnother message"
    assert tracker.append_note("note") == "\n\nnote"
    assert tracker.sent == "Draft text, completed\n\nAnother message\n\nnote"


@pytest.mark.parametrize("payload", [
    '{"messageId":"m1","runId":"run_1","streamId":"assistant:s0","revision":2,"content":"abc"}',  # live shape
    '{"type":"draft","messageId":"m1","revision":2,"content":"abc"}',
    '{"type":"draft.updated","data":{"messageId":"m1","revision":2,"text":"abc"}}',
    '{"messageId":"m1","revision":2,"parts":[{"type":"text","text":"a"},{"type":"text","text":"bc"}]}',
])
def test_classify_draft_shapes(payload: str) -> None:
    event = classify("draft", None, payload)
    assert (event.kind, event.message_id, event.text, event.revision) == ("draft", "m1", "abc", 2)


def test_classify_durable_events() -> None:
    run = classify("run.completed", "cur-1", '{"id":"evt","type":"run.completed","runId":"run_1",'
                                             '"data":{"id":"run_1","status":"completed"},"cursor":"cur-1"}')
    assert (run.kind, run.type, run.run_id, run.status, run.cursor) == ("event", "run.completed", "run_1",
                                                                       "completed", "cur-1")
    persisted = classify("message.persisted", "c2",
                         '{"type":"message.persisted","data":{"id":"m9","role":"assistant"}}')
    assert (persisted.message_id, persisted.role) == ("m9", "assistant")
    assert classify("heartbeat", "c3", '{"cursor":"c3"}').kind == "heartbeat"
    removed = classify("draft.removed", None, '{"messageId":"m1"}')
    assert (removed.kind, removed.message_id) == ("draft_removed", "m1")


def test_failure_classification_and_questions() -> None:
    assert classify_failure({"message": "The thread spend limit has been reached"})[0] == "spend_limit"
    assert classify_failure({"message": "The context limit has been reached"})[1].code == "context_length_exceeded"
    assert classify_failure({"message": "You are out of credits"})[1].status_code == 402
    assert classify_failure({"message": "Sorry, an error occurred"})[1].status_code == 502
    question = {"questionId": "q1", "request": {"questions": [
        {"header": "DB", "question": "Which database?", "multiSelect": False,
         "options": [{"label": "Postgres", "description": "default"}, "SQLite"]}]}}
    text = format_question(question)
    assert "**DB:** Which database?" in text and "1. Postgres — default" in text and "2. SQLite" in text
    assert answer_body(question, "Postgres") == {"answer": "Postgres"}
    double = {"request": {"questions": [{"question": "A?"}, {"question": "B?"}]}}
    assert answer_body(double, "yes") == {"answers": {"A?": "yes", "B?": "yes"}}
