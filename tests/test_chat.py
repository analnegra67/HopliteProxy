"""End-to-end chat completions against the fake Hoplite."""

from __future__ import annotations

import logging

import pytest

from tests.conftest import parse_sse
from tests.fake_hoplite import Ask, Fail, Say, Think, Tool

pytestmark = pytest.mark.anyio


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


async def chat(client, auth, messages, **extra):
    return await client.post("/v1/chat/completions", headers={**auth, **extra.pop("headers", {})},
                             json={"model": extra.pop("model", "gpt-6-sol"), "messages": messages, **extra})


async def test_non_streaming_completion(client, auth, fake) -> None:
    resp = await chat(client, auth, [user("ping")])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["object"] == "chat.completion" and body["id"].startswith("chatcmpl-")
    assert body["model"] == "gpt-6-sol"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "You said: ping", "refusal": None}
    assert body["choices"][0]["finish_reason"] == "stop"
    usage = body["usage"]
    assert usage["prompt_tokens"] == 100 and usage["completion_tokens"] == 3
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    thread_id = resp.headers["x-hoplite-thread-id"]
    assert body["hoplite"]["thread_id"] == thread_id and body["hoplite"]["cost_micros"] == 2500

    [created] = fake.thread_creations()
    assert created["projectId"] == "proj_1" and created["prompt"] == "ping" and created["model"] == "gpt-6-sol"
    assert created["autoMerge"] is False and created["spendLimitMicros"] == 1_000_000
    assert created["metadata"]["source"] == "openai-proxy"


async def test_streaming_completion(client, auth) -> None:
    resp = await chat(client, auth, [user("pong")], stream=True, stream_options={"include_usage": True})
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(resp.text)
    assert events[-1] == "[DONE]"
    chunks = events[:-1]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert text == "You said: pong"
    deltas = [c for c in chunks if c["choices"] and c["choices"][0]["delta"].get("content")]
    assert len(deltas) >= 3  # streamed from draft snapshots, not one final blob
    finish = [c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
    assert finish[0]["choices"][0]["finish_reason"] == "stop"
    usage_chunk = chunks[-1]
    assert usage_chunk["choices"] == [] and usage_chunk["usage"]["total_tokens"] > 0
    assert len({c["id"] for c in chunks}) == 1


async def test_follow_up_continues_the_same_thread(client, auth, fake) -> None:
    first = await chat(client, auth, [user("one")])
    reply = first.json()["choices"][0]["message"]["content"]
    second = await chat(client, auth, [user("one"), assistant(reply), user("two")])
    assert second.status_code == 200, second.text
    assert second.json()["choices"][0]["message"]["content"] == "You said: two"
    assert second.headers["x-hoplite-thread-id"] == first.headers["x-hoplite-thread-id"]
    assert len(fake.thread_creations()) == 1
    [appended] = fake.calls_to("POST", f"/api/threads/{first.headers['x-hoplite-thread-id']}/messages")
    assert appended[2]["content"] == "two"
    assert second.json()["usage"]["prompt_tokens"] == 100  # usage is per run, not cumulative

    # Regenerating the second answer (same history again) must not re-append to the same thread.
    third = await chat(client, auth, [user("one"), assistant(reply), user("two")])
    assert third.status_code == 200
    assert third.headers["x-hoplite-thread-id"] != first.headers["x-hoplite-thread-id"]
    assert "<conversation_history>" in fake.thread_creations()[-1]["prompt"]


async def test_persisted_text_completes_what_the_drafts_missed(client, auth, fake) -> None:
    text = "The drafts only reach the first half of this sentence; the rest arrives on persistence."
    fake.behaviour = lambda _: [Say(text, chunks=4, draft_coverage=0.5)]
    stream = await chat(client, auth, [user("s")], stream=True)
    streamed = "".join(c["choices"][0]["delta"].get("content", "") for c in parse_sse(stream.text)[:-1]
                       if c.get("choices"))
    assert streamed == text
    plain = await chat(client, auth, [user("p")])
    assert plain.json()["choices"][0]["message"]["content"] == text


@pytest.mark.parametrize("fake", [{"announce_assistant_messages": True}], indirect=True)
async def test_mid_run_message_announcements_are_handled(client, auth, fake) -> None:
    fake.behaviour = lambda _: [Say("first part"), Tool("bash"), Say("second part")]
    stream = await chat(client, auth, [user("x")], stream=True)
    streamed = "".join(c["choices"][0]["delta"].get("content", "") for c in parse_sse(stream.text)[:-1]
                       if c.get("choices"))
    assert streamed == "first part\n\nsecond part"


async def test_streamed_reply_can_be_continued(client, auth, fake) -> None:
    first = await chat(client, auth, [user("alpha")], stream=True)
    reply = "".join(c["choices"][0]["delta"].get("content", "")
                    for c in parse_sse(first.text)[:-1] if c.get("choices"))
    second = await chat(client, auth, [user("alpha"), assistant(reply), user("beta")], stream=True)
    assert second.headers["x-hoplite-thread-id"] == first.headers["x-hoplite-thread-id"]
    assert len(fake.thread_creations()) == 1


async def test_thread_headers(client, auth, fake) -> None:
    first = await chat(client, auth, [user("a")])
    thread_id = first.headers["x-hoplite-thread-id"]
    pinned = await chat(client, auth, [user("unrelated")], headers={"X-Hoplite-Thread-Id": thread_id})
    assert pinned.headers["x-hoplite-thread-id"] == thread_id
    assert fake.calls_to("POST", f"/api/threads/{thread_id}/messages")[-1][2]["content"] == "unrelated"

    reply = first.json()["choices"][0]["message"]["content"]
    forced = await chat(client, auth, [user("a"), assistant(reply), user("b")], headers={"X-Hoplite-New-Thread": "1"})
    assert forced.headers["x-hoplite-thread-id"] != thread_id


async def test_instructions_and_history_in_new_thread(client, auth, fake) -> None:
    resp = await chat(client, auth, [{"role": "system", "content": "Answer in French."}, user("hi"),
                                     assistant("salut"), user("again")])
    assert resp.status_code == 200
    prompt = fake.thread_creations()[0]["prompt"]
    assert prompt.startswith("<instructions>\nAnswer in French.\n</instructions>")
    assert prompt.endswith("Respond to the latest message:\n\nagain")


async def test_tools_are_rejected(client, auth, fake) -> None:
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    resp = await chat(client, auth, [user("x")], tools=tools)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "tools_not_supported" and resp.json()["error"]["param"] == "tools"
    assert fake.thread_creations() == []
    declined = await chat(client, auth, [user("x")], tools=tools, tool_choice="none")
    assert declined.status_code == 200


async def test_model_resolution(client, auth, fake) -> None:
    unknown = await chat(client, auth, [user("x")], model="gpt-4o")
    assert unknown.status_code == 404 and unknown.json()["error"]["code"] == "model_not_found"
    assert "claude-sonnet-5" in unknown.json()["error"]["message"]

    default = await chat(client, auth, [user("x")], model="hoplite-default")
    assert default.status_code == 200 and default.json()["model"] == "gpt-6-sol"  # reported by Hoplite
    assert "model" not in fake.thread_creations()[-1]

    prefixed = await chat(client, auth, [user("x")], model="hoplite/claude-sonnet-5", reasoning_effort="high")
    assert prefixed.json()["model"] == "claude-sonnet-5"
    assert fake.thread_creations()[-1]["reasoning"] == {"mode": "high"}

    models = (await client.get("/v1/models", headers=auth)).json()
    ids = [m["id"] for m in models["data"]]
    assert ids[0] == "hoplite-default" and {"gpt-6-sol", "claude-sonnet-5"} <= set(ids)
    one = await client.get("/v1/models/claude-sonnet-5", headers=auth)
    assert one.json()["owned_by"] == "anthropic"
    assert (await client.get("/v1/models/nope", headers=auth)).status_code == 404


@pytest.mark.parametrize("settings_overrides", [{"model_aliases": {"gpt-4o": "claude-sonnet-5"}}])
async def test_model_aliases(client, auth) -> None:
    resp = await chat(client, auth, [user("x")], model="gpt-4o")
    assert resp.status_code == 200 and resp.json()["model"] == "claude-sonnet-5"
    ids = [m["id"] for m in (await client.get("/v1/models", headers=auth)).json()["data"]]
    assert "gpt-4o" in ids


async def test_unsupported_fields_are_logged(client, auth, caplog) -> None:
    with caplog.at_level(logging.INFO):
        resp = await chat(client, auth, [user("x")], temperature=0.1, max_tokens=5, seed=1)
    assert resp.status_code == 200
    records = [r for r in caplog.records if r.getMessage() == "ignoring unsupported request fields"]
    assert records and records[0].ignored_fields == ["max_tokens", "seed", "temperature"]


async def test_validation_errors_use_openai_format(client, auth) -> None:
    resp = await client.post("/v1/chat/completions", headers=auth, json={"model": "gpt-6-sol", "messages": []})
    assert resp.status_code == 400 and resp.json()["error"]["type"] == "invalid_request_error"
    resp = await client.post("/v1/chat/completions", headers=auth,
                             json={"messages": [{"role": "system", "content": "only system"}]})
    assert resp.status_code == 400


async def test_multiple_messages_thinking_and_activity(client, auth, fake, settings) -> None:
    fake.behaviour = lambda text: [Say("Looking into it."), Think("hidden reasoning"), Tool("bash"),
                                   Say("Done: fixed.")]
    resp = await chat(client, auth, [user("fix")])
    assert resp.json()["choices"][0]["message"]["content"] == "Looking into it.\n\nDone: fixed."
    assert "reasoning_content" not in resp.json()["choices"][0]["message"]


@pytest.mark.parametrize("settings_overrides", [{"include_reasoning": True}])
async def test_reasoning_content_when_enabled(client, auth, fake) -> None:
    fake.behaviour = lambda text: [Think("step 1"), Say("answer")]
    resp = await chat(client, auth, [user("q")])
    message = resp.json()["choices"][0]["message"]
    assert message["content"] == "answer" and message["reasoning_content"] == "step 1"
    stream = await chat(client, auth, [user("q2")], stream=True)
    reasoning = "".join(c["choices"][0]["delta"].get("reasoning_content", "")
                        for c in parse_sse(stream.text)[:-1] if c.get("choices"))
    assert reasoning == "step 1"


async def test_agent_question_is_surfaced_and_answered(client, auth, fake) -> None:
    fake.behaviour = lambda text: [Say("I need a detail."), Ask("Which database?", header="DB",
                                                                 options=["Postgres", "SQLite"]), Say("Using it.")]
    first = await chat(client, auth, [user("build it")])
    content = first.json()["choices"][0]["message"]["content"]
    assert content.startswith("I need a detail.\n\n**The agent needs your input:**")
    assert "**DB:** Which database?" in content and "1. Postgres" in content
    assert first.json()["choices"][0]["finish_reason"] == "stop"

    second = await chat(client, auth, [user("build it"), assistant(content), user("Postgres")])
    assert second.status_code == 200, second.text
    assert fake.answers == ["Postgres"]
    assert second.json()["choices"][0]["message"]["content"] == "Using it."
    assert second.headers["x-hoplite-thread-id"] == first.headers["x-hoplite-thread-id"]


@pytest.mark.parametrize("settings_overrides", [{"question_policy": "auto"}])
async def test_agent_question_auto_answer(client, auth, fake, settings) -> None:
    fake.behaviour = lambda text: [Ask("Proceed?"), Say("Proceeding.")]
    resp = await chat(client, auth, [user("go")])
    assert resp.json()["choices"][0]["message"]["content"] == "Proceeding."
    assert fake.answers == [settings.question_auto_answer]


async def test_spend_limit_failures(client, auth, fake) -> None:
    fake.behaviour = lambda text: [Fail("The thread spend limit has been reached", "spend_limit")]
    empty = await chat(client, auth, [user("x")])
    assert empty.status_code == 402 and empty.json()["error"]["type"] == "insufficient_quota"

    fake.behaviour = lambda text: [Say("Partial work."), Fail("The thread spend limit has been reached")]
    partial = await chat(client, auth, [user("y")])
    assert partial.status_code == 200
    assert partial.json()["choices"][0] == {"index": 0, "logprobs": None, "finish_reason": "length",
                                            "message": {"role": "assistant", "content": "Partial work.",
                                                        "refusal": None}}


async def test_run_failure_mapping(client, auth, fake) -> None:
    fake.behaviour = lambda text: [Fail("The model connection failed", "model_connection")]
    resp = await chat(client, auth, [user("x")])
    assert resp.status_code == 502 and resp.json()["error"]["code"] == "run_failed"
    assert "model connection failed" in resp.json()["error"]["message"]

    stream = await chat(client, auth, [user("y")], stream=True)
    events = parse_sse(stream.text)
    assert events[-1] == "[DONE]" and events[-2]["error"]["code"] == "run_failed"


async def test_upstream_errors(client, auth, fake) -> None:
    fake.inject("POST", "/api/threads", 429, {"error": "rate_limited"}, {"Retry-After": "7"}, times=10)
    limited = await chat(client, auth, [user("x")])
    assert limited.status_code == 429 and limited.headers["retry-after"] == "7"
    assert len(fake.thread_creations()) == 4  # retried with the same idempotency key, then surfaced

    fake._injected.clear()
    fake.inject("POST", "/api/threads", 401, {"error": "unauthorized"})
    denied = await chat(client, auth, [user("x")])
    assert denied.status_code == 502 and denied.json()["error"]["code"] == "upstream_auth_error"

    fake.inject("POST", "/api/threads", 503, {"error": "unavailable"})
    recovered = await chat(client, auth, [user("x")])
    assert recovered.status_code == 200
    attempts = [key for path, key in fake.write_keys if path == "/api/threads"][-2:]
    assert attempts[0] == attempts[1] and attempts[0].endswith(":thread")  # retried with the same Idempotency-Key


@pytest.mark.parametrize("fake", [{"stream_enabled": False}], indirect=True)
async def test_polling_fallback_when_events_are_unavailable(client, auth, fake) -> None:
    resp = await chat(client, auth, [user("poll")])
    assert resp.status_code == 200, resp.text
    assert resp.json()["choices"][0]["message"]["content"] == "You said: poll"
