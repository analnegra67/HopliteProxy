"""Wire compatibility with the official `openai` Python SDK."""

from __future__ import annotations

import httpx
import openai
import pytest

pytestmark = pytest.mark.anyio


@pytest.fixture
async def sdk(app, api_key):
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test")
    client = openai.AsyncOpenAI(api_key=api_key, base_url="http://proxy.test/v1", http_client=http, max_retries=0)
    yield client
    await client.close()


async def test_models_and_non_streaming(sdk) -> None:
    models = await sdk.models.list()
    assert {"hoplite-default", "gpt-6-sol"} <= {m.id for m in models.data}

    raw = await sdk.chat.completions.with_raw_response.create(
        model="gpt-6-sol", messages=[{"role": "user", "content": "ping"}]
    )
    completion = raw.parse()
    assert completion.choices[0].message.content == "You said: ping"
    assert completion.choices[0].finish_reason == "stop"
    assert completion.usage is not None and completion.usage.total_tokens > 0
    assert raw.headers["x-hoplite-thread-id"].startswith("thr_")


async def test_streaming_with_usage(sdk) -> None:
    stream = await sdk.chat.completions.create(
        model="gpt-6-sol", messages=[{"role": "user", "content": "pong"}], stream=True,
        stream_options={"include_usage": True},
    )
    parts, usage, finish = [], None, None
    async for chunk in stream:
        if chunk.choices:
            parts.append(chunk.choices[0].delta.content or "")
            finish = chunk.choices[0].finish_reason or finish
        if chunk.usage:
            usage = chunk.usage
    assert "".join(parts) == "You said: pong"
    assert finish == "stop" and usage is not None and usage.total_tokens > 0


READ_TOOL = {"type": "function", "function": {
    "name": "read", "description": "Read a file",
    "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}}, "required": ["filePath"]}}}
CALL = '<tool_calls>\n[{"name": "read", "arguments": {"filePath": "notes.txt"}}]\n</tool_calls>'


async def test_tool_calls_round_trip(sdk, fake) -> None:
    from tests.fake_hoplite import Say

    fake.behaviour = lambda text: [Say("It says: hello")] if "<tool_result id=" in text else [Say("Reading.\n" + CALL)]
    messages = [{"role": "user", "content": "What is in notes.txt?"}]
    first = await sdk.chat.completions.create(model="gpt-6-sol", messages=messages, tools=[READ_TOOL])
    choice = first.choices[0]
    assert choice.finish_reason == "tool_calls" and choice.message.content == "Reading."
    call = choice.message.tool_calls[0]
    assert call.function.name == "read" and call.function.arguments == '{"filePath": "notes.txt"}'

    messages += [choice.message.model_dump(exclude_none=True),
                 {"role": "tool", "tool_call_id": call.id, "content": "hello"}]
    stream = await sdk.chat.completions.create(model="gpt-6-sol", messages=messages, tools=[READ_TOOL], stream=True)
    text = "".join([c.choices[0].delta.content or "" async for c in stream if c.choices])
    assert text == "It says: hello"
    assert len(fake.thread_creations()) == 1  # the tool result went into the same Hoplite thread


async def test_streamed_tool_calls_accumulate_in_the_sdk(sdk, fake) -> None:
    from tests.fake_hoplite import Say

    fake.behaviour = lambda text: [Say(CALL, chunks=5)]
    stream = await sdk.chat.completions.create(
        model="gpt-6-sol", messages=[{"role": "user", "content": "x"}], tools=[READ_TOOL], stream=True)
    calls, finish, content = {}, None, ""
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        content += delta.content or ""
        for part in delta.tool_calls or []:
            entry = calls.setdefault(part.index, {"id": part.id, "name": "", "arguments": ""})
            entry["name"] += part.function.name or ""
            entry["arguments"] += part.function.arguments or ""
        finish = chunk.choices[0].finish_reason or finish
    assert content == "" and finish == "tool_calls"
    assert calls[0]["name"] == "read" and calls[0]["arguments"] == '{"filePath": "notes.txt"}'
    assert calls[0]["id"].startswith("call_")


async def test_errors_map_to_sdk_exceptions(sdk, app) -> None:
    with pytest.raises(openai.NotFoundError) as missing:
        await sdk.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    assert missing.value.code == "model_not_found"

    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test")
    anonymous = openai.AsyncOpenAI(api_key="sk-hop-wrong", base_url="http://proxy.test/v1", http_client=http,
                                   max_retries=0)
    with pytest.raises(openai.AuthenticationError):
        await anonymous.models.list()
    await anonymous.close()
