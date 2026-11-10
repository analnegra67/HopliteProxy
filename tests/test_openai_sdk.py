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


async def test_errors_map_to_sdk_exceptions(sdk, app) -> None:
    with pytest.raises(openai.BadRequestError) as bad:
        await sdk.chat.completions.create(
            model="gpt-6-sol", messages=[{"role": "user", "content": "x"}],
            tools=[{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}],
        )
    assert bad.value.code == "tools_not_supported"

    with pytest.raises(openai.NotFoundError) as missing:
        await sdk.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    assert missing.value.code == "model_not_found"

    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test")
    anonymous = openai.AsyncOpenAI(api_key="sk-hop-wrong", base_url="http://proxy.test/v1", http_client=http,
                                   max_retries=0)
    with pytest.raises(openai.AuthenticationError):
        await anonymous.models.list()
    await anonymous.close()
