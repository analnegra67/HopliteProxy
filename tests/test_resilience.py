"""Long-running runs: retries attaching to in-flight runs, disconnects, timeouts and concurrency."""

from __future__ import annotations

import asyncio

import pytest

from tests.conftest import parse_sse, wait_for
from tests.fake_hoplite import Hang, Say

pytestmark = pytest.mark.anyio

MESSAGES = [{"role": "user", "content": "long task"}]


def body(**extra) -> dict:
    return {"model": "gpt-6-sol", "messages": MESSAGES, **extra}


async def test_client_retry_attaches_to_the_running_run(client, auth, fake) -> None:
    fake.behaviour = lambda text: [Hang(gate="go"), Say("finished")]
    first = asyncio.create_task(client.post("/v1/chat/completions", headers=auth, json=body()))
    await wait_for(lambda: len(fake.threads) == 1)
    retry = asyncio.create_task(client.post("/v1/chat/completions", json=body(),
                                            headers={**auth, "X-Stainless-Retry-Count": "1"}))
    await asyncio.sleep(0.2)
    fake.open_gate("go")
    a, b = await asyncio.gather(first, retry)
    assert a.status_code == b.status_code == 200
    assert a.json()["id"] == b.json()["id"]
    assert a.json()["choices"][0]["message"]["content"] == b.json()["choices"][0]["message"]["content"] == "finished"
    assert len(fake.thread_creations()) == 1


async def test_completed_result_is_replayed_to_retries_only(client, auth, fake) -> None:
    first = await client.post("/v1/chat/completions", headers=auth, json=body())
    replay = await client.post("/v1/chat/completions", json=body(stream=True),
                               headers={**auth, "X-Stainless-Retry-Count": "2"})
    assert len(fake.thread_creations()) == 1
    chunks = parse_sse(replay.text)
    assert chunks[0]["id"] == first.json()["id"]
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks[:-1] if c["choices"]) == \
        first.json()["choices"][0]["message"]["content"]

    fresh = await client.post("/v1/chat/completions", headers=auth, json=body())  # not a retry: a new run
    assert fresh.json()["id"] != first.json()["id"]
    assert len(fake.thread_creations()) == 2


async def test_disconnect_stops_the_run_after_the_grace_period(client, auth, fake) -> None:
    fake.behaviour = lambda text: [Hang()]
    request = asyncio.create_task(client.post("/v1/chat/completions", headers=auth, json=body(stream=True)))
    await wait_for(lambda: len(fake.threads) == 1)
    await wait_for(lambda: fake.stream_connections >= 1)
    request.cancel()  # the client goes away
    await wait_for(lambda: len(fake.stops) == 1, within=5)
    thread_id = next(iter(fake.threads))
    assert fake.stops[0][0] == thread_id


async def test_orphaned_run_is_picked_up_by_a_reconnecting_client(client, auth, fake) -> None:
    fake.behaviour = lambda text: [Hang(gate="go"), Say("done")]
    request = asyncio.create_task(client.post("/v1/chat/completions", headers=auth, json=body()))
    await wait_for(lambda: fake.stream_connections >= 1)
    request.cancel()
    await asyncio.sleep(0.05)  # within the 0.2 s grace period
    again = asyncio.create_task(client.post("/v1/chat/completions", headers=auth, json=body()))
    await asyncio.sleep(0.3)  # past the grace period: the run must not be abandoned
    fake.open_gate("go")
    resp = await again
    assert resp.status_code == 200 and resp.json()["choices"][0]["message"]["content"] == "done"
    assert fake.stops == [] and len(fake.thread_creations()) == 1


@pytest.mark.parametrize("settings_overrides", [{"run_timeout_seconds": 1.0}])
async def test_run_timeout_stops_the_run(client, auth, fake) -> None:
    fake.behaviour = lambda text: [Hang()]
    resp = await client.post("/v1/chat/completions", headers=auth, json=body())
    assert resp.status_code == 504 and resp.json()["error"]["code"] == "run_timeout"
    assert len(fake.stops) == 1


async def test_per_key_concurrency_limit(client, admin_headers, fake) -> None:
    created = await client.post("/admin/keys", headers=admin_headers,
                                json={"hoplite_api_key": "hop_test_key", "max_concurrency": 1})
    headers = {"Authorization": f"Bearer {created.json()['key']}"}
    fake.behaviour = lambda text: [Hang(gate=text), Say("ok")]
    first = asyncio.create_task(client.post("/v1/chat/completions", headers=headers,
                                            json={"messages": [{"role": "user", "content": "a"}]}))
    await wait_for(lambda: len(fake.threads) == 1)
    second = await client.post("/v1/chat/completions", headers=headers,
                               json={"messages": [{"role": "user", "content": "b"}]})
    assert second.status_code == 429 and second.json()["error"]["code"] == "concurrency_limit_exceeded"
    fake.open_gate("a")
    assert (await first).status_code == 200
    fake.open_gate("c")
    third = await client.post("/v1/chat/completions", headers=headers,
                              json={"messages": [{"role": "user", "content": "c"}]})
    assert third.status_code == 200  # the slot was released when the first run finished
