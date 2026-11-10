from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI

from app.config import Settings
from app.main import create_app
from tests.fake_hoplite import FakeHoplite

MASTER_KEY = "m" * 40
HOPLITE_KEY = "hop_test_key"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def fake(request: pytest.FixtureRequest) -> FakeHoplite:
    options = getattr(request, "param", None) or {}
    return FakeHoplite(api_keys={HOPLITE_KEY, "hop_second_key"}, **options)


@pytest.fixture
def settings_overrides() -> dict[str, Any]:
    return {}


@pytest.fixture
def settings(tmp_path, settings_overrides) -> Settings:
    values: dict[str, Any] = {
        "master_key": MASTER_KEY,
        "encryption_keys": Fernet.generate_key().decode(),
        "database_path": str(tmp_path / "proxy.db"),
        "hoplite_base_url": "https://api.hoplite.test",
        "bcrypt_rounds": 4,
        "keepalive_seconds": 1.0,
        "disconnect_grace_seconds": 0.2,
        "run_timeout_seconds": 20.0,
        "hoplite_max_backoff_seconds": 0.01,
        "message_sync_interval_seconds": 0.0,
        "log_format": "text",
        "log_level": "DEBUG",
    }
    values.update(settings_overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
async def app(settings: Settings, fake: FakeHoplite) -> AsyncIterator[FastAPI]:
    application = create_app(settings, hoplite_transport=httpx.ASGITransport(app=fake.app))
    async with application.router.lifespan_context(application):
        yield application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test",
                                 timeout=30) as http:
        yield http


@pytest.fixture
def admin_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {MASTER_KEY}"}


@pytest.fixture
async def api_key(client: httpx.AsyncClient, admin_headers: dict[str, str]) -> str:
    resp = await client.post("/admin/keys", json={"hoplite_api_key": HOPLITE_KEY, "name": "tests"},
                             headers=admin_headers)
    assert resp.status_code == 201, resp.text
    return resp.json()["key"]


@pytest.fixture
def auth(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def parse_sse(text: str) -> list[Any]:
    """Return the `data:` payloads of an SSE body (JSON-decoded, except the literal [DONE])."""
    payloads: list[Any] = []
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                data = line[len("data: "):]
                payloads.append(data if data == "[DONE]" else json.loads(data))
    return payloads


async def wait_for(predicate: Callable[[], bool], within: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + within
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.02)
