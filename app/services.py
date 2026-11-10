"""Application-wide service container, created in the FastAPI lifespan."""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from app.config import Settings
from app.hoplite.client import HopliteClient
from app.security.crypto import SecretBox
from app.security.keys import VerificationCache
from app.service.conversation import ConversationMapper
from app.service.inflight import InflightRegistry, KeySlots
from app.service.models import ModelResolver
from app.service.runner import Runner
from app.storage.sqlite import Store

log = logging.getLogger(__name__)

TOUCH_INTERVAL_SECONDS = 60.0
CLEANUP_INTERVAL_SECONDS = 3600.0


class Services:
    def __init__(self, settings: Settings, store: Store, http: httpx.AsyncClient, box: SecretBox) -> None:
        self.settings = settings
        self.store = store
        self.http = http
        self.box = box
        self.auth_cache = VerificationCache(settings.auth_cache_seconds)
        self.models = ModelResolver(settings)
        self.registry = InflightRegistry(settings)
        self.slots = KeySlots()
        self.mapper = ConversationMapper(store, settings)
        self.runner = Runner(settings=settings, http=http, mapper=self.mapper, slots=self.slots)
        self.started_at = time.time()
        self._last_touch: dict[str, float] = {}

    def hoplite(self, hoplite_api_key: str) -> HopliteClient:
        return self.runner.client_for(hoplite_api_key)

    async def touch_key(self, key_id: str) -> None:
        now = time.monotonic()
        if now - self._last_touch.get(key_id, -TOUCH_INTERVAL_SECONDS) < TOUCH_INTERVAL_SECONDS:
            return
        self._last_touch[key_id] = now
        try:
            await self.store.touch_key(key_id)
        except Exception:
            log.exception("failed to record key usage")

    def forget_key(self, key_id: str) -> None:
        """Drop cached state for a key after it was changed or revoked."""
        self.auth_cache.invalidate(key_id)
        self.models.forget(key_id)

    async def purge_once(self) -> None:
        cutoff = time.time() - self.settings.conversation_ttl_days * 86400
        removed = await self.store.purge_conversations(cutoff)
        self.registry.prune()
        if removed:
            log.info("purged old conversation fingerprints", extra={"removed": removed})

    async def cleanup_forever(self) -> None:
        while True:
            await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
            try:
                await self.purge_once()
            except Exception:
                log.exception("periodic cleanup failed")
