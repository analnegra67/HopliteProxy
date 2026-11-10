"""Hoplite model catalog and resolution of the OpenAI `model` / `reasoning_effort` parameters."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.context import KeyContext
from app.errors import OpenAIError
from app.hoplite.client import HopliteClient, HopliteError

log = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "hoplite-default"
REASONING_MODES = ("off", "none", "minimal", "low", "medium", "high", "xhigh", "max")
_STRIPPED_PREFIXES = ("hoplite/",)


@dataclass
class Catalog:
    models: list[dict[str, Any]]
    default_model_id: str | None
    fetched_at: float

    def ids(self) -> set[str]:
        return {m["id"] for m in self.models}


@dataclass(frozen=True)
class ResolvedModel:
    hoplite_id: str | None  # None: let Hoplite use the project/thread default
    requested: str | None

    @property
    def label(self) -> str:
        return self.hoplite_id or DEFAULT_MODEL_ID


class ModelResolver:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cache: dict[str, Catalog] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _fresh(self, catalog: Catalog | None) -> bool:
        return catalog is not None and time.monotonic() - catalog.fetched_at < self._settings.model_catalog_ttl_seconds

    async def catalog(self, ctx: KeyContext, client: HopliteClient) -> Catalog:
        cached = self._cache.get(ctx.key_id)
        if self._fresh(cached):
            return cached  # type: ignore[return-value]
        lock = self._locks.setdefault(ctx.key_id, asyncio.Lock())
        async with lock:
            cached = self._cache.get(ctx.key_id)
            if self._fresh(cached):
                return cached  # type: ignore[return-value]
            data = await client.list_models()
            models = [m for m in data.get("models") or [] if isinstance(m, dict) and isinstance(m.get("id"), str)]
            run_config = data.get("runConfig") or {}
            catalog = Catalog(models, run_config.get("defaultModelId"), time.monotonic())
            self._cache[ctx.key_id] = catalog
            return catalog

    def forget(self, key_id: str) -> None:
        self._cache.pop(key_id, None)

    async def resolve(self, requested: str | None, ctx: KeyContext, client: HopliteClient) -> ResolvedModel:
        name = (requested or "").strip()
        for prefix in _STRIPPED_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix):]
        if not name or name == DEFAULT_MODEL_ID:
            return ResolvedModel(ctx.default_model, requested)
        aliased = self._settings.model_aliases.get(name)
        if aliased:
            return ResolvedModel(aliased, requested)
        try:
            catalog = await self.catalog(ctx, client)
        except HopliteError as exc:
            log.warning("model catalog unavailable; passing the model through unvalidated",
                        extra={"status": exc.status, "code": exc.code})
            return ResolvedModel(name, requested)
        if name in catalog.ids():
            return ResolvedModel(name, requested)
        available = sorted(catalog.ids())
        raise OpenAIError(
            404,
            f"The model `{requested}` does not exist or is not available through this Hoplite key. "
            f"Available: {', '.join(available[:40]) or 'none'}; or use `{DEFAULT_MODEL_ID}` for the project default.",
            code="model_not_found",
            param="model",
        )


def reasoning_for(effort: str | None) -> dict[str, str] | None:
    """Map OpenAI `reasoning_effort` onto Hoplite `reasoning.mode` (same vocabulary)."""
    if effort is None:
        return None
    mode = effort.strip().lower()
    if mode in REASONING_MODES:
        return {"mode": mode}
    log.info("ignoring unsupported reasoning_effort", extra={"reasoning_effort": effort})
    return None
