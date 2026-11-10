"""Per-request view of an authenticated proxy key."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.storage.sqlite import KeyRecord


@dataclass(frozen=True)
class KeyContext:
    key_id: str
    name: str | None
    project_id: str
    default_model: str | None
    spend_limit_micros: int | None
    max_concurrency: int
    repos: list[dict[str, Any]] | None
    hoplite_api_key: str = field(repr=False)

    @classmethod
    def from_record(cls, rec: KeyRecord, hoplite_api_key: str, settings: Settings) -> KeyContext:
        spend = rec.spend_limit_micros if rec.spend_limit_micros is not None else settings.default_spend_limit_micros
        return cls(
            key_id=rec.id,
            name=rec.name,
            project_id=rec.project_id,
            default_model=rec.default_model,
            spend_limit_micros=spend or None,
            max_concurrency=rec.max_concurrency or settings.max_concurrency_per_key,
            repos=rec.repos,
            hoplite_api_key=hoplite_api_key,
        )
