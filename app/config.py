"""Runtime configuration, read from environment variables and an optional `.env` file."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

MIN_MASTER_KEY_LENGTH = 32


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    # --- secrets -----------------------------------------------------------------------------
    master_key: str = ""
    encryption_keys: str = ""  # comma-separated Fernet keys; the first one encrypts new data

    # --- storage and network -----------------------------------------------------------------
    database_path: str = "data/proxy.db"
    hoplite_base_url: str = "https://api.hoplite.sh"
    hoplite_timeout_seconds: float = Field(default=30.0, gt=0)
    hoplite_max_attempts: int = Field(default=4, ge=1, le=10)
    hoplite_max_backoff_seconds: float = Field(default=20.0, ge=0)
    sse_read_timeout_seconds: float = Field(default=45.0, gt=0)  # Hoplite heartbeats every 15 s
    message_sync_interval_seconds: float = Field(default=1.0, ge=0)  # min gap between message-list fetches
    host: str = "127.0.0.1"
    port: int = 8000

    # --- behaviour ---------------------------------------------------------------------------
    thread_reuse: Literal["fingerprint", "off"] = "fingerprint"
    question_policy: Literal["surface", "auto"] = "surface"
    question_auto_answer: str = (
        "No human is available to answer right now. Proceed using your best judgment "
        "and state the assumptions you made."
    )
    default_spend_limit_micros: int = Field(default=1_000_000, ge=0)  # per new thread; 0 = no limit
    run_timeout_seconds: float = Field(default=1800.0, gt=0)
    stop_on_disconnect: bool = True
    disconnect_grace_seconds: float = Field(default=60.0, ge=0)
    keepalive_seconds: float = Field(default=15.0, gt=0)
    result_cache_seconds: float = Field(default=600.0, ge=0)
    model_aliases: Annotated[dict[str, str], NoDecode] = Field(default_factory=dict)
    model_catalog_ttl_seconds: float = Field(default=300.0, ge=0)
    include_reasoning: bool = False
    max_concurrency_per_key: int = Field(default=4, ge=1)
    conversation_ttl_days: float = Field(default=14.0, gt=0)

    # --- security ----------------------------------------------------------------------------
    bcrypt_rounds: int = Field(default=12, ge=4, le=16)
    auth_cache_seconds: float = Field(default=300.0, ge=0)
    cors_origins: str = ""  # comma-separated; empty disables CORS
    max_body_bytes: int = Field(default=10 * 1024 * 1024, gt=0)

    # --- logging -----------------------------------------------------------------------------
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"
    log_prompts: bool = False

    @field_validator("model_aliases", mode="before")
    @classmethod
    def _parse_aliases(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return {}
            parsed = json.loads(value)
            if not isinstance(parsed, dict):
                raise ValueError("MODEL_ALIASES must be a JSON object, e.g. {\"gpt-4o\": \"gpt-6-sol\"}")
            return parsed
        return value

    @property
    def admin_enabled(self) -> bool:
        return len(self.master_key) >= MIN_MASTER_KEY_LENGTH

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def spend_limit_default(self) -> int | None:
        return self.default_spend_limit_micros or None


@lru_cache
def get_settings() -> Settings:
    return Settings()
