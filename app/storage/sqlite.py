"""SQLite persistence (aiosqlite): proxy API keys and conversation fingerprints."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiosqlite

SCHEMA_VERSION = 1

_MIGRATIONS: dict[int, list[str]] = {
    1: [
        """
        CREATE TABLE api_keys (
            id                 TEXT PRIMARY KEY,
            name               TEXT,
            secret_hash        TEXT NOT NULL,
            hoplite_key_enc    TEXT NOT NULL,
            hoplite_key_hint   TEXT NOT NULL,
            project_id         TEXT NOT NULL,
            default_model      TEXT,
            spend_limit_micros INTEGER,
            max_concurrency    INTEGER,
            repos_json         TEXT,
            created_at         REAL NOT NULL,
            updated_at         REAL NOT NULL,
            last_used_at       REAL,
            revoked_at         REAL
        )
        """,
        """
        CREATE TABLE conversations (
            fingerprint         TEXT PRIMARY KEY,
            key_id              TEXT NOT NULL,
            thread_id           TEXT NOT NULL,
            run_id              TEXT,
            pending_question_id TEXT,
            created_at          REAL NOT NULL,
            consumed_at         REAL
        )
        """,
        "CREATE INDEX conversations_created_at ON conversations (created_at)",
    ],
}

UPDATABLE_KEY_FIELDS = frozenset(
    {"name", "hoplite_key_enc", "hoplite_key_hint", "project_id", "default_model",
     "spend_limit_micros", "max_concurrency", "repos"}
)


@dataclass
class KeyRecord:
    id: str
    secret_hash: str
    hoplite_key_enc: str
    hoplite_key_hint: str
    project_id: str
    name: str | None = None
    default_model: str | None = None
    spend_limit_micros: int | None = None
    max_concurrency: int | None = None
    repos: list[dict[str, Any]] | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_used_at: float | None = None
    revoked_at: float | None = None

    @property
    def revoked(self) -> bool:
        return self.revoked_at is not None


@dataclass
class ConversationRecord:
    fingerprint: str
    key_id: str
    thread_id: str
    run_id: str | None = None
    pending_question_id: str | None = None
    created_at: float = field(default_factory=time.time)
    consumed_at: float | None = None


def _key_from_row(row: aiosqlite.Row) -> KeyRecord:
    data = dict(row)
    repos_json = data.pop("repos_json", None)
    data["repos"] = json.loads(repos_json) if repos_json else None
    return KeyRecord(**data)


class Store:
    """A single shared aiosqlite connection in autocommit mode with WAL journaling."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._db: aiosqlite.Connection | None = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Store is not open")
        return self._db

    def _ensure_parent_dir(self) -> None:
        if self._path != ":memory:":
            Path(self._path).expanduser().parent.mkdir(parents=True, exist_ok=True)

    async def open(self) -> None:
        self._ensure_parent_dir()
        self._db = await aiosqlite.connect(self._path, isolation_level=None)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._migrate()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def ping(self) -> bool:
        try:
            await self.db.execute_fetchall("SELECT 1")
            return True
        except Exception:
            return False

    async def _migrate(self) -> None:
        rows = await self.db.execute_fetchall("PRAGMA user_version")
        version = int(list(rows)[0][0])
        for target in range(version + 1, SCHEMA_VERSION + 1):
            await self.db.execute("BEGIN")
            try:
                for statement in _MIGRATIONS[target]:
                    await self.db.execute(statement)
                await self.db.execute(f"PRAGMA user_version = {target}")
                await self.db.execute("COMMIT")
            except BaseException:
                await self.db.execute("ROLLBACK")
                raise

    # --- api keys --------------------------------------------------------------------------

    async def insert_key(self, rec: KeyRecord) -> None:
        await self.db.execute(
            """
            INSERT INTO api_keys (id, name, secret_hash, hoplite_key_enc, hoplite_key_hint, project_id,
                                  default_model, spend_limit_micros, max_concurrency, repos_json,
                                  created_at, updated_at, last_used_at, revoked_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec.id, rec.name, rec.secret_hash, rec.hoplite_key_enc, rec.hoplite_key_hint, rec.project_id,
                rec.default_model, rec.spend_limit_micros, rec.max_concurrency,
                json.dumps(rec.repos) if rec.repos is not None else None,
                rec.created_at, rec.updated_at, rec.last_used_at, rec.revoked_at,
            ),
        )

    async def get_key(self, key_id: str) -> KeyRecord | None:
        rows = list(await self.db.execute_fetchall("SELECT * FROM api_keys WHERE id = ?", (key_id,)))
        return _key_from_row(rows[0]) if rows else None

    async def list_keys(self, *, include_revoked: bool = False) -> list[KeyRecord]:
        sql = "SELECT * FROM api_keys"
        if not include_revoked:
            sql += " WHERE revoked_at IS NULL"
        sql += " ORDER BY created_at"
        return [_key_from_row(row) for row in await self.db.execute_fetchall(sql)]

    async def update_key(self, key_id: str, **changes: Any) -> KeyRecord | None:
        unknown = set(changes) - UPDATABLE_KEY_FIELDS
        if unknown:
            raise ValueError(f"cannot update fields: {sorted(unknown)}")
        if changes:
            assignments: list[str] = []
            params: list[Any] = []
            for name, value in changes.items():
                if name == "repos":
                    assignments.append("repos_json = ?")
                    params.append(json.dumps(value) if value is not None else None)
                else:
                    assignments.append(f"{name} = ?")
                    params.append(value)
            assignments.append("updated_at = ?")
            params.extend([time.time(), key_id])
            cursor = await self.db.execute(
                f"UPDATE api_keys SET {', '.join(assignments)} WHERE id = ? AND revoked_at IS NULL", params
            )
            if cursor.rowcount == 0:
                return None
        return await self.get_key(key_id)

    async def revoke_key(self, key_id: str) -> bool:
        now = time.time()
        cursor = await self.db.execute(
            "UPDATE api_keys SET revoked_at = ?, updated_at = ? WHERE id = ? AND revoked_at IS NULL",
            (now, now, key_id),
        )
        return cursor.rowcount > 0

    async def touch_key(self, key_id: str) -> None:
        await self.db.execute("UPDATE api_keys SET last_used_at = ? WHERE id = ?", (time.time(), key_id))

    async def all_keys(self) -> list[KeyRecord]:
        return await self.list_keys(include_revoked=True)

    async def set_key_ciphertext(self, key_id: str, ciphertext: str) -> None:
        await self.db.execute("UPDATE api_keys SET hoplite_key_enc = ? WHERE id = ?", (ciphertext, key_id))

    # --- conversations ---------------------------------------------------------------------

    async def save_conversation(self, rec: ConversationRecord) -> None:
        await self.db.execute(
            """
            INSERT INTO conversations (fingerprint, key_id, thread_id, run_id, pending_question_id,
                                       created_at, consumed_at)
            VALUES (?, ?, ?, ?, ?, ?, NULL)
            ON CONFLICT (fingerprint) DO UPDATE SET
                key_id = excluded.key_id,
                thread_id = excluded.thread_id,
                run_id = excluded.run_id,
                pending_question_id = excluded.pending_question_id,
                created_at = excluded.created_at,
                consumed_at = NULL
            """,
            (rec.fingerprint, rec.key_id, rec.thread_id, rec.run_id, rec.pending_question_id, rec.created_at),
        )

    async def consume_conversation(self, fingerprint: str, key_id: str) -> ConversationRecord | None:
        """Atomically claim a conversation continuation; returns None if absent or already claimed."""
        rows = list(
            await self.db.execute_fetchall(
                """
                UPDATE conversations SET consumed_at = ?
                WHERE fingerprint = ? AND key_id = ? AND consumed_at IS NULL
                RETURNING fingerprint, key_id, thread_id, run_id, pending_question_id, created_at, consumed_at
                """,
                (time.time(), fingerprint, key_id),
            )
        )
        return ConversationRecord(**dict(rows[0])) if rows else None

    async def release_conversation(self, fingerprint: str) -> None:
        await self.db.execute("UPDATE conversations SET consumed_at = NULL WHERE fingerprint = ?", (fingerprint,))

    async def get_conversation(self, fingerprint: str) -> ConversationRecord | None:
        rows = list(
            await self.db.execute_fetchall("SELECT * FROM conversations WHERE fingerprint = ?", (fingerprint,))
        )
        return ConversationRecord(**dict(rows[0])) if rows else None

    async def purge_conversations(self, older_than: float) -> int:
        cursor = await self.db.execute("DELETE FROM conversations WHERE created_at < ?", (older_than,))
        return cursor.rowcount
