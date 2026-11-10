"""Issuing proxy keys: shared by the admin API and the CLI."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import anyio

from app.config import Settings
from app.errors import invalid_request
from app.hoplite.client import HopliteClient, HopliteError, to_openai_error
from app.security.crypto import SecretBox
from app.security.keys import display_prefix, generate_key, hash_secret, hoplite_key_hint
from app.storage.sqlite import KeyRecord, Store


async def resolve_project(client: HopliteClient, project_id: str | None) -> str:
    """Check the Hoplite key works and the project is reachable; pick the only project if none is given."""
    try:
        if project_id:
            await client.get_project(project_id)
            return project_id
        projects = await client.list_projects()
    except HopliteError as exc:
        if exc.status in (401, 403):
            raise invalid_request(
                f"Hoplite rejected hoplite_api_key (HTTP {exc.status}). Check the key and that it has "
                "the project:read, thread:create, thread:read, thread:update and thread:stop permissions.",
                param="hoplite_api_key", code="invalid_hoplite_key",
            ) from exc
        if exc.status == 404 and project_id:
            raise invalid_request(
                f"Project {project_id!r} was not found or is not accessible with this Hoplite key.",
                param="project_id", code="project_not_found",
            ) from exc
        raise to_openai_error(exc) from exc
    if len(projects) == 1:
        return str(projects[0]["id"])
    listing = ", ".join(f"{p.get('id')} ({p.get('name') or 'unnamed'})" for p in projects[:20]) or "none"
    raise invalid_request(
        f"project_id is required: this Hoplite key can access {len(projects)} project(s): {listing}",
        param="project_id", code="project_required",
    )


async def issue_key(
    *,
    store: Store,
    box: SecretBox,
    settings: Settings,
    hoplite_api_key: str,
    project_id: str,
    name: str | None = None,
    default_model: str | None = None,
    spend_limit_micros: int | None = None,
    max_concurrency: int | None = None,
    repos: list[dict[str, Any]] | None = None,
) -> tuple[KeyRecord, str]:
    """Create and store a new proxy key. Returns the record and the plaintext key (shown once)."""
    generated = generate_key()
    secret_hash = await anyio.to_thread.run_sync(hash_secret, generated.secret, settings.bcrypt_rounds)
    record = KeyRecord(
        id=generated.key_id,
        secret_hash=secret_hash,
        hoplite_key_enc=box.encrypt(hoplite_api_key),
        hoplite_key_hint=hoplite_key_hint(hoplite_api_key),
        project_id=project_id,
        name=name,
        default_model=default_model,
        spend_limit_micros=spend_limit_micros,
        max_concurrency=max_concurrency,
        repos=repos,
    )
    await store.insert_key(record)
    return record, generated.token


def _iso(ts: float | None) -> str | None:
    return datetime.fromtimestamp(ts, UTC).isoformat().replace("+00:00", "Z") if ts is not None else None


def key_view(rec: KeyRecord) -> dict[str, Any]:
    """Public representation of a key: never includes the secret or the Hoplite key."""
    return {
        "id": rec.id,
        "object": "api_key",
        "name": rec.name,
        "key_prefix": display_prefix(rec.id),
        "project_id": rec.project_id,
        "hoplite_key_hint": rec.hoplite_key_hint,
        "default_model": rec.default_model,
        "spend_limit_micros": rec.spend_limit_micros,
        "max_concurrency": rec.max_concurrency,
        "repos": [{"repo_full_name": r.get("repoFullName"), "branch": r.get("branch")} for r in rec.repos]
        if rec.repos else None,
        "created_at": _iso(rec.created_at),
        "updated_at": _iso(rec.updated_at),
        "last_used_at": _iso(rec.last_used_at),
        "revoked": rec.revoked,
        "revoked_at": _iso(rec.revoked_at),
    }
