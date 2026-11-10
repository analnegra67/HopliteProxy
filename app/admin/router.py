"""Admin API: issue, list, update and revoke proxy keys. Protected by MASTER_KEY."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.admin.keys_service import issue_key, key_view, resolve_project
from app.auth import get_services, require_admin
from app.errors import invalid_request, not_found
from app.security.keys import hoplite_key_hint, parse_key
from app.services import Services

log = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


class RepoSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_full_name: str = Field(min_length=3, max_length=200, pattern=r"^[^/\s]+/[^/\s]+$")
    branch: str | None = Field(default=None, min_length=1, max_length=200)

    def to_hoplite(self) -> dict[str, Any]:
        repo: dict[str, Any] = {"repoFullName": self.repo_full_name}
        if self.branch:
            repo["branch"] = self.branch
        return repo


def _check_hoplite_key(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value.startswith("hop_"):
        raise ValueError("hoplite_api_key must be a Hoplite API key starting with 'hop_'")
    return value


class KeyCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hoplite_api_key: str = Field(min_length=8, max_length=512)
    project_id: str | None = Field(default=None, min_length=1, max_length=128)
    name: str | None = Field(default=None, max_length=200)
    default_model: str | None = Field(default=None, min_length=1, max_length=128)
    spend_limit_micros: int | None = Field(default=None, ge=0, description="Per new thread; 0 = no limit")
    max_concurrency: int | None = Field(default=None, ge=1, le=64)
    repos: list[RepoSpec] | None = None
    verify: bool = Field(default=True, description="Check the Hoplite key and project before saving")

    @field_validator("hoplite_api_key")
    @classmethod
    def check_hoplite_key(cls, value: str | None) -> str | None:
        return _check_hoplite_key(value)


class KeyUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hoplite_api_key: str | None = Field(default=None, min_length=8, max_length=512)
    project_id: str | None = Field(default=None, min_length=1, max_length=128)
    name: str | None = Field(default=None, max_length=200)
    default_model: str | None = Field(default=None, min_length=1, max_length=128)
    spend_limit_micros: int | None = Field(default=None, ge=0)
    max_concurrency: int | None = Field(default=None, ge=1, le=64)
    repos: list[RepoSpec] | None = None
    verify: bool = True

    @field_validator("hoplite_api_key")
    @classmethod
    def check_hoplite_key(cls, value: str | None) -> str | None:
        return _check_hoplite_key(value)


@router.post("/keys", status_code=201)
async def create_key(body: KeyCreate, services: Services = Depends(get_services)) -> dict[str, Any]:
    project_id = body.project_id
    if body.verify:
        project_id = await resolve_project(services.hoplite(body.hoplite_api_key), body.project_id)
    elif not project_id:
        raise invalid_request("project_id is required when verify is false.", param="project_id")
    assert project_id is not None
    record, token = await issue_key(
        store=services.store,
        box=services.box,
        settings=services.settings,
        hoplite_api_key=body.hoplite_api_key,
        project_id=project_id,
        name=body.name,
        default_model=body.default_model,
        spend_limit_micros=body.spend_limit_micros,
        max_concurrency=body.max_concurrency,
        repos=[r.to_hoplite() for r in body.repos] if body.repos else None,
    )
    log.info("proxy key created", extra={"key_id": record.id, "project_id": project_id})
    return {**key_view(record), "key": token}


@router.get("/keys")
async def list_keys(include_revoked: bool = False, services: Services = Depends(get_services)) -> dict[str, Any]:
    records = await services.store.list_keys(include_revoked=include_revoked)
    return {"object": "list", "data": [key_view(r) for r in records]}


@router.get("/keys/{key_id}")
async def get_key(key_id: str, services: Services = Depends(get_services)) -> dict[str, Any]:
    record = await services.store.get_key(key_id)
    if record is None:
        raise not_found(f"API key {key_id!r} not found.", code="key_not_found")
    return key_view(record)


@router.patch("/keys/{key_id}")
async def update_key(key_id: str, body: KeyUpdate, services: Services = Depends(get_services)) -> dict[str, Any]:
    record = await services.store.get_key(key_id)
    if record is None or record.revoked:
        raise not_found(f"API key {key_id!r} not found or revoked.", code="key_not_found")
    fields = body.model_fields_set - {"verify"}
    if "project_id" in fields and body.project_id is None:
        raise invalid_request("project_id cannot be null.", param="project_id")
    if "hoplite_api_key" in fields and body.hoplite_api_key is None:
        raise invalid_request("hoplite_api_key cannot be null.", param="hoplite_api_key")

    changes: dict[str, Any] = {}
    if body.verify and fields & {"hoplite_api_key", "project_id"}:
        hoplite_key = body.hoplite_api_key or services.box.decrypt(record.hoplite_key_enc)
        await resolve_project(services.hoplite(hoplite_key), body.project_id or record.project_id)
    if body.hoplite_api_key:
        changes["hoplite_key_enc"] = services.box.encrypt(body.hoplite_api_key)
        changes["hoplite_key_hint"] = hoplite_key_hint(body.hoplite_api_key)
    for name in ("project_id", "name", "default_model", "spend_limit_micros", "max_concurrency"):
        if name in fields:
            changes[name] = getattr(body, name)
    if "repos" in fields:
        changes["repos"] = [r.to_hoplite() for r in body.repos] if body.repos else None

    updated = await services.store.update_key(key_id, **changes)
    if updated is None:
        raise not_found(f"API key {key_id!r} not found or revoked.", code="key_not_found")
    services.forget_key(key_id)
    log.info("proxy key updated", extra={"key_id": key_id, "fields": sorted(fields)})
    return key_view(updated)


@router.delete("/keys/{key_ref}")
async def revoke_key(key_ref: str, services: Services = Depends(get_services)) -> dict[str, Any]:
    """Revoke a key by its id (preferred) or by the full `sk-hop-...` value."""
    parsed = parse_key(key_ref)
    key_id = parsed[0] if parsed else key_ref
    if not await services.store.revoke_key(key_id):
        raise not_found(f"API key {key_id!r} not found or already revoked.", code="key_not_found")
    services.forget_key(key_id)
    log.info("proxy key revoked", extra={"key_id": key_id})
    return {"id": key_id, "object": "api_key", "revoked": True}
