"""FastAPI dependencies: admin (MASTER_KEY) and client (sk-hop-...) authentication."""

from __future__ import annotations

import hmac
import logging

import anyio
from fastapi import Request

from app.context import KeyContext
from app.errors import OpenAIError, authentication_error
from app.security.crypto import InvalidToken
from app.security.keys import check_secret, parse_key, token_digest
from app.services import Services

log = logging.getLogger(__name__)


def get_services(request: Request) -> Services:
    return request.app.state.services


def bearer_token(header: str | None) -> str | None:
    if not header:
        return None
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()


async def require_admin(request: Request) -> None:
    settings = get_services(request).settings
    if not settings.admin_enabled:
        raise OpenAIError(
            503,
            "The admin API is disabled: set MASTER_KEY to a random string of at least 32 characters.",
            type_="api_error",
            code="admin_disabled",
        )
    token = bearer_token(request.headers.get("authorization")) or request.headers.get("x-admin-key")
    if not token or not hmac.compare_digest(token.encode("utf-8"), settings.master_key.encode("utf-8")):
        raise OpenAIError(
            401, "Invalid admin credentials.", type_="authentication_error", code="invalid_admin_key",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def require_client_key(request: Request) -> KeyContext:
    services = get_services(request)
    token = bearer_token(request.headers.get("authorization"))
    if token is None:
        raise authentication_error(
            "You didn't provide an API key. Send it as 'Authorization: Bearer sk-hop-...'."
        )
    parsed = parse_key(token)
    if parsed is None:
        raise authentication_error()
    key_id, secret = parsed
    record = await services.store.get_key(key_id)
    if record is None or record.revoked:
        raise authentication_error()
    digest = token_digest(token)
    if not services.auth_cache.hit(key_id, digest):
        if not await anyio.to_thread.run_sync(check_secret, secret, record.secret_hash):
            raise authentication_error()
        services.auth_cache.add(key_id, digest)
    try:
        hoplite_key = services.box.decrypt(record.hoplite_key_enc)
    except InvalidToken:
        log.error("cannot decrypt the stored Hoplite key; check ENCRYPTION_KEYS", extra={"key_id": key_id})
        raise OpenAIError(500, "The proxy cannot decrypt its upstream credential.", type_="api_error",
                          code="credential_decryption_failed") from None
    await services.touch_key(key_id)
    return KeyContext.from_record(record, hoplite_key, services.settings)
