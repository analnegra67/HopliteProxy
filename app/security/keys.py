"""Proxy API keys: `sk-hop-<id><secret>`.

The 12-character id is a public lookup handle (primary key in the database); only the
40-character secret is bcrypt-hashed. Looking a key up by id keeps authentication O(1)
instead of bcrypt-checking every stored hash.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import string
import time
from dataclasses import dataclass

import bcrypt

KEY_PREFIX = "sk-hop-"
ALPHABET = string.ascii_letters + string.digits
ID_LENGTH = 12
SECRET_LENGTH = 40
_ALPHABET_SET = frozenset(ALPHABET)


@dataclass(frozen=True)
class GeneratedKey:
    key_id: str
    secret: str

    @property
    def token(self) -> str:
        return f"{KEY_PREFIX}{self.key_id}{self.secret}"


def _random_string(length: int) -> str:
    return "".join(secrets.choice(ALPHABET) for _ in range(length))


def generate_key() -> GeneratedKey:
    return GeneratedKey(key_id=_random_string(ID_LENGTH), secret=_random_string(SECRET_LENGTH))


def parse_key(token: str) -> tuple[str, str] | None:
    """Split a proxy key into (key_id, secret), or return None when it is malformed."""
    if not token.startswith(KEY_PREFIX):
        return None
    body = token[len(KEY_PREFIX):]
    if len(body) != ID_LENGTH + SECRET_LENGTH or not set(body) <= _ALPHABET_SET:
        return None
    return body[:ID_LENGTH], body[ID_LENGTH:]


def display_prefix(key_id: str) -> str:
    return f"{KEY_PREFIX}{key_id}…"


def hash_secret(secret: str, rounds: int) -> str:
    return bcrypt.hashpw(secret.encode("ascii"), bcrypt.gensalt(rounds)).decode("ascii")


def check_secret(secret: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(secret.encode("ascii"), hashed.encode("ascii"))
    except ValueError:
        return False


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def hoplite_key_hint(hoplite_key: str) -> str:
    kind = "hop_svc_" if hoplite_key.startswith("hop_svc_") else "hop_"
    return f"{kind}…{hoplite_key[-4:]}"


class VerificationCache:
    """Remembers keys whose bcrypt check passed recently, so hot paths skip bcrypt.

    Only a SHA-256 digest of the full key is kept. Revocation is still checked against the
    database on every request; this cache only saves the expensive hash comparison.
    """

    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = ttl_seconds
        self._items: dict[str, tuple[str, float]] = {}

    def hit(self, key_id: str, digest: str) -> bool:
        item = self._items.get(key_id)
        if item is None:
            return False
        cached_digest, expires_at = item
        if expires_at < time.monotonic():
            self._items.pop(key_id, None)
            return False
        return hmac.compare_digest(cached_digest, digest)

    def add(self, key_id: str, digest: str) -> None:
        if self._ttl > 0:
            self._items[key_id] = (digest, time.monotonic() + self._ttl)

    def invalidate(self, key_id: str) -> None:
        self._items.pop(key_id, None)
