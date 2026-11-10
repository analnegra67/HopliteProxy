"""Symmetric encryption of stored Hoplite keys (Fernet, with key rotation via MultiFernet)."""

from __future__ import annotations

from cryptography.fernet import Fernet, InvalidToken, MultiFernet

__all__ = ["InvalidToken", "SecretBox"]


class SecretBox:
    """Encrypts with the first configured key; decrypts with any of them."""

    def __init__(self, keys_csv: str) -> None:
        keys = [key.strip() for key in keys_csv.split(",") if key.strip()]
        if not keys:
            raise ValueError(
                "ENCRYPTION_KEYS is empty. Generate one with: python -m app.cli gen-secrets"
            )
        try:
            fernets = [Fernet(key.encode("ascii")) for key in keys]
        except (ValueError, UnicodeEncodeError) as exc:
            raise ValueError("ENCRYPTION_KEYS must contain url-safe base64 Fernet keys (32 bytes)") from exc
        self._multi = MultiFernet(fernets)

    def encrypt(self, plaintext: str) -> str:
        return self._multi.encrypt(plaintext.encode("utf-8")).decode("ascii")

    def decrypt(self, token: str) -> str:
        return self._multi.decrypt(token.encode("ascii")).decode("utf-8")

    def rotate(self, token: str) -> str:
        """Re-encrypt a token with the current primary key."""
        return self._multi.rotate(token.encode("ascii")).decode("ascii")

    @staticmethod
    def generate_key() -> str:
        return Fernet.generate_key().decode("ascii")
