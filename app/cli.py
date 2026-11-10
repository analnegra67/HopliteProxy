"""Operational commands.

    python -m app.cli gen-secrets          print fresh MASTER_KEY and ENCRYPTION_KEYS values
    python -m app.cli create-key [...]     issue a proxy key without going through the HTTP admin API
    python -m app.cli rotate-encryption    re-encrypt stored Hoplite keys with the first ENCRYPTION_KEYS entry

`create-key` reads the Hoplite key from the HOPLITE_API_KEY variable (environment or .env), so the
secret never appears in shell history.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import secrets
import sys

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.admin.keys_service import issue_key, resolve_project
from app.config import get_settings
from app.errors import OpenAIError
from app.hoplite.client import HopliteClient
from app.security.crypto import SecretBox
from app.storage.sqlite import Store


class _CliEnv(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_ignore_empty=True)

    hoplite_api_key: str = ""
    hoplite_project_id: str = ""


def gen_secrets() -> None:
    print(f"MASTER_KEY={secrets.token_urlsafe(48)}")
    print(f"ENCRYPTION_KEYS={SecretBox.generate_key()}")


async def create_key(args: argparse.Namespace) -> int:
    settings = get_settings()
    env = _CliEnv()
    hoplite_key = os.environ.get(args.hoplite_key_env, "")
    if not hoplite_key and args.hoplite_key_env == "HOPLITE_API_KEY":
        hoplite_key = env.hoplite_api_key  # fall back to the .env file
    if not hoplite_key.startswith("hop_"):
        print(f"error: set {args.hoplite_key_env} (environment or .env) to a Hoplite key starting with hop_",
              file=sys.stderr)
        return 2
    project_id = args.project_id or env.hoplite_project_id or None
    box = SecretBox(settings.encryption_keys)
    store = Store(settings.database_path)
    await store.open()
    try:
        async with httpx.AsyncClient(base_url=settings.hoplite_base_url, timeout=30) as http:
            client = HopliteClient(http, hoplite_key)
            try:
                project_id = await resolve_project(client, project_id)
            except OpenAIError as exc:
                print(f"error: {exc.message}", file=sys.stderr)
                return 1
        record, token = await issue_key(
            store=store, box=box, settings=settings, hoplite_api_key=hoplite_key, project_id=project_id,
            name=args.name, default_model=args.default_model, spend_limit_micros=args.spend_limit_micros,
            max_concurrency=args.max_concurrency,
        )
    finally:
        await store.close()
    print(f"key id:   {record.id}")
    print(f"project:  {record.project_id}")
    print(f"api key:  {token}")
    print("Store the api key now: it is not shown again.")
    return 0


async def rotate_encryption() -> int:
    settings = get_settings()
    box = SecretBox(settings.encryption_keys)
    store = Store(settings.database_path)
    await store.open()
    try:
        records = await store.all_keys()
        for record in records:
            await store.set_key_ciphertext(record.id, box.rotate(record.hoplite_key_enc))
    finally:
        await store.close()
    print(f"re-encrypted {len(records)} key(s) with the primary encryption key")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("gen-secrets", help="print new MASTER_KEY and ENCRYPTION_KEYS values")
    create = sub.add_parser("create-key", help="issue a proxy key")
    create.add_argument("--project-id", help="Hoplite project id (default: HOPLITE_PROJECT_ID or the only project)")
    create.add_argument("--name")
    create.add_argument("--default-model")
    create.add_argument("--spend-limit-micros", type=int)
    create.add_argument("--max-concurrency", type=int)
    create.add_argument("--hoplite-key-env", default="HOPLITE_API_KEY",
                        help="name of the variable holding the Hoplite key (default: HOPLITE_API_KEY)")
    sub.add_parser("rotate-encryption", help="re-encrypt stored Hoplite keys with the primary key")
    args = parser.parse_args(argv)

    if args.command == "gen-secrets":
        gen_secrets()
        return 0
    if args.command == "create-key":
        return asyncio.run(create_key(args))
    return asyncio.run(rotate_encryption())


if __name__ == "__main__":
    raise SystemExit(main())
