"""Application factory. Run with `python -m app` or `uvicorn app.main:create_app --factory`."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.admin.router import router as admin_router
from app.config import Settings, get_settings
from app.errors import install_error_handlers
from app.log import setup_logging
from app.middleware import RequestContextMiddleware
from app.openai_api.router import router as openai_router
from app.security.crypto import SecretBox
from app.services import Services
from app.storage.sqlite import Store

log = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None, *, hoplite_transport: httpx.AsyncBaseTransport | None = None
) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level, settings.log_format)
    box = SecretBox(settings.encryption_keys)  # fail fast on a missing or malformed ENCRYPTION_KEYS
    if not settings.admin_enabled:
        log.warning("MASTER_KEY is empty or shorter than 32 characters: the admin API is disabled")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store = Store(settings.database_path)
        await store.open()
        http = httpx.AsyncClient(
            base_url=settings.hoplite_base_url,
            timeout=httpx.Timeout(settings.hoplite_timeout_seconds, connect=10.0),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
            headers={"User-Agent": f"hoplite-openai-proxy/{__version__}"},
            transport=hoplite_transport,
            follow_redirects=False,
        )
        services = Services(settings, store, http, box)
        app.state.services = services
        await services.purge_once()
        cleanup = asyncio.create_task(services.cleanup_forever(), name="cleanup")
        log.info("proxy ready", extra={"hoplite_base_url": settings.hoplite_base_url,
                                       "database": settings.database_path, "thread_reuse": settings.thread_reuse})
        try:
            yield
        finally:
            cleanup.cancel()
            await services.registry.shutdown()
            await http.aclose()
            await store.close()

    app = FastAPI(
        title="Hoplite OpenAI-compatible proxy",
        version=__version__,
        description="OpenAI `/v1/chat/completions` in front of Hoplite cloud coding agents.",
        lifespan=lifespan,
    )
    install_error_handlers(app)
    if settings.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["X-Hoplite-Thread-Id", "X-Hoplite-Run-Id", "X-Request-Id"],
        )
    app.add_middleware(RequestContextMiddleware, max_body_bytes=settings.max_body_bytes)
    app.include_router(openai_router)
    app.include_router(admin_router)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz(request: Request) -> JSONResponse:
        ok = await request.app.state.services.store.ping()
        return JSONResponse({"status": "ok" if ok else "degraded", "database": ok}, status_code=200 if ok else 503)

    return app
