"""`python -m app`: run the proxy with uvicorn using HOST/PORT from the configuration."""

from __future__ import annotations

import uvicorn

from app.config import get_settings
from app.log import setup_logging


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_format)
    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_config=None,
        proxy_headers=True,
        timeout_graceful_shutdown=20,
    )


if __name__ == "__main__":
    main()
