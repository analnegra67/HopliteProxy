FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install . \
    && useradd --create-home --uid 10001 proxy \
    && mkdir -p /data \
    && chown proxy /data

USER proxy
ENV HOST=0.0.0.0 \
    PORT=8000 \
    DATABASE_PATH=/data/proxy.db \
    LOG_FORMAT=json
EXPOSE 8000
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).status == 200 else 1)"]
CMD ["python", "-m", "app"]
