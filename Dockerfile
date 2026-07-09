FROM ghcr.io/astral-sh/uv:0.9.8-python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.14-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

RUN groupadd --gid 10001 atlas \
    && useradd --uid 10001 --gid atlas --shell /usr/sbin/nologin --create-home atlas

COPY --from=builder --chown=atlas:atlas /app/.venv /app/.venv
COPY --chown=atlas:atlas alembic.ini ./
COPY --chown=atlas:atlas database ./database

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2)"

CMD ["uvicorn", "rag_platform.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
