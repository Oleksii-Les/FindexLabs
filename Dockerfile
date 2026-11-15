# syntax=docker/dockerfile:1
# findex web service, two stages.
#
#   docker build -t findex .                                  # 150 Gutenberg books, indexed at build time
#   docker build --build-arg GUTENBERG_COUNT=0 -t findex .    # no corpus: bring your own index
#   docker run --rm -p 8000:8000 findex
#   docker run --rm -p 8000:8000 -v "$PWD:/data:ro" -e INDEX_PATH=/data/index.bin findex
#
# Stage 1 (builder) has uv and the build tools; stage 2 (runtime) only gets the finished
# virtualenv, the index and the books. Dependencies are installed exactly as in uv.lock.

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# 1) dependencies only: this layer is cached until uv.lock or pyproject.toml change
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# 2) the project itself, as a regular (non-editable) install into the same venv
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# 3) the data: download books and build the index with the code we just installed
ARG GUTENBERG_COUNT=150
COPY scripts ./scripts
RUN mkdir -p /data/books && \
    if [ "$GUTENBERG_COUNT" -gt 0 ]; then \
        .venv/bin/python scripts/fetch_gutenberg.py --count "$GUTENBERG_COUNT" --out /data/books && \
        .venv/bin/findex index /data/books --out /data/index.bin --positions; \
    fi


FROM python:3.12-slim AS runtime
RUN groupadd --system --gid 10001 app && \
    useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder --chown=app:app /data /data
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    INDEX_PATH=/data/index.bin CORPUS_ROOT=/data/books \
    PORT=8000 WEB_CONCURRENCY=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os,sys,urllib.request as u; sys.exit(0 if u.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=2).status == 200 else 1)"]
CMD ["findex", "serve", "--host", "0.0.0.0"]
