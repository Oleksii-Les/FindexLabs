"""The application factory: ``uvicorn findex.web.app:create_app --factory``.

Start-up (``lifespan``) loads the index ONCE, in a background task, and the app starts answering
at once: ``/health`` says 503 until the index is in memory (that is what a platform's health
check polls), the pages say "starting up". If loading fails the process stays alive, logs the
reason and keeps answering 503 -- the platform shows the log instead of restarting in a loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from findex.web import __version__, api, pages
from findex.web.errors import install_error_handlers
from findex.web.logs import configure_service_logging
from findex.web.middleware import RequestContextMiddleware
from findex.web.service import SearchService
from findex.web.settings import Settings
from findex.web.templating import STATIC_DIR

log = logging.getLogger(__name__)


async def _load_index(app: FastAPI) -> None:
    settings: Settings = app.state.settings
    try:
        service = await asyncio.to_thread(
            SearchService.from_path,
            settings.index_path,
            corpus_root=settings.corpus_root,
            embeddings=settings.embeddings_path,
            vectorized=settings.vectorized,
            cache_size=settings.cache_size,
            window=settings.snippet_window,
        )
    except Exception as exc:
        app.state.load_error = f"{type(exc).__name__}: {exc}"
        log.error(
            "cannot load the index %s: %s", settings.index_path, app.state.load_error, exc_info=exc
        )
        return
    app.state.service = service
    app.state.load_error = None
    log.info("ready: %d documents, %d terms", service.index.num_docs, len(service.index))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    loader = asyncio.create_task(_load_index(app), name="load-index")
    try:
        yield
    finally:
        loader.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await loader
        service: SearchService | None = app.state.service
        app.state.service = None
        if service is not None:
            service.close()
            log.info("index closed")


def create_app(settings: Settings | None = None, *, setup_logging: bool = True) -> FastAPI:
    settings = settings or Settings()
    if setup_logging:
        configure_service_logging(settings.log_level)
    app = FastAPI(
        title=f"{settings.site_name} search API",
        version=__version__,
        description="Full-text search (BM25 / TF-IDF) over an inverted index. Query language: "
        '`words`, `AND` / `OR` / `NOT`, parentheses, `"exact phrase"`.',
        lifespan=lifespan,
        docs_url="/api/docs",  # /docs/{doc_id} is a document, not Swagger
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.settings = settings
    app.state.service = None
    app.state.load_error = None
    app.add_middleware(RequestContextMiddleware)
    install_error_handlers(app)
    app.include_router(api.router)
    app.include_router(pages.router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
