"""The same search in three route styles, for the def-vs-async-def measurement (Lab 07).

    uv run python experiments/web_variants.py --index index.bin --port 8001 [--cache 0]

    /search-def    def                      FastAPI runs it in a worker thread  (what the real app does)
    /search-async  async def, work inline   the work runs ON the event loop     (blocks it)
    /search-pool   async def + threadpool   explicit hop to a thread            (equivalent to def)
    /health        async def, no work       answers instantly -- if the loop is free

``--cache 0`` switches the per-query cache off, so a repeated URL does the full work every time
(otherwise a load test of one fixed query measures a dictionary lookup).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Query
from fastapi.concurrency import run_in_threadpool

from findex.web.schemas import SearchResponse
from findex.web.service import SearchService


def build(index: Path, cache_size: int) -> FastAPI:
    service = SearchService.from_path(index, cache_size=cache_size)
    app = FastAPI()

    def work(q: str, k: int) -> SearchResponse:
        return SearchResponse.from_page(service.search(q, k=k))

    @app.get("/search-def")
    def search_def(q: str = Query(), k: int = 10) -> SearchResponse:
        return work(q, k)

    @app.get("/search-async")
    async def search_async(q: str = Query(), k: int = 10) -> SearchResponse:
        return work(q, k)  # CPU-bound Python on the event loop: nothing else runs meanwhile

    @app.get("/search-pool")
    async def search_pool(q: str = Query(), k: int = 10) -> SearchResponse:
        return await run_in_threadpool(work, q, k)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--cache", type=int, default=0, help="query cache size (0 = off)")
    args = ap.parse_args()
    uvicorn.run(
        build(args.index, args.cache), host="127.0.0.1", port=args.port, log_level="warning"
    )


if __name__ == "__main__":
    main()
