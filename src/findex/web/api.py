"""JSON routes. Thin: validate (Pydantic, before the body runs) -> ask the service -> shape the answer.

``def`` and not ``async def``: searching is CPU-bound Python. FastAPI runs a ``def`` route in a
worker thread, so the event loop stays free to accept connections and answer ``/health`` while
searches run. An ``async def`` route that did the same work inline would freeze the whole
server for the duration of every query (measured in ``experiments/web_variants.py``, README).
Only ``/health`` is ``async def``: it does no work at all, a thread hop would be pure cost.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Path, Query

from findex.web import __version__
from findex.web.deps import MaybeService, Service
from findex.web.schemas import (
    DocParams,
    DocumentResponse,
    ErrorResponse,
    HealthResponse,
    SearchParams,
    SearchResponse,
    StatsResponse,
)
from findex.web.service import DocumentNotFound, DocumentUnavailable

router = APIRouter(tags=["api"])

Responses = dict[int | str, dict[str, Any]]
_422: Responses = {422: {"description": "Invalid parameter or malformed query"}}
_503: Responses = {503: {"model": ErrorResponse}}
_404: Responses = {404: {"model": ErrorResponse}}
_501: Responses = {
    501: {"model": ErrorResponse, "description": "mode needs embeddings, none loaded"}
}


@router.get("/search", response_model=SearchResponse, responses=_422 | _503 | _501)
def search_route(params: Annotated[SearchParams, Query()], service: Service) -> SearchResponse:
    """Ranked, paginated full-text search. ``k`` is the page size, ``page`` starts at 1."""
    return SearchResponse.from_page(
        service.search(
            params.q, k=params.k, scorer=params.scorer, page=params.page, mode=params.mode
        )
    )


@router.get(
    "/docs/{doc_id}",
    response_model=DocumentResponse,
    responses=_404 | _422,
)
def document_route(
    doc_id: Annotated[int, Path(ge=0, description="Id from a search hit.")],
    params: Annotated[DocParams, Query()],
    service: Service,
) -> DocumentResponse:
    """The text of a document, in slices (``offset`` / ``limit`` characters)."""
    try:
        return DocumentResponse.from_page(
            service.document(doc_id, offset=params.offset, limit=params.limit)
        )
    except (DocumentNotFound, DocumentUnavailable):
        raise HTTPException(status_code=404, detail=f"document {doc_id} not found") from None


@router.get("/stats", response_model=StatsResponse, responses=_503)
def stats_route(service: Service) -> StatsResponse:
    """Size of the index, the query cache and how long the service has been up."""
    return StatsResponse.from_stats(service.stats(), __version__)


@router.get("/health", response_model=HealthResponse, responses=_503)
async def health_route(service: MaybeService) -> HealthResponse:
    """For the platform's health check: 200 once the index is loaded, 503 before (and if loading failed)."""
    if service is None:
        raise HTTPException(
            status_code=503, detail="index not loaded", headers={"Retry-After": "2"}
        )
    return HealthResponse(status="ok", version=__version__, documents=service.index.num_docs)
