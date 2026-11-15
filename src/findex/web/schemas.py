"""Pydantic models: the ONLY place where the web layer's data has a schema.

* input  -- ``SearchParams`` / ``PageParams`` / ``DocParams``: query parameters, validated before
  any route code runs (a bad one is a 422 with the field named, never a 500);
* output -- ``SearchResponse`` & co.: what the JSON looks like (and what ``/api/docs`` shows).

Inside the program there are no Pydantic objects: the service returns the dataclasses of
``service.py`` and the routes convert them here, once, on the way out.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from findex.scoring import ScorerName
from findex.semantic import Mode
from findex.web.service import MARK_OFF, MARK_ON, DocPage, SearchPage, ServiceStats

Query = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
MAX_K = 50
MAX_PAGE = 100


# ----------------------------------------------------------------------------- input
class SearchParams(BaseModel):
    """``GET /search?q=...&k=...&scorer=...&page=...`` -- unknown parameters are an error."""

    model_config = ConfigDict(extra="forbid")

    q: Query = Field(description='Query: words, AND / OR / NOT, parentheses, "exact phrase".')
    k: int = Field(10, ge=1, le=MAX_K, description="Results per page.")
    scorer: ScorerName = Field("bm25", description="Ranking function.")
    page: int = Field(1, ge=1, le=MAX_PAGE, description="Page number, from 1.")
    mode: Mode = Field(
        "lexical",
        description="lexical = words and operators (BM25/TF-IDF); semantic = by meaning (best fragment); "
        "hybrid = both fused with RRF. The last two need embeddings (501 otherwise).",
    )


class PageParams(BaseModel):
    """The HTML front page: the same fields, but ``q`` may be absent (the start page) and
    strangers' extra parameters (``?utm_source=...``) are ignored."""

    model_config = ConfigDict(extra="ignore")

    q: Annotated[str, StringConstraints(strip_whitespace=True, max_length=200)] = ""
    k: int = Field(10, ge=1, le=MAX_K)
    scorer: ScorerName = "bm25"
    page: int = Field(1, ge=1, le=MAX_PAGE)
    mode: Mode = "lexical"


class DocParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    offset: int | None = Field(None, ge=0, description="First character to return.")
    limit: int = Field(20_000, ge=1, le=200_000, description="Characters to return.")


class ViewParams(BaseModel):
    model_config = ConfigDict(extra="ignore")

    q: Annotated[str, StringConstraints(strip_whitespace=True, max_length=200)] = ""
    offset: int | None = Field(None, ge=0)
    limit: int = Field(8_000, ge=1, le=50_000)


# ---------------------------------------------------------------------------- output
class Hit(BaseModel):
    doc_id: int
    title: str
    score: float
    snippet: str = Field(description="Excerpt; matches are wrapped in **double asterisks**.")
    url: str = Field(description="Where to read the document (JSON).")


class SearchResponse(BaseModel):
    query: str
    scorer: ScorerName
    mode: Mode = "lexical"
    k: int
    page: int
    pages: int
    total: int = Field(
        description="Documents matching the query, over all pages (semantic / hybrid: the ranking depth, at most 100)."
    )
    took_ms: float
    hits: list[Hit]

    @classmethod
    def from_page(cls, p: SearchPage) -> SearchResponse:
        return cls(
            query=p.query, scorer=p.scorer, mode=p.mode, k=p.k, page=p.page, pages=p.pages, total=p.total,
            took_ms=round(p.took_ms, 2),
            hits=[
                Hit(
                    doc_id=h.doc_id, title=h.title, score=round(h.score, 4),
                    snippet=h.snippet.replace(MARK_ON, "**").replace(MARK_OFF, "**"),
                    url=f"/docs/{h.doc_id}",
                )
                for h in p.hits
            ],
        )  # fmt: skip


class DocumentResponse(BaseModel):
    doc_id: int
    title: str
    source: str
    tokens: int
    offset: int
    end: int
    total_chars: int
    next_offset: int | None = Field(description="Where the next slice starts; null at the end.")
    text: str

    @classmethod
    def from_page(cls, d: DocPage) -> DocumentResponse:
        return cls(
            doc_id=d.doc_id, title=d.title, source=d.source, tokens=d.tokens, offset=d.offset,
            end=d.end, total_chars=d.total_chars,
            next_offset=d.end if d.end < d.total_chars else None, text=d.text,
        )  # fmt: skip


class CacheInfo(BaseModel):
    hits: int
    misses: int
    size: int
    max_size: int | None


class StatsResponse(BaseModel):
    version: str
    documents: int
    tokens: int
    vocabulary: int
    postings: int
    avg_doc_length: float
    has_positions: bool
    storage: str
    index_bytes: int | None
    loaded_at: datetime
    uptime_s: float
    query_cache: CacheInfo
    semantic_model: str | None = Field(
        None, description="Embedding model, or null if semantic search is off."
    )

    @classmethod
    def from_stats(cls, s: ServiceStats, version: str) -> StatsResponse:
        return cls(
            version=version, documents=s.documents, tokens=s.tokens, vocabulary=s.vocabulary,
            postings=s.postings, avg_doc_length=round(s.avg_doc_length, 2),
            has_positions=s.has_positions, storage=s.storage, index_bytes=s.index_bytes,
            loaded_at=s.loaded_at, uptime_s=round(s.uptime_s, 1),
            semantic_model=s.semantic_model,
            query_cache=CacheInfo(hits=s.cache_hits, misses=s.cache_misses, size=s.cache_size,
                                  max_size=s.cache_max),
        )  # fmt: skip


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    documents: int


class ErrorResponse(BaseModel):
    """Shape of every non-validation error: ``{"detail": ..., "request_id": ...}``."""

    detail: str
    request_id: str
