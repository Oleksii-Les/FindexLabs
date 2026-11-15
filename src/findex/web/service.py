"""The search logic of the web app. Plain classes and dataclasses: no FastAPI, no Pydantic.

Pydantic models exist at the edges only (``schemas.py``: parsing the request, shaping the JSON);
here everything is what Labs 02-03 gave us: ``Index``, ``SearchEngine``, ``SearchResult``.
All methods are synchronous and CPU-bound (``def`` routes run them in FastAPI's thread pool).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from findex.engine import SearchEngine, search
from findex.model import Index
from findex.query import parse
from findex.scoring import ScorerName, make_scorer
from findex.semantic import (
    DEPTH,
    Mode,
    SemanticIndex,
    SemanticUnavailable,
    fragment_snippet,
    informative_terms,
    query_terms,
    search_mode,
)
from findex.snippets import first_match, highlight, make_snippet, read_text
from findex.store import IndexFormatError, detect_format, load

log = logging.getLogger(__name__)

# Highlight marks that cannot occur in text: the HTML layer escapes first, then turns them into <mark>.
MARK_ON, MARK_OFF = "\x02", "\x03"


class DocumentNotFound(LookupError):
    """No document with this id in the index."""


class DocumentUnavailable(LookupError):
    """The document is in the index but its text cannot be read (file missing, or inside a .jsonl)."""


@dataclass(frozen=True, slots=True)
class HitData:
    doc_id: int
    title: str
    score: float
    snippet: str  # contains MARK_ON / MARK_OFF around matches


@dataclass(frozen=True, slots=True)
class SearchPage:
    query: str
    scorer: ScorerName
    k: int  # results per page
    page: int
    total: int  # documents matching the query, over ALL pages
    pages: int
    took_ms: float
    hits: tuple[HitData, ...]
    mode: Mode = "lexical"


@dataclass(frozen=True, slots=True)
class DocPage:
    doc_id: int
    title: str
    source: str  # file name, never a server path
    tokens: int
    text: str  # the requested slice (with marks if a query was given)
    offset: int
    end: int
    total_chars: int
    first_match: int | None


@dataclass(frozen=True, slots=True)
class ServiceStats:
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
    cache_hits: int
    cache_misses: int
    cache_size: int
    cache_max: int | None
    semantic_model: str | None = None


class SearchService:
    def __init__(
        self,
        index: Index,
        *,
        corpus_root: Path | None = None,
        cache_size: int = 256,
        window: int = 110,
        index_bytes: int | None = None,
        semantic: SemanticIndex | None = None,
        vectorized: bool = True,
    ) -> None:
        self.index = index
        self.semantic = semantic
        self.engine = SearchEngine(index, cache_size=cache_size, vectorized=vectorized)
        self.corpus_root = corpus_root
        self.window = window
        self.index_bytes = index_bytes
        self.loaded_at = datetime.now(UTC)
        self._t0 = time.perf_counter()
        self._tokens = sum(index.doc_lengths.values())

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        corpus_root: Path | None = None,
        cache_size: int = 256,
        window: int = 110,
        embeddings: Path | None = None,
        vectorized: bool = True,
    ) -> SearchService:
        """Load an index file. Refuses the pickle format: unpickling is arbitrary code
        execution, and the path of a deployed service comes from an environment variable."""
        if not path.is_file():
            raise FileNotFoundError(f"index file not found: {path}")
        if detect_format(path) != "binary":
            raise IndexFormatError(
                f"{path} is not a findex binary index (pickle files are refused)"
            )
        t0 = time.perf_counter()
        index = load(path)
        log.info("loaded %r from %s in %.2f s", index, path, time.perf_counter() - t0)
        semantic = None
        if embeddings is not None:
            try:  # semantic modes are optional: a missing or stale file must not take the service down
                semantic = SemanticIndex.load(embeddings, index)
                log.info("semantic search: %s, %d documents", semantic.model, len(semantic))
            except (SemanticUnavailable, OSError, ValueError) as exc:
                log.warning("semantic search disabled: %s", exc)
        return cls(
            index,
            corpus_root=corpus_root,
            cache_size=cache_size,
            window=window,
            index_bytes=path.stat().st_size,
            semantic=semantic,
            vectorized=vectorized,
        )

    @property
    def semantic_ready(self) -> bool:
        return self.semantic is not None

    def close(self) -> None:
        self.index.close()

    # ------------------------------------------------------------------ documents
    def _text(self, doc_id: int) -> str | None:
        meta = self.index.doc_meta[doc_id]
        text = read_text(meta.path)
        if text is None and self.corpus_root is not None:
            # The index remembers the path from the machine that built it; on another machine
            # (or in a container) the same file name lives under CORPUS_ROOT.
            text = read_text(self.corpus_root / Path(meta.path).name)
        return text

    def document(
        self,
        doc_id: int,
        *,
        offset: int | None = None,
        limit: int = 20_000,
        query: str | None = None,
    ) -> DocPage:
        """A slice of one document's text. With ``query``, matches are marked and an unspecified
        ``offset`` starts a little before the first match."""
        meta = self.index.doc_meta.get(doc_id)
        if meta is None:
            raise DocumentNotFound(doc_id)
        text = self._text(doc_id)
        if text is None:
            raise DocumentUnavailable(doc_id)
        terms = list(dict.fromkeys(parse(query).terms())) if query else []
        first = first_match(text, terms) if terms else None
        if offset is None:
            offset = 0
            if first is not None:
                back = max(0, first - 300)
                offset = text.rfind("\n", back, first) + 1 or back  # start on a line if one is near
        offset = min(max(offset, 0), len(text))
        end = min(len(text), offset + limit)
        chunk = text[offset:end]
        if terms:
            chunk = highlight(chunk, terms, (MARK_ON, MARK_OFF))
        return DocPage(
            doc_id=doc_id,
            title=meta.title,
            source=Path(meta.path).name,
            tokens=self.index.doc_lengths[doc_id],
            text=chunk,
            offset=offset,
            end=end,
            total_chars=len(text),
            first_match=first,
        )

    # ------------------------------------------------------------------ search
    def search(
        self,
        query: str,
        *,
        k: int = 10,
        scorer: ScorerName = "bm25",
        page: int = 1,
        mode: Mode = "lexical",
    ) -> SearchPage:
        """One page of ranked results. Raises ``QuerySyntaxError`` / ``PositionsRequired``.

        The matching documents come from the engine's per-query cache (so page 2 of a query does
        not evaluate it again); only the top ``page * k`` are ranked; snippets are made for the
        ``k`` hits that are actually shown.
        """
        t0 = time.perf_counter()
        if mode != "lexical":
            return self._search_semantic(query, k=k, scorer=scorer, page=page, mode=mode, t0=t0)
        terms = list(dict.fromkeys(parse(query).terms()))
        ids = self.engine.match(query)
        total = len(ids)
        start = (page - 1) * k
        hits: list[HitData] = []
        if start < total:
            ranked = search(
                self.index, query, scorer=make_scorer(scorer), k=page * k,
                matcher=lambda _q: ids, snippets=False, vectors=self.engine.vectors,
            )  # fmt: skip
            for r in ranked[start : start + k]:
                text = self._text(r.doc_id)
                snippet = (
                    make_snippet(text, terms, window=self.window, marks=(MARK_ON, MARK_OFF))
                    if text
                    else ""
                )
                hits.append(HitData(r.doc_id, r.title, r.score, snippet))
        return SearchPage(
            query=query,
            scorer=scorer,
            k=k,
            page=page,
            total=total,
            pages=math.ceil(total / k),
            took_ms=(time.perf_counter() - t0) * 1000,
            hits=tuple(hits),
        )

    def _search_semantic(
        self, query: str, *, k: int, scorer: ScorerName, page: int, mode: Mode, t0: float
    ) -> SearchPage:
        """``semantic`` / ``hybrid``: the best ``DEPTH`` documents, paged like the lexical results.
        ``total`` is how many documents the ranking holds (at most ``DEPTH``), not "all that match":
        a similarity search has no notion of a non-matching document."""
        ranked = search_mode(
            self.engine, self.semantic, query, mode, DEPTH, scorer=make_scorer(scorer)
        )
        terms = informative_terms(self.index, query_terms(query))
        start = (page - 1) * k
        hits: list[HitData] = []
        for h in ranked[start : start + k]:
            text = self._text(h.doc_id)
            snippet = (
                fragment_snippet(
                    text, h.start, h.end, terms, window=self.window, marks=(MARK_ON, MARK_OFF)
                )
                if text
                else ""
            )
            hits.append(HitData(h.doc_id, self.index.doc_meta[h.doc_id].title, h.score, snippet))
        return SearchPage(
            query=query, scorer=scorer, k=k, page=page, total=len(ranked),
            pages=math.ceil(len(ranked) / k), took_ms=(time.perf_counter() - t0) * 1000,
            hits=tuple(hits), mode=mode,
        )  # fmt: skip

    # ------------------------------------------------------------------ stats
    def stats(self) -> ServiceStats:
        cache = self.engine.cache_info()
        return ServiceStats(
            documents=self.index.num_docs,
            tokens=self._tokens,
            vocabulary=len(self.index),
            postings=self.index.n_postings(),
            avg_doc_length=self.index.avg_doc_length,
            has_positions=self.index.has_positions,
            storage=self.index.storage,
            index_bytes=self.index_bytes,
            loaded_at=self.loaded_at,
            uptime_s=time.perf_counter() - self._t0,
            cache_hits=cache.hits,
            cache_misses=cache.misses,
            cache_size=cache.currsize,
            cache_max=cache.maxsize,
            semantic_model=self.semantic.model if self.semantic else None,
        )
