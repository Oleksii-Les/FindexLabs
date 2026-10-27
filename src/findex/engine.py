"""Ranked search: parse -> evaluate (matching docs) -> score -> top-k -> snippets."""

from __future__ import annotations

import functools
import heapq
import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import replace

from findex.decorators import timed
from findex.model import Index
from findex.query import parse
from findex.scoring import BM25, Scorer, SearchResult
from findex.snippets import Marks, make_snippet, read_text

log = logging.getLogger(__name__)


@timed
def search(
    index: Index,
    query: str,
    *,
    scorer: Scorer | None = None,
    k: int = 10,
    engine: str = "merge",
    matcher: Callable[[str], Sequence[int]] | None = None,
    snippets: bool = False,
    window: int = 80,
    marks: Marks = ("**", "**"),
) -> list[SearchResult]:
    """Top-``k`` documents for ``query``, best first.

    ``matcher`` (query -> doc ids) lets a caller supply a cached matcher; by default
    the query tree is evaluated here. Scores are accumulated per document in a dict
    and the top-k is taken with ``heapq.nlargest`` -- O(n log k), not a full sort.
    """
    scorer = scorer or BM25()
    tree = parse(query)
    ids = matcher(query) if matcher is not None else tree.evaluate(index, engine)
    if not ids:
        return []
    wanted = set(ids)
    terms = list(dict.fromkeys(tree.terms()))  # unique, in query order

    scores: dict[int, float] = defaultdict(float)
    for term in terms:
        postings = index.get(term)
        if postings is None:
            continue
        for posting in postings:
            if posting.doc_id in wanted:
                scores[posting.doc_id] += scorer.score(term, posting, index)
    if not terms:  # e.g. only "NOT x": nothing to rank by, every match ties at 0
        scores = dict.fromkeys(ids, 0.0)

    meta = index.doc_meta
    hits = (
        SearchResult(score=s, doc_id=d, title=meta[d].title) for d, s in scores.items()
    )
    top = heapq.nlargest(k, hits)  # uses SearchResult's order=True

    if snippets:
        out = []
        for r in top:
            text = read_text(meta[r.doc_id].path)
            snip = make_snippet(text, terms, window=window, marks=marks) if text else ""
            out.append(
                replace(r, snippet=snip)
            )  # frozen dataclass: build a modified copy
        top = out
    return top


class SearchEngine:
    """An open index + a default scorer + a per-instance cache of ``query -> doc ids``."""

    def __init__(
        self,
        index: Index,
        scorer: Scorer | None = None,
        *,
        engine: str = "merge",
        cache_size: int = 256,
    ) -> None:
        self.index = index
        self.scorer = scorer or BM25()
        self.engine = engine
        # lru_cache needs hashable args (a str is) and returns the same tuple on a hit.
        # It is created per instance, not with @lru_cache on the method, so the cache
        # does not keep every engine (and its index) alive forever through ``self``.
        self.match = functools.lru_cache(maxsize=cache_size)(self._match)

    def _match(self, query: str) -> tuple[int, ...]:
        return tuple(parse(query).evaluate(self.index, self.engine))

    def search(
        self,
        query: str,
        k: int = 10,
        *,
        scorer: Scorer | None = None,
        snippets: bool = True,
    ) -> list[SearchResult]:
        hits_before = self.match.cache_info().hits
        ids = self.match(query)
        hit = self.match.cache_info().hits > hits_before
        log.info(
            "match %r -> %d docs [%s]",
            query,
            len(ids),
            "cache HIT" if hit else "cache miss",
        )
        return search(
            self.index,
            query,
            scorer=scorer or self.scorer,
            k=k,
            engine=self.engine,
            matcher=lambda _q: ids,
            snippets=snippets,
        )

    def cache_info(self):
        return self.match.cache_info()
