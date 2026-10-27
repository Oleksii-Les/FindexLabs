"""Ranking: a ``Scorer`` protocol with two interchangeable implementations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from findex.model import Index, Posting


@runtime_checkable
class Scorer(Protocol):
    """Anything with this method is a Scorer -- no inheritance needed (duck typing)."""

    def score(self, term: str, posting: Posting, index: Index) -> float:
        """Contribution of ``term`` to the document of ``posting``."""
        ...


@dataclass(frozen=True)
class TfIdf:
    """``(1 + ln tf) * ln(N / df)``  (``sublinear=False`` -> raw ``tf * idf``)."""

    sublinear: bool = True

    def idf(self, term: str, index: Index) -> float:
        df = index.df(term)
        return math.log(index.num_docs / df) if df else 0.0

    def score(self, term: str, posting: Posting, index: Index) -> float:
        tf = posting.tf
        weight = (1.0 + math.log(tf)) if self.sublinear else float(tf)
        return weight * self.idf(term, index) if tf > 0 else 0.0

    __call__ = score


@dataclass(frozen=True)
class BM25:
    """Okapi BM25 (Lucene's non-negative idf).

    ``idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))``

    * ``k1`` -- term-frequency saturation: contribution tends to ``idf * (k1 + 1)``
      as tf grows, so the 20th repetition adds almost nothing.
    * ``b``  -- length normalization: 0 = ignore document length, 1 = fully normalize.
    """

    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        if self.k1 < 0 or not 0.0 <= self.b <= 1.0:
            raise ValueError("need k1 >= 0 and 0 <= b <= 1")

    def idf(self, term: str, index: Index) -> float:
        df, n = index.df(term), index.num_docs
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def score(self, term: str, posting: Posting, index: Index) -> float:
        tf = posting.tf
        if tf <= 0:
            return 0.0
        avg = index.avg_doc_length  # cached_property: computed once per index
        dl = index.doc_length(posting.doc_id)
        norm = 1.0 - self.b + (self.b * dl / avg if avg else 0.0)
        return self.idf(term, index) * tf * (self.k1 + 1.0) / (tf + self.k1 * norm)

    __call__ = score


@dataclass(frozen=True, order=True)
class SearchResult:
    """One ranked hit. ``order=True`` compares fields in definition order, so
    ``score`` must come first: ``sorted(results)`` is then ascending by score
    (ties: by doc_id). ``title`` and ``snippet`` are display-only (``compare=False``).
    """

    score: float
    doc_id: int
    title: str = field(default="", compare=False)
    snippet: str = field(default="", compare=False, repr=False)

    def __str__(self) -> str:
        return f"{self.score:8.3f}  [{self.doc_id}] {self.title}"
