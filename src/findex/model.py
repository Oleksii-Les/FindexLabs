"""Data model of the inverted index.

Kept in its own module on purpose: if these classes lived in ``index.py`` and you
ran ``python -m findex.index``, they would be defined in ``__main__`` and pickle
would store them as ``__main__.Posting`` -- unloadable from any other program.
"""

from __future__ import annotations

from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

STORAGES = ("dataclass", "slots", "array")


@dataclass(frozen=True, slots=True, order=True)
class Posting:
    """One (document, term-frequency) pair. Frozen => hashable; slots => no __dict__."""

    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(order=True)
class PlainPosting:
    """Baseline for the memory study: an ordinary dataclass (has __dict__)."""

    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    path: str
    title: str


class PostingList:
    """Postings of one term as parallel raw arrays -- no per-posting Python objects.

    ``positions`` (optional) is one flat array: the positions of posting *i* are the
    next ``tfs[i]`` values, because a term occurs exactly ``tf`` times.
    """

    __slots__ = ("doc_ids", "positions", "tfs")

    def __init__(self, with_positions: bool = False) -> None:
        self.doc_ids = array("I")
        self.tfs = array("I")
        self.positions: array | None = array("I") if with_positions else None

    def __len__(self) -> int:
        return len(self.doc_ids)


@dataclass(slots=True)
class Index:
    storage: str  # "dataclass" | "slots" | "array"
    has_positions: bool
    postings: dict[str, Any]  # term -> list[Posting|PlainPosting] | PostingList
    doc_lengths: dict[int, int]  # tokens per document (for BM25 in Lab 3)
    doc_meta: dict[int, DocMeta]  # for displaying results

    @property
    def n_docs(self) -> int:
        return len(self.doc_meta)

    def df(self, term: str) -> int:
        p = self.postings.get(term)
        return 0 if p is None else len(p)

    def n_postings(self) -> int:
        return sum(len(p) for p in self.postings.values())

    def doc_ids(self, term: str) -> Sequence[int]:
        """Sorted doc ids containing ``term`` (empty if unknown)."""
        p = self.postings.get(term)
        if p is None:
            return ()
        if isinstance(p, PostingList):
            return p.doc_ids
        return [x.doc_id for x in p]
