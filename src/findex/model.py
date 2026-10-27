"""Data model of the inverted index.

Kept in its own module on purpose: if these classes lived in ``index.py`` and you
ran ``python -m findex.index``, they would be defined in ``__main__`` and pickle
would store them as ``__main__.Posting`` -- unloadable from any other program.
"""

from __future__ import annotations

from array import array
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, overload

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


class PostingList(Sequence[Posting]):
    """Postings of one term as parallel raw arrays -- no per-posting Python objects.

    It still *behaves* like ``list[Posting]`` (a ``Sequence``): iterating or indexing
    builds ``Posting`` views on demand. ``positions`` (optional) is one flat array:
    the positions of posting *i* are the next ``tfs[i]`` values, because a term
    occurs exactly ``tf`` times.
    """

    __slots__ = ("doc_ids", "positions", "tfs")

    def __init__(self, with_positions: bool = False) -> None:
        self.doc_ids = array("I")
        self.tfs = array("I")
        self.positions: array | None = array("I") if with_positions else None

    def __len__(self) -> int:
        return len(self.doc_ids)

    def __iter__(self) -> Iterator[Posting]:
        pos = self.positions
        if pos is None:
            for d, t in zip(self.doc_ids, self.tfs, strict=True):
                yield Posting(d, t)
            return
        off = 0
        for d, t in zip(self.doc_ids, self.tfs, strict=True):
            yield Posting(d, t, tuple(pos[off : off + t]))
            off += t

    @overload
    def __getitem__(self, i: int) -> Posting: ...
    @overload
    def __getitem__(self, i: slice) -> list[Posting]: ...
    def __getitem__(self, i):
        n = len(self)
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(n))]
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError("posting index out of range")
        tf = self.tfs[i]
        if self.positions is None:
            return Posting(self.doc_ids[i], tf)
        off = sum(self.tfs[:i])  # O(i): random access with positions is the slow path
        return Posting(self.doc_ids[i], tf, tuple(self.positions[off : off + tf]))

    def positions_by_doc(self, wanted: Collection[int]) -> dict[int, tuple[int, ...]]:
        """doc_id -> positions, only for ``wanted`` docs (one sequential pass)."""
        if self.positions is None:
            raise ValueError("this index was built without positions")
        out: dict[int, tuple[int, ...]] = {}
        off = 0
        for d, t in zip(self.doc_ids, self.tfs, strict=True):
            if d in wanted:
                out[d] = tuple(self.positions[off : off + t])
            off += t
        return out

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PostingList):
            return NotImplemented
        return (self.doc_ids, self.tfs, self.positions) == (
            other.doc_ids,
            other.tfs,
            other.positions,
        )

    __hash__ = None  # type: ignore[assignment]  # mutable container


class Index(Mapping[str, Sequence[Posting]]):
    """The inverted index as a read-only mapping ``term -> postings``.

    Python protocols implemented here (each is just a dunder method):
    ``len(ix)`` terms, ``term in ix``, ``ix[term]`` (``KeyError`` if absent),
    ``for term in ix``, ``repr(ix)``. Inheriting ``Mapping`` adds ``keys()``,
    ``items()``, ``values()``, ``get()`` and ``==`` for free.
    """

    def __init__(
        self,
        storage: str,
        has_positions: bool,
        postings: dict[str, Any],
        doc_lengths: dict[int, int],
        doc_meta: dict[int, DocMeta],
    ) -> None:
        self.storage = storage  # "dataclass" | "slots" | "array"
        self.has_positions = has_positions
        self.postings = postings  # term -> list[Posting|PlainPosting] | PostingList
        self.doc_lengths = doc_lengths  # tokens per document
        self.doc_meta = doc_meta  # for displaying results
        self._closed = False

    # --- resource lifecycle -------------------------------------------------------
    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Release the data (idempotent). Used by ``open_index`` on exit, even on errors."""
        self._closed = True
        self.postings = {}
        self.doc_lengths = {}
        self.doc_meta = {}
        self.__dict__.pop("avg_doc_length", None)  # drop the cached_property value

    def _ensure_open(self) -> None:
        if self._closed:
            raise ValueError("operation on a closed Index")

    # --- Mapping protocol ---------------------------------------------------------
    def __len__(self) -> int:
        self._ensure_open()
        return len(self.postings)

    def __contains__(
        self, term: object
    ) -> bool:  # O(1); Mapping's default would try __getitem__
        self._ensure_open()
        return term in self.postings

    def __getitem__(self, term: str) -> Sequence[Posting]:
        self._ensure_open()
        return self.postings[term]  # KeyError for unknown terms, like a dict

    def __iter__(self) -> Iterator[str]:
        self._ensure_open()
        return iter(self.postings)

    def __repr__(self) -> str:
        if self._closed:
            return "Index(closed)"
        return (
            f"Index(terms={len(self.postings):_}, docs={self.num_docs:_}, "
            f"storage={self.storage!r}, positions={self.has_positions})"
        )

    # --- derived attributes (properties, no get_* methods) -------------------------
    @property
    def num_docs(self) -> int:
        return len(self.doc_meta)

    @cached_property
    def avg_doc_length(self) -> float:
        """Computed once on first access, then stored on the instance (BM25 needs it per doc)."""
        return sum(self.doc_lengths.values()) / self.num_docs if self.num_docs else 0.0

    def doc_length(self, doc_id: int) -> int:
        return self.doc_lengths[doc_id]

    def df(self, term: str) -> int:
        """Document frequency: in how many documents does ``term`` occur (0 if unknown)."""
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
