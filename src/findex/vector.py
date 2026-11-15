"""Ranking with NumPy: the same BM25 / TF-IDF as ``scoring.py``, as expressions over arrays.

Why it is faster (see docs/profiling.md): the pure-Python loop calls ``scorer.score`` once per posting
-- a function call, a ``Posting`` object and a ``math.log`` for each of ~15 000 postings per query. Here
a term's postings are two ``int32`` arrays and the score of ALL its documents is one expression:

    idf * tf * (k1 + 1) / (tf + k1 * norm[doc_ids])         # ufuncs: a C loop, no Python per element

``doc_ids`` / ``tfs`` are NOT copied: the index already keeps them as ``array('I')`` (contiguous 4-byte
integers), and ``np.frombuffer`` looks at the same memory as ``int32`` (ids and counts are far below 2**31).
Lengths of documents are one ``int32`` array indexed by doc id.

The result must be the **same ranking** as lab 3, tie-breaks included (``heapq.nlargest`` over
``SearchResult`` orders by score, then by doc id, both descending). Two details keep it exact:

* the arithmetic is done in the same order as the scalar code, so the floats are identical;
* ``(1 + ln tf)`` is read from a table built with ``math.log`` (NumPy's ``log`` may differ in the last bit).
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence

import numpy as np
from numpy.typing import NDArray

from findex.model import Index, PostingList, PostingView
from findex.scoring import BM25, Scorer, TfIdf

I32 = NDArray[np.int32]
F64 = NDArray[np.float64]

_EMPTY_I = np.zeros(0, dtype=np.int32)
_EMPTY_F = np.zeros(0, dtype=np.float64)


def top_k(ids: I32, scores: F64, k: int) -> tuple[I32, F64]:
    """The ``k`` best by (score, doc id), both descending -- what ``heapq.nlargest`` gives lab 3.

    ``np.argpartition`` finds the k-th best score in O(n) without sorting everything; only the k
    survivors are sorted. Documents tied with the k-th score are cut by doc id, as the heap would.
    """
    n = scores.size
    if n == 0 or k <= 0:
        return _EMPTY_I, _EMPTY_F
    if k >= n:
        sel = np.arange(n)
    else:
        part = np.argpartition(scores, n - k)[n - k :]
        threshold = scores[part].min()
        above = np.flatnonzero(scores > threshold)
        ties = np.flatnonzero(scores == threshold)
        need = k - above.size
        if ties.size > need:  # more equal scores than places: the larger doc ids win
            ties = ties[np.argsort(ids[ties], kind="stable")[ties.size - need :]]
        sel = np.concatenate((above, ties))
    order = np.lexsort((-ids[sel].astype(np.int64), -scores[sel]))  # last key is the primary one
    sel = sel[order]
    return ids[sel], scores[sel]


class VectorIndex:
    """NumPy view of an ``Index``: per-term ``int32`` arrays (cached views) + one length array."""

    #: below this many postings per query the loop is faster: NumPy has a fixed cost per call
    #: (three arrays the size of the corpus, ``fromiter``) of ~15-30 us that tiny lists cannot repay
    MIN_POSTINGS = 128

    def __init__(self, index: Index, min_postings: int | None = None) -> None:
        self.index = index
        self.min_postings = self.MIN_POSTINGS if min_postings is None else min_postings
        self.size = max(index.doc_meta, default=-1) + 1
        if self.size >= 2**31:
            raise ValueError("int32 doc ids need fewer than 2**31 documents")
        self.doc_len: I32 = np.zeros(self.size, dtype=np.int32)
        if index.doc_lengths:
            ids = np.fromiter(
                index.doc_lengths.keys(), dtype=np.int64, count=len(index.doc_lengths)
            )
            lengths = np.fromiter(index.doc_lengths.values(), dtype=np.int32, count=len(ids))
            self.doc_len[ids] = lengths
        self._terms: dict[str, tuple[I32, I32] | None] = {}
        self._norm: dict[tuple[float, float], F64] = {}
        self._log_weight: F64 = np.zeros(1, dtype=np.float64)

    # ------------------------------------------------------------------ the arrays
    def term(self, term: str) -> tuple[I32, I32] | None:
        """``(doc_ids, tfs)`` of a term as int32 arrays, or None for an unknown term."""
        if term in self._terms:
            return self._terms[term]
        plist = self.index.postings.get(term)
        pair = None if plist is None else _as_int32(plist)
        self._terms[term] = pair
        return pair

    def _length_norm(self, k1: float, b: float) -> F64:
        """``1 - b + b * dl / avgdl`` for every document (BM25's length normalisation), cached."""
        key = (k1, b)
        if key not in self._norm:
            avg = self.index.avg_doc_length
            dl = self.doc_len.astype(np.float64)
            self._norm[key] = (1.0 - b) + b * dl / avg if avg else np.full(self.size, 1.0 - b)
        return self._norm[key]

    def _tf_weight(self, tfs: I32) -> F64:
        """``1 + ln tf`` through a table built with ``math.log`` (bit-identical to the scalar scorer)."""
        top = int(tfs.max()) if tfs.size else 0
        if top >= self._log_weight.size:
            have = self._log_weight.size
            extra = np.array([1.0 + math.log(t) for t in range(max(have, 1), top + 1)])
            head = self._log_weight if have > 1 else np.zeros(1)
            self._log_weight = np.concatenate((head, extra))
        return self._log_weight[tfs]

    # ------------------------------------------------------------------ scoring
    @staticmethod
    def supports(scorer: Scorer) -> bool:
        """Only the two built-in scorers have a vector form; a custom ``Scorer`` stays on the loop."""
        return type(scorer) in (BM25, TfIdf)

    def worth_it(self, terms: Sequence[str]) -> bool:
        """Is the query long enough (postings of all its terms) to repay the fixed cost of NumPy?"""
        return sum(self.index.df(t) for t in terms) >= self.min_postings

    def rank(
        self, wanted: Collection[int], terms: Sequence[str], scorer: Scorer, k: int
    ) -> list[tuple[int, float]]:
        """Top-``k`` ``(doc_id, score)`` among ``wanted`` documents, summing the query terms' scores."""
        ids = np.fromiter(wanted, dtype=np.int64, count=len(wanted))
        in_wanted = np.zeros(self.size, dtype=np.bool_)
        in_wanted[ids] = True
        total = np.zeros(self.size, dtype=np.float64)
        touched = np.zeros(self.size, dtype=np.bool_)  # a document needs a posting for some term
        n = self.index.num_docs
        for term in terms:
            pair = self.term(term)
            if pair is None:
                continue
            docs, tfs = pair
            keep = in_wanted[docs]
            docs, tfs = docs[keep], tfs[keep]
            if not docs.size:
                continue
            if isinstance(scorer, BM25):
                df = self.index.df(term)
                idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
                tf = tfs.astype(np.float64)
                norm = self._length_norm(scorer.k1, scorer.b)[docs]
                contribution = idf * tf * (scorer.k1 + 1.0) / (tf + scorer.k1 * norm)
            else:
                assert isinstance(scorer, TfIdf)
                df = self.index.df(term)
                idf = math.log(n / df) if df else 0.0
                weight = self._tf_weight(tfs) if scorer.sublinear else tfs.astype(np.float64)
                contribution = weight * idf
            total[docs] += contribution  # doc ids are unique inside one term: no lost updates
            touched[docs] = True
        candidates = np.flatnonzero(touched).astype(np.int32)
        best_ids, best_scores = top_k(candidates, total[candidates], k)
        return list(zip(best_ids.tolist(), best_scores.tolist(), strict=True))


def _as_int32(plist: Sequence[PostingView]) -> tuple[I32, I32]:
    if isinstance(plist, PostingList):
        # array('I') -> int32 without copying; read-only use only (the index is immutable once built)
        return (
            np.frombuffer(plist.doc_ids, dtype=np.int32),
            np.frombuffer(plist.tfs, dtype=np.int32),
        )
    return (
        np.fromiter((p.doc_id for p in plist), dtype=np.int32, count=len(plist)),
        np.fromiter((p.tf for p in plist), dtype=np.int32, count=len(plist)),
    )
