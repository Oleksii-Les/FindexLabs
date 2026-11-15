"""The NumPy ranking must give the SAME results as the loop of lab 3 -- ids, order, tie-breaks, scores."""

from __future__ import annotations

import random
from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from findex.engine import SearchEngine, search
from findex.model import Index, PostingList, StorageName
from findex.query import PositionsRequired
from findex.scoring import BM25, SearchResult, TfIdf
from findex.vector import VectorIndex, top_k

WORDS = [
    "whale",
    "sea",
    "ship",
    "ocean",
    "storm",
    "sail",
    "captain",
    "wave",
    "harbor",
    "rope",
    "mast",
    "deck",
]


def corpus(n: int = 120, seed: int = 7) -> list[str]:
    """Random texts of varying length; every 5th text is a duplicate of the previous one (exact ties)."""
    rng = random.Random(seed)
    texts: list[str] = []
    for i in range(n):
        if i % 5 == 4:
            texts.append(texts[-1])  # same text -> same score for every query
            continue
        weights = [1 / (r + 1) for r in range(len(WORDS))]
        texts.append(" ".join(rng.choices(WORDS, weights, k=rng.randint(3, 60))))
    return texts


QUERIES = [
    "whale", "whale OR sea", "whale AND sea", "whale AND NOT sea", "NOT whale", "sea OR ship OR storm",
    "nothing", "whale OR nothing", "mast deck rope", "ocean AND (storm OR wave)",
]  # fmt: skip
SCORERS = [BM25(), BM25(k1=0.4, b=0.0), BM25(k1=2.0, b=1.0), TfIdf(), TfIdf(sublinear=False)]


def pairs(hits: list[SearchResult]) -> list[tuple[int, float]]:
    return [(h.doc_id, h.score) for h in hits]


@pytest.mark.parametrize("storage", ["array", "slots", "dataclass"])
@pytest.mark.parametrize("scorer", SCORERS, ids=repr)
@pytest.mark.parametrize("k", [1, 3, 10, 500])
def test_same_ranking_as_the_loop(
    mk: Callable[..., Index], storage: StorageName, scorer: BM25 | TfIdf, k: int
) -> None:
    index = mk(corpus(), storage, positions=False)
    vectors = VectorIndex(index, min_postings=0)  # force NumPy even on this small corpus
    for q in QUERIES:
        expected = search(index, q, scorer=scorer, k=k)
        got = search(index, q, scorer=scorer, k=k, vectors=vectors)
        assert [d for d, _ in pairs(got)] == [d for d, _ in pairs(expected)], (
            q
        )  # order, ties included
        assert [s for _, s in pairs(got)] == [s for _, s in pairs(expected)], (
            q
        )  # bit-identical floats


def test_phrase_query_with_positions(mk: Callable[..., Index]) -> None:
    index = mk(corpus(), "array", positions=True)
    vectors = VectorIndex(index, min_postings=0)
    q = '"white whale" OR sea'
    index2 = mk(
        ["the white whale swam", "a whale", "white sea whale", "white whale white whale"],
        "array",
        True,
    )
    v2 = VectorIndex(index2, min_postings=0)
    assert pairs(search(index2, q, vectors=v2)) == pairs(search(index2, q))
    with pytest.raises(PositionsRequired):  # matching is unchanged: no positions, no phrases
        search(mk(corpus(), "array", positions=False), '"white whale"', vectors=vectors)


def test_engine_default_is_vectorized_and_can_be_switched_off(mk: Callable[..., Index]) -> None:
    index = mk(corpus(), "array", positions=False)
    fast, slow = SearchEngine(index), SearchEngine(index, vectorized=False)
    assert fast.vectors is not None and slow.vectors is None
    for q in QUERIES:
        assert pairs(fast.search(q, 7, snippets=False)) == pairs(slow.search(q, 7, snippets=False))


def test_custom_scorer_falls_back_to_the_loop(mk: Callable[..., Index]) -> None:
    class Constant:
        def score(self, term: str, posting: object, index: Index) -> float:
            return 1.0

    index = mk(corpus(), "array", positions=False)
    hits = search(
        index, "whale OR sea", scorer=Constant(), k=5, vectors=VectorIndex(index, min_postings=0)
    )
    assert len(hits) == 5 and {h.score for h in hits} <= {1.0, 2.0}


def test_arrays_are_views_not_copies(mk: Callable[..., Index]) -> None:
    index = mk(corpus(), "array", positions=False)
    vectors = VectorIndex(index)
    pair = vectors.term("whale")
    assert pair is not None
    docs, tfs = pair
    assert docs.dtype == np.int32 and tfs.dtype == np.int32 and vectors.doc_len.dtype == np.int32
    assert (
        not docs.flags.owndata and not tfs.flags.owndata
    )  # memory belongs to the index's array('I')
    postings = index["whale"]
    assert isinstance(postings, PostingList) and docs.tolist() == list(postings.doc_ids)
    assert vectors.term("no-such-term") is None


def test_doc_lengths_array_matches_the_index(mk: Callable[..., Index]) -> None:
    index = mk(corpus(30), "array", positions=False)
    vectors = VectorIndex(index)
    assert vectors.doc_len.tolist() == [index.doc_lengths[d] for d in range(index.num_docs)]


def test_empty_index_and_empty_result(mk: Callable[..., Index]) -> None:
    index = mk([""], "array", positions=False)
    assert search(index, "whale", vectors=VectorIndex(index, min_postings=0)) == []


# --- top_k -------------------------------------------------------------------------------------
def reference(ids: list[int], scores: list[float], k: int) -> list[tuple[int, float]]:
    return sorted(zip(ids, scores, strict=True), key=lambda t: (t[1], t[0]), reverse=True)[:k]


@given(
    scores=st.lists(st.sampled_from([0.0, 0.5, 1.0, 1.5, 2.0, 3.25]), min_size=0, max_size=60),
    k=st.integers(0, 70),
)
def test_top_k_equals_a_full_sort_with_ties(scores: list[float], k: int) -> None:
    ids = list(range(0, 3 * len(scores), 3))  # sorted, unique, like the output of np.flatnonzero
    got_ids, got_scores = top_k(
        np.array(ids, dtype=np.int32), np.array(scores, dtype=np.float64), k
    )
    assert list(zip(got_ids.tolist(), got_scores.tolist(), strict=True)) == reference(
        ids, scores, k
    )


def test_top_k_boundary_ties_prefer_larger_doc_ids() -> None:
    ids = np.arange(6, dtype=np.int32)
    scores = np.array([1.0, 1.0, 1.0, 1.0, 5.0, 1.0])
    got, _ = top_k(ids, scores, 3)
    assert got.tolist() == [4, 5, 3]


def test_short_queries_stay_on_the_loop_and_long_ones_use_numpy(mk: Callable[..., Index]) -> None:
    index = mk(corpus(), "array", positions=False)
    default = VectorIndex(index)
    assert default.min_postings == VectorIndex.MIN_POSTINGS
    assert not default.worth_it(["nothing"])  # unknown term: 0 postings
    assert default.worth_it(
        ["whale", "sea", "ship", "ocean"]
    )  # a few hundred postings in this corpus
    assert VectorIndex(index, min_postings=0).worth_it([])
