from pathlib import Path

import pytest

from findex.build import build_index
from findex.corpus import Document
from findex.engine import search
from findex.evaluate import evaluate, is_relevant, precision_at_k
from findex.model import DocMeta
from findex.scoring import BM25, TfIdf


def test_is_relevant_matches_name_stem_or_title():
    m = DocMeta(0, "data/g/2701.txt", "Moby Dick; Or, The Whale")
    assert (
        is_relevant(m, ["2701.txt"])
        and is_relevant(m, ["2701"])
        and is_relevant(m, ["whale"])
    )
    assert not is_relevant(m, ["27"]) and not is_relevant(m, [])


@pytest.fixture
def ix():
    texts = [
        "Title: Cats\ncat cat",
        "Title: Dogs\ncat",
        "Title: Birds\nbird",
        "Title: Cars\ncar",
    ]
    docs = [Document(i, Path(f"d{i}.txt"), t) for i, t in enumerate(texts)]
    return build_index(iter(docs), positions=False)


def test_precision_at_k(ix):
    hits = search(ix, "cat", k=5)  # docs 0 and 1
    assert precision_at_k(hits, ix, ["cats"], 5) == pytest.approx(1 / 5)
    assert precision_at_k(hits, ix, ["cats", "dogs"], 2) == 1.0
    assert precision_at_k(hits, ix, ["birds"], 5) == 0.0


def test_evaluate_rows_and_missing_positions(ix):
    queries = [
        {"query": "cat", "relevant": ["cats"]},
        {"query": '"cat cat"', "relevant": ["cats"]},
    ]
    rows = evaluate(ix, queries, {"tfidf": TfIdf(), "bm25": BM25()}, k=1)
    assert rows[0][1] == {"tfidf": 1.0, "bm25": 1.0}
    assert rows[1][1] == {"tfidf": None, "bm25": None}  # phrase, but no positions
