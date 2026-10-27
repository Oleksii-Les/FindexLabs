import math
from dataclasses import FrozenInstanceError

import pytest

from findex.engine import SearchEngine, search
from findex.model import Posting
from findex.scoring import BM25, Scorer, SearchResult, TfIdf


def test_search_result_ordering():
    lo, hi = SearchResult(1.0, 7, "x"), SearchResult(2.0, 3, "y")
    assert lo < hi and sorted([hi, lo]) == [lo, hi] and max(hi, lo) is hi
    assert SearchResult(1.0, 7, "title A") == SearchResult(
        1.0, 7, "title B"
    )  # title not compared
    with pytest.raises(FrozenInstanceError):
        lo.score = 5  # type: ignore[misc]
    assert "[3]" in str(hi) and "y" in str(hi)


def test_scorers_share_a_protocol_and_are_swappable(mk):
    assert isinstance(BM25(), Scorer) and isinstance(TfIdf(), Scorer)

    class ByTermFrequency:  # not inheriting from anything: duck typing
        def score(self, term, posting, index):
            return float(posting.tf)

    ix = mk(["cat cat cat", "cat dog", "dog"])
    assert isinstance(ByTermFrequency(), Scorer)
    for scorer in (BM25(), TfIdf(), TfIdf(sublinear=False), ByTermFrequency()):
        assert search(ix, "cat", scorer=scorer)[0].doc_id == 0


def test_tfidf_formula(mk):
    ix = mk(["x y", "x", "z", "z"])  # N=4, df(x)=2
    p = Posting(0, 3)
    assert TfIdf().score("x", p, ix) == pytest.approx(
        (1 + math.log(3)) * math.log(4 / 2)
    )
    assert TfIdf(sublinear=False).score("x", p, ix) == pytest.approx(3 * math.log(2))
    assert TfIdf().score("nope", p, ix) == 0.0


def test_bm25_formula_and_parameters(mk):
    ix = mk(["x y", "x", "z", "z"])  # avgdl = 1.25, df(x)=2, N=4
    p = Posting(0, 2)
    idf = math.log(1 + (4 - 2 + 0.5) / (2 + 0.5))
    norm = 1 - 0.75 + 0.75 * 2 / 1.25
    assert BM25().score("x", p, ix) == pytest.approx(idf * 2 * 2.5 / (2 + 1.5 * norm))
    # b = 0: document length is ignored
    s0 = BM25(b=0)
    assert s0.score("x", Posting(0, 1), ix) == pytest.approx(
        s0.score("x", Posting(1, 1), ix)
    )
    # b = 1: the longer document is penalized
    assert BM25(b=1).score("x", Posting(0, 1), ix) < BM25(b=1).score(
        "x", Posting(1, 1), ix
    )
    assert BM25() == BM25(1.5, 0.75) and "k1=1.5" in repr(BM25())
    for bad in ({"k1": -1}, {"b": 1.5}, {"b": -0.1}):
        with pytest.raises(ValueError):
            BM25(**bad)


def corpus_for_checks():
    docs = ["zebra pad pad pad"]  # 0: rare term only
    docs += ["common pad pad pad"] * 5  # 1-5: common term only
    docs += ["common common pad pad"] * 4  # 6-9
    docs += ["filler pad pad pad"] * 10  # 10-19
    return docs


def test_sanity_rare_term_outranks_common(mk):
    ix = mk(corpus_for_checks(), positions=False)
    for scorer in (BM25(), TfIdf()):
        ranked = [r.doc_id for r in search(ix, "zebra OR common", scorer=scorer, k=20)]
        assert ranked.index(0) < ranked.index(1)  # tf=1 of rare beats tf=1 of common


def test_sanity_saturation(mk):
    ix = mk(["term " + "pad " * 20] * 3 + ["other"], positions=False)
    s = BM25()

    def at(tf):
        return s.score("term", Posting(0, tf), ix)

    first_gain, last_gain = at(2) - at(1), at(20) - at(19)
    assert last_gain < 0.05 * first_gain
    linear = TfIdf(sublinear=False)
    assert linear.score("term", Posting(0, 20), ix) == pytest.approx(
        20 * linear.score("term", Posting(0, 1), ix)
    )


def test_sanity_short_document_beats_long_one(mk):
    ix = mk(["needle"] + ["needle " + "pad " * 400] + ["pad"] * 8, positions=False)
    assert ix.doc_length(0) == 1 and ix.doc_length(1) == 401
    hits = search(ix, "needle", scorer=BM25())
    assert [h.doc_id for h in hits] == [0, 1] and hits[0].score > hits[1].score
    tf = {h.doc_id: h.score for h in search(ix, "needle", scorer=TfIdf())}
    assert tf[0] == pytest.approx(tf[1])  # TF-IDF has no length normalization


def test_search_returns_topk_best_first(mk):
    ix = mk(["a " * n + "pad" for n in range(1, 8)] + ["b"] * 3, positions=False)
    top3 = search(ix, "a", k=3)
    assert [r.doc_id for r in top3] == [6, 5, 4]
    assert (
        top3 == sorted(search(ix, "a", k=100), reverse=True)[:3]
    )  # nlargest == sorted()[:k]
    assert search(ix, "zebra") == [] and search(ix, "a", k=0) == []


def test_search_matches_only_boolean_hits_and_handles_not(mk):
    ix = mk(["cat dog", "cat", "dog", "bird"], positions=False)
    assert {r.doc_id for r in search(ix, "cat NOT dog")} == {1}
    assert {r.doc_id for r in search(ix, "cat OR dog")} == {0, 1, 2}
    only_not = search(ix, "NOT cat")
    assert {r.doc_id for r in only_not} == {2, 3} and all(
        r.score == 0 for r in only_not
    )


def test_engine_cache_hit_is_logged(mk, caplog):
    import logging

    ix = mk(["cat dog", "cat"], positions=False)
    eng = SearchEngine(ix, cache_size=8)
    with caplog.at_level(logging.INFO):
        first = eng.search("cat", snippets=False)
        second = eng.search("cat", snippets=False)
        eng.search("dog", snippets=False)
    text = [r.getMessage() for r in caplog.records]
    assert any("cat" in m and "cache miss" in m for m in text)
    assert any("cat" in m and "cache HIT" in m for m in text)
    assert first == second
    info = eng.cache_info()
    assert (info.hits, info.misses) == (1, 2)
    assert any("search took" in m for m in text)  # @timed evidence
