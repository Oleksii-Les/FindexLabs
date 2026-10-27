from collections.abc import Mapping, Sequence

import pytest

from findex.model import STORAGES, Index, Posting, PostingList
from findex.store import open_index, save

TEXTS = ["cat dog cat", "dog bird", "cat bird fish", "fish"]


@pytest.mark.parametrize("storage", STORAGES)
def test_mapping_protocols(mk, storage):
    ix = mk(TEXTS, storage)
    assert isinstance(ix, Mapping)
    assert len(ix) == 4
    assert "cat" in ix and "zebra" not in ix
    assert sorted(ix) == ["bird", "cat", "dog", "fish"]  # __iter__
    assert [p.doc_id for p in ix["cat"]] == [0, 2]
    with pytest.raises(KeyError):
        ix["zebra"]
    assert ix.get("zebra") is None  # free from Mapping
    assert set(ix.keys()) == set(ix)
    assert {t: ix.df(t) for t in ix} == {
        "cat": 2,
        "dog": 2,
        "bird": 2,
        "fish": 2,
    }


def test_repr_and_properties(mk):
    ix = mk(TEXTS, "array", positions=False)
    assert repr(ix) == "Index(terms=4, docs=4, storage='array', positions=False)"
    assert ix.num_docs == 4
    assert ix.doc_length(0) == 3
    assert ix.avg_doc_length == pytest.approx((3 + 2 + 3 + 1) / 4)


def test_avg_doc_length_is_cached(mk):
    ix = mk(TEXTS)
    assert "avg_doc_length" not in vars(ix)
    first = ix.avg_doc_length
    ix.doc_lengths[0] = 10_000  # would change a recomputation...
    assert ix.avg_doc_length == first  # ...but cached_property computed it once
    assert "avg_doc_length" in vars(ix)


def test_postinglist_is_a_sequence_of_postings(mk):
    pl = mk(TEXTS, "array")["cat"]
    sl = mk(TEXTS, "slots")["cat"]
    assert isinstance(pl, PostingList) and isinstance(pl, Sequence)
    assert list(pl) == sl  # same Posting values, array storage builds them on demand
    assert pl[1] == sl[1] == Posting(2, 1, (0,))
    assert pl[-1] == sl[-1]
    assert pl[0:2] == sl[0:2]
    assert pl == mk(TEXTS, "array")["cat"]
    with pytest.raises(IndexError):
        pl[5]


def test_positions_by_doc(mk):
    pl = mk(["a b a", "b", "a"], "array")["a"]
    assert pl.positions_by_doc({0, 2}) == {0: (0, 2), 2: (0,)}
    with pytest.raises(ValueError):
        mk(["a"], "array", positions=False)["a"].positions_by_doc({0})


def test_open_index_context_manager(mk, tmp_path):
    p = tmp_path / "i.bin"
    save(mk(TEXTS), p)
    with open_index(p) as ix:
        assert isinstance(ix, Index) and not ix.closed
        assert "cat" in ix
    assert ix.closed
    assert repr(ix) == "Index(closed)"
    with pytest.raises(ValueError):
        ix["cat"]


def test_open_index_releases_on_exception(mk, tmp_path):
    p = tmp_path / "i.bin"
    save(mk(TEXTS), p)
    holder = []
    with pytest.raises(RuntimeError, match="boom"), open_index(p) as ix:
        holder.append(ix)
        raise RuntimeError("boom")
    assert holder[0].closed  # cleanup ran although the body raised


def test_close_is_idempotent(mk):
    ix = mk(TEXTS)
    _ = ix.avg_doc_length
    ix.close()
    ix.close()
    assert ix.closed and "avg_doc_length" not in vars(ix)
