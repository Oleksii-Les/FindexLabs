import dataclasses
import pickle

import pytest

from findex.model import DocMeta, PlainPosting, Posting


def test_posting_is_hashable_and_in_sets():
    a, b = Posting(1, 2), Posting(1, 2)
    assert a == b and hash(a) == hash(b)  # the hashing contract
    assert len({a, b, Posting(2, 1)}) == 2
    assert hash(DocMeta(1, "p", "t")) == hash(DocMeta(1, "p", "t"))


def test_frozen_and_slots():
    p = Posting(1, 1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.tf = 5  # type: ignore[misc]
    assert not hasattr(p, "__dict__")
    assert hasattr(PlainPosting(1, 1), "__dict__")


def test_ordering_sorts_by_doc_id():
    ps = [Posting(3, 1), Posting(1, 9), Posting(2, 4)]
    assert [p.doc_id for p in sorted(ps)] == [1, 2, 3]


def test_pickle_roundtrip_of_frozen_slots():
    p = Posting(7, 3, (1, 5, 9))
    assert pickle.loads(pickle.dumps(p)) == p
