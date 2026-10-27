import json
import random
from pathlib import Path

import pytest

from findex.boolean import ENGINES, evaluate, merge_and, merge_andnot, merge_or
from findex.build import build_index, extract_title
from findex.corpus import Document, iter_documents
from findex.model import STORAGES, Index
from findex.search import main as search_main
from findex.store import load, save

TEXTS = [
    "Title: Cats\ncat dog cat",  # 0
    "dog bird",  # 1
    "cat bird fish",  # 2
    "Привіт світе cat",  # 3
]


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    for i, t in enumerate(TEXTS):
        (tmp_path / f"{i}.txt").write_text(t, encoding="utf-8")
    return tmp_path


def docs():
    return [Document(i, Path(f"{i}.txt"), t) for i, t in enumerate(TEXTS)]


def build(storage="slots", positions=False) -> Index:
    return build_index(iter(docs()), storage=storage, positions=positions)


def snapshot(idx: Index):
    return {t: (list(idx.doc_ids(t))) for t in idx.postings}


def test_all_storages_agree():
    ref = snapshot(build("slots"))
    assert ref["cat"] == [0, 2, 3]
    for s in STORAGES:
        assert snapshot(build(s)) == ref


def test_tf_and_lengths():
    idx = build("slots")
    assert [(p.doc_id, p.tf) for p in idx.postings["cat"]] == [(0, 2), (2, 1), (3, 1)]
    assert idx.doc_lengths == {0: 5, 1: 2, 2: 3, 3: 3}  # "Title: Cats" adds two tokens
    assert idx.doc_meta[0].title == "Cats"
    assert idx.doc_meta[1].title == "dog bird"


def test_positions_consistent_across_storages():
    a = build("slots", positions=True)
    b = build("array", positions=True)
    cat_a = a.postings["cat"]
    assert cat_a[0].positions == (2, 4)  # tokens: title, cats, cat, dog, cat
    pl = b.postings["cat"]
    assert list(pl.positions) == [p for x in cat_a for p in x.positions]
    assert sum(pl.tfs) == len(pl.positions)


def test_ascending_doc_ids_enforced():
    bad = [Document(1, Path("a"), "x"), Document(0, Path("b"), "y")]
    with pytest.raises(ValueError):
        build_index(bad)


def test_extract_title_fallbacks():
    assert extract_title("\n\n  hello world \nrest", Path("x.txt")) == "hello world"
    assert extract_title("", Path("dir/x.txt")) == "x.txt"


def test_merge_primitives():
    assert merge_and([1, 3, 5, 7], [3, 4, 5, 8]) == [3, 5]
    assert merge_or([1, 3], [2, 3, 9]) == [1, 2, 3, 9]
    assert merge_andnot([1, 2, 3, 4], [2, 4, 9]) == [1, 3]
    assert merge_and([], [1]) == [] and merge_or([], []) == []


def test_engines_agree_on_random_input():
    rng = random.Random(0)
    for _ in range(200):
        a = sorted(rng.sample(range(60), rng.randint(0, 30)))
        b = sorted(rng.sample(range(60), rng.randint(0, 30)))
        for f_merge, f_set in zip(ENGINES["merge"], ENGINES["set"], strict=True):
            assert f_merge(a, b) == f_set(a, b)


@pytest.mark.parametrize("engine", ["merge", "set"])
@pytest.mark.parametrize("storage", STORAGES)
def test_queries(engine, storage):
    idx = build(storage)
    q = lambda s: evaluate(s, idx, engine)
    assert q("cat") == [0, 2, 3]
    assert q("cat dog") == [0]  # implicit AND
    assert q("cat AND bird") == [2]
    assert q("dog OR fish") == [0, 1, 2]
    assert q("cat NOT dog") == [2, 3]
    assert q("NOT cat") == [1]
    assert q("dog OR bird NOT cat") == [1]  # ((dog OR bird) AND NOT cat)
    assert q("CAT") == [0, 2, 3]  # same normalization as indexing
    assert q("привіт") == [3]
    assert q("unknownterm") == [] and q("") == [] and q("cat zzz") == []


@pytest.mark.parametrize("fmt", ["binary", "pickle"])
@pytest.mark.parametrize("storage", STORAGES)
@pytest.mark.parametrize("positions", [False, True])
def test_save_load_roundtrip(tmp_path, fmt, storage, positions):
    idx = build(storage, positions)
    p = tmp_path / "i.bin"
    assert save(idx, p, fmt) == p.stat().st_size
    back = load(p)
    assert snapshot(back) == snapshot(idx)
    assert back.doc_lengths == idx.doc_lengths and back.doc_meta == idx.doc_meta
    assert back.has_positions == positions
    if fmt == "binary":
        assert back.storage == "array"
        if positions:
            ref = build("array", True)
            assert list(back.postings["cat"].positions) == list(
                ref.postings["cat"].positions
            )


def test_binary_rejects_truncated(tmp_path):
    p = tmp_path / "i.bin"
    save(build("array"), p, "binary")
    p.write_bytes(p.read_bytes()[:-5])
    with pytest.raises(ValueError):
        load(p)


def test_end_to_end_cli(corpus, tmp_path, capsys):
    from findex.index import main as index_main

    out = tmp_path / "index.bin"
    index_main([str(corpus), "--out", str(out), "--no-tracemalloc"])
    capsys.readouterr()
    search_main([str(out), "cat NOT dog", "--no-tracemalloc"])
    text = capsys.readouterr().out
    # rglob order is arbitrary, so identify hits by content rather than by doc_id
    assert (
        "2 matching documents" in text and "cat bird fish" in text and "Привіт" in text
    )
    assert "Cats" not in text
    search_main([str(out), "cat", "--engine", "set", "--no-tracemalloc"])
    assert "3 matching documents" in capsys.readouterr().out


def test_json_header_is_plain_json(tmp_path):
    p = tmp_path / "i.bin"
    save(build("array"), p, "binary")
    raw = p.read_bytes()
    assert raw.startswith(b"FINDEXBN")
    hlen = int.from_bytes(raw[8:16], "little")
    assert json.loads(raw[16 : 16 + hlen])["version"] == 1


def test_iter_documents_still_lazy(corpus):
    assert iter_documents(corpus).__next__().doc_id == 0
