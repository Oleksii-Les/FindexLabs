import json

from findex.corpus import iter_documents
from findex.stats import compute_stats


def test_txt_tree_and_bad_encoding(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("alpha beta", encoding="utf-8")
    (tmp_path / "sub" / "b.txt").write_bytes(b"gamma \xff\xfe delta")
    docs = list(iter_documents(tmp_path))
    assert len(docs) == 2
    assert {d.doc_id for d in docs} == {0, 1}


def test_jsonl(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(
        json.dumps({"text": "one two"})
        + "\nnot json\n"
        + json.dumps({"text": "three"})
        + "\n",
        encoding="utf-8",
    )
    assert [d.text for d in iter_documents(p)] == ["one two", "three"]


def test_stats_and_limit(tmp_path):
    for i in range(5):
        (tmp_path / f"{i}.txt").write_text("the cat the dog", encoding="utf-8")
    s = compute_stats(tmp_path)
    assert (s.documents, s.tokens, s.vocabulary) == (5, 20, 3)
    assert s.top(1) == [("the", 10)]
    assert compute_stats(tmp_path, limit=2).documents == 2
