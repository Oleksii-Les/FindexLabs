import json

import pytest

from findex import evaluate, sanity
from findex.index import main as index_main
from findex.search import main as search_main


@pytest.fixture
def index_file(tmp_path):
    corpus = tmp_path / "c"
    corpus.mkdir()
    for i in range(30):
        filler = " ".join(f"w{j}" for j in range(i % 7 * 20))
        extra = "event loop " if i % 3 == 0 else "loop only "
        rare = "zebra " if i == 4 else ""
        (corpus / f"{i}.txt").write_text(
            f"Title: Doc {i}\n{extra * (1 + i % 4)} python common {rare}{filler}",
            encoding="utf-8",
        )
    out = tmp_path / "ix.bin"
    index_main(
        [
            str(corpus),
            "--out",
            str(out),
            "--storage",
            "array",
            "--positions",
            "--no-tracemalloc",
        ]
    )
    return out


def test_ranked_search_with_phrase_and_snippets(index_file, capsys):
    capsys.readouterr()
    code = search_main(
        [
            str(index_file),
            'python "event loop" NOT zebra',
            "--no-tracemalloc",
            "--top",
            "3",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "matching documents" in out and "scorer=bm25" in out
    assert "**event**" in out and "**loop**" in out  # highlighted snippet
    assert "1." in out and "2." in out


def test_cli_reports_syntax_error_and_missing_positions(index_file, tmp_path, capsys):
    assert search_main([str(index_file), "(oops", "--no-tracemalloc"]) == 2
    assert "missing ')'" in capsys.readouterr().err
    assert search_main([str(tmp_path / "nope.bin"), "x"]) == 2

    corpus = tmp_path / "c2"
    corpus.mkdir()
    (corpus / "a.txt").write_text("event loop", encoding="utf-8")
    plain = tmp_path / "plain.bin"
    index_main([str(corpus), "--out", str(plain), "--no-tracemalloc"])
    assert search_main([str(plain), '"event loop"', "--no-tracemalloc"]) == 2
    assert "--positions" in capsys.readouterr().err


def test_sanity_and_evaluate_run(index_file, tmp_path, capsys):
    capsys.readouterr()
    assert sanity.main([str(index_file)]) == 0
    assert "Check 1" in capsys.readouterr().out
    q = tmp_path / "q.json"
    q.write_text(
        json.dumps([{"query": "zebra", "relevant": ["Doc 4"]}]), encoding="utf-8"
    )
    assert evaluate.main([str(index_file), str(q), "--show"]) == 0
    out = capsys.readouterr().out
    assert "P@5 BM25" in out and "| 1 | `zebra` | 1 | 0.20 | 0.20 |" in out
