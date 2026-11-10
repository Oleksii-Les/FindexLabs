"""Smoke tests for the measurement tools of Labs 1-3 (they print Markdown for the README)."""

import json
from pathlib import Path

import pytest

from findex import bench_search, benchmark, evaluate, memstudy, sanity, stats
from findex.build import build_index
from findex.corpus import iter_documents
from findex.store import save


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    for i in range(24):
        words = " ".join(f"w{j}" for j in range(i % 6 * 15))
        rare = "zebra " if i == 5 else ""
        sea = "sea " if i % 3 == 0 else ""  # in a third of the docs, once: a mid-frequency term
        (root / f"{i}.txt").write_text(
            f"Title: Doc {i}\ncat dog {'cat ' * (i % 4)} {rare}{sea}common {words}",
            encoding="utf-8",
        )
    return root


@pytest.fixture
def index_file(corpus: Path, tmp_path: Path) -> Path:
    path = tmp_path / "ix.bin"
    save(build_index(iter_documents(corpus), storage="array", positions=True), path)
    return path


def test_corpus_stats_tool(corpus: Path, capsys: pytest.CaptureFixture[str]):
    stats.main([str(corpus), "--top", "3"])
    out = capsys.readouterr().out
    assert "documents:  24" in out and "top 3 terms" in out and "peak memory" in out


def test_eager_and_lazy_agree(corpus: Path):
    from findex.eager import compute_stats_eager

    lazy, eager = stats.compute_stats(corpus), compute_stats_eager(corpus)
    assert (lazy.documents, lazy.tokens, lazy.vocabulary) == (
        eager.documents,
        eager.tokens,
        eager.vocabulary,
    )
    assert stats.compute_stats(corpus, limit=5).documents == 5


def test_benchmark_tool(corpus: Path, capsys: pytest.CaptureFixture[str]):
    benchmark.main([str(corpus), "--limit", "10"])
    out = capsys.readouterr().out
    assert "eager (lists)" in out and "lazy (generators)" in out


def test_bench_search_tool(index_file: Path, capsys: pytest.CaptureFixture[str]):
    bench_search.main([str(index_file), "--quick"])
    out = capsys.readouterr().out
    assert "common terms" in out and "| AND |" in out and "| NOT |" in out


def test_sanity_tool_passes_its_three_checks(index_file: Path, capsys: pytest.CaptureFixture[str]):
    assert sanity.main([str(index_file)]) == 0
    out = capsys.readouterr().out
    assert "Check 1" in out and "Check 2" in out and "Check 3" in out
    assert out.count("PASS") == 3 and "FAIL" not in out


def test_evaluate_tool(index_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    queries = tmp_path / "q.json"
    queries.write_text(json.dumps([{"query": "zebra", "relevant": ["Doc 5"]}]), encoding="utf-8")
    assert evaluate.main([str(index_file), str(queries), "--show"]) == 0
    out = capsys.readouterr().out
    assert "P@5 BM25" in out and "| 1 | `zebra` | 1 | 0.20 | 0.20 |" in out


@pytest.mark.slow
def test_memstudy_tool(corpus: Path, capsys: pytest.CaptureFixture[str]):
    memstudy.main([str(corpus), "--limit", "12"])
    out = capsys.readouterr().out
    assert "Table 1" in out and "Table 2" in out and "Table 3" in out
    assert "plain `@dataclass`" in out and "Shallow sizes" in out
