"""The ``findex`` command, exercised in-process with typer's CliRunner."""

import json
import logging
import pickle
from pathlib import Path

import pytest
from conftest import TINY_CORPUS
from typer.testing import CliRunner

from findex.build import build_index
from findex.cli import app
from findex.corpus import iter_documents
from findex.store import save
from findex.tokenize import tokenize

runner = CliRunner()


def test_help_lists_the_three_subcommands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for word in ("index", "search", "stats"):
        assert word in result.stdout


def test_no_arguments_shows_help():
    result = runner.invoke(app, [])
    assert "Usage" in result.output and "search" in result.output


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and result.stdout.startswith("findex ")


def test_index_command_builds_and_summarizes(tiny_corpus: Path, tmp_path: Path):
    out = tmp_path / "i.bin"
    result = runner.invoke(app, ["index", str(tiny_corpus), "--out", str(out), "--positions"])
    assert result.exit_code == 0, result.output
    assert out.stat().st_size > 0
    assert "documents" in result.stdout and "6" in result.stdout  # 6 files incl. the empty one
    assert "indexing" in result.stderr  # the progress bar is diagnostics -> stderr


def test_index_limit_and_storage_options(tiny_corpus: Path, tmp_path: Path):
    out = tmp_path / "i.bin"
    result = runner.invoke(
        app, ["index", str(tiny_corpus), "-o", str(out), "--limit", "2", "--storage", "slots"]
    )
    assert result.exit_code == 0, result.output
    stats = runner.invoke(app, ["stats", str(out), "--json"])
    assert json.loads(stats.stdout)["documents"] == 2


def test_search_table(tiny_index_file: Path):
    result = runner.invoke(app, ["search", str(tiny_index_file), "white whale", "-k", "2"])
    assert result.exit_code == 0, result.output
    assert "Whales" in result.stdout and "a_whales.txt" in result.stdout
    assert "matching documents" in result.stdout


def test_search_json_lines_are_clean_on_stdout(tiny_index_file: Path):
    result = runner.invoke(app, ["-vv", "search", str(tiny_index_file), "sea", "--json"])
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in result.stdout.splitlines()]  # EVERY line is JSON
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
    assert {"doc_id", "score", "title", "path", "snippet"} <= rows[0].keys()
    assert rows == sorted(rows, key=lambda r: -r["score"])
    assert any("**sea**" in r["snippet"] for r in rows)
    assert "took" in result.stderr  # -vv diagnostics went to stderr, not into the JSON


@pytest.mark.parametrize("scorer", ["bm25", "tfidf"])
@pytest.mark.parametrize("engine", ["merge", "set"])
def test_search_options(tiny_index_file: Path, scorer: str, engine: str):
    result = runner.invoke(
        app,
        [
            "search",
            str(tiny_index_file),
            "sea NOT cat",
            "--scorer",
            scorer,
            "--engine",
            engine,
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    titles = {json.loads(line)["title"] for line in result.stdout.splitlines()}
    assert titles == {"Whales", "Ships", "Oceans"}


def test_phrase_and_brackets_query(tiny_index_file: Path):
    q = 'python AND (async OR await) NOT java "event loop"'
    result = runner.invoke(app, ["search", str(tiny_index_file), q, "--json"])
    assert result.exit_code == 0, result.output
    assert [json.loads(line)["title"] for line in result.stdout.splitlines()] == ["Python"]


def test_no_results_is_not_an_error(tiny_index_file: Path):
    result = runner.invoke(app, ["search", str(tiny_index_file), "zebra"])
    assert result.exit_code == 0 and "No documents match" in result.stdout
    assert runner.invoke(app, ["search", str(tiny_index_file), "zebra", "--json"]).stdout == ""


def test_stats_table_and_json(tiny_index_file: Path):
    table = runner.invoke(app, ["stats", str(tiny_index_file), "--top", "3"])
    assert table.exit_code == 0 and "vocabulary" in table.stdout and "the" in table.stdout
    data = json.loads(runner.invoke(app, ["stats", str(tiny_index_file), "--json"]).stdout)
    assert data["documents"] == 6 and data["top_terms"][0]["term"] == "the"
    assert data["tokens"] == sum(len(list(tokenize(t))) for t in TINY_CORPUS.values())


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["search", "/no/such/index.bin", "x"], "index file not found"),
        (["stats", "/no/such/index.bin"], "index file not found"),
        (["index", "/no/such/corpus", "--out", "x.bin"], "corpus not found"),
    ],
)
def test_expected_errors_are_one_line_and_nonzero(args: list[str], message: str):
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert result.stderr.strip().count("\n") == 0  # exactly one line
    assert result.stderr.startswith("error: ") and message in result.stderr
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("query", "message"),
    [("(oops", "missing ')'"), ("a OR", "unexpected end"), ("", "empty query"), ('"abc', "quote")],
)
def test_bad_query_is_a_clean_error(tiny_index_file: Path, query: str, message: str):
    result = runner.invoke(app, ["search", str(tiny_index_file), query])
    assert result.exit_code == 1 and message in result.stderr
    assert "Traceback" not in result.output and result.stdout == ""


def test_phrase_without_positions_explains_the_fix(tiny_corpus: Path, tmp_path: Path):
    out = tmp_path / "nopos.bin"
    runner.invoke(app, ["index", str(tiny_corpus), "--out", str(out)])
    result = runner.invoke(app, ["search", str(out), '"event loop"'])
    assert result.exit_code == 1 and "findex index --positions" in result.stderr


def test_corrupt_and_foreign_files_are_clean_errors(tiny_index_file: Path, tmp_path: Path):
    truncated = tmp_path / "trunc.bin"
    truncated.write_bytes(tiny_index_file.read_bytes()[:200])
    garbage = tmp_path / "garbage.bin"
    garbage.write_text("not an index")
    wrong_pickle = tmp_path / "wrong.bin"
    wrong_pickle.write_bytes(pickle.dumps({"not": "an index"}))
    for path in (truncated, garbage, wrong_pickle):
        result = runner.invoke(app, ["search", str(path), "x"])
        assert result.exit_code == 1, path
        assert "error:" in result.stderr and "Traceback" not in result.output


def test_empty_corpus_is_an_error(tmp_path: Path):
    result = runner.invoke(app, ["index", str(tmp_path), "--out", str(tmp_path / "o.bin")])
    assert result.exit_code == 1 and "no documents found" in result.stderr


def test_invalid_choice_is_a_usage_error(tiny_index_file: Path):
    result = runner.invoke(app, ["search", str(tiny_index_file), "x", "--scorer", "nope"])
    assert result.exit_code == 2


def test_pickle_index_works_but_warns(tiny_corpus: Path, tmp_path: Path):
    index = build_index(iter_documents(tiny_corpus), storage="slots", positions=True)
    path = tmp_path / "i.pkl"
    save(index, path, "pickle")
    result = runner.invoke(app, ["search", str(path), "whale", "--json"])
    assert result.exit_code == 0 and "pickle" in result.stderr
    assert result.stdout.strip() != ""


@pytest.mark.parametrize(
    ("argv", "level"),
    [([], logging.WARNING), (["-v"], logging.INFO), (["-vv"], logging.DEBUG)],
)
def test_verbosity_flags_set_the_root_logger(tiny_index_file: Path, argv: list[str], level: int):
    runner.invoke(app, [*argv, "stats", str(tiny_index_file)])
    assert logging.getLogger().level == level


def test_verbose_flag_is_also_accepted_after_the_subcommand(tiny_index_file: Path):
    result = runner.invoke(app, ["search", str(tiny_index_file), "sea", "-vv", "--json"])
    assert result.exit_code == 0 and "cache miss" in result.stderr and "took" in result.stderr


def test_repeat_shows_the_query_cache(tiny_index_file: Path):
    result = runner.invoke(
        app, ["-v", "search", str(tiny_index_file), "sea", "--repeat", "3", "--json"]
    )
    assert result.exit_code == 0
    assert result.stderr.count("cache miss") == 1 and result.stderr.count("cache HIT") == 2
