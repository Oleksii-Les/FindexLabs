import gzip
import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

from findex.corpus import count_documents, iter_documents
from findex.measure import describe, run


def test_jsonl_skips_bad_lines_with_warnings(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    f = tmp_path / "c.jsonl"
    f.write_text(
        "\n".join(
            [
                json.dumps({"text": "one"}),
                "",
                "{broken",
                json.dumps({"no_text": 1}),
                json.dumps([1, 2]),
                json.dumps({"text": "two"}),
            ]
        ),
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        docs = list(iter_documents(f))
    assert [(d.doc_id, d.text) for d in docs] == [(0, "one"), (1, "two")]
    assert len(caplog.records) == 3  # broken JSON, missing field, wrong type


def test_gzipped_jsonl_and_counting(tmp_path: Path):
    f = tmp_path / "c.jsonl.gz"
    with gzip.open(f, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"text": "a b"}) + "\n")
    assert [d.text for d in iter_documents(f)] == ["a b"]
    assert count_documents(f) is None  # unknown without reading it
    (tmp_path / "x.txt").write_text("t")
    assert count_documents(tmp_path) == 1


def test_unsupported_corpus_path_is_reported(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        next(iter_documents(tmp_path / "missing"))


def test_bad_utf8_is_replaced_not_fatal(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    (tmp_path / "a.txt").write_bytes(b"ok \xff\xfe bytes")
    with caplog.at_level(logging.WARNING):
        (doc,) = iter_documents(tmp_path)
    assert "\ufffd" in doc.text and "bad UTF-8" in caplog.text


def test_run_measures_time_and_optionally_memory():
    result, elapsed, peak = run(lambda: [0] * 100_000, trace=True)
    assert len(result) == 100_000 and elapsed >= 0 and peak is not None and peak > 100_000
    _, _, no_peak = run(lambda: 1, trace=False)
    assert no_peak is None
    assert "tracemalloc" in describe(0.5, 2048) and describe(0.5, None) == "0.500 s"


def test_python_dash_m_findex_works():
    done = subprocess.run(
        [sys.executable, "-m", "findex", "--version"], capture_output=True, text=True, check=True
    )
    assert done.stdout.startswith("findex ")
