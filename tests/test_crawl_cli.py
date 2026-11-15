"""``findex crawl`` end to end against the local mirror site (``findex.mirror``) on 127.0.0.1.

No outside network: the mirror runs in a background thread with its own event loop, the crawler
runs in the test's thread with its own, as it would against any remote server."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.corpus import iter_documents
from findex.mirror import Mirror, MirrorConfig, run_in_thread

runner = CliRunner()
FAST = ["--delay", "0", "--per-host", "5", "-c", "5", "--contact", "tests@example.org"]


@pytest.fixture
def site() -> Iterator[tuple[str, Mirror]]:
    base, mirror, stop = run_in_thread(MirrorConfig(pages=80, latency_ms=2))
    try:
        yield base, mirror
    finally:
        stop()


def test_crawl_writes_jsonl_log_and_a_summary(site: tuple[str, Mirror], tmp_path: Path):
    base, mirror = site
    out, log = tmp_path / "pages.jsonl", tmp_path / "crawl.log"
    result = runner.invoke(
        app, ["crawl", base + "/", "--max-pages", "25", *FAST, "--out", str(out), "--log", str(log)]
    )
    assert result.exit_code == 0, result.output
    assert "Crawl finished" in result.output and "pages / second" in result.output

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 25  # the ceiling
    assert len({r["url"] for r in records}) == 25  # never the same canonical URL twice
    assert all(
        {"url", "title", "text", "status", "depth", "bytes", "fetched_at"} <= r.keys()
        for r in records
    )
    assert all(r["status"] == 200 and r["text"].startswith(r["title"]) for r in records)
    assert not any("/private/" in r["url"] for r in records)  # robots.txt
    assert mirror.hits["/private/secret"] == 0  # never even requested

    lines = log.read_text(encoding="utf-8").splitlines()
    gets = [ln for ln in lines if " GET " in ln]
    assert len(gets) == 25 and all(
        "status=200" in ln and "bytes=" in ln and "time=" in ln for ln in gets
    )
    assert any("SKIP" in ln and "robots.txt" in ln for ln in lines)


def test_the_jsonl_is_a_corpus_that_index_and_search_understand(
    site: tuple[str, Mirror], tmp_path: Path
):
    base, _ = site
    out, idx = tmp_path / "pages.jsonl", tmp_path / "idx.bin"
    assert runner.invoke(app, ["crawl", base + "/", "--max-pages", "30", *FAST, "--out", str(out),
                               "--log", str(tmp_path / "c.log")]).exit_code == 0  # fmt: skip
    docs = list(iter_documents(out))
    assert len(docs) == 30 and all(d.text for d in docs)
    built = runner.invoke(app, ["index", str(out), "--out", str(idx)])
    assert built.exit_code == 0, built.output
    found = runner.invoke(app, ["search", str(idx), "whale", "--json", "--no-snippets"])
    assert found.exit_code == 0 and found.stdout.strip()
    assert all(
        "whale" in json.loads(line)["title"].lower() for line in found.stdout.splitlines()[:3]
    )


def test_crawl_never_leaves_the_seed_host(site: tuple[str, Mirror], tmp_path: Path):
    base, _ = site
    out = tmp_path / "p.jsonl"
    runner.invoke(app, ["crawl", base + "/", "--max-pages", "60", *FAST, "--out", str(out),
                        "--log", str(tmp_path / "c.log")])  # fmt: skip
    hosts = {
        json.loads(ln)["url"].split("/")[2] for ln in out.read_text(encoding="utf-8").splitlines()
    }
    assert hosts == {base.removeprefix("http://")}


def test_contact_is_mandatory_and_goes_into_the_user_agent(tmp_path: Path):
    result = runner.invoke(
        app, ["crawl", "http://127.0.0.1:9/", "--out", str(tmp_path / "x.jsonl")]
    )
    assert result.exit_code == 1 and "error:" in result.output and "--contact" in result.output
    bad = runner.invoke(app, ["crawl", "http://127.0.0.1:9/", "--contact", "no way to reach me"])
    assert bad.exit_code == 1 and "e-mail" in bad.output

    from findex.crawl_run import build_user_agent

    ua = build_user_agent("me@example.org")
    assert ua.startswith("findex-crawler/") and "(+mailto:me@example.org)" in ua
    assert build_user_agent("https://example.org/bot").endswith("(+https://example.org/bot)")


def test_invalid_arguments_are_rejected(tmp_path: Path):
    args = ["--contact", "a@b.co", "--out", str(tmp_path / "x.jsonl")]
    assert runner.invoke(app, ["crawl", "ftp://example.org/", *args]).exit_code == 1
    assert (
        runner.invoke(app, ["crawl", "http://127.0.0.1:9/", "--max-pages", "0", *args]).exit_code
        == 2
    )
    assert (
        runner.invoke(app, ["crawl", "http://127.0.0.1:9/", "--retries", "9", *args]).exit_code == 2
    )
    assert (
        runner.invoke(
            app, ["crawl", "http://127.0.0.1:9/", "--max-pages", "999999", *args]
        ).exit_code
        == 2
    )


def test_nothing_reachable_is_an_error_not_an_empty_success(tmp_path: Path):
    result = runner.invoke(
        app,
        ["crawl", "http://127.0.0.1:9/", "--contact", "a@b.co", "--timeout", "1", "--retries", "0",
         "--out", str(tmp_path / "x.jsonl"), "--log", str(tmp_path / "x.log")],
    )  # fmt: skip
    assert (
        result.exit_code == 1 and "no pages" in result.output
    )  # robots.txt unreachable: disallow all


def test_debug_loop_flag_reports_the_blocked_loop_counter(site: tuple[str, Mirror], tmp_path: Path):
    base, _ = site
    result = runner.invoke(app, ["crawl", base + "/", "--max-pages", "15", *FAST, "--debug-loop",
                                 "--out", str(tmp_path / "p.jsonl"), "--log", str(tmp_path / "c.log")])  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "event-loop blocked warnings" in result.output


@pytest.mark.slow
def test_failures_are_survived_and_explained_in_the_log(tmp_path: Path):
    base, _, stop = run_in_thread(MirrorConfig(pages=20, latency_ms=2, faults=True))
    out, log = tmp_path / "p.jsonl", tmp_path / "crawl.log"
    try:
        result = runner.invoke(
            app,
            ["crawl", base + "/", "--max-pages", "20", *FAST, "--timeout", "0.5", "--retries", "1",
             "--out", str(out), "--log", str(log)],
        )  # fmt: skip
    finally:
        stop()
    assert result.exit_code == 0, result.output
    text = log.read_text(encoding="utf-8")
    # 429: waited the Retry-After the server asked for, then succeeded
    assert (
        "retry 1/1 for " in text and "/flaky/429 after HTTP 429 (Retry-After=1): sleeping 1" in text
    )
    urls = {json.loads(ln)["url"] for ln in out.read_text(encoding="utf-8").splitlines()}
    assert base + "/flaky/429" in urls
    # the others are reported, each for its own reason, and none of them stopped the crawl
    assert "FAIL " + base + '/missing error="HTTP 404"' in text
    assert "/redirect-loop" in text and "too many redirects" in text
    assert '/slow error="timeout"' in text and "attempts=2" in text
    assert "reason=content-type 'application/pdf'" in text
    assert len(urls) >= 15


def test_ctrl_c_leaves_a_clean_message_and_status_130(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    def interrupted(*_a: object, **_k: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("findex.cli.run_crawl", interrupted)
    result = runner.invoke(app, ["crawl", "http://127.0.0.1:9/", "--contact", "a@b.co",
                                 "--out", str(tmp_path / "p.jsonl")])  # fmt: skip
    assert result.exit_code == 130 and "interrupted" in result.output
