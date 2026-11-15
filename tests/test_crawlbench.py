"""The crawl benchmark, the mirror site and the glue code around the crawler."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
from pathlib import Path

import pytest
from rich.console import Console

from findex import crawlbench
from findex.crawl_run import crawl_logging, live_counters, record_of, render_stats, run_crawl
from findex.crawlbench import RunRecord, Setup, measure_crawl, render_table, summarize
from findex.crawler import CrawlConfig, CrawlStats
from findex.mirror import Mirror, MirrorConfig, run_in_thread

UA = "findex-test/1.0 (+mailto:tests@example.org)"


def rec(setup: str, c: int, wall: float, pages: int = 200, errors: int = 0) -> RunRecord:
    return {
        "setup": setup, "concurrency": c, "per_host": c, "delay": 0.0, "run": 0,
        "pages": pages, "errors": errors, "retries": 0, "wall_s": wall, "cpu_s": 0.5,
        "rss_bytes": 40 * 2**20, "bytes": 1000, "statuses": {"200": pages},
    }  # fmt: skip


def test_summarize_takes_the_median_and_computes_speedup_against_concurrency_1():
    runs = [rec("c=1", 1, w) for w in (10.0, 12.0, 11.0)] + [
        rec("c=5", 5, w) for w in (2.0, 2.2, 9.0)
    ]
    rows = {r.setup: r for r in summarize(runs)}
    assert rows["c=1"].wall == 11.0 and rows["c=1"].speedup == 1.0
    assert (
        rows["c=5"].wall == 2.2 and rows["c=5"].wall_max == 9.0
    )  # the outlier is visible, not the median
    assert rows["c=5"].speedup == pytest.approx(11.0 / 2.2)
    assert rows["c=5"].pages_per_s == pytest.approx(200 / 2.2)


def test_render_table_is_markdown():
    table = render_table(summarize([rec("c=1", 1, 10.0), rec("c=20", 20, 1.0, errors=2)]))
    lines = table.splitlines()
    assert lines[0].startswith("| Setup") and set(lines[1]) <= set("| -:")
    assert "| c=20 | 1.00 |" in lines[3] and "10.00x" in lines[3] and "40 MiB" in lines[3]


def test_default_setups_are_the_labs_grid_plus_a_polite_row():
    setups = crawlbench.default_setups()
    assert [s.concurrency for s in setups[:3]] == [1, 5, 20]
    assert setups[3].per_host == 2 and setups[3].delay > 0


def test_measure_crawl_reports_pages_wall_and_memory():
    base, _, stop = run_in_thread(MirrorConfig(pages=30, latency_ms=1))
    try:
        res = measure_crawl(base + "/", 12, Setup("t", 4, 4, 0.0))
    finally:
        stop()
    assert res["pages"] == 12 and res["errors"] == 0 and res["wall_s"] > 0
    assert res["bytes"] > 0 and res["statuses"]["200"] >= 12


def test_child_and_report_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    base, _, stop = run_in_thread(MirrorConfig(pages=20, latency_ms=1))
    try:
        crawlbench.main(["child", base + "/", "--max-pages", "5", "--concurrency", "2",
                         "--per-host", "2", "--delay", "0"])  # fmt: skip
    finally:
        stop()
    child = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert child["pages"] == 5

    saved = tmp_path / "crawl.json"
    saved.write_text(
        json.dumps({"machine": {"cpu": "x", "logical_cpus": 4, "os": "t", "python": "3"},
                    "target": "t", "latency_ms": 30, "max_pages": 200,
                    "runs": [rec("c=1", 1, 4.0), rec("c=5", 5, 1.0)]}),
        encoding="utf-8",
    )  # fmt: skip
    crawlbench.main(["report", str(saved)])
    out = capsys.readouterr().out
    assert "4 logical CPUs" in out and "| c=5 | 1.00 |" in out and "4.00x" in out


@pytest.mark.slow
def test_run_command_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    out = tmp_path / "bench" / "crawl.json"
    crawlbench.main(["run", "--pages", "10", "--runs", "1", "--latency-ms", "1", "--out", str(out)])
    data = json.loads(out.read_text(encoding="utf-8"))
    assert {r["setup"] for r in data["runs"]} == {s.name for s in crawlbench.default_setups()}
    assert all(r["pages"] == 10 for r in data["runs"]) and data["machine"]["logical_cpus"] >= 1
    assert "| c=1 |" in capsys.readouterr().out


# ----------------------------------------------------------------------- crawl_run glue
def test_record_of_puts_the_title_into_the_searchable_text():
    line = json.loads(record_of("http://h/", "Title", "body text", status=200))
    assert line["text"] == "Title\nbody text" and line["title"] == "Title" and line["status"] == 200
    assert json.loads(record_of("http://h/", "", "only body"))["text"] == "only body"


def test_render_stats_shows_the_live_counters():
    stats = CrawlStats(pages=7, errors=2, retries=3, in_flight=5, queue_depth=40, bytes=2048)
    buf = io.StringIO()
    Console(file=buf, width=60).print(render_stats(stats, 200))
    text = buf.getvalue()
    for label in ("pages", "7 / 200", "pages/s", "requests in flight", "errors", "queue depth"):
        assert label in text


def test_live_counters_can_be_disabled_or_enabled():
    stats = CrawlStats()
    with live_counters(stats, 10, enabled=False):
        pass
    with live_counters(stats, 10, enabled=True):  # a non-terminal stderr: must not crash
        time.sleep(0.05)


@pytest.mark.parametrize("verbosity", [0, 1, 2])
def test_crawl_logging_routes_records_to_the_file_and_restores_the_loggers(
    tmp_path: Path, verbosity: int
):
    crawler_logger = logging.getLogger("findex.crawler")
    before = (crawler_logger.level, crawler_logger.propagate, list(crawler_logger.handlers))
    log = tmp_path / "c.log"
    with crawl_logging(log, verbosity) as counter:
        logging.getLogger("findex.crawler.access").info("GET http://h/ status=200")
        logging.getLogger("asyncio").warning("Executing <Task> took 0.500 seconds")
        assert not crawler_logger.propagate  # INFO lines must not leak to the root handler
    text = log.read_text(encoding="utf-8")
    assert "GET http://h/ status=200" in text and "took 0.500 seconds" in text
    assert counter.count == 1
    assert (crawler_logger.level, crawler_logger.propagate, list(crawler_logger.handlers)) == before


def test_run_crawl_returns_stats_and_the_blocked_loop_count(tmp_path: Path):
    base, _, stop = run_in_thread(MirrorConfig(pages=30, latency_ms=1))
    config = CrawlConfig((base + "/",), UA, max_pages=10, delay=0.0, per_host=5)
    try:
        stats, blocked = run_crawl(
            config,
            tmp_path / "p.jsonl",
            tmp_path / "c.log",
            live=True,
            debug_loop=True,
            verbosity=2,
        )
    finally:
        stop()
    assert stats.pages == 10 and blocked == 0
    assert len((tmp_path / "p.jsonl").read_text(encoding="utf-8").splitlines()) == 10


# ----------------------------------------------------------------------------- mirror
async def raw_request(port: int, request: bytes) -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(request)
    await writer.drain()
    data = await reader.read()  # the request asks for "Connection: close"
    writer.close()
    return data


def test_mirror_serves_robots_pages_and_rejects_other_methods():
    base, mirror, stop = run_in_thread(MirrorConfig(pages=10, latency_ms=0, crawl_delay=3))
    port = int(base.rsplit(":", 1)[1])
    try:
        get = asyncio.run(
            raw_request(port, b"GET /robots.txt HTTP/1.1\r\nConnection: close\r\n\r\n")
        )
        post = asyncio.run(raw_request(port, b"POST / HTTP/1.1\r\nConnection: close\r\n\r\n"))
        page = asyncio.run(raw_request(port, b"GET /page/3/ HTTP/1.1\r\nConnection: close\r\n\r\n"))
        gone = asyncio.run(raw_request(port, b"GET /page/99 HTTP/1.1\r\nConnection: close\r\n\r\n"))
    finally:
        stop()
    assert b"200 OK" in get and b"Disallow: /private/" in get and b"Crawl-delay: 3" in get
    assert b"405" in post
    assert (
        b"<title>Page 3 about" in page and b"/page/4" in page
    )  # children 5*3+1.. are >= 10: random links
    assert b"404" in gone
    assert mirror.hits["/robots.txt"] == 1


def test_mirror_fault_routes():
    mirror = Mirror(MirrorConfig(pages=5, latency_ms=0, faults=True))

    async def go() -> list[int]:
        out = []
        for path in ("/flaky/429", "/flaky/429", "/flaky/503", "/flaky/503", "/flaky/503",
                     "/redirect-loop", "/file.pdf", "/nothing"):  # fmt: skip
            status, _, _ = await mirror.respond(path)
            out.append(status)
        return out

    assert asyncio.run(go()) == [429, 200, 503, 503, 200, 302, 200, 404]
    plain = Mirror(MirrorConfig(pages=5, latency_ms=0))
    assert asyncio.run(plain.respond("/flaky/429"))[0] == 404  # faults are opt-in


def test_mirror_can_take_its_text_from_a_corpus(tmp_path: Path):
    (tmp_path / "book.txt").write_text("Call me Ishmael. " * 500, encoding="utf-8")
    mirror = Mirror(MirrorConfig(pages=4, latency_ms=0, corpus=tmp_path))
    status, _, body = asyncio.run(mirror.respond("/page/2"))
    assert status == 200 and b"Ishmael" in body


def test_mirror_pages_are_deterministic():
    a, b = (
        Mirror(MirrorConfig(pages=20, latency_ms=0)),
        Mirror(MirrorConfig(pages=20, latency_ms=0)),
    )
    assert asyncio.run(a.respond("/page/7"))[2] == asyncio.run(b.respond("/page/7"))[2]
