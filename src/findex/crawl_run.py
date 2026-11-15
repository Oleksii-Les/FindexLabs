"""Glue between ``findex crawl`` and :mod:`findex.crawler`: JSONL output, the live counters,
the ``crawl.log`` file and the event-loop-blocking detector. No crawling logic lives here."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
from collections.abc import Generator
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table

from findex.crawler import CrawlConfig, CrawlStats, crawl
from findex.stats import fmt_bytes

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def build_user_agent(contact: str) -> str:
    """An honest User-Agent: who we are and how to reach us, e.g.
    ``findex-crawler/0.6.0 (+mailto:me@example.org)``. ``contact`` is an e-mail or an http(s) URL."""
    contact = contact.strip()
    if contact.startswith(("http://", "https://", "mailto:")):
        target = contact
    elif "@" in contact and " " not in contact:
        target = f"mailto:{contact}"
    else:
        raise ValueError("--contact must be an e-mail address or an http(s) URL")
    try:
        ver = version("findex")
    except PackageNotFoundError:
        ver = "dev"
    return f"findex-crawler/{ver} (+{target})"


def record_of(page_url: str, title: str, text: str, **extra: object) -> str:
    """One JSONL line. ``text`` is what ``iter_documents`` reads, so the title is part of it."""
    body = f"{title}\n{text}" if title else text
    return json.dumps({"url": page_url, "title": title, "text": body, **extra}, ensure_ascii=False)


async def crawl_to_jsonl(config: CrawlConfig, out: Path, stats: CrawlStats) -> None:
    """Crawl and append each page to ``out`` as it arrives -- the pages are never all in memory.

    One short synchronous ``write`` per page (a few KiB, microseconds) is cheaper than a thread
    hop; indexing, which is CPU-heavy, is NOT done here but afterwards by ``findex index``.
    """
    with out.open("w", encoding="utf-8") as f:
        async with contextlib.aclosing(crawl(config, stats=stats)) as pages:
            async for page in pages:
                f.write(
                    record_of(
                        page.url,
                        page.title,
                        page.text,
                        status=page.status,
                        depth=page.depth,
                        bytes=page.nbytes,
                        fetched_at=page.fetched_at,
                    )
                    + "\n"
                )
                f.flush()


# ------------------------------------------------------------------------ live display
def render_stats(stats: CrawlStats, max_pages: int) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column(justify="right")
    table.add_row("pages", f"{stats.pages} / {max_pages}")
    table.add_row("pages/s", f"{stats.pages_per_second:.1f}")
    table.add_row("requests in flight", str(stats.in_flight))
    table.add_row("errors", str(stats.errors))
    table.add_row("retries", str(stats.retries))
    table.add_row("queue depth", str(stats.queue_depth))
    table.add_row("robots-skipped", str(stats.robots_skipped))
    table.add_row("downloaded", fmt_bytes(stats.bytes))
    table.add_row("elapsed", f"{stats.elapsed:.1f} s")
    return table


@contextlib.contextmanager
def live_counters(stats: CrawlStats, max_pages: int, enabled: bool) -> Generator[None]:
    """Rich ``Live`` on stderr, refreshed 5 times a second from a helper thread that only
    *reads* the counters (a stale number for one frame is harmless)."""
    if not enabled:
        yield
        return
    console = Console(stderr=True)
    with Live(
        get_renderable=lambda: Group(render_stats(stats, max_pages)),
        console=console,
        refresh_per_second=5,
        transient=False,
    ):
        yield


# ----------------------------------------------------------------------------- logging
class SlowCallbackCounter(logging.Handler):
    """Counts asyncio's "Executing <...> took 0.2 seconds" warnings (debug mode): each one means
    some code ran for too long WITHOUT an ``await`` and froze every other request."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if "took" in record.getMessage() and "seconds" in record.getMessage():
            self.count += 1


@contextlib.contextmanager
def crawl_logging(log_path: Path, verbosity: int) -> Generator[SlowCallbackCounter]:
    """Send the crawler's records (one line per URL, retries, robots) to ``log_path`` and keep them
    out of the progress display; restore everything on exit (the CLI is also called in tests)."""
    crawler_logger = logging.getLogger("findex.crawler")
    asyncio_logger = logging.getLogger("asyncio")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    counter = SlowCallbackCounter()
    saved = (crawler_logger.level, crawler_logger.propagate)
    handlers: list[logging.Handler] = [file_handler]
    if verbosity >= 1:  # -v: also show the crawler's lines on stderr
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        handlers.append(console_handler)
    crawler_logger.setLevel(logging.DEBUG if verbosity >= 2 else logging.INFO)
    crawler_logger.propagate = False  # otherwise INFO lines would also reach the root handler
    for h in handlers:
        crawler_logger.addHandler(h)
    asyncio_logger.addHandler(file_handler)
    asyncio_logger.addHandler(counter)
    quiet = ("httpx", "httpcore")
    saved_quiet = [logging.getLogger(n).level for n in quiet]
    if verbosity < 2:  # httpx logs every request at INFO: that is crawl.log's job
        for n in quiet:
            logging.getLogger(n).setLevel(logging.WARNING)
    try:
        yield counter
    finally:
        for n, level in zip(quiet, saved_quiet, strict=True):
            logging.getLogger(n).setLevel(level)
        asyncio_logger.removeHandler(counter)
        asyncio_logger.removeHandler(file_handler)
        for h in handlers:
            crawler_logger.removeHandler(h)
            h.close()
        crawler_logger.setLevel(saved[0])
        crawler_logger.propagate = saved[1]


def run_crawl(
    config: CrawlConfig,
    out: Path,
    log_path: Path,
    *,
    live: bool,
    debug_loop: bool,
    verbosity: int,
) -> tuple[CrawlStats, int]:
    """Run the crawl to completion. Returns the final counters and the number of
    "event loop was blocked" warnings (only meaningful with ``debug_loop``)."""
    stats = CrawlStats()

    async def main() -> None:
        if debug_loop:
            asyncio.get_running_loop().slow_callback_duration = 0.1  # seconds (the default)
        await crawl_to_jsonl(config, out, stats)

    with (
        crawl_logging(log_path, verbosity) as slow,
        live_counters(stats, config.max_pages, live),
    ):
        asyncio.run(main(), debug=debug_loop)
    return stats, slow.count
