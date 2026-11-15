"""What happens to a crawler when CPU-heavy work (indexing) runs INSIDE a coroutine?

    uv run python experiments/loop_blocking.py            # starts its own local mirror

A "heartbeat" coroutine wakes up every 10 ms and records how late it was. On a healthy event loop
it is a few ms late at most; if some code runs for 200 ms without an ``await``, the heartbeat -- and
every network read, every timeout, every other worker -- is 200 ms late too.

Three ways to give each downloaded page `--work-ms` of CPU work (a stand-in for indexing it):

  none      the baseline: just crawl
  inline    the work is done in the consumer coroutine: the loop stands still while it runs
  thread    ``await asyncio.to_thread(work)``: the loop keeps turning (the thread shares the GIL,
            but CPython hands it over every 5 ms)
  process   ``await loop.run_in_executor(ProcessPoolExecutor, work)``: no GIL contention at all,
            pays the pickle tax of Lab 05 for the text

Run it with `-X dev` or PYTHONASYNCIODEBUG=1 to see asyncio itself report the inline case:
"Executing <Task ...> took 0.021 seconds".
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from findex.crawlbench import started_mirror
from findex.crawler import CrawlConfig, CrawlStats, crawl
from findex.tokenize import tokenize


def index_work(text: str, work_ms: float) -> int:
    """Burn `work_ms` of CPU the way indexing would: tokenize and count, again and again."""
    deadline = time.perf_counter() + work_ms / 1000
    n = 0
    while time.perf_counter() < deadline:
        n += sum(Counter(tokenize(text)).values())
    return n


async def heartbeat(gaps: list[float], period: float = 0.01) -> None:
    loop = asyncio.get_running_loop()
    expected = loop.time() + period
    while True:
        await asyncio.sleep(period)
        now = loop.time()
        gaps.append(now - expected)  # how late we were woken up
        expected = now + period


async def run_mode(
    url: str, mode: str, pages: int, concurrency: int, work_ms: float
) -> dict[str, float]:
    config = CrawlConfig(
        (url,), "loop-blocking-demo/1.0 (+mailto:demo@localhost)",
        max_pages=pages, concurrency=concurrency, per_host=concurrency, delay=0.0,
    )  # fmt: skip
    stats = CrawlStats()
    gaps: list[float] = []
    pool = ProcessPoolExecutor(2) if mode == "process" else None
    if pool is not None:  # start the workers before the clock does
        await asyncio.get_running_loop().run_in_executor(pool, index_work, "warm up", 1.0)
    beat = asyncio.create_task(heartbeat(gaps))
    t0 = time.perf_counter()
    peak_in_flight = 0
    async with contextlib.aclosing(crawl(config, stats=stats)) as it:
        async for page in it:
            peak_in_flight = max(peak_in_flight, stats.in_flight)
            if mode == "inline":
                index_work(page.text, work_ms)  # <- no await: the whole loop waits for it
            elif mode == "thread":
                await asyncio.to_thread(index_work, page.text, work_ms)
            elif mode == "process":
                await asyncio.get_running_loop().run_in_executor(
                    pool, index_work, page.text, work_ms
                )
    wall = time.perf_counter() - t0
    beat.cancel()
    if pool is not None:
        pool.shutdown()
    return {
        "wall": wall,
        "pages": stats.pages,
        "pages_per_s": stats.pages / wall,
        "max_gap_ms": max(gaps, default=0.0) * 1000,
        "p95_gap_ms": sorted(gaps)[int(len(gaps) * 0.95)] * 1000 if gaps else 0.0,
        "peak_in_flight": peak_in_flight,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pages", type=int, default=60)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--latency-ms", type=float, default=30.0)
    ap.add_argument("--work-ms", type=float, default=20.0, help="CPU per page, ms")
    ap.add_argument("--modes", nargs="+", default=["none", "inline", "thread", "process"])
    args = ap.parse_args()
    print(f"{args.pages} pages, concurrency {args.concurrency}, server latency {args.latency_ms:.0f} ms, "
          f"{args.work_ms:.0f} ms of CPU work per page\n")  # fmt: skip
    print(f"{'mode':8} {'wall, s':>8} {'pages/s':>8} {'heartbeat max, ms':>18} {'p95, ms':>8}")
    with started_mirror(args.pages, args.latency_ms, None) as url:
        for mode in args.modes:
            r = asyncio.run(run_mode(url, mode, args.pages, args.concurrency, args.work_ms))
            print(
                f"{mode:8} {r['wall']:8.2f} {r['pages_per_s']:8.1f} {r['max_gap_ms']:18.1f} {r['p95_gap_ms']:8.1f}"
            )


if __name__ == "__main__":
    main()
