"""Lab 05 (optional): the opposite experiment. I/O-bound work: threads DO scale.

    uv run python experiments/io_bound_demo.py --mode sleep               # simulated waiting
    uv run python experiments/io_bound_demo.py --mode urls --tasks 30     # real downloads
    uv run python experiments/io_bound_demo.py --mode files data/gutenberg

While a thread waits for the network or the disk, CPython RELEASES the GIL, so another thread
can run. With 20 waits of 50 ms each, one thread needs 1 s; 20 threads need about 50 ms.

Modes
  sleep  time.sleep(delay): a stand-in for a network round-trip (deterministic, no network)
  urls   download Project Gutenberg plain-text books (be polite: keep --tasks small)
  files  read every .txt file of a corpus (only waits if the OS file cache is cold)
"""

from __future__ import annotations

import argparse
import statistics
import time
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def make_tasks(args: argparse.Namespace) -> list[Callable[[], object]]:
    if args.mode == "sleep":
        return [lambda: time.sleep(args.delay) for _ in range(args.tasks)]
    if args.mode == "urls":

        def fetch(book_id: int) -> Callable[[], object]:
            url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
            req = urllib.request.Request(url, headers={"User-Agent": "findex-lab/0.5"})

            def run() -> int:
                with urllib.request.urlopen(req, timeout=30) as response:
                    return len(response.read())

            return run

        return [fetch(args.start + i) for i in range(args.tasks)]
    if args.mode == "files":
        paths = sorted(Path(args.corpus).rglob("*.txt"))[: args.tasks]

        def reader(path: Path) -> Callable[[], object]:
            return lambda: path.read_bytes()

        return [reader(p) for p in paths]
    raise SystemExit(f"unknown mode {args.mode}")


def timed(tasks: list[Callable[[], object]], workers: int, repeat: int) -> float:
    runs = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(workers) as pool:
            for future in [pool.submit(t) for t in tasks]:
                future.result()
        runs.append(time.perf_counter() - t0)
    return statistics.median(runs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="?", default="data/gutenberg")
    ap.add_argument("--mode", choices=["sleep", "urls", "files"], default="sleep")
    ap.add_argument("--tasks", type=int, default=40)
    ap.add_argument("--delay", type=float, default=0.05, help="seconds per simulated request")
    ap.add_argument("--start", type=int, default=1000, help="first Gutenberg book id (urls)")
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    tasks = make_tasks(args)
    print(f"mode={args.mode}, {len(tasks)} tasks, median of {args.repeat} runs\n")
    base = timed(tasks, 1, args.repeat)
    print(f"{'threads':>8} {'wall, s':>9} {'speedup':>9}")
    for workers in (1, 2, 4, 8, 16, 32):
        wall = base if workers == 1 else timed(tasks, workers, args.repeat)
        print(f"{workers:>8} {wall:9.3f} {base / wall:8.1f}x")


if __name__ == "__main__":
    main()
