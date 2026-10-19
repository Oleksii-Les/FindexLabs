"""M4: eager vs lazy on the same slice.

Usage:  python -m findex.benchmark data/ --limit 500
Each version is run twice: once WITHOUT tracemalloc (honest wall-clock time),
once WITH tracemalloc (peak memory). Output is a Markdown table for the README.
"""

from __future__ import annotations

import argparse
import gc
import logging
import time
from pathlib import Path

from findex.eager import compute_stats_eager
from findex.stats import compute_stats, fmt_bytes, measure


def timed(fn, *args):
    gc.collect()
    t0 = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - t0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m findex.benchmark")
    ap.add_argument("root", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.ERROR)

    rows = []
    reference = None
    for name, fn in (
        ("eager (lists)", compute_stats_eager),
        ("lazy (generators)", compute_stats),
    ):
        result, elapsed = timed(fn, args.root, args.limit)
        del result
        gc.collect()
        result, _, peak = measure(fn, args.root, args.limit)
        summary = (result.documents, result.tokens, result.vocabulary)
        if reference is None:
            reference = summary
        assert summary == reference, f"{name} disagrees: {summary} vs {reference}"
        rows.append((name, result.documents, result.tokens, peak, elapsed))
        del result
        gc.collect()

    print("| Version | Documents | Tokens | Peak memory | Elapsed |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for name, docs, toks, peak, elapsed in rows:
        print(f"| {name} | {docs:,} | {toks:,} | {fmt_bytes(peak)} | {elapsed:.2f} s |")
    if rows[1][3]:
        print(f"\neager / lazy peak memory: {rows[0][3] / rows[1][3]:.1f}x")


if __name__ == "__main__":
    main()
