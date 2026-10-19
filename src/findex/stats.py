"""Corpus statistics in a single streaming pass.

Usage:  python -m findex.stats data/ [--limit N] [--top 50]
"""

from __future__ import annotations

import argparse
import logging
import time
import tracemalloc
from collections import Counter
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


@dataclass
class CorpusStats:
    documents: int
    tokens: int
    vocabulary: int
    counts: Counter[str]

    def top(self, n: int = 50) -> list[tuple[str, int]]:
        return self.counts.most_common(n)


def compute_stats(root: Path, limit: int | None = None) -> CorpusStats:
    """Lazy pipeline: iter_documents -> tokenize -> Counter (the only thing that grows)."""
    docs = iter_documents(root)
    if limit is not None:
        docs = islice(docs, limit)
    counts: Counter[str] = Counter()
    n_docs = 0
    for doc in docs:
        n_docs += 1
        counts.update(tokenize(doc.text))  # consumes the token stream lazily
    return CorpusStats(n_docs, counts.total(), len(counts), counts)


def measure(fn, *args, **kwargs):
    """Run fn under tracemalloc; return (result, elapsed_s, peak_bytes).

    Note: tracemalloc slows code down, so this elapsed time is inflated.
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    raise AssertionError  # unreachable


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m findex.stats")
    ap.add_argument(
        "root", type=Path, help="directory of .txt files or a .jsonl[.gz] file"
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="only the first N documents"
    )
    ap.add_argument("--top", type=int, default=50)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    stats, elapsed, peak = measure(compute_stats, args.root, args.limit)

    print(f"documents:  {stats.documents:,}")
    print(f"tokens:     {stats.tokens:,}")
    print(f"vocabulary: {stats.vocabulary:,}")
    print(f"\ntop {args.top} terms:")
    for rank, (term, n) in enumerate(stats.top(args.top), start=1):
        print(f"{rank:>3}. {term:<20} {n:>10,}")
    print(f"\nelapsed (with tracemalloc): {elapsed:.2f} s")
    print(f"peak memory:                {fmt_bytes(peak)}")


if __name__ == "__main__":
    main()
