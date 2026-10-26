"""Load a saved index and answer a Boolean query.

Usage:  python -m findex.search index.bin "cat dog"            # AND
        python -m findex.search index.bin "cat OR dog NOT bird" --engine set
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from findex.boolean import ENGINES, evaluate
from findex.measure import describe, run
from findex.store import MAGIC, load


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m findex.search")
    ap.add_argument("index", type=Path)
    ap.add_argument("query")
    ap.add_argument("--engine", choices=sorted(ENGINES), default="merge")
    ap.add_argument("--top", type=int, default=10, help="how many hits to print")
    ap.add_argument(
        "--no-tracemalloc", action="store_true", help="clean timing, no peak memory"
    )
    args = ap.parse_args(argv)

    with args.index.open("rb") as f:
        if f.read(len(MAGIC)) != MAGIC:
            print(
                "warning: pickle file -- loading it executes code; only open your own files",
                file=sys.stderr,
            )

    def job():
        index = load(args.index)
        t_loaded = time.perf_counter()
        hits = evaluate(args.query, index, args.engine)
        return index, hits, time.perf_counter() - t_loaded

    (index, hits, t_query), elapsed, peak = run(job, trace=not args.no_tracemalloc)

    print(f"query: {args.query!r}  engine={args.engine}  ->  {len(hits):,} documents")
    for doc_id in hits[: args.top]:
        m = index.doc_meta[doc_id]
        print(f"  [{doc_id}] {m.title}  ({m.path})")
    if len(hits) > args.top:
        print(f"  ... and {len(hits) - args.top:,} more")
    print(
        f"\nload + search: {describe(elapsed, peak)}; the search alone took {t_query * 1e3:.3f} ms"
    )


if __name__ == "__main__":
    main()
