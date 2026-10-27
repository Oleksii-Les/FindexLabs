"""Load a saved index and answer a ranked query.

Usage:  python -m findex.search index.bin 'python AND (async OR await) NOT java "event loop"'
        python -m findex.search index.bin "whale sea" --scorer tfidf --top 5 -v --repeat 2
Phrases ("...") need an index built with --positions.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

from findex.boolean import ENGINES
from findex.engine import SearchEngine
from findex.measure import describe, run
from findex.query import PositionsRequired, QuerySyntaxError
from findex.scoring import BM25, TfIdf
from findex.store import MAGIC, open_index

SCORERS = {"bm25": BM25, "tfidf": TfIdf}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m findex.search")
    ap.add_argument("index", type=Path)
    ap.add_argument("query")
    ap.add_argument("--scorer", choices=sorted(SCORERS), default="bm25")
    ap.add_argument(
        "--engine", choices=sorted(ENGINES), default="merge", help="set operations"
    )
    ap.add_argument("--top", type=int, default=10, help="how many hits to print")
    ap.add_argument(
        "--window", type=int, default=80, help="snippet half-width in characters"
    )
    ap.add_argument("--no-snippets", action="store_true")
    ap.add_argument(
        "--repeat", type=int, default=1, help="run the query N times (shows cache)"
    )
    ap.add_argument(
        "-v", "--verbose", action="store_true", help="show @timed / cache log lines"
    )
    ap.add_argument(
        "--no-tracemalloc", action="store_true", help="clean timing, no peak memory"
    )
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(name)s: %(message)s",
    )

    if not args.index.is_file():
        print(f"error: index file not found: {args.index}", file=sys.stderr)
        return 2
    with args.index.open("rb") as f:
        if f.read(len(MAGIC)) != MAGIC:
            print(
                "warning: pickle file -- loading it executes code; only open your own files",
                file=sys.stderr,
            )

    use_color = sys.stdout.isatty()
    marks = ("\033[1;33m", "\033[0m") if use_color else ("**", "**")

    def job():
        with open_index(args.index) as index:  # released even if the query raises
            t_loaded = time.perf_counter()
            eng = SearchEngine(index, SCORERS[args.scorer](), engine=args.engine)
            for _ in range(args.repeat):
                hits = eng.search(args.query, args.top, snippets=False)
            if not args.no_snippets:
                from findex.engine import search

                hits = search(
                    index,
                    args.query,
                    scorer=eng.scorer,
                    k=args.top,
                    engine=args.engine,
                    matcher=eng.match,
                    snippets=True,
                    window=args.window,
                    marks=marks,
                )
            return (
                repr(index),
                len(eng.match(args.query)),
                hits,
                time.perf_counter() - t_loaded,
            )

    try:
        (desc, n_matches, hits, t_query), elapsed, peak = run(
            job, trace=not args.no_tracemalloc
        )
    except (QuerySyntaxError, PositionsRequired) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"{desc}\nquery: {args.query!r}  scorer={args.scorer}  ->  {n_matches:,} matching documents\n"
    )
    for rank, r in enumerate(hits, start=1):
        print(f"{rank:>2}. {r}")
        if r.snippet:
            print(f"      {re.sub(chr(10), ' ', r.snippet)}")
    print(
        f"\nload + search: {describe(elapsed, peak)}; query part {t_query * 1e3:.2f} ms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
