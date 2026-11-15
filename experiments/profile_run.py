"""Lab 08, step 1: where does the time go? (cProfile; py-spy and Scalene wrap this same script)

    uv run python experiments/profile_run.py index  CORPUS_DIR [--positions]
    uv run python experiments/profile_run.py search INDEX.bin "function OR value OR error" [--repeat 200]

    py-spy record -f flamegraph -o docs/flame-search.svg --rate 500 -- python experiments/profile_run.py search ...
    scalene --cli --reduced-profile experiments/profile_run.py search ...      (Python vs native time)

``search`` loads the index once (profiled separately: it is a different cost from a query) and
then runs the query ``--repeat`` times, with snippets, the way ``findex search`` does it.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from pathlib import Path

from findex.build import build_index
from findex.corpus import iter_documents
from findex.engine import SearchEngine
from findex.store import load


def report(prof: cProfile.Profile, top: int) -> str:
    out = io.StringIO()
    for key in ("tottime", "cumulative"):
        out.write(f"\n--- top {top} by {key} ---\n")
        pstats.Stats(prof, stream=out).strip_dirs().sort_stats(key).print_stats(top)
    return out.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--top", type=int, default=12)
    common.add_argument(
        "--profile", action="store_true", help="wrap the work in cProfile (off for py-spy/Scalene)"
    )
    common.add_argument("--out", type=Path, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("index", parents=[common])
    pi.add_argument("corpus", type=Path)
    pi.add_argument("--positions", action="store_true")
    ps = sub.add_parser("search", parents=[common])
    ps.add_argument("index", type=Path)
    ps.add_argument("query")
    ps.add_argument("--repeat", type=int, default=200)
    args = ap.parse_args()

    prof = cProfile.Profile()
    text = ""
    if args.cmd == "index":
        t0 = time.perf_counter()
        if args.profile:
            prof.enable()
        ix = build_index(iter_documents(args.corpus), storage="array", positions=args.positions)
        if args.profile:
            prof.disable()
        text = f"index: {len(ix):,} terms, {ix.num_docs:,} docs in {time.perf_counter() - t0:.2f} s"
    else:
        t0 = time.perf_counter()
        ix = load(args.index)
        load_s = time.perf_counter() - t0
        engine = SearchEngine(ix, cache_size=0)  # cache off: every repetition does the whole job
        engine.search(args.query, k=10)  # warm-up
        t0 = time.perf_counter()
        if args.profile:
            prof.enable()
        for _ in range(args.repeat):
            hits = engine.search(args.query, k=10)
        if args.profile:
            prof.disable()
        per = (time.perf_counter() - t0) / args.repeat * 1000
        text = f"load: {load_s:.2f} s; query {args.query!r}: {per:.2f} ms per query ({len(hits)} hits, {args.repeat} runs)"
    print(text)
    if args.profile:
        body = text + "\n" + report(prof, args.top)
        if args.out:
            args.out.write_text(body, encoding="utf-8")
        print(body)


if __name__ == "__main__":
    main()
