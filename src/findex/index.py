"""Build an inverted index from a corpus and save it.

Usage:  python -m findex.index data/ --out index.bin [--limit N] [--positions]
                                     [--storage {dataclass,slots,array}] [--format {binary,pickle}]
"""

from __future__ import annotations

import argparse
import logging
from itertools import islice
from pathlib import Path

from findex.build import build_index
from findex.corpus import iter_documents
from findex.measure import describe, fmt_bytes, run
from findex.model import STORAGES
from findex.store import FORMATS, save


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m findex.index")
    ap.add_argument(
        "root", type=Path, help="directory of .txt files or a .jsonl[.gz] file"
    )
    ap.add_argument("--out", type=Path, default=Path("index.bin"))
    ap.add_argument(
        "--limit", type=int, default=None, help="only the first N documents"
    )
    ap.add_argument(
        "--positions", action="store_true", help="store token positions (bigger)"
    )
    ap.add_argument("--storage", choices=STORAGES, default="slots")
    ap.add_argument("--format", choices=FORMATS, default="binary", dest="fmt")
    ap.add_argument(
        "--no-tracemalloc", action="store_true", help="clean timing, no peak memory"
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    def job():
        docs = iter_documents(args.root)
        if args.limit is not None:
            docs = islice(docs, args.limit)
        return build_index(docs, storage=args.storage, positions=args.positions)

    index, elapsed, peak = run(job, trace=not args.no_tracemalloc)
    print(f"documents:  {index.n_docs:,}")
    print(f"vocabulary: {len(index.postings):,}")
    print(
        f"postings:   {index.n_postings():,}  (storage={args.storage}, positions={args.positions})"
    )
    print(f"build:      {describe(elapsed, peak)}")

    import time

    t0 = time.perf_counter()
    size = save(index, args.out, args.fmt)
    print(
        f"saved:      {args.out}  [{args.fmt}] {fmt_bytes(size)} in {time.perf_counter() - t0:.3f} s"
    )


if __name__ == "__main__":
    main()
