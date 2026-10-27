"""precision@k for TF-IDF vs BM25 on hand-labeled queries.

Usage:  python -m findex.evaluate index.bin eval/queries.json [-k 5] [--show]

queries.json:  [{"query": "white whale", "relevant": ["2701.txt", "moby"]}, ...]
A result is relevant if an entry equals its file name (with or without extension)
or is a case-insensitive substring of its title. Label BEFORE looking at rankings,
from what you know about the books; ``--show`` prints the top hits to help you spot
labeling mistakes afterwards, not to copy the system's answers into the labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

from findex.engine import search
from findex.model import DocMeta, Index
from findex.query import PositionsRequired
from findex.scoring import BM25, Scorer, SearchResult, TfIdf
from findex.store import open_index


def is_relevant(meta: DocMeta, relevant: Iterable[str]) -> bool:
    name = Path(meta.path).name.lower()
    stem = Path(meta.path).stem.lower()
    title = meta.title.lower()
    for r in (x.lower() for x in relevant):
        if r in (name, stem) or r in title:
            return True
    return False


def precision_at_k(
    results: list[SearchResult], index: Index, relevant: list[str], k: int
) -> float:
    """Fraction of the top-k that is relevant (denominator is always k)."""
    return sum(is_relevant(index.doc_meta[r.doc_id], relevant) for r in results[:k]) / k


def evaluate(index: Index, queries: list[dict], scorers: dict[str, Scorer], k: int = 5):
    """-> list of (query, {scorer_name: P@k or None}); None = query unsupported (no positions)."""
    rows = []
    for q in queries:
        row: dict[str, float | None] = {}
        for name, scorer in scorers.items():
            try:
                hits = search(index, q["query"], scorer=scorer, k=k)
            except PositionsRequired:
                row[name] = None
                continue
            row[name] = precision_at_k(hits, index, q["relevant"], k)
        rows.append((q, row))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m findex.evaluate")
    ap.add_argument("index", type=Path)
    ap.add_argument("queries", type=Path)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument(
        "--show", action="store_true", help="print the top-k hits per scorer"
    )
    args = ap.parse_args(argv)
    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    scorers: dict[str, Scorer] = {"TF-IDF": TfIdf(), "BM25": BM25()}

    with open_index(args.index) as index:
        rows = evaluate(index, queries, scorers, args.k)
        if args.show:
            for q, _ in rows:
                print(f"\n== {q['query']}   (relevant: {', '.join(q['relevant'])})")
                for name, sc in scorers.items():
                    hits = search(index, q["query"], scorer=sc, k=args.k)
                    print(f"  {name}:")
                    for r in hits:
                        mark = (
                            "+"
                            if is_relevant(index.doc_meta[r.doc_id], q["relevant"])
                            else " "
                        )
                        print(
                            f"   {mark} {Path(index.doc_meta[r.doc_id].path).name}  {r.title}"
                        )
            print()

    print(f"| # | Query | Relevant | P@{args.k} TF-IDF | P@{args.k} BM25 |")
    print("| ---: | --- | ---: | ---: | ---: |")
    sums = {n: [] for n in scorers}
    for i, (q, row) in enumerate(rows, start=1):
        cells = []
        for n in scorers:
            v = row[n]
            cells.append("n/a" if v is None else f"{v:.2f}")
            if v is not None:
                sums[n].append(v)
        print(
            f"| {i} | `{q['query']}` | {len(q['relevant'])} | {cells[0]} | {cells[1]} |"
        )
    means = [f"{sum(v) / len(v):.2f}" if v else "n/a" for v in sums.values()]
    print(f"| | **mean** | | **{means[0]}** | **{means[1]}** |")
    if any(v is None for _, row in rows for v in row.values()):
        print(
            "\nn/a = phrase query on an index built without --positions (not in the mean)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
