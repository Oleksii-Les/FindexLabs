"""Lab 08: scorer of lab 3 (Python loop) vs NumPy, on a frequent term, a rare term and a 3-term query.

    uv run python experiments/numpy_table.py INDEX.bin [--repeat 30] [--terms the,whale,...]

For each query the matching document ids are computed ONCE (not timed); then ``score + top-10`` is timed
for both implementations, ``repeat`` times, and the **median** is shown. Both must return the same ranking
(checked on every row). The last column is the whole query (match + score + top-10, no snippets).
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from pathlib import Path

from findex.engine import search
from findex.model import Index
from findex.query import parse
from findex.scoring import BM25, SearchResult
from findex.store import load
from findex.vector import VectorIndex


def median_ms(fn: Callable[[], object], repeat: int) -> float:
    fn()  # warm-up (also fills the per-term array views)
    runs = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        runs.append((time.perf_counter() - t0) * 1000)
    return statistics.median(runs)


def row(index: Index, vectors: VectorIndex, label: str, q: str, repeat: int) -> str:
    scorer = BM25()
    ids = tuple(parse(q).evaluate(index, "merge"))
    terms = list(dict.fromkeys(parse(q).terms()))
    postings = sum(index.df(t) for t in terms)

    def slow() -> list[SearchResult]:
        return search(index, q, scorer=scorer, k=10, matcher=lambda _: ids)

    def fast() -> list[SearchResult]:
        return search(index, q, scorer=scorer, k=10, matcher=lambda _: ids, vectors=vectors)

    assert [(r.doc_id, r.score) for r in slow()] == [(r.doc_id, r.score) for r in fast()], q
    t_slow, t_fast = median_ms(slow, repeat), median_ms(fast, repeat)
    w_slow = median_ms(lambda: search(index, q, scorer=scorer, k=10), repeat)
    w_fast = median_ms(lambda: search(index, q, scorer=scorer, k=10, vectors=vectors), repeat)
    return (f"| {label}: `{q}` | {postings:,} | {t_slow:.3f} | {t_fast:.3f} | **{t_slow / t_fast:.1f}x** | "
            f"{w_slow:.2f} -> {w_fast:.2f} |")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("index", type=Path)
    ap.add_argument("--repeat", type=int, default=30)
    ap.add_argument("--queries", nargs="*", help='override: "label=query" pairs')
    args = ap.parse_args()
    index = load(args.index)
    vectors = VectorIndex(
        index, min_postings=0
    )  # forced: show the raw cost, also where the engine would not use it
    by_df = sorted(((index.df(t), t) for t in index), reverse=True)
    frequent = next(t for df, t in by_df if t.isalpha() and len(t) >= 3)
    mid = next(t for df, t in by_df if t.isalpha() and df <= len(index.doc_meta) // 50)
    rare = next(t for df, t in reversed(by_df) if t.isalpha() and df >= 3)
    tiny = next(t for df, t in reversed(by_df) if t.isalpha() and df == 1)
    cases = [("frequent term", frequent), ("mid-frequency term", mid), ("rare term", rare), ("term in 1 document", tiny),
             ("3 terms", f"{frequent} OR {mid} OR {rare}"), ("3 common terms", "function OR value OR error")]  # fmt: skip
    if args.queries:
        cases = [tuple(x.split("=", 1)) for x in args.queries]  # type: ignore[misc]
    print(
        f"{index.num_docs:,} documents, {len(index):,} terms; median of {args.repeat} runs, BM25, k=10\n"
    )
    print(
        "| Query | Postings scored | Scorer of lab 3, ms | NumPy, ms | Speed-up | Whole query: lab 3 → NumPy, ms |"
    )
    print("| --- | ---: | ---: | ---: | ---: | --- |")
    for label, q in cases:
        print(row(index, vectors, label, q, args.repeat))


if __name__ == "__main__":
    main()
