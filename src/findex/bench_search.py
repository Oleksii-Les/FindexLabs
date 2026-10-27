"""merge vs set on the two most frequent and the two rarest terms of an index.

Usage:  python -m findex.bench_search index.bin
Both engines get the same pre-extracted doc-id sequences, so only the algorithm is timed.
"""

from __future__ import annotations

import argparse
import timeit
from functools import partial
from pathlib import Path

from findex.boolean import ENGINES
from findex.store import load


def best_time(fn, repeat: int = 5) -> float:
    """Best-of-``repeat`` seconds per call (timeit picks the loop count)."""
    t = timeit.Timer(fn)
    number, _ = t.autorange()
    return min(t.repeat(repeat=repeat, number=number)) / number


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m findex.bench_search")
    ap.add_argument("index", type=Path)
    args = ap.parse_args(argv)
    index = load(args.index)

    by_df = sorted(index.postings, key=lambda t: (-index.df(t), t))
    common = by_df[:2]
    rare = sorted(index.postings, key=lambda t: (index.df(t), t))[:2]
    ids = {t: index.doc_ids(t) for t in common + rare}
    c1, c2 = common
    r1, r2 = rare

    cases = [
        ("AND", "common & common", c1, c2),
        ("AND", "common & rare", c1, r1),
        ("AND", "rare & rare", r1, r2),
        ("OR", "common | common", c1, c2),
        ("OR", "common | rare", c1, r1),
        ("NOT", "common - common", c1, c2),
        ("NOT", "common - rare", c1, r1),
    ]
    print(f"common terms: {c1!r} (df={index.df(c1):,}), {c2!r} (df={index.df(c2):,})")
    print(f"rare terms:   {r1!r} (df={index.df(r1)}), {r2!r} (df={index.df(r2)})\n")
    print("| Op | Case | len(a) | len(b) | merge | set | winner |")
    print("| --- | --- | ---: | ---: | ---: | ---: | --- |")
    slot = {"AND": 0, "OR": 1, "NOT": 2}
    for op, label, ta, tb in cases:
        a, b = ids[ta], ids[tb]
        f_merge, f_set = ENGINES["merge"][slot[op]], ENGINES["set"][slot[op]]
        assert f_merge(a, b) == f_set(a, b), "engines disagree!"
        tm = best_time(partial(f_merge, a, b))
        ts = best_time(partial(f_set, a, b))
        win = f"set x{tm / ts:.1f}" if ts < tm else f"merge x{ts / tm:.1f}"
        print(
            f"| {op} | {label} | {len(a):,} | {len(b):,} | {tm * 1e6:,.1f} µs | {ts * 1e6:,.1f} µs | {win} |"
        )


if __name__ == "__main__":
    main()
