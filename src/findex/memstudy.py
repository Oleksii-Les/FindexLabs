"""M4: where do the bytes go? Three ways to store postings, same corpus.

Usage:  python -m findex.memstudy data/ [--limit N] [--skip-positions]
Prints Markdown tables for the README. Peak/resident memory come from tracemalloc;
wall-clock times come from separate runs WITHOUT tracemalloc.
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import tempfile
import time
import tracemalloc
from itertools import islice
from pathlib import Path

from findex.build import build_index
from findex.corpus import iter_documents
from findex.measure import fmt_bytes
from findex.model import Index, PlainPosting, Posting
from findex.store import load, save


def _build(root: Path, limit: int | None, storage: str, positions: bool) -> Index:
    docs = iter_documents(root)
    if limit is not None:
        docs = islice(docs, limit)
    return build_index(docs, storage=storage, positions=positions)


def traced_build(root, limit, storage, positions):
    """-> (index, peak_bytes, resident_bytes_after_build)."""
    gc.collect()
    tracemalloc.start()
    index = _build(root, limit, storage, positions)
    resident, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return index, peak, resident


def clean_build_time(root, limit, storage, positions) -> float:
    gc.collect()
    t0 = time.perf_counter()
    _build(root, limit, storage, positions)
    return time.perf_counter() - t0


def file_cost(index: Index, fmt: str, tmp: Path, repeat: int = 3):
    """-> (size_bytes, best save s, best load s)."""
    best_save = best_load = float("inf")
    size = 0
    for _ in range(repeat):
        t0 = time.perf_counter()
        size = save(index, tmp, fmt)
        best_save = min(best_save, time.perf_counter() - t0)
    for _ in range(repeat):
        gc.collect()
        t0 = time.perf_counter()
        loaded = load(tmp)
        best_load = min(best_load, time.perf_counter() - t0)
        del loaded
    return size, best_save, best_load


def per_object_sizes() -> None:
    plain = PlainPosting(123456, 7)
    slotted = Posting(123456, 7)
    print("\nShallow sizes of ONE posting (bytes), via sys.getsizeof:")
    print(f"- plain dataclass object:          {sys.getsizeof(plain)}")
    print(f"  + its __dict__:                  {sys.getsizeof(plain.__dict__)}")
    print(f"- slots dataclass object:          {sys.getsizeof(slotted)}")
    print(f"- int (doc_id > 256):              {sys.getsizeof(123456)}")
    print(f"- empty tuple (shared singleton):  {sys.getsizeof(())}")
    print("- array('I') element:              4 (x2 for doc_id and tf)")
    print(
        f"- pointer in a list[...]:          {sys.getsizeof([0]) - sys.getsizeof([])}"
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m findex.memstudy")
    ap.add_argument("root", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--skip-positions", action="store_true", help="skip the positions-cost table"
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.ERROR)

    names = {
        "dataclass": "`list[Posting]`, plain `@dataclass`",
        "slots": "`list[Posting]`, `slots=True`",
        "array": "`array('I')` pairs (no objects)",
    }
    built: dict[str, Index] = {}
    rows = []
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "idx"
        for storage in ("dataclass", "slots", "array"):
            t_build = clean_build_time(args.root, args.limit, storage, False)
            index, peak, resident = traced_build(args.root, args.limit, storage, False)
            size, t_save, t_load = file_cost(index, "pickle", tmp)
            built[storage] = index
            n = index.n_postings()
            rows.append(
                (storage, peak, resident, resident / n, t_build, size, t_save, t_load)
            )
            print(f"[done] {storage}", file=sys.stderr)
        first = built["array"]
        print(
            f"Corpus: {first.num_docs:,} documents, {len(first.postings):,} terms, "
            f"{first.n_postings():,} postings.\n"
        )
        print("### Table 1 -- postings representation (pickle file, no positions)\n")
        print(
            "| Postings representation | Peak (build) | Resident index | B/posting | Build | File | Save | Load |"
        )
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for st, peak, res, bpp, tb, size, ts, tl in rows:
            print(
                f"| {names[st]} | {fmt_bytes(peak)} | {fmt_bytes(res)} | {bpp:.1f} | "
                f"{tb:.2f} s | {fmt_bytes(size)} | {ts:.2f} s | {tl:.2f} s |"
            )

        print("\n### Table 2 -- file formats\n")
        print("| Format | In-memory storage | File size | Save | Load |")
        print("| --- | --- | ---: | ---: | ---: |")
        for fmt, st in (("pickle", "slots"), ("pickle", "array"), ("binary", "array")):
            size, ts, tl = file_cost(built[st], fmt, tmp)
            print(f"| {fmt} | {st} | {fmt_bytes(size)} | {ts:.2f} s | {tl:.2f} s |")

        if not args.skip_positions:
            print(
                "\n### Table 3 -- cost of token positions (needed for phrase search, Lab 3)\n"
            )
            print(
                "| Storage | Positions | Resident index | Pickle file | vs. no positions (file) |"
            )
            print("| --- | --- | ---: | ---: | ---: |")
            for st in ("slots", "array"):
                base = file_cost(built[st], "pickle", tmp, repeat=1)[0]
                print(
                    f"| {st} | off | {fmt_bytes(next(r for r in rows if r[0] == st)[2])} | {fmt_bytes(base)} | 1.0x |"
                )
                idx, _, res = traced_build(args.root, args.limit, st, True)
                size = file_cost(idx, "pickle", tmp, repeat=1)[0]
                print(
                    f"| {st} | on | {fmt_bytes(res)} | {fmt_bytes(size)} | {size / base:.1f}x |"
                )
                del idx
    per_object_sizes()


if __name__ == "__main__":
    main()
