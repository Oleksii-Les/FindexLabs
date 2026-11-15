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
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from findex.engine import search
from findex.model import DocMeta, Index
from findex.query import PositionsRequired
from findex.scoring import BM25, Scorer, SearchResult, TfIdf
from findex.store import open_index


class LabeledQuery(TypedDict):
    """One line of ``queries.json``."""

    query: str
    relevant: list[str]
    kind: NotRequired[str]  # "keyword" | "paraphrase" (only used by the semantic table)


def is_relevant(meta: DocMeta, relevant: Iterable[str]) -> bool:
    name = Path(meta.path).name.lower()
    stem = Path(meta.path).stem.lower()
    title = meta.title.lower()
    return any(r in (name, stem) or r in title for r in (x.lower() for x in relevant))


def precision_at_k(results: list[SearchResult], index: Index, relevant: list[str], k: int) -> float:
    """Fraction of the top-k that is relevant (denominator is always k)."""
    return sum(is_relevant(index.doc_meta[r.doc_id], relevant) for r in results[:k]) / k


def evaluate(
    index: Index, queries: Sequence[LabeledQuery], scorers: dict[str, Scorer], k: int = 5
) -> list[tuple[LabeledQuery, dict[str, float | None]]]:
    """-> list of (query, {scorer_name: P@k or None}); None = query unsupported (no positions)."""
    rows: list[tuple[LabeledQuery, dict[str, float | None]]] = []
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


System = Callable[[str, int], list[int]]  # (query, k) -> ranked doc ids


def evaluate_systems(
    index: Index, queries: Sequence[LabeledQuery], systems: dict[str, System], k: int = 5
) -> list[tuple[LabeledQuery, dict[str, float | None]]]:
    """P@k of every system on every query (None: the system cannot answer, e.g. a phrase without positions)."""
    rows: list[tuple[LabeledQuery, dict[str, float | None]]] = []
    for q in queries:
        row: dict[str, float | None] = {}
        for name, system in systems.items():
            try:
                ids = system(q["query"], k)
            except PositionsRequired:
                row[name] = None
                continue
            row[name] = sum(is_relevant(index.doc_meta[d], q["relevant"]) for d in ids[:k]) / k
        rows.append((q, row))
    return rows


def semantic_systems(
    index: Index, prefix: Path, corpus_root: Path | None
) -> dict[str, System]:  # pragma: no cover
    """BM25 as typed (lab 3), BM25 over the words (OR), semantic, hybrid: the four rows of the table."""
    from findex.engine import SearchEngine
    from findex.semantic import SemanticIndex, bag_of_words, search_mode

    engine = SearchEngine(index, BM25(), cache_size=0)
    sem = SemanticIndex.load(prefix, index)

    def lexical(q: str, k: int) -> list[int]:
        return [r.doc_id for r in engine.search(q, k, snippets=False)]

    return {
        "BM25": lexical,
        "BM25 (OR)": lambda q, k: lexical(bag_of_words(q), k),
        "semantic": lambda q, k: [h.doc_id for h in search_mode(engine, sem, q, "semantic", k)],
        "hybrid": lambda q, k: [h.doc_id for h in search_mode(engine, sem, q, "hybrid", k)],
    }


def render_semantic_table(
    rows: Sequence[tuple[LabeledQuery, dict[str, float | None]]], names: Sequence[str], k: int
) -> str:
    out = [
        f"| # | Kind | Query | {' | '.join(f'P@{k} {n}' for n in names)} |",
        f"| ---: | --- | --- | {' | '.join('---:' for _ in names)} |",
    ]
    for i, (q, row) in enumerate(rows, start=1):
        cells = ["n/a" if row[n] is None else f"{row[n]:.2f}" for n in names]
        out.append(f"| {i} | {q.get('kind', '')} | `{q['query']}` | {' | '.join(cells)} |")
    for kind in dict.fromkeys(q.get("kind", "") for q, _ in rows):
        part = [row for q, row in rows if q.get("kind", "") == kind]
        means = [
            sum(v for r in part if (v := r[n]) is not None)
            / max(1, sum(r[n] is not None for r in part))
            for n in names
        ]
        out.append(
            f"| | **mean, {kind or 'all'}** ({len(part)}) | | {' | '.join(f'**{m:.2f}**' for m in means)} |"
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m findex.evaluate")
    ap.add_argument("index", type=Path)
    ap.add_argument("queries", type=Path)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--show", action="store_true", help="print the top-k hits per scorer")
    ap.add_argument(
        "--embeddings", type=Path, help="prefix made by `findex embed`: adds semantic and hybrid"
    )
    ap.add_argument("--corpus-root", type=Path, default=None)
    args = ap.parse_args(argv)
    queries = cast(list[LabeledQuery], json.loads(args.queries.read_text(encoding="utf-8")))
    scorers: dict[str, Scorer] = {"TF-IDF": TfIdf(), "BM25": BM25()}

    if args.embeddings is not None:  # pragma: no cover - needs a built embedding file
        with open_index(args.index) as index:
            systems = semantic_systems(index, args.embeddings, args.corpus_root)
            print(
                render_semantic_table(
                    evaluate_systems(index, queries, systems, args.k), list(systems), args.k
                )
            )
        return 0

    with open_index(args.index) as index:
        rows = evaluate(index, queries, scorers, args.k)
        if args.show:
            for q, _ in rows:
                print(f"\n== {q['query']}   (relevant: {', '.join(q['relevant'])})")
                for name, sc in scorers.items():
                    hits = search(index, q["query"], scorer=sc, k=args.k)
                    print(f"  {name}:")
                    for r in hits:
                        mark = "+" if is_relevant(index.doc_meta[r.doc_id], q["relevant"]) else " "
                        print(f"   {mark} {Path(index.doc_meta[r.doc_id].path).name}  {r.title}")
            print()

    print(f"| # | Query | Relevant | P@{args.k} TF-IDF | P@{args.k} BM25 |")
    print("| ---: | --- | ---: | ---: | ---: |")
    sums: dict[str, list[float]] = {n: [] for n in scorers}
    for i, (q, row) in enumerate(rows, start=1):
        cells: list[str] = []
        for n in scorers:
            v = row[n]
            cells.append("n/a" if v is None else f"{v:.2f}")
            if v is not None:
                sums[n].append(v)
        print(f"| {i} | `{q['query']}` | {len(q['relevant'])} | {cells[0]} | {cells[1]} |")
    means = [f"{sum(v) / len(v):.2f}" if v else "n/a" for v in sums.values()]
    print(f"| | **mean** | | **{means[0]}** | **{means[1]}** |")
    if any(v is None for _, row in rows for v in row.values()):
        print("\nn/a = phrase query on an index built without --positions (not in the mean)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
