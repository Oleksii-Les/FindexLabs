"""Lab 05: the race that the GIL hides, shown on the real indexing workload.

Several threads count tokens of different documents into ONE shared structure. The correct
total is known (a serial count), so any difference is a lost update.

    uv run python experiments/race_demo.py data/gutenberg --docs 40 --threads 8
    uv run --python 3.13t python experiments/race_demo.py data/gutenberg --docs 40 --threads 8

Variants:
  A  shared Counter, ``counts.update(tokens)``            C-level loop, GIL hides the race
  B  shared dict,    ``d[t] = d.get(t, 0) + 1``           read ... write: a real race
  C  shared dict, one Lock around every document          correct, but threads queue up
  D  one Counter per thread, merged at the end             correct and lock-free (= build_partial/merge)
"""

from __future__ import annotations

import argparse
import sys
import threading
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def load_texts(root: Path, n_docs: int) -> list[str]:
    texts: list[str] = []
    for doc in iter_documents(root):
        texts.append(doc.text)
        if len(texts) == n_docs:
            break
    return texts


def run_threads(texts: list[str], n_threads: int, worker: Callable[[list[str]], None]) -> None:
    slices = [texts[i::n_threads] for i in range(n_threads)]
    threads = [threading.Thread(target=worker, args=(s,)) for s in slices]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def variant_a(texts: list[str], n_threads: int) -> int:
    shared: Counter[str] = Counter()

    def worker(chunk: list[str]) -> None:
        for text in chunk:
            shared.update(tokenize(text))

    run_threads(texts, n_threads, worker)
    return shared.total()


def variant_b(texts: list[str], n_threads: int) -> int:
    shared: dict[str, int] = {}

    def worker(chunk: list[str]) -> None:
        for text in chunk:
            for tok in tokenize(text):
                shared[tok] = shared.get(tok, 0) + 1  # read, add, write: three steps

    run_threads(texts, n_threads, worker)
    return sum(shared.values())


def variant_c(texts: list[str], n_threads: int) -> int:
    shared: dict[str, int] = {}
    lock = threading.Lock()

    def worker(chunk: list[str]) -> None:
        for text in chunk:
            with lock:
                for tok in tokenize(text):
                    shared[tok] = shared.get(tok, 0) + 1

    run_threads(texts, n_threads, worker)
    return sum(shared.values())


def variant_d(texts: list[str], n_threads: int) -> int:
    partials: list[Counter[str]] = [Counter() for _ in range(n_threads)]

    def make_worker(i: int) -> Callable[[list[str]], None]:
        def worker(chunk: list[str]) -> None:
            for text in chunk:
                partials[i].update(tokenize(text))  # touches only its own Counter

        return worker

    slices = [texts[i::n_threads] for i in range(n_threads)]
    threads = [threading.Thread(target=make_worker(i), args=(s,)) for i, s in enumerate(slices)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    merged: Counter[str] = Counter()
    for p in partials:
        merged.update(p)  # the sequential 'reduce' step
    return merged.total()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--docs", type=int, default=40)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    texts = load_texts(args.corpus, args.docs)
    expected = sum(1 for text in texts for _ in tokenize(text))
    probe = getattr(sys, "_is_gil_enabled", None)
    gil = "always on" if probe is None else str(probe())
    print(f"Python {sys.version.split()[0]}  GIL enabled: {gil}  threads: {args.threads}")
    print(f"{len(texts)} documents, correct total = {expected:,} tokens, {args.repeat} runs each\n")
    for name, fn in (
        ("A shared Counter.update", variant_a),
        ("B shared dict get/set", variant_b),
        ("C shared dict + Lock", variant_c),
        ("D per-thread Counters + merge", variant_d),
    ):
        totals = [fn(texts, args.threads) for _ in range(args.repeat)]
        worst = expected - min(totals)
        verdict = "OK" if worst == 0 else f"LOST UPDATES (up to {worst:,} tokens)"
        print(f"  {name:32} {verdict}")


if __name__ == "__main__":
    main()
