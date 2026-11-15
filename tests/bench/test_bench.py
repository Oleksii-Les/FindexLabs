"""pytest-benchmark: ``build_index`` and ``search`` on a fixed synthetic corpus, so a slowdown shows up in CI.

    uv run pytest tests/bench --benchmark-enable --benchmark-only          # numbers (ordinary runs only execute once)

The corpus is generated from a seed (Zipf-like word frequencies, like real text), never read from disk, so the
workload is identical on every machine. Absolute times depend on the machine; the CI job therefore also keeps
the *ratio* NumPy / pure Python (``test_numpy_stays_much_faster_than_the_loop``), which does not.
"""

from __future__ import annotations

import random
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from pytest_benchmark.fixture import BenchmarkFixture  # type: ignore[import-untyped]

from findex.build import build_index
from findex.corpus import Document
from findex.engine import search
from findex.model import Index
from findex.scoring import BM25
from findex.vector import VectorIndex

N_DOCS, WORDS_PER_DOC = 2_000, 120
QUERY = "w0 OR w3 OR w17"  # frequent words: thousands of postings, the case NumPy is for


def synthetic_docs(n: int = N_DOCS, seed: int = 11) -> list[Document]:
    rng = random.Random(seed)
    vocabulary = [f"w{i}" for i in range(3_000)]
    weights = [1 / (r + 1) for r in range(len(vocabulary))]
    return [
        Document(i, Path(f"{i}.txt"), " ".join(rng.choices(vocabulary, weights, k=WORDS_PER_DOC)))
        for i in range(n)
    ]


@pytest.fixture(scope="module")
def docs() -> list[Document]:
    return synthetic_docs()


@pytest.fixture(scope="module")
def index(docs: list[Document]) -> Iterator[Index]:
    yield build_index(iter(docs), storage="array", positions=False)


@pytest.mark.benchmark(group="build")
def test_bench_build_index(benchmark: BenchmarkFixture, docs: list[Document]) -> None:
    built = benchmark(lambda: build_index(iter(docs), storage="array", positions=True))
    assert built.num_docs == N_DOCS


@pytest.mark.benchmark(group="search")
def test_bench_search_python_loop(benchmark: BenchmarkFixture, index: Index) -> None:
    hits = benchmark(lambda: search(index, QUERY, scorer=BM25(), k=10))
    assert len(hits) == 10


@pytest.mark.benchmark(group="search")
def test_bench_search_numpy(benchmark: BenchmarkFixture, index: Index) -> None:
    vectors = VectorIndex(index)
    hits = benchmark(lambda: search(index, QUERY, scorer=BM25(), k=10, vectors=vectors))
    assert len(hits) == 10


def best_of(fn, repeat: int = 7) -> float:  # type: ignore[no-untyped-def]
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def test_numpy_stays_much_faster_than_the_loop(index: Index) -> None:
    """A regression guard that is independent of the machine: both sides run on the same one."""
    vectors = VectorIndex(index)
    slow = best_of(lambda: search(index, QUERY, scorer=BM25(), k=10))
    fast = best_of(lambda: search(index, QUERY, scorer=BM25(), k=10, vectors=vectors))
    assert slow / fast >= 3, (
        f"NumPy only {slow / fast:.1f}x faster ({slow * 1000:.1f} ms vs {fast * 1000:.1f} ms)"
    )
