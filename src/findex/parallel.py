"""Parallel index construction as map-reduce.

    paths -> plan_chunks -> [build_partial(chunk) ...]  (map, runs in workers)
                         -> merge(partials)             (reduce, runs in the parent)

* ``build_partial`` is a module-level function (picklable by reference) that receives
  **paths**, not documents: a worker reads its own files, so the only data crossing a process
  boundary is a few path strings going in and a compact partial index coming back.
* Every chunk gets its own range of ``doc_id`` up front, so partial indexes never collide and
  merging a term is plain concatenation of already sorted arrays.
* The same two functions run under a plain loop, a ``ThreadPoolExecutor`` or a
  ``ProcessPoolExecutor``; only ``run_chunks`` knows which.

Process pools use the ``spawn`` start method on every OS: a fresh interpreter re-imports this
module, so nothing may start workers at import time (see ``__main__.py`` and the entry point).
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import (
    Executor,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from findex.build import build_index
from findex.corpus import Document, list_documents, read_document
from findex.model import DocMeta, Index, PostingList

log = logging.getLogger(__name__)

ExecutorName = Literal["serial", "threads", "processes"]
EXECUTORS: tuple[ExecutorName, ...] = ("serial", "threads", "processes")

#: Below this much text, starting processes costs more than it saves (see the benchmark).
MIN_PARALLEL_BYTES = 2_000_000
#: What ``executor=None`` ("auto") picks for a big enough corpus. Chosen from the Lab 05
#: benchmark: processes win on a CPU-bound workload, threads do not (the GIL).
DEFAULT_PARALLEL_EXECUTOR: ExecutorName = "processes"


class MergeError(ValueError):
    """Partial indexes cannot be combined (overlapping doc_id ranges, mixed settings)."""


@dataclass(slots=True)
class PartialIndex:
    """The result of one worker: an index over a contiguous range of doc ids.

    It travels through a pipe (processes) as a pickle, so it is kept compact: postings are
    ``array('I')`` buffers, never lists of per-posting objects.
    """

    first_doc_id: int
    n_docs: int
    has_positions: bool
    postings: dict[str, PostingList]
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]
    worker_pid: int  # which process built it
    cpu_seconds: float  # that process's total CPU time when it finished (absolute)
    wall_seconds: float  # how long build_partial ran, measured inside the worker


@dataclass(frozen=True, slots=True)
class Chunk:
    """A contiguous slice of the corpus and the doc_id range reserved for it."""

    first_doc_id: int
    paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class BuildTimings:
    """Where the time went in one ``build_parallel`` call."""

    executor: ExecutorName
    workers: int
    chunks: int
    docs: int
    plan_s: float  # listing files, sizes, chunking
    map_s: float  # all build_partial calls, including pool start-up and result transfer
    merge_s: float  # the sequential reduce: the part Amdahl's law is about
    wall_s: float  # total wall-clock
    cpu_s: float  # CPU seconds: parent process + worker processes


# --------------------------------------------------------------------------- map side
def build_partial(
    doc_paths: Sequence[Path],
    first_doc_id: int = 0,
    *,
    positions: bool = False,
    on_doc: Callable[[], None] | None = None,
) -> PartialIndex:
    """Index the documents at ``doc_paths``; document *i* gets ``doc_id = first_doc_id + i``.

    Module-level on purpose: ``ProcessPoolExecutor`` sends a function to a worker *by name*
    (``findex.parallel.build_partial``); lambdas and nested functions have no importable name.
    ``on_doc`` (progress callback) only makes sense in-process; it cannot cross to a worker.
    """
    t0 = time.perf_counter()

    def documents() -> Iterator[Document]:
        for offset, path in enumerate(doc_paths):
            yield Document(first_doc_id + offset, path, read_document(path))
            if on_doc is not None:
                on_doc()

    index = build_index(documents(), storage="array", positions=positions)
    postings: dict[str, PostingList] = {}
    for term, plist in index.postings.items():
        if not isinstance(plist, PostingList):  # cannot happen with storage="array"
            raise TypeError(f"expected PostingList for {term!r}, got {type(plist).__name__}")
        postings[term] = plist
    return PartialIndex(
        first_doc_id=first_doc_id,
        n_docs=len(doc_paths),
        has_positions=positions,
        postings=postings,
        doc_lengths=index.doc_lengths,
        doc_meta=index.doc_meta,
        worker_pid=os.getpid(),
        cpu_seconds=time.process_time(),
        wall_seconds=time.perf_counter() - t0,
    )


def plan_chunks(
    paths: Sequence[Path], n_chunks: int, sizes: Sequence[int] | None = None
) -> list[Chunk]:
    """Cut ``paths`` into at most ``n_chunks`` contiguous, non-empty chunks.

    Cuts are placed so each chunk holds about the same number of *bytes* (``sizes``), not the
    same number of files: books differ in length by 100x, and one huge chunk would leave the
    other workers idle at the end (load imbalance). Chunk *k* owns doc ids
    ``first_doc_id .. first_doc_id + len(paths) - 1``; the ranges tile ``0 .. n-1`` exactly.
    """
    n = len(paths)
    if n == 0:
        return []
    weights = list(sizes) if sizes is not None else [1] * n
    if len(weights) != n:
        raise ValueError("sizes must have one entry per path")
    n_chunks = max(1, min(n_chunks, n))
    total = sum(weights) or n

    chunks: list[Chunk] = []
    start = 0
    acc = 0
    for i, w in enumerate(weights):
        acc += w
        made = len(chunks) + 1
        docs_left = n - (i + 1)
        if made < n_chunks and acc * n_chunks >= total * made and docs_left >= n_chunks - made:
            chunks.append(Chunk(start, tuple(paths[start : i + 1])))
            start = i + 1
    if start < n:
        chunks.append(Chunk(start, tuple(paths[start:])))
    return chunks


# ------------------------------------------------------------------------- reduce side
def merge(partials: Iterable[PartialIndex]) -> Index:
    """Combine partial indexes into one ``Index`` (storage ``"array"``).

    Ranges of doc ids are disjoint, so after ordering the partials by range, the postings of
    a term are simply the partials' arrays one after another -- already sorted. Terms enter the
    result in first-appearance order over the whole corpus, exactly like the serial build, so
    the saved file is byte-identical however the work was split.

    Consumes its input: the first partial that mentions a term donates its arrays (no copy),
    later ones are appended to them.
    """
    parts = sorted(partials, key=lambda p: p.first_doc_id)
    if not parts:
        return Index("array", False, {}, {}, {})
    has_positions = parts[0].has_positions
    prev_end = parts[0].first_doc_id
    for p in parts:
        if p.has_positions != has_positions:
            raise MergeError("cannot merge indexes built with and without positions")
        if p.first_doc_id < prev_end:
            raise MergeError(
                f"doc_id ranges overlap: a chunk starts at {p.first_doc_id} "
                f"but the previous one ends at {prev_end}"
            )
        if len(p.doc_meta) != p.n_docs:
            raise MergeError("a partial index does not hold the documents it claims")
        prev_end = p.first_doc_id + p.n_docs

    postings: dict[str, PostingList] = {}
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    for part in parts:
        for term, src in part.postings.items():
            dst = postings.get(term)
            if dst is None:
                postings[term] = src
                continue
            dst.doc_ids.extend(src.doc_ids)
            dst.tfs.extend(src.tfs)
            if dst.positions is not None and src.positions is not None:
                dst.positions.extend(src.positions)
        doc_lengths.update(part.doc_lengths)
        doc_meta.update(part.doc_meta)
    return Index("array", has_positions, dict(postings), doc_lengths, doc_meta)


# ---------------------------------------------------------------------- the executors
def run_chunks(
    chunks: Sequence[Chunk],
    *,
    executor: ExecutorName,
    workers: int,
    positions: bool = False,
    start_method: str = "spawn",
    on_chunk_done: Callable[[PartialIndex], None] | None = None,
    on_doc: Callable[[], None] | None = None,
) -> list[PartialIndex]:
    """Run ``build_partial`` on every chunk under the chosen executor.

    A worker that raises makes the whole call raise *the same exception* (for processes it
    arrives with the worker's traceback attached as ``__cause__``); the remaining queued
    chunks are cancelled instead of being left to run. A worker that dies outright surfaces as
    ``BrokenProcessPool``. Neither case hangs.
    """
    if executor == "serial":
        done: list[PartialIndex] = []
        for chunk in chunks:
            part = build_partial(
                chunk.paths, chunk.first_doc_id, positions=positions, on_doc=on_doc
            )
            done.append(part)
            if on_chunk_done is not None:
                on_chunk_done(part)
        return done

    workers = max(1, min(workers, len(chunks)))
    pool: Executor
    if executor == "threads":
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="findex")
    elif executor == "processes":
        pool = ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context(start_method)
        )
    else:
        raise ValueError(f"unknown executor {executor!r}; choose from {EXECUTORS}")

    results: list[PartialIndex] = []
    with pool:
        futures = [
            pool.submit(build_partial, c.paths, c.first_doc_id, positions=positions) for c in chunks
        ]
        try:
            for fut in as_completed(futures):
                part = fut.result()  # re-raises the worker's exception right here
                results.append(part)
                if on_chunk_done is not None:
                    on_chunk_done(part)
        except BaseException:
            for f in futures:
                f.cancel()
            raise
    return results


def default_workers() -> int:
    """Usable CPUs: honours CPU affinity / container limits where the OS reports them."""
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        return max(1, len(affinity(0)))
    return os.cpu_count() or 1


def resolve_executor(
    requested: ExecutorName | None, workers: int, total_bytes: int
) -> ExecutorName:
    """``None`` means auto: processes for a corpus big enough to repay their start-up cost."""
    if requested is not None:
        return requested
    if workers > 1 and total_bytes >= MIN_PARALLEL_BYTES:
        return DEFAULT_PARALLEL_EXECUTOR
    return "serial"


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:  # vanished or unreadable: build_partial will report it properly
        return 1


def _worker_cpu(partials: Sequence[PartialIndex]) -> float:
    """CPU seconds spent by worker *processes* (not the parent). Each worker reports its
    absolute process CPU time after every task, so the last/largest value per pid is its total."""
    me = os.getpid()
    per_pid: dict[int, float] = {}
    for p in partials:
        if p.worker_pid != me:
            per_pid[p.worker_pid] = max(per_pid.get(p.worker_pid, 0.0), p.cpu_seconds)
    return sum(per_pid.values())


def build_parallel(
    root: Path,
    *,
    executor: ExecutorName | None = None,
    workers: int | None = None,
    positions: bool = False,
    limit: int | None = None,
    chunks_per_worker: int = 2,
    start_method: str = "spawn",
    on_chunk_done: Callable[[PartialIndex], None] | None = None,
    on_doc: Callable[[], None] | None = None,
) -> tuple[Index, BuildTimings]:
    """Build an index of the ``*.txt`` files below ``root`` and report where the time went.

    ``executor=None`` chooses automatically (see ``resolve_executor``). More chunks than
    workers (``chunks_per_worker``) lets fast workers pick up extra chunks, which evens out
    uneven file sizes, at the price of a larger merge. Progress: ``on_doc`` fires per document
    (serial only, it cannot cross a process boundary), ``on_chunk_done`` per finished chunk.
    """
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    paths = list_documents(root)[:limit]
    sizes = [_file_size(p) for p in paths]
    n_workers = max(1, workers if workers is not None else default_workers())
    chosen = resolve_executor(executor, n_workers, sum(sizes))
    if chosen == "serial":
        n_workers = 1
    chunks = plan_chunks(paths, 1 if chosen == "serial" else n_workers * chunks_per_worker, sizes)
    t_plan = time.perf_counter()
    log.info(
        "%d documents in %d chunks; executor=%s workers=%d",
        len(paths),
        len(chunks),
        chosen,
        n_workers,
    )

    partials = run_chunks(
        chunks,
        executor=chosen,
        workers=n_workers,
        positions=positions,
        start_method=start_method,
        on_chunk_done=None if chosen == "serial" and on_doc is not None else on_chunk_done,
        on_doc=on_doc if chosen == "serial" else None,
    )
    t_map = time.perf_counter()
    worker_cpu = _worker_cpu(partials)
    index = merge(partials)
    t_end = time.perf_counter()

    timings = BuildTimings(
        executor=chosen,
        workers=n_workers,
        chunks=len(chunks),
        docs=len(paths),
        plan_s=t_plan - t0,
        map_s=t_map - t_plan,
        merge_s=t_end - t_map,
        wall_s=t_end - t0,
        cpu_s=(time.process_time() - cpu0) + worker_cpu,
    )
    log.info(
        "wall %.2fs = plan %.2f + map %.2f + merge %.2f; cpu %.2fs",
        timings.wall_s,
        timings.plan_s,
        timings.map_s,
        timings.merge_s,
        timings.cpu_s,
    )
    return index, timings
