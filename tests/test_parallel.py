"""build_partial / merge / executors: parallel and serial builds must be the SAME index."""

import pickle
import random
import subprocess
import sys
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from findex.build import build_index
from findex.corpus import iter_documents, list_documents
from findex.model import Index, PostingList
from findex.parallel import (
    MIN_PARALLEL_BYTES,
    Chunk,
    MergeError,
    PartialIndex,
    build_parallel,
    build_partial,
    default_workers,
    merge,
    plan_chunks,
    resolve_executor,
    run_chunks,
)
from findex.store import save


def classic(root: Path, positions: bool) -> Index:
    """The Lab 04 way: one lazy pass over iter_documents."""
    return build_index(iter_documents(root), storage="array", positions=positions)


def saved_bytes(index: Index, where: Path) -> bytes:
    save(index, where, "binary")
    return where.read_bytes()


# ------------------------------------------------------------------ stable corpus order
def test_list_documents_is_sorted_and_matches_iter_documents(tiny_corpus: Path):
    paths = list_documents(tiny_corpus)
    assert paths == sorted(paths) and len(paths) == 6
    assert [d.path for d in iter_documents(tiny_corpus)] == paths
    assert [d.doc_id for d in iter_documents(tiny_corpus)] == list(range(6))


def test_a_missing_corpus_directory_is_an_error_not_an_empty_index(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="not found"):
        list_documents(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        build_parallel(tmp_path / "nope", executor="serial")


# ------------------------------------------------------------------------ plan_chunks
@pytest.mark.parametrize(
    ("n_paths", "n_chunks", "expected_sizes"),
    [(10, 1, [10]), (10, 2, [5, 5]), (10, 3, [4, 3, 3]), (3, 8, [1, 1, 1]), (0, 4, [])],
)
def test_plan_chunks_equal_weights(n_paths: int, n_chunks: int, expected_sizes: list[int]):
    paths = [Path(f"{i}.txt") for i in range(n_paths)]
    chunks = plan_chunks(paths, n_chunks)
    assert [len(c.paths) for c in chunks] == expected_sizes


def test_plan_chunks_balances_bytes_not_file_counts():
    paths = [Path(f"{i}.txt") for i in range(9)]
    sizes = [1000, 1, 1, 1, 1, 1, 1, 1, 1]  # one huge book first
    chunks = plan_chunks(paths, 3, sizes)
    assert len(chunks[0].paths) == 1  # the huge file stands alone instead of dragging 3 files in
    assert sum(len(c.paths) for c in chunks) == 9


def test_plan_chunks_rejects_mismatched_sizes():
    with pytest.raises(ValueError, match="one entry per path"):
        plan_chunks([Path("a")], 2, [1, 2])


@given(
    sizes=st.lists(st.integers(0, 10_000), min_size=0, max_size=60),
    n_chunks=st.integers(1, 20),
)
def test_chunks_tile_the_corpus_exactly(sizes: list[int], n_chunks: int):
    paths = [Path(f"{i}.txt") for i in range(len(sizes))]
    chunks = plan_chunks(paths, n_chunks, sizes)
    assert [p for c in chunks for p in c.paths] == paths  # nothing lost, nothing repeated
    assert all(c.paths for c in chunks)  # no empty chunk
    assert len(chunks) <= max(1, min(n_chunks, len(paths))) or not paths
    expected_first = 0
    for c in chunks:  # doc_id ranges are contiguous and disjoint
        assert c.first_doc_id == expected_first
        expected_first += len(c.paths)


# ---------------------------------------------------------------------- build_partial
def test_build_partial_assigns_its_own_doc_id_range(tiny_corpus: Path):
    paths = list_documents(tiny_corpus)[2:5]
    part = build_partial(paths, first_doc_id=100, positions=True)
    assert (part.first_doc_id, part.n_docs, part.has_positions) == (100, 3, True)
    assert sorted(part.doc_meta) == [100, 101, 102]
    assert all(isinstance(p, PostingList) for p in part.postings.values())
    assert all(100 <= d <= 102 for p in part.postings.values() for d in p.doc_ids)


def test_build_partial_reports_progress_per_document(tiny_corpus: Path):
    seen: list[int] = []
    build_partial(list_documents(tiny_corpus), on_doc=lambda: seen.append(1))
    assert len(seen) == 6


def test_partial_index_survives_pickling(tiny_corpus: Path):
    """This is exactly what a process worker does with its result."""
    part = build_partial(list_documents(tiny_corpus), 7, positions=True)
    back: PartialIndex = pickle.loads(pickle.dumps(part))
    assert back.postings == part.postings
    assert (back.doc_lengths, back.doc_meta) == (part.doc_lengths, part.doc_meta)
    assert merge([back]) == merge([part])


def test_empty_chunk_is_fine():
    part = build_partial([], 5)
    assert part.n_docs == 0 and part.postings == {}
    assert len(merge([part])) == 0


# ---------------------------------------------------------------------------- merge
@pytest.mark.parametrize("positions", [False, True])
def test_serial_pipeline_equals_the_lab4_builder(
    tiny_corpus: Path, tmp_path: Path, positions: bool
):
    paths = list_documents(tiny_corpus)
    via_partials = merge([build_partial(paths, 0, positions=positions)])
    reference = classic(tiny_corpus, positions)
    assert via_partials == reference  # same terms, same postings
    assert via_partials.doc_lengths == reference.doc_lengths
    assert via_partials.doc_meta == reference.doc_meta
    # ... and the SAME FILE, byte for byte
    assert saved_bytes(via_partials, tmp_path / "a") == saved_bytes(reference, tmp_path / "b")


@pytest.mark.parametrize("positions", [False, True])
@pytest.mark.parametrize("n_chunks", [2, 3, 6])
def test_any_chunking_gives_the_same_bytes(
    tiny_corpus: Path, tmp_path: Path, positions: bool, n_chunks: int
):
    paths = list_documents(tiny_corpus)
    partials = [
        build_partial(c.paths, c.first_doc_id, positions=positions)
        for c in plan_chunks(paths, n_chunks)
    ]
    assert saved_bytes(merge(partials), tmp_path / "m") == saved_bytes(
        classic(tiny_corpus, positions), tmp_path / "c"
    )


def test_merge_does_not_depend_on_arrival_order(tiny_corpus: Path, tmp_path: Path):
    """Futures finish in any order (as_completed); merge sorts partials by doc_id range."""
    chunks = plan_chunks(list_documents(tiny_corpus), 3)
    in_order = [build_partial(c.paths, c.first_doc_id) for c in chunks]
    shuffled = [build_partial(c.paths, c.first_doc_id) for c in chunks]
    random.Random(1).shuffle(shuffled)
    assert saved_bytes(merge(in_order), tmp_path / "x") == saved_bytes(
        merge(shuffled), tmp_path / "y"
    )


def test_merge_rejects_overlapping_ranges(tiny_corpus: Path):
    paths = list_documents(tiny_corpus)
    a = build_partial(paths[:3], 0)
    b = build_partial(paths[3:], 2)  # starts inside a's range
    with pytest.raises(MergeError, match="overlap"):
        merge([a, b])


def test_merge_rejects_mixed_positions(tiny_corpus: Path):
    paths = list_documents(tiny_corpus)
    with pytest.raises(MergeError, match="positions"):
        merge([build_partial(paths[:3], 0, positions=True), build_partial(paths[3:], 3)])


def test_merge_rejects_a_partial_that_lies_about_its_size(tiny_corpus: Path):
    part = build_partial(list_documents(tiny_corpus), 0)
    part.n_docs = 99
    with pytest.raises(MergeError, match="claims"):
        merge([part])


def test_merge_of_nothing_is_an_empty_index():
    assert len(merge([])) == 0


# --------------------------------------------------------------------- the executors
@pytest.mark.parametrize(("executor", "workers"), [("serial", 1), ("threads", 1), ("threads", 3)])
@pytest.mark.parametrize("positions", [False, True])
def test_executors_give_identical_files(
    tiny_corpus: Path, tmp_path: Path, executor: str, workers: int, positions: bool
):
    index, timings = build_parallel(
        tiny_corpus,
        executor=executor,
        workers=workers,
        positions=positions,
    )
    assert timings.executor == executor and timings.docs == 6
    assert saved_bytes(index, tmp_path / "p") == saved_bytes(
        classic(tiny_corpus, positions), tmp_path / "c"
    )


@pytest.mark.slow
@pytest.mark.parametrize("positions", [False, True])
def test_process_pool_gives_identical_files(tiny_corpus: Path, tmp_path: Path, positions: bool):
    index, timings = build_parallel(
        tiny_corpus, executor="processes", workers=2, positions=positions
    )
    assert timings.executor == "processes" and timings.workers == 2
    assert saved_bytes(index, tmp_path / "p") == saved_bytes(
        classic(tiny_corpus, positions), tmp_path / "c"
    )


def test_build_parallel_timings_add_up(tiny_corpus: Path):
    _, t = build_parallel(tiny_corpus, executor="threads", workers=2)
    assert t.wall_s >= t.plan_s + t.map_s + t.merge_s - 1e-3
    assert t.cpu_s > 0 and t.chunks == 4 and t.workers == 2


def test_limit_takes_the_first_documents(tiny_corpus: Path):
    index, t = build_parallel(tiny_corpus, executor="serial", limit=2)
    assert t.docs == 2 and index.num_docs == 2 and sorted(index.doc_meta) == [0, 1]


def test_progress_callbacks(tiny_corpus: Path):
    per_doc: list[int] = []
    build_parallel(tiny_corpus, executor="serial", on_doc=lambda: per_doc.append(1))
    assert len(per_doc) == 6
    per_chunk: list[int] = []
    build_parallel(
        tiny_corpus,
        executor="threads",
        workers=2,
        on_chunk_done=lambda p: per_chunk.append(p.n_docs),
    )
    assert sum(per_chunk) == 6


def test_unknown_executor_is_rejected():
    with pytest.raises(ValueError, match="unknown executor"):
        run_chunks([Chunk(0, (Path("x"),))], executor="gpu", workers=2)


# ------------------------------------------------------------------ failing workers
BAD_CHUNKS = [
    Chunk(0, (Path("/definitely/not/here.txt"),)),
    Chunk(1, (Path("/neither/here.txt"),)),
]


@pytest.mark.parametrize("executor", ["serial", "threads"])
def test_worker_exception_reaches_the_caller(executor: str):
    with pytest.raises(FileNotFoundError):
        run_chunks(BAD_CHUNKS, executor=executor, workers=2)


@pytest.mark.slow
def test_process_worker_exception_arrives_with_its_remote_traceback():
    with pytest.raises(FileNotFoundError) as info:
        run_chunks(BAD_CHUNKS, executor="processes", workers=2)
    cause = str(info.value.__cause__)
    assert "Traceback" in cause and "build_partial" in cause  # the WORKER's stack, not a hang


# ---------------------------------------------------------------- executor selection
@pytest.mark.parametrize(
    ("requested", "workers", "nbytes", "expected"),
    [
        ("threads", 4, 10, "threads"),  # an explicit choice is always honoured
        ("serial", 8, 10**9, "serial"),
        (None, 8, MIN_PARALLEL_BYTES, "processes"),  # big enough: processes win
        (None, 8, MIN_PARALLEL_BYTES - 1, "serial"),  # too small to repay start-up cost
        (None, 1, 10**9, "serial"),  # one worker is just serial
    ],
)
def test_resolve_executor(requested: str | None, workers: int, nbytes: int, expected: str):
    assert resolve_executor(requested, workers, nbytes) == expected


def test_default_workers_is_positive():
    assert default_workers() >= 1


# ----------------------------------------------------------------------- spawn safety
def test_importing_the_main_module_does_not_start_the_cli():
    """Under 'spawn' every worker re-imports __main__: it must be side-effect free."""
    done = subprocess.run(
        [sys.executable, "-c", "import findex.__main__"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0 and done.stdout == "" and done.stderr == ""


@pytest.mark.slow
def test_python_dash_m_findex_index_with_processes_does_not_recurse(
    tiny_corpus: Path, tmp_path: Path
):
    out = tmp_path / "i.bin"
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "findex",
            "index",
            str(tiny_corpus),
            "--out",
            str(out),
            "--executor",
            "processes",
            "--workers",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "processes x 2" in done.stdout and out.stat().st_size > 0
