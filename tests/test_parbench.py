"""The benchmark tool: statistics, table, plot, and the child-process protocol."""

import json
import sys
from pathlib import Path

import pytest

from findex import parbench
from findex.parbench import (
    ChildResult,
    Config,
    RunRecord,
    amdahl_fraction,
    default_worker_counts,
    gil_enabled,
    measure_build,
    peak_rss_bytes,
    plan_matrix,
    render_table,
    summarize,
)


def record(
    label: str, executor: str, workers: int, run: int, wall: float, **extra: float
) -> RunRecord:
    base: ChildResult = {
        "python": "3.12.3",
        "gil_enabled": True,
        "executor": executor,
        "workers": workers,
        "chunks": workers * 2,
        "docs": 100,
        "terms": 1000,
        "postings": 5000,
        "wall_s": wall,
        "cpu_s": extra.get("cpu", wall),
        "plan_s": extra.get("plan", 0.1),
        "map_s": wall - extra.get("merge", 0.5),
        "merge_s": extra.get("merge", 0.5),
        "rss_bytes": 100 * 2**20,
    }
    return {**base, "label": label, "run": run}


FAKE_RUNS = [
    # serial: 10 s, three runs (median 10)
    *[record("A", "serial", 1, i, w) for i, w in enumerate([9.0, 10.0, 12.0])],
    # processes x4: 3 s (median), merge 0.5 s
    *[record("A", "processes", 4, i, w, cpu=11.0) for i, w in enumerate([2.9, 3.0, 3.4])],
    # threads x4 on the GIL build: slower than serial
    *[record("A", "threads", 4, i, w) for i, w in enumerate([10.5, 11.0, 11.5])],
]


def test_summarize_takes_the_median_and_computes_speedup():
    rows = {(r.executor, r.workers): r for r in summarize(FAKE_RUNS)}
    assert rows[("serial", 1)].wall == 10.0 and rows[("serial", 1)].speedup == pytest.approx(1.0)
    assert rows[("processes", 4)].wall == 3.0
    assert rows[("processes", 4)].speedup == pytest.approx(10.0 / 3.0)
    assert rows[("processes", 4)].cpu == 11.0 and rows[("processes", 4)].n == 3
    assert (rows[("serial", 1)].wall_min, rows[("serial", 1)].wall_max) == (9.0, 12.0)
    assert rows[("threads", 4)].speedup < 1.0  # threads under the GIL: no gain


def test_summarize_orders_serial_threads_processes():
    order = [(r.executor, r.workers) for r in summarize(FAKE_RUNS)]
    assert order == [("serial", 1), ("threads", 4), ("processes", 4)]


def test_each_label_is_compared_with_its_own_serial_baseline():
    other = [record("B", "serial", 1, 0, 20.0), record("B", "threads", 2, 0, 10.0)]
    rows = {(r.label, r.executor): r for r in summarize([*FAKE_RUNS, *other])}
    assert rows[("B", "threads")].speedup == pytest.approx(2.0)  # 20 / 10, not 10 / 10
    assert rows[("A", "processes")].speedup == pytest.approx(10.0 / 3.0)


def test_render_table_has_the_labs_columns():
    table = render_table(summarize(FAKE_RUNS))
    header = table.splitlines()[0]
    for column in ("Executor", "Workers", "Wall", "CPU", "Peak RSS", "Merge", "Speedup"):
        assert column in header
    assert "| processes | 4 | 3.00 |" in table and "3.33x" in table and "100 MiB" in table
    assert "GIL on" in table


def test_amdahl_fraction_is_plan_plus_merge_over_serial_wall():
    rows = summarize(FAKE_RUNS)
    # (plan 0.1 + merge 0.5) / serial 10.0
    assert amdahl_fraction(rows) == pytest.approx(0.06)
    assert amdahl_fraction([r for r in rows if r.executor != "processes"]) is None


def test_plan_matrix_runs_serial_once():
    configs = plan_matrix(["serial", "threads", "processes"], [4, 1, 2, 2])
    assert configs[0] == Config("serial", 1)
    assert [c.workers for c in configs if c.executor == "threads"] == [1, 2, 4]
    assert len(configs) == 1 + 3 + 3


def test_default_worker_counts_follow_the_labs_grid():
    counts = default_worker_counts()
    assert counts == sorted(set(counts)) and {1, 2, 4, 8} <= set(counts)


def test_environment_probes():
    assert isinstance(gil_enabled(), bool)
    rss = peak_rss_bytes()
    if sys.platform != "win32":  # the Windows branch is exercised on a Windows machine
        assert rss is not None and rss > 10 * 2**20


def test_measure_build_in_process(tiny_corpus: Path):
    result = measure_build(tiny_corpus, executor="threads", workers=2)
    assert result["docs"] == 6 and result["executor"] == "threads" and result["workers"] == 2
    assert result["terms"] > 10 and result["postings"] >= result["terms"]
    assert result["wall_s"] > 0 and result["gil_enabled"] in (True, False)


def test_child_protocol_prints_one_json_line(tiny_corpus: Path, capsys: pytest.CaptureFixture[str]):
    code = parbench.main(["child", str(tiny_corpus), "--executor", "serial", "--workers", "1"])
    assert code == 0
    (line,) = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("{")]
    assert json.loads(line)["docs"] == 6


@pytest.mark.slow
def test_run_then_report_end_to_end(
    tiny_corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    pytest.importorskip("matplotlib")
    out = tmp_path / "bench" / "r.json"
    code = parbench.main(
        [
            "run",
            str(tiny_corpus),
            "--out",
            str(out),
            "--label",
            "test",
            "--executors",
            "serial",
            "threads",
            "--workers",
            "1",
            "2",
            "--runs",
            "2",
            "--no-warmup",
        ]
    )
    assert code == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert len(saved["runs"]) == 2 * 3  # 2 runs x (serial + threads x1 + threads x2)
    assert {"cpu", "os", "logical_cpus"} <= saved["machine"].keys()
    assert "| threads | 2 |" in capsys.readouterr().out

    png = tmp_path / "plot" / "speedup.png"
    assert parbench.main(["report", str(out), "--plot", str(png), "--title", "t"]) == 0
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert "Machine:" in capsys.readouterr().out


def test_run_rejects_a_missing_corpus(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    code = parbench.main(["run", str(tmp_path / "nope"), "--out", str(tmp_path / "o.json")])
    assert code == 1 and "corpus directory not found" in capsys.readouterr().err


def test_a_failing_child_reports_its_stderr(tmp_path: Path):
    with pytest.raises(RuntimeError, match="child failed"):
        parbench.run_child(
            sys.executable, tmp_path / "missing", Config("serial", 1), parbench.ChildSettings()
        )


def test_plot_speedup_writes_a_png(tmp_path: Path):
    pytest.importorskip("matplotlib")
    png = tmp_path / "s.png"
    parbench.plot_speedup(summarize(FAKE_RUNS), png, "title")
    assert png.stat().st_size > 1000
