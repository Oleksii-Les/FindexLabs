"""Benchmark parallel indexing: serial vs threads vs processes, 1..N workers.

    python -m findex.parbench run    data/gutenberg --out bench/gil.json  --label "3.12 (GIL)"
    python -m findex.parbench run    data/gutenberg --out bench/nogil.json --label "3.13t (no GIL)" \\
            --python C:/path/to/python3.13t.exe --gil 0 --executors threads
    python -m findex.parbench report bench/gil.json bench/nogil.json --plot docs/speedup.png

Method (Lab 05, section "Benchmarking honestly"):

* every measurement is a **fresh child process**, so peak memory is that run's own and
  nothing is warm from the previous run except the operating system's file cache;
* one unmeasured warm-up run first (fills the file cache), then ``--runs`` measured runs per
  configuration, interleaved round-robin so slow drift (heat, background jobs) is spread over
  all configurations instead of punishing the last one; the table reports the **median**;
* wall time and CPU time are separate columns: the gap between them is the cost of parallelism.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import statistics
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

from findex.logging_setup import configure_logging
from findex.parallel import EXECUTORS, ExecutorName, build_parallel, default_workers

log = logging.getLogger(__name__)

# The directory that contains the ``findex`` package. Children get it on PYTHONPATH, so they
# can run under ANY interpreter (for example python3.13t) without installing findex into it.
_PACKAGE_PARENT = Path(__file__).resolve().parents[1]


class ChildResult(TypedDict):
    """What one measured build reports (one JSON line on the child's stdout)."""

    python: str
    gil_enabled: bool
    executor: str
    workers: int
    chunks: int
    docs: int
    terms: int
    postings: int
    wall_s: float
    cpu_s: float
    plan_s: float
    map_s: float
    merge_s: float
    rss_bytes: int | None


class RunRecord(ChildResult):
    """A child result tagged by the orchestrator."""

    label: str
    run: int


# ------------------------------------------------------------------------ measuring
def peak_rss_bytes() -> int | None:
    """Peak resident memory of THIS process, or None if the OS gives no way to ask."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            get_process = ctypes.windll.kernel32.GetCurrentProcess
            get_process.restype = wintypes.HANDLE
            get_info = ctypes.windll.psapi.GetProcessMemoryInfo
            get_info.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            get_info.restype = wintypes.BOOL
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(ProcessMemoryCounters)
            if get_info(get_process(), ctypes.byref(counters), counters.cb):
                return int(counters.PeakWorkingSetSize)
        except Exception:  # best effort: a missing number must not break a benchmark
            log.debug("could not read the peak working set", exc_info=True)
        return None
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024  # Linux reports KiB, macOS bytes


def gil_enabled() -> bool:
    """False only on a free-threaded build that is really running without the GIL."""
    probe = getattr(sys, "_is_gil_enabled", None)  # exists since Python 3.13
    return True if probe is None else bool(probe())


def measure_build(
    corpus: Path,
    *,
    executor: ExecutorName,
    workers: int,
    positions: bool = False,
    limit: int | None = None,
    chunks_per_worker: int = 2,
) -> ChildResult:
    """Build once in THIS process and report timings (this is what a child runs)."""
    index, t = build_parallel(
        corpus,
        executor=executor,
        workers=workers,
        positions=positions,
        limit=limit,
        chunks_per_worker=chunks_per_worker,
    )
    return {
        "python": platform.python_version() + ("t" if "free-threading" in sys.version else ""),
        "gil_enabled": gil_enabled(),
        "executor": t.executor,
        "workers": t.workers,
        "chunks": t.chunks,
        "docs": t.docs,
        "terms": len(index),
        "postings": index.n_postings(),
        "wall_s": t.wall_s,
        "cpu_s": t.cpu_s,
        "plan_s": t.plan_s,
        "map_s": t.map_s,
        "merge_s": t.merge_s,
        "rss_bytes": peak_rss_bytes(),
    }


# ----------------------------------------------------------------------- orchestrate
@dataclass(frozen=True, slots=True)
class Config:
    executor: ExecutorName
    workers: int


@dataclass(frozen=True, slots=True)
class ChildSettings:
    """Everything about a measured run except which executor / how many workers."""

    positions: bool = False
    limit: int | None = None
    chunks_per_worker: int = 2
    gil: str | None = None  # value for PYTHON_GIL in the child ("0" = off on a 3.13t build)
    timeout: float = 3600.0


def plan_matrix(executors: Sequence[ExecutorName], worker_counts: Sequence[int]) -> list[Config]:
    """serial runs once (1 worker); the others once per worker count."""
    configs: list[Config] = []
    for ex in executors:
        counts = [1] if ex == "serial" else sorted(set(worker_counts))
        configs.extend(Config(ex, w) for w in counts)
    return configs


def default_worker_counts() -> list[int]:
    """1, 2, 4, 8 and the number of usable CPUs (the lab's grid), without duplicates."""
    return sorted({1, 2, 4, 8, default_workers()})


def run_child(
    python: str,
    corpus: Path,
    config: Config,
    settings: ChildSettings,
) -> ChildResult:
    cmd = [
        python,
        "-m",
        "findex.parbench",
        "child",
        str(corpus),
        "--executor",
        config.executor,
        "--workers",
        str(config.workers),
        "--chunks-per-worker",
        str(settings.chunks_per_worker),
    ]
    if settings.positions:
        cmd.append("--positions")
    if settings.limit is not None:
        cmd += ["--limit", str(settings.limit)]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_PACKAGE_PARENT), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    env["PYTHONUTF8"] = "1"
    if settings.gil is not None:
        env["PYTHON_GIL"] = settings.gil  # "0" turns the GIL off on a free-threaded build
    done = subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=settings.timeout, check=False
    )
    if done.returncode != 0:
        raise RuntimeError(
            f"child failed ({config.executor} x {config.workers}, exit {done.returncode}):\n"
            f"{done.stderr.strip()}"
        )
    lines = [ln for ln in done.stdout.splitlines() if ln.startswith("{")]
    if not lines:
        raise RuntimeError(f"child printed no result:\n{done.stdout}\n{done.stderr}")
    return cast(ChildResult, json.loads(lines[-1]))


def machine_info() -> dict[str, str | int]:
    cpu = platform.processor() or "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {
        "cpu": cpu,
        "logical_cpus": os.cpu_count() or 0,
        "usable_cpus": default_workers(),
        "os": platform.platform(),
    }


# -------------------------------------------------------------------------- reporting
@dataclass(frozen=True, slots=True)
class Row:
    label: str
    python: str
    gil: bool
    executor: str
    workers: int
    n: int
    wall: float
    wall_min: float
    wall_max: float
    cpu: float
    merge: float
    plan: float
    rss: int | None
    speedup: float


_EXECUTOR_ORDER = {name: i for i, name in enumerate(EXECUTORS)}


def summarize(runs: Sequence[RunRecord]) -> list[Row]:
    """Median over repeated runs per (label, executor, workers); speedup vs the serial row
    of the same label (or, if that label has none, the first serial row available)."""
    groups: dict[tuple[str, str, int], list[RunRecord]] = {}
    for r in runs:
        groups.setdefault((r["label"], r["executor"], r["workers"]), []).append(r)

    serial_wall: dict[str, float] = {}
    for (label, executor, _), rs in groups.items():
        if executor == "serial":
            serial_wall[label] = statistics.median(x["wall_s"] for x in rs)
    fallback = next(iter(serial_wall.values()), None)

    rows: list[Row] = []
    for (label, executor, workers), rs in groups.items():
        walls = [x["wall_s"] for x in rs]
        rss = [x["rss_bytes"] for x in rs if x["rss_bytes"] is not None]
        base = serial_wall.get(label, fallback)
        wall = statistics.median(walls)
        rows.append(
            Row(
                label=label,
                python=rs[0]["python"],
                gil=rs[0]["gil_enabled"],
                executor=executor,
                workers=workers,
                n=len(rs),
                wall=wall,
                wall_min=min(walls),
                wall_max=max(walls),
                cpu=statistics.median(x["cpu_s"] for x in rs),
                merge=statistics.median(x["merge_s"] for x in rs),
                plan=statistics.median(x["plan_s"] for x in rs),
                rss=int(statistics.median(rss)) if rss else None,
                speedup=(base / wall) if base else float("nan"),
            )
        )
    rows.sort(key=lambda r: (r.label, _EXECUTOR_ORDER.get(r.executor, 99), r.workers))
    return rows


def _fmt_rss(n: int | None) -> str:
    return "n/a" if n is None else f"{n / 2**20:,.0f} MiB"


def render_table(rows: Sequence[Row]) -> str:
    out = [
        "| Interpreter | Executor | Workers | Wall, s (median) | min-max | CPU, s | Peak RSS (parent)"
        " | Merge, s | Speedup |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        gil = "on" if r.gil else "OFF"
        out.append(
            f"| {r.label} [py {r.python}, GIL {gil}] | {r.executor} | {r.workers} | {r.wall:.2f} | "
            f"{r.wall_min:.2f}-{r.wall_max:.2f} | {r.cpu:.2f} | {_fmt_rss(r.rss)} | "
            f"{r.merge:.2f} | {r.speedup:.2f}x |"
        )
    return "\n".join(out)


def amdahl_fraction(rows: Sequence[Row]) -> float | None:
    """Sequential fraction of the serial build measured from the process-pool run with the
    most workers: (planning + merge) / serial wall. Used to draw Amdahl's predicted ceiling."""
    serial = next((r for r in rows if r.executor == "serial"), None)
    procs = [
        r for r in rows if r.executor == "processes" and r.label == (serial.label if serial else "")
    ]
    if serial is None or not procs or serial.wall <= 0:
        return None
    biggest = max(procs, key=lambda r: r.workers)
    return min(1.0, (biggest.plan + biggest.merge) / serial.wall)


def plot_speedup(rows: Sequence[Row], path: Path, title: str, *, amdahl: bool = True) -> None:
    """Speedup vs workers for every (interpreter, executor), with the ideal linear line."""
    import matplotlib

    matplotlib.use("Agg")  # no window: just write a PNG
    import matplotlib.pyplot as plt

    # matplotlib's stubs leave **kwargs untyped; treat the figure as Any at this boundary so
    # the rest of the module stays strictly checked.
    fig: Any
    ax: Any
    fig, ax = plt.subplots(figsize=(7.5, 5))
    series: dict[tuple[str, str], list[Row]] = {}
    for r in rows:
        if r.executor != "serial":
            series.setdefault((r.label, r.executor), []).append(r)
    xs_all = sorted({r.workers for rs in series.values() for r in rs} | {1})
    for (label, executor), rs in sorted(series.items()):
        rs = sorted(rs, key=lambda r: r.workers)
        xs = [1, *[r.workers for r in rs]]
        ys = [1.0, *[r.speedup for r in rs]]
        ax.plot(xs, ys, marker="o", label=f"{executor} - {label}")
    top = max(xs_all)
    ax.plot([1, top], [1, top], "k--", linewidth=1, label="ideal (linear)")
    frac = amdahl_fraction(rows) if amdahl else None
    if frac is not None:
        grid = [1 + i * (top - 1) / 100 for i in range(101)]
        ax.plot(
            grid,
            [1 / (frac + (1 - frac) / n) for n in grid],
            ":",
            color="gray",
            label=f"Amdahl, sequential part = {frac:.0%}",
        )
    ax.set_xlabel("workers")
    ax.set_ylabel("speedup over serial (wall clock)")
    ax.set_xticks(xs_all)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------ CLI
def _cmd_child(args: argparse.Namespace) -> int:
    result = measure_build(
        Path(args.corpus),
        executor=cast(ExecutorName, args.executor),
        workers=int(args.workers),
        positions=bool(args.positions),
        limit=cast("int | None", args.limit),
        chunks_per_worker=int(args.chunks_per_worker),
    )
    sys.stdout.write(json.dumps(result) + "\n")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    corpus = Path(args.corpus)
    if not corpus.is_dir():
        sys.stderr.write(f"error: corpus directory not found: {corpus}\n")
        return 1
    executors = cast("list[ExecutorName]", args.executors)
    configs = plan_matrix(executors, cast("list[int]", args.workers) or default_worker_counts())
    python = str(args.python or sys.executable)
    runs = int(args.runs)
    settings = ChildSettings(
        positions=bool(args.positions),
        limit=cast("int | None", args.limit),
        chunks_per_worker=int(args.chunks_per_worker),
        gil=cast("str | None", args.gil),
        timeout=float(args.timeout),
    )
    records: list[RunRecord] = []
    if args.warmup:
        log.info("warm-up run (not recorded)")
        run_child(python, corpus, Config("serial", 1), settings)
    for i in range(1, runs + 1):
        for cfg in configs:  # round-robin: one run of every configuration per pass
            res = run_child(python, corpus, cfg, settings)
            sys.stderr.write(
                f"[run {i}/{runs}] {cfg.executor:9} x{cfg.workers:<2} wall {res['wall_s']:.2f}s "
                f"cpu {res['cpu_s']:.2f}s merge {res['merge_s']:.2f}s\n"
            )
            records.append(cast(RunRecord, {**res, "label": str(args.label), "run": i}))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"machine": machine_info(), "runs": records}, indent=1), encoding="utf-8"
    )
    sys.stderr.write(f"saved {len(records)} measurements to {out}\n")
    print(render_table(summarize(records)))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    runs: list[RunRecord] = []
    machine: dict[str, str | int] = {}
    for name in cast("list[str]", args.files):
        data = cast(dict[str, object], json.loads(Path(name).read_text(encoding="utf-8")))
        machine = machine or cast("dict[str, str | int]", data.get("machine", {}))
        runs.extend(cast("list[RunRecord]", data["runs"]))
    rows = summarize(runs)
    print(render_table(rows))
    if machine:
        print(
            f"\nMachine: {machine.get('cpu')}, {machine.get('logical_cpus')} logical CPUs "
            f"({machine.get('usable_cpus')} usable), {machine.get('os')}"
        )
    frac = amdahl_fraction(rows)
    if frac is not None:
        print(
            f"Sequential fraction (plan + merge) of the serial build: {frac:.1%}; "
            f"Amdahl ceiling with infinite workers: {1 / frac if frac else float('inf'):.1f}x"
        )
    if args.plot:
        title = args.title or "Parallel indexing speedup"
        plot_speedup(rows, Path(args.plot), title)
        sys.stderr.write(f"plot written to {args.plot}\n")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m findex.parbench")
    ap.add_argument("-v", "--verbose", action="count", default=0)
    sub = ap.add_subparsers(dest="command", required=True)

    def add_build_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--positions", action="store_true")
        p.add_argument("--limit", type=int, default=None, help="only the first N documents")
        p.add_argument("--chunks-per-worker", type=int, default=2)

    child = sub.add_parser("child", help="(internal) one measured build; prints a JSON line")
    child.add_argument("corpus")
    child.add_argument("--executor", choices=EXECUTORS, required=True)
    child.add_argument("--workers", type=int, required=True)
    add_build_options(child)
    child.set_defaults(func=_cmd_child)

    run = sub.add_parser("run", help="measure a grid of executors x worker counts")
    run.add_argument("corpus")
    run.add_argument("--out", required=True, help="JSON file for the raw measurements")
    run.add_argument("--label", default=f"Python {platform.python_version()}")
    run.add_argument("--executors", nargs="+", choices=EXECUTORS, default=list(EXECUTORS))
    run.add_argument("--workers", nargs="+", type=int, default=None, help="default: 1 2 4 8 + CPUs")
    run.add_argument("--runs", type=int, default=3, help="measured runs per configuration (>= 3)")
    run.add_argument("--python", default=None, help="interpreter for the children (e.g. 3.13t)")
    run.add_argument("--gil", choices=["0", "1"], default=None, help="PYTHON_GIL for the children")
    run.add_argument("--timeout", type=float, default=3600.0, help="seconds per child")
    run.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    add_build_options(run)
    run.set_defaults(func=_cmd_run)

    rep = sub.add_parser("report", help="table (and plot) from saved measurements")
    rep.add_argument("files", nargs="+")
    rep.add_argument("--plot", default=None, help="write a speedup PNG here")
    rep.add_argument("--title", default=None)
    rep.set_defaults(func=_cmd_report)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    configure_logging(int(args.verbose))
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
