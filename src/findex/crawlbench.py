"""Benchmark the crawler: the same start URL and the same ``--max-pages`` for concurrency 1, 5, 20.

    python -m findex.crawlbench run    --pages 200 --out bench/crawl.json         # starts its own mirror
    python -m findex.crawlbench run    --url http://127.0.0.1:8765/ --pages 200   # a server you started
    python -m findex.crawlbench report bench/crawl.json

Method (the same as ``parbench``): every measurement is a **fresh child process** (so peak RSS is
that run's own), configurations are run round-robin ``--runs`` times, the table shows the median.
The default target is ``findex.mirror``: your own server on 127.0.0.1 with a fixed latency per
request (``--latency-ms``), which stands in for the network round trip the crawler overlaps.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

from findex.parbench import machine_info, peak_rss_bytes

_PACKAGE_PARENT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class Setup:
    """One row of the table."""

    name: str
    concurrency: int
    per_host: int
    delay: float


def default_setups() -> list[Setup]:
    """Concurrency 1, 5 and 20 with the host limit out of the way, plus one polite row that shows
    what the per-host limit and pause cost when ONE host serves everything."""
    plain = [Setup(f"c={c}", c, c, 0.0) for c in (1, 5, 20)]
    return [*plain, Setup("c=20, polite (2/host, 50 ms)", 20, 2, 0.05)]


class ChildResult(TypedDict):
    pages: int
    errors: int
    retries: int
    wall_s: float
    cpu_s: float
    rss_bytes: int | None
    bytes: int
    statuses: dict[str, int]


class RunRecord(ChildResult):
    setup: str
    concurrency: int
    per_host: int
    delay: float
    run: int


# ------------------------------------------------------------------------------- child
def measure_crawl(url: str, max_pages: int, setup: Setup) -> ChildResult:
    """Crawl once in THIS process and report the numbers (this is what a child runs)."""
    from findex.crawler import CrawlConfig, CrawlStats, crawl

    config = CrawlConfig(
        seeds=(url,),
        user_agent="findex-crawlbench/1.0 (+mailto:bench@localhost)",
        max_pages=max_pages,
        concurrency=setup.concurrency,
        per_host=setup.per_host,
        delay=setup.delay,
    )
    stats = CrawlStats()

    async def main() -> None:
        async with contextlib.aclosing(crawl(config, stats=stats)) as pages:
            async for _ in pages:  # pages are counted, not kept: memory = the frontier only
                pass

    cpu0, t0 = time.process_time(), time.perf_counter()
    asyncio.run(main())
    return {
        "pages": stats.pages,
        "errors": stats.errors,
        "retries": stats.retries,
        "wall_s": time.perf_counter() - t0,
        "cpu_s": time.process_time() - cpu0,
        "rss_bytes": peak_rss_bytes(),
        "bytes": stats.bytes,
        "statuses": dict(stats.statuses),
    }


def run_child(url: str, max_pages: int, setup: Setup, timeout: float = 600.0) -> ChildResult:
    cmd = [
        sys.executable, "-m", "findex.crawlbench", "child", url,
        "--max-pages", str(max_pages), "--concurrency", str(setup.concurrency),
        "--per-host", str(setup.per_host), "--delay", str(setup.delay),
    ]  # fmt: skip
    env = dict(os.environ, PYTHONUTF8="1")
    env["PYTHONPATH"] = os.pathsep.join([str(_PACKAGE_PARENT), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    done = subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=timeout, check=False
    )
    if done.returncode != 0:
        raise RuntimeError(f"child failed ({setup.name}):\n{done.stderr.strip()}")
    lines = [ln for ln in done.stdout.splitlines() if ln.startswith("{")]
    if not lines:
        raise RuntimeError(f"child printed no result:\n{done.stdout}\n{done.stderr}")
    return cast(ChildResult, json.loads(lines[-1]))


@contextlib.contextmanager
def started_mirror(pages: int, latency_ms: float, corpus: Path | None):
    """Launch ``python -m findex.mirror`` as a separate process and yield its base URL."""
    cmd = [
        sys.executable, "-m", "findex.mirror", "--pages", str(pages * 2),
        "--latency-ms", str(latency_ms),
    ]  # fmt: skip
    if corpus is not None:
        cmd += ["--corpus", str(corpus)]
    env = dict(os.environ, PYTHONUTF8="1")
    env["PYTHONPATH"] = os.pathsep.join([str(_PACKAGE_PARENT), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, env=env)
    try:
        assert proc.stdout is not None
        first = proc.stdout.readline().strip()
        if not first.startswith("LISTENING "):
            raise RuntimeError(f"mirror did not start: {first!r}")
        yield f"http://127.0.0.1:{first.split()[1]}/"
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(5)
        proc.kill()
        if proc.stdout is not None:
            proc.stdout.close()


# ---------------------------------------------------------------------------- reporting
@dataclass(frozen=True, slots=True)
class Row:
    setup: str
    concurrency: int
    n: int
    wall: float
    wall_min: float
    wall_max: float
    pages: int
    pages_per_s: float
    errors: int
    cpu: float
    rss: int | None
    speedup: float


def summarize(runs: Sequence[RunRecord]) -> list[Row]:
    groups: dict[str, list[RunRecord]] = {}
    for r in runs:
        groups.setdefault(r["setup"], []).append(r)
    base = next(
        (statistics.median(x["wall_s"] for x in rs) for rs in groups.values() if rs[0]["concurrency"] == 1),
        None,
    )  # fmt: skip
    rows: list[Row] = []
    for name, rs in groups.items():
        walls = [x["wall_s"] for x in rs]
        rss = [x["rss_bytes"] for x in rs if x["rss_bytes"] is not None]
        wall = statistics.median(walls)
        pages = int(statistics.median(x["pages"] for x in rs))
        rows.append(
            Row(
                setup=name,
                concurrency=rs[0]["concurrency"],
                n=len(rs),
                wall=wall,
                wall_min=min(walls),
                wall_max=max(walls),
                pages=pages,
                pages_per_s=pages / wall if wall else 0.0,
                errors=int(statistics.median(x["errors"] for x in rs)),
                cpu=statistics.median(x["cpu_s"] for x in rs),
                rss=int(statistics.median(rss)) if rss else None,
                speedup=(base / wall) if base else float("nan"),
            )
        )
    return rows


def render_table(rows: Sequence[Row]) -> str:
    out = [
        "| Setup | Wall, s (median) | min-max | Pages | Pages/s | Errors | CPU, s | Peak RSS | Speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        rss = "n/a" if r.rss is None else f"{r.rss / 2**20:,.0f} MiB"
        out.append(
            f"| {r.setup} | {r.wall:.2f} | {r.wall_min:.2f}-{r.wall_max:.2f} | {r.pages} | "
            f"{r.pages_per_s:.1f} | {r.errors} | {r.cpu:.2f} | {rss} | {r.speedup:.2f}x |"
        )
    return "\n".join(out)


# -------------------------------------------------------------------------------- CLI
def _cmd_run(args: argparse.Namespace) -> None:
    setups = default_setups()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records: list[RunRecord] = []

    def measure(url: str) -> None:
        print(f"warm-up ({setups[1].name}) ...", file=sys.stderr)
        run_child(url, min(args.pages, 40), setups[1])  # unmeasured: imports, connection set-up
        for k in range(args.runs):
            for s in setups:  # round-robin: drift hits every setup equally
                res = run_child(url, args.pages, s)
                print(f"run {k + 1}/{args.runs} {s.name}: {res['wall_s']:.2f} s, "
                      f"{res['pages']} pages, {res['errors']} errors", file=sys.stderr)  # fmt: skip
                records.append(
                    cast(
                        RunRecord,
                        {
                            **res,
                            "setup": s.name,
                            "concurrency": s.concurrency,
                            "per_host": s.per_host,
                            "delay": s.delay,
                            "run": k,
                        },
                    )
                )

    if args.url:
        measure(args.url)
        target = args.url
    else:
        with started_mirror(args.pages, args.latency_ms, args.corpus) as target:
            measure(target)
    payload = {
        "machine": machine_info() | {"python": platform.python_version()},
        "target": target,
        "latency_ms": None if args.url else args.latency_ms,
        "max_pages": args.pages,
        "runs": records,
    }
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(render_table(summarize(records)))
    print(f"saved {out}", file=sys.stderr)


def _cmd_report(args: argparse.Namespace) -> None:
    for path in args.files:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        m = data["machine"]
        print(f"{path}: {m['cpu']}, {m['logical_cpus']} logical CPUs, {m['os']}, "
              f"Python {m['python']}; target {data['target']}, latency {data['latency_ms']} ms, "
              f"max_pages {data['max_pages']}\n")  # fmt: skip
        print(render_table(summarize(data["runs"])))
        print()


def _cmd_child(args: argparse.Namespace) -> None:
    setup = Setup("child", args.concurrency, args.per_host, args.delay)
    print(json.dumps(measure_crawl(args.url, args.max_pages, setup)))


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="findex.crawlbench", description=(__doc__ or "").splitlines()[0]
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="measure concurrency 1/5/20 (+ a polite row)")
    run.add_argument("--url", help="start URL of a server you run; default: start findex.mirror")
    run.add_argument("--pages", type=int, default=200, help="max pages per crawl (default 200)")
    run.add_argument("--runs", type=int, default=3, help="measured runs per setup (default 3)")
    run.add_argument("--latency-ms", type=float, default=30.0, help="mirror latency per request")
    run.add_argument("--corpus", type=Path, default=None, help="mirror: take page text from here")
    run.add_argument("--out", default="bench/crawl.json")
    run.set_defaults(func=_cmd_run)
    rep = sub.add_parser("report", help="print the table from saved runs")
    rep.add_argument("files", nargs="+")
    rep.set_defaults(func=_cmd_report)
    child = sub.add_parser("child", help=argparse.SUPPRESS)
    child.add_argument("url")
    child.add_argument("--max-pages", type=int, required=True)
    child.add_argument("--concurrency", type=int, required=True)
    child.add_argument("--per-host", type=int, required=True)
    child.add_argument("--delay", type=float, required=True)
    child.set_defaults(func=_cmd_child)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
