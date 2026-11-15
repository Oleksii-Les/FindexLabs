"""20 simultaneous requests to /search: ``def`` vs ``async def`` (Lab 07).

    uv run python experiments/def_vs_async.py --index index.bin [--rounds 10] [--clients 20]

For each variant a FRESH server process is started (cold caches), then ROUNDS times:
CLIENTS requests (different queries) are sent at the same instant. While this goes on, a probe
asks ``/health`` every 10 ms. Reported: how long a round takes, the latency of the search
requests, and -- the number that matters -- how late /health is answered.

The searches do the same work in every variant, so throughput is about equal (one interpreter,
one GIL, whichever thread runs it). What differs is whether the loop stays free: with
``async def`` the loop is busy for the whole of every search, so the probe, new connections and
every other request wait for it.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
TERMS = ["function", "object", "value", "return", "string", "default", "list", "class", "file", "type",
         "data", "name", "error", "call", "path", "method", "number", "result", "time", "format"]  # fmt: skip


def queries(n: int) -> list[str]:
    """n different OR-queries over common words: each matches many documents (real work)."""
    out: list[str] = []
    for i in range(n):
        a, b, c = TERMS[i % 20], TERMS[(i * 7 + 3) % 20], TERMS[(i * 11 + 5) % 20]
        out.append(f"{a} OR {b} OR {c}" if a != b != c else f"{a} OR {b}")
    return out


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * p))] if s else 0.0


def start_server(index: Path, port: int) -> subprocess.Popen[bytes]:
    env = dict(os.environ, PYTHONPATH=str(HERE.parent / "src"), PYTHONUTF8="1")
    cmd = [
        sys.executable,
        str(HERE / "web_variants.py"),
        "--index",
        str(index),
        "--port",
        str(port),
        "--cache",
        "0",
    ]
    proc = subprocess.Popen(cmd, env=env)
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                return proc
        except httpx.HTTPError:
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError("server did not start")


async def measure(base: str, path: str, rounds: int, clients: int) -> dict[str, float]:
    qs = queries(rounds * clients)
    limits = httpx.Limits(max_connections=clients + 2)
    async with httpx.AsyncClient(base_url=base, limits=limits, timeout=60) as client:
        await client.get("/health")
        stop = asyncio.Event()
        probes: list[float] = []

        async def probe() -> None:
            while not stop.is_set():
                t0 = time.perf_counter()
                await client.get("/health")
                probes.append((time.perf_counter() - t0) * 1000)
                await asyncio.sleep(0.01)

        async def one(q: str, lat: list[float]) -> None:
            t0 = time.perf_counter()
            r = await client.get(path, params={"q": q, "k": 50})
            r.raise_for_status()
            lat.append((time.perf_counter() - t0) * 1000)

        probe_task = asyncio.create_task(probe())
        lat: list[float] = []
        round_times: list[float] = []
        for r in range(rounds):
            t0 = time.perf_counter()
            await asyncio.gather(*(one(q, lat) for q in qs[r * clients : (r + 1) * clients]))
            round_times.append((time.perf_counter() - t0) * 1000)
        stop.set()
        await probe_task
    return {
        "round_ms": statistics.median(round_times), "rps": clients * 1000 / statistics.median(round_times),
        "p50": pct(lat, 0.5), "p95": pct(lat, 0.95), "max": max(lat),
        "h_p50": pct(probes, 0.5), "h_p95": pct(probes, 0.95), "h_max": max(probes), "h_n": len(probes),
    }  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument(
        "--repeat", type=int, default=3, help="independent runs per variant; the median is shown"
    )
    args = ap.parse_args()
    variants = ["/search-def", "/search-async", "/search-pool"]
    results: dict[str, list[dict[str, float]]] = {v: [] for v in variants}
    for _ in range(args.repeat):  # round-robin over variants: slow drift hits all of them
        for v in variants:
            proc = start_server(args.index, args.port)
            try:
                results[v].append(
                    asyncio.run(
                        measure(f"http://127.0.0.1:{args.port}", v, args.rounds, args.clients)
                    )
                )
            finally:
                proc.terminate()
                proc.wait(10)
    print(
        f"{args.clients} simultaneous requests x {args.rounds} rounds, cold cache, median of {args.repeat} runs\n"
    )
    print(
        "| Route | 20 requests take, ms | req/s | search p50, ms | search p95, ms | /health p50, ms | /health p95, ms | /health worst, ms |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for v in variants:
        m = {k: statistics.median(r[k] for r in results[v]) for k in results[v][0]}
        label = {
            "/search-def": "`def`",
            "/search-async": "`async def` (inline)",
            "/search-pool": "`async def` + thread pool",
        }[v]
        print(f"| {label} | {m['round_ms']:.0f} | {m['rps']:.0f} | {m['p50']:.0f} | {m['p95']:.0f} | "
              f"{m['h_p50']:.1f} | {m['h_p95']:.1f} | {m['h_max']:.0f} |")  # fmt: skip


if __name__ == "__main__":
    main()
