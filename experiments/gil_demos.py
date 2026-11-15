"""Lab 05 'Prove it to yourself': five small experiments about the GIL.

    uv run python experiments/gil_demos.py                  # on the normal build
    uv run --python 3.13t python experiments/gil_demos.py   # on the free-threaded build

Every experiment prints numbers you should be able to explain at the whiteboard.
"""

from __future__ import annotations

import argparse
import dis
import statistics
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def cpu_task(n: int) -> int:
    """Pure-Python CPU-bound work: never waits for anything, holds the GIL all the time."""
    return sum(i * i for i in range(n))


def timed(fn: Callable[[], object], repeat: int = 3) -> float:
    best = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best.append(time.perf_counter() - t0)
    return statistics.median(best)


def gil_state() -> str:
    probe = getattr(sys, "_is_gil_enabled", None)
    return "always on (Python < 3.13)" if probe is None else f"enabled = {probe()}"


def exp1_cpu_bound(n: int) -> None:
    print(f"\n[1] CPU-bound: sum(i*i for i in range({n:,}))   GIL: {gil_state()}")
    once = timed(lambda: cpu_task(n))
    twice = timed(lambda: (cpu_task(n), cpu_task(n)))

    def two_threads() -> None:
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(cpu_task, [n, n]))

    def two_processes() -> None:
        with ProcessPoolExecutor(2) as pool:
            list(pool.map(cpu_task, [n, n]))

    threads = timed(two_threads)
    procs = timed(two_processes)
    print(f"    one call                : {once:6.2f} s")
    print(f"    two calls, sequential   : {twice:6.2f} s   (expect ~2x one call)")
    print(
        f"    two calls, 2 threads    : {threads:6.2f} s   (GIL build: ~same as sequential or worse)"
    )
    print(
        f"    two calls, 2 processes  : {procs:6.2f} s   (needs >= 2 free cores to be ~1x one call)"
    )


def exp2_lost_updates(n: int, threads: int) -> None:
    print(f"\n[2] Lost updates: {threads} threads x {n:,} increments of a shared int")
    box = {"x": 0}
    lock = threading.Lock()

    def plain() -> None:
        for _ in range(n):
            box["x"] += 1

    def split_by_a_call() -> None:  # read ... a function call ... write
        for _ in range(n):
            tmp = box["x"]
            tmp = abs(tmp) + 1
            box["x"] = tmp

    def locked() -> None:
        for _ in range(n):
            with lock:
                tmp = box["x"]
                tmp = abs(tmp) + 1
                box["x"] = tmp

    def run(fn: Callable[[], None]) -> int:
        box["x"] = 0
        ts = [threading.Thread(target=fn) for _ in range(threads)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return box["x"]

    want = threads * n
    for name, fn in (
        ("x += 1", plain),
        ("read; call; write", split_by_a_call),
        ("same, with Lock", locked),
    ):
        got = [run(fn) for _ in range(3)]
        print(
            f"    {name:18}: results {[f'{g:,}' for g in got]}  expected {want:,}  "
            f"worst loss {want - min(got):,}"
        )


def exp3_switch_interval(n: int) -> None:
    print("\n[3] sys.getswitchinterval() and the threaded CPU-bound case")
    original = sys.getswitchinterval()

    def two_threads() -> None:
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(cpu_task, [n, n]))

    print(f"    default interval {original * 1000:.3f} ms : {timed(two_threads):6.2f} s")
    sys.setswitchinterval(0.0001)
    print(
        f"    interval 0.100 ms           : {timed(two_threads):6.2f} s   "
        "(more switches, more contention)"
    )
    sys.setswitchinterval(original)


def exp4_lambda() -> None:
    print("\n[4] A lambda cannot be sent to a process (pickle stores functions BY NAME)")
    try:
        with ProcessPoolExecutor(1) as pool:
            pool.submit(lambda: 1).result()
    except Exception as exc:  # the point of the experiment is to read this message
        print(f"    {type(exc).__name__}: {exc}")


def exp5_bytecode() -> None:
    print("\n[5] Why `counter += 1` is not atomic: the bytecode")
    counter = 0

    def bump() -> None:
        nonlocal counter
        counter += 1

    dis.dis(bump)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5_000_000, help="size of the CPU-bound task")
    ap.add_argument("--increments", type=int, default=300_000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--only", type=int, nargs="*", help="run only these experiment numbers")
    args = ap.parse_args()
    print(
        f"Python {sys.version.split()[0]}  GIL: {gil_state()}  switchinterval: {sys.getswitchinterval()}"
    )
    wanted = set(args.only or [1, 2, 3, 4, 5])
    if 1 in wanted:
        exp1_cpu_bound(args.n)
    if 2 in wanted:
        exp2_lost_updates(args.increments, args.threads)
    if 3 in wanted:
        exp3_switch_interval(args.n)
    if 4 in wanted:
        exp4_lambda()
    if 5 in wanted:
        exp5_bytecode()


if __name__ == "__main__":  # required: process pools below re-import this file under 'spawn'
    main()
