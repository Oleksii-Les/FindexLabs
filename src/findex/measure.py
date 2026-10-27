"""Tiny helper shared by the CLIs: run a callable, report time and (optionally) peak memory."""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable
from typing import Any

from findex.stats import fmt_bytes

__all__ = ["describe", "fmt_bytes", "run"]


def run(fn: Callable[[], Any], trace: bool = True) -> tuple[Any, float, int | None]:
    """Return (result, elapsed_s, peak_bytes or None). tracemalloc inflates elapsed ~3-5x."""
    if trace:
        tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    peak = None
    if trace:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return result, elapsed, peak


def describe(elapsed: float, peak: int | None) -> str:
    if peak is None:
        return f"{elapsed:.3f} s"
    return f"{elapsed:.3f} s (inflated by tracemalloc), peak memory {fmt_bytes(peak)}"
