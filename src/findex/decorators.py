"""Decorators. A decorator is just ``fn = decorator(fn)`` -- a closure around ``fn``."""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from typing import Any

log = logging.getLogger("findex.timing")


def timed(fn: Callable | None = None, *, level: int = logging.INFO) -> Any:
    """Log how long the wrapped function took.

    Works bare (``@timed``) and with arguments (``@timed(level=logging.DEBUG)``):
    the second form needs one more level of nesting -- ``timed(level=...)`` returns
    the real decorator.

    ``functools.wraps`` copies ``__name__``, ``__doc__`` and sets ``__wrapped__`` on
    the wrapper. Without it every timed function would be called "wrapper" in logs,
    ``help()`` and tracebacks.
    """

    def decorate(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:  # logged even if func raises
                ms = (time.perf_counter() - t0) * 1000
                log.log(level, "%s took %.2f ms", func.__qualname__, ms)

        return wrapper

    return decorate if fn is None else decorate(fn)
