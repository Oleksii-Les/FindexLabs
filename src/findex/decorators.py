"""Decorators. A decorator is just ``fn = decorator(fn)`` -- a closure around ``fn``."""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from typing import Any, Protocol, overload

log = logging.getLogger("findex.timing")


class _Decorator(Protocol):
    """What ``timed(level=...)`` returns: a decorator that keeps the wrapped signature."""

    def __call__[**P, R](self, fn: Callable[P, R], /) -> Callable[P, R]: ...


@overload
def timed[**P, R](fn: Callable[P, R], /) -> Callable[P, R]: ...
@overload
def timed(*, level: int = ...) -> _Decorator: ...
def timed(fn: Callable[..., Any] | None = None, /, *, level: int = logging.DEBUG) -> Any:
    """Log how long the wrapped function took (at DEBUG by default: ``-vv``).

    Works bare (``@timed``) and with arguments (``@timed(level=logging.INFO)``):
    the second form needs one more level of nesting -- ``timed(level=...)`` returns
    the real decorator. ``ParamSpec`` (``**P``) makes the wrapper advertise exactly the
    parameters and return type of the wrapped function, so type checking is not lost.

    ``functools.wraps`` copies ``__name__``, ``__doc__`` and sets ``__wrapped__`` on
    the wrapper. Without it every timed function would be called "wrapper" in logs,
    ``help()`` and tracebacks.
    """

    def decorate[**P, R](func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:  # logged even if func raises
                ms = (time.perf_counter() - t0) * 1000
                log.log(level, "%s took %.2f ms", func.__qualname__, ms)

        return wrapper

    return decorate if fn is None else decorate(fn)
