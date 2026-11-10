import logging

import pytest

from findex.build import build_index
from findex.decorators import timed
from findex.engine import search
from findex.store import load


def test_wraps_preserves_metadata():
    @timed
    def f(x):
        """doc of f"""
        return x

    assert f.__name__ == "f" and f.__doc__ == "doc of f"
    assert f.__wrapped__(3) == 3  # the original is reachable
    assert f(5) == 5


def test_real_functions_keep_their_names():
    assert search.__name__ == "search"
    assert build_index.__name__ == "build_index"
    assert load.__name__ == "load"
    assert hasattr(search, "__wrapped__")


def test_logs_elapsed_time(caplog):
    @timed
    def work():
        return 1

    with caplog.at_level(logging.DEBUG, logger="findex.timing"):
        work()
    assert any("work took" in r.getMessage() and " ms" in r.getMessage() for r in caplog.records)


def test_logs_even_when_function_raises(caplog):
    @timed
    def bad():
        raise KeyError("x")

    with caplog.at_level(logging.DEBUG, logger="findex.timing"), pytest.raises(KeyError):
        bad()
    assert any("bad took" in r.getMessage() for r in caplog.records)


def test_decorator_with_arguments(caplog):
    @timed(level=logging.INFO)
    def loud():
        return "ok"

    @timed
    def quiet():
        return "ok"

    with caplog.at_level(logging.INFO, logger="findex.timing"):
        assert quiet() == "ok"
        assert not caplog.records  # the default level is DEBUG: filtered out at INFO
        assert loud() == "ok"
    assert [r.levelno for r in caplog.records] == [logging.INFO]
