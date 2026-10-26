import inspect
import unicodedata

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def toks(s: str) -> list[str]:
    return list(tokenize(s))


def test_is_generator():
    assert inspect.isgeneratorfunction(tokenize)
    assert inspect.isgeneratorfunction(iter_documents)


def test_empty_and_punctuation_only():
    assert toks("") == []
    assert toks("  ...!?  —  ") == []


def test_mixed_case():
    assert toks("Hello WORLD hElLo") == ["hello", "world", "hello"]


def test_casefold_vs_lower():
    assert toks("Straße") == toks("STRASSE") == ["strasse"]


def test_cyrillic():
    assert toks("Привіт, Світе! Їжак та ЄВРОПА.") == [
        "привіт",
        "світе",
        "їжак",
        "та",
        "європа",
    ]


def test_combining_mark_is_normalized():
    composed = "caf\u00e9"
    decomposed = "cafe\u0301"
    assert composed != decomposed
    assert toks(composed) == toks(decomposed) == [unicodedata.normalize("NFC", "café")]


def test_apostrophes():
    assert toks("Don't don’t 'quoted' п'ять") == ["don't", "don't", "quoted", "п'ять"]


def test_hyphens():
    assert toks("well-known - state-of-the-art — end") == [
        "well-known",
        "state-of-the-art",
        "end",
    ]


def test_digits_and_underscore():
    assert toks("mp3 2026 3.14 snake_case") == [
        "mp3",
        "2026",
        "3",
        "14",
        "snake",
        "case",
    ]


def test_lazy_returns_before_consuming():
    it = tokenize("a b c")
    assert next(it) == "a"
    assert list(it) == ["b", "c"]
    assert list(it) == []  # exhausted: single-pass
