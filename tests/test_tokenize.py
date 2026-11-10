import inspect
import unicodedata

import pytest

from findex.corpus import iter_documents
from findex.tokenize import normalize, tokenize


def toks(s: str) -> list[str]:
    return list(tokenize(s))


def test_pipeline_pieces_are_generators():
    assert inspect.isgeneratorfunction(tokenize)
    assert inspect.isgeneratorfunction(iter_documents)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", []),
        ("  ...!?  —  ", []),
        ("Hello WORLD hElLo", ["hello", "world", "hello"]),
        # apostrophes: inside a word they stay, typographic ones are normalized, edges go away
        ("Don't don’t 'quoted' ʼthisʼ", ["don't", "don't", "quoted", "this"]),
        # hyphens: inside a word they stay; a spaced dash is punctuation
        ("well-known - state-of-the-art — end", ["well-known", "state-of-the-art", "end"]),
        # digits are ordinary token characters; decimals are NOT special; "_" separates
        ("mp3 2026 3.14 snake_case", ["mp3", "2026", "3", "14", "snake", "case"]),
        # Unicode: Cyrillic, German sharp s, accents, CJK, Greek final sigma
        ("Привіт, Світе! Їжак та ЄВРОПА.", ["привіт", "світе", "їжак", "та", "європа"]),
        ("п'ять і п’ять", ["п'ять", "і", "п'ять"]),
        ("Straße STRASSE", ["strasse", "strasse"]),
        ("ΟΔΟΣ", ["οδοσ"]),
        ("naïve café", ["naïve", "café"]),
        ("日本語のテキスト", ["日本語のテキスト"]),
        ("emoji 🙂 and ½", ["emoji", "and", "½"]),
    ],
)
def test_tokenize_table(text: str, expected: list[str]):
    assert toks(text) == expected


@pytest.mark.parametrize("composed", ["café", "ångström", "Ёлка", "ǆ"])
def test_unicode_equivalent_forms_give_same_tokens(composed: str):
    decomposed = unicodedata.normalize("NFD", composed)
    assert toks(composed) == toks(decomposed)


def test_normalize_is_nfc_casefolded():
    assert normalize("CAFE\u0301") == "caf\u00e9"
    assert unicodedata.is_normalized("NFC", normalize("ǅ Ⅷ ﬃ"))


def test_tokenizer_is_lazy_and_single_pass():
    it = tokenize("a b c")
    assert next(it) == "a"
    assert list(it) == ["b", "c"]
    assert list(it) == []
