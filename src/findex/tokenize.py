"""Streaming tokenizer.

Rules (the contract that tests/test_tokenize.py checks)
------------------------------------------------------
1. Normalization: NFC -> ``casefold()`` -> NFC again (casefold can emit
   sequences that are not NFC). ``"Straße"`` and ``"STRASSE"`` give the same token.
2. A token is a run of letters/digits (Unicode-aware, ``_`` excluded), so
   Cyrillic, accented Latin etc. work out of the box.
3. Apostrophes: an apostrophe *between* letters/digits stays inside the
   token (``don't``, ``п'ять``). Typographic ’ and ʼ are mapped to ``'`` first,
   so ``don’t`` == ``don't``. Leading/trailing apostrophes (quotes) are dropped.
4. Hyphens: a hyphen *between* letters/digits stays inside the token
   (``well-known``, ``кого-небудь``) — one concept, one term. A dash with
   spaces (`` - ``, ``—``) is punctuation and splits.
5. Digits: digits are kept as ordinary token characters (``2026``, ``mp3``,
   ``b2b``). Decimal/thousand separators are NOT special: ``3.14`` -> ``3``, ``14``.
6. Everything else (punctuation, whitespace, symbols, emoji) separates tokens.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator

_APOSTROPHES = str.maketrans({"\u2019": "'", "\u02bc": "'"})

# [^\W_]+ : Unicode word chars minus underscore (letters + digits)
_TOKEN_RE = re.compile(r"[^\W_]+(?:['\-][^\W_]+)*")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.casefold()
    text = unicodedata.normalize("NFC", text)
    return text.translate(_APOSTROPHES)


def tokenize(text: str) -> Iterator[str]:
    """Yield normalized tokens lazily (``finditer``, not ``findall``)."""
    for match in _TOKEN_RE.finditer(normalize(text)):
        yield match.group()


def tokenize_all(text: str) -> list[str]:
    """The same tokens as ``tokenize``, but as a list from ONE native call (``re.findall``).

    ``tokenize`` resumes a generator and calls ``match.group()`` for every token -- 7 million times on a
    47 MB corpus (docs/profiling.md, hotspot 2 of ``findex index``). The index builder consumes every token
    of a document anyway, so it asks for the whole list of ONE document (bounded memory); ``tokenize``
    stays lazy for callers that stream a corpus (``stats``) and is what the tests of lab 1 pin down.
    """
    return _TOKEN_RE.findall(normalize(text))
