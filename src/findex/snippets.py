"""Snippets: a +-``window``-character excerpt around the best match, matches highlighted."""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from functools import lru_cache
from itertools import islice
from pathlib import Path

from findex.tokenize import normalize

Marks = tuple[str, str]


@lru_cache(maxsize=16)
def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def read_text(path: str | Path) -> str | None:
    """Original text of a ``.txt`` document, or None (e.g. documents inside a .jsonl)."""
    p = Path(path)
    if p.suffix.lower() != ".txt" or not p.is_file():
        return None
    return _read(str(p))


@lru_cache(maxsize=256)
def _pattern(terms: tuple[str, ...]) -> re.Pattern[str] | None:
    alts = sorted(
        {re.escape(t).replace("'", "['\u2019\u02bc]") for t in terms if t}, key=len
    )
    if not alts:
        return None
    body = "|".join(reversed(alts))  # longest first
    # whole tokens only: not preceded/followed by a letter or digit
    return re.compile(rf"(?<![^\W_])(?:{body})(?![^\W_])", re.IGNORECASE)


def _matches(text: str, pat: re.Pattern[str], limit: int) -> list[tuple[int, int, str]]:
    return [
        (m.start(), m.end(), m.group().casefold())
        for m in islice(pat.finditer(text), limit)
    ]


def make_snippet(
    text: str,
    terms: Sequence[str],
    *,
    window: int = 80,
    marks: Marks = ("**", "**"),
    max_matches: int = 5000,
) -> str:
    """Excerpt of ``text`` around the densest cluster of query ``terms``.

    The "best" window contains the most *distinct* terms, then the most matches.
    Matching is done on the original text (case-insensitive, whole tokens); if that
    finds nothing (e.g. ``Straße`` vs the casefolded ``strasse``) we retry on the
    normalized text, which shows the excerpt lower-cased but still highlighted.
    """
    pat = _pattern(tuple(terms))
    ms: list[tuple[int, int, str]] = []
    if pat is not None:
        ms = _matches(text, pat, max_matches)
        if not ms:
            text = normalize(text)
            ms = _matches(text, pat, max_matches)
    if not ms:
        return " ".join(text[: 2 * window].split())

    starts = [m[0] for m in ms]
    best_key, best_lo, best_hi = (-1, -1), 0, 0
    for s, e, _ in ms:
        lo, hi = s - window, e + window
        inside = ms[bisect_left(starts, lo) : bisect_right(starts, hi)]
        key = (len({m[2] for m in inside}), len(inside))
        if key > best_key:
            best_key, best_lo, best_hi = key, lo, hi
    lo, hi = max(0, best_lo), min(len(text), best_hi)

    parts: list[str] = []
    cur = lo
    for s, e, _ in ms[bisect_left(starts, lo) : bisect_right(starts, hi)]:
        if s >= cur and e <= hi:
            parts.append(text[cur:s])
            parts.append(f"{marks[0]}{text[s:e]}{marks[1]}")
            cur = e
    parts.append(text[cur:hi])
    body = " ".join("".join(parts).split())  # collapse newlines / runs of spaces
    return f"{'...' if lo > 0 else ''}{body}{'...' if hi < len(text) else ''}"
