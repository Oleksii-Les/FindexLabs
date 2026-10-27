"""Build the inverted index in ONE pass over a lazily streamed corpus."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from findex.corpus import Document
from findex.decorators import timed
from findex.model import (
    STORAGES,
    DocMeta,
    Index,
    PlainPosting,
    Posting,
    PostingList,
)
from findex.tokenize import tokenize

_TITLE_RE = re.compile(r"^Title:[ \t]*(.+)$", re.MULTILINE)


def extract_title(text: str, path: Path) -> str:
    """Gutenberg-style ``Title:`` header, else the first non-empty line, else the file name."""
    m = _TITLE_RE.search(text, 0, 5000)
    if m:
        return m.group(1).strip()[:120]
    for line in text[:2000].splitlines():
        line = line.strip()
        if line:
            return line[:80]
    return Path(path).name


@timed
def build_index(
    docs: Iterable[Document], *, storage: str = "slots", positions: bool = False
) -> Index:
    """Consume ``docs`` (a stream, ascending doc_id) and return the index.

    Memory while building = the index + the Counter of ONE document.
    Because doc_ids arrive ascending, every postings list is sorted for free.
    """
    if storage not in STORAGES:
        raise ValueError(f"storage must be one of {STORAGES}, got {storage!r}")

    if storage == "array":
        postings: defaultdict = defaultdict(lambda: PostingList(positions))
    else:
        postings = defaultdict(list)  # term -> list; first access creates the list
    make = Posting if storage == "slots" else PlainPosting
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    last_id = -1

    for doc in docs:
        doc_id = doc.doc_id
        if doc_id <= last_id:
            raise ValueError("documents must arrive in strictly ascending doc_id order")
        last_id = doc_id
        doc_meta[doc_id] = DocMeta(
            doc_id, str(doc.path), extract_title(doc.text, doc.path)
        )

        if positions:
            occ: defaultdict[str, list[int]] = defaultdict(list)
            for pos, tok in enumerate(tokenize(doc.text)):
                occ[tok].append(pos)
            items = ((t, len(p), p) for t, p in occ.items())
            doc_lengths[doc_id] = sum(len(p) for p in occ.values())
        else:
            counts = Counter(tokenize(doc.text))
            items = ((t, c, None) for t, c in counts.items())
            doc_lengths[doc_id] = counts.total()

        if storage == "array":
            for term, tf, pos in items:
                pl = postings[term]
                pl.doc_ids.append(doc_id)
                pl.tfs.append(tf)
                if pos is not None:
                    pl.positions.extend(pos)
        else:
            for term, tf, pos in items:
                postings[term].append(
                    make(doc_id, tf, tuple(pos) if pos is not None else ())
                )

    # plain dict: no auto-creation on lookup, and (unlike a lambda-defaultdict) picklable
    return Index(storage, positions, dict(postings), doc_lengths, doc_meta)
