"""Lazy corpus intake: documents are read one at a time, never all at once."""

from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple, cast

log = logging.getLogger(__name__)


class Document(NamedTuple):
    doc_id: int
    path: Path
    text: str


def read_document(path: Path) -> str:
    """Read one document as UTF-8; on bad bytes log it and substitute U+FFFD.

    Public since Lab 5: worker processes receive *paths* and read their own documents.
    """
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        log.warning("bad UTF-8 in %s (%s); decoding with errors='replace'", path, exc)
        return path.read_text(encoding="utf-8", errors="replace")


def list_documents(root: Path) -> list[Path]:
    """All ``*.txt`` files below ``root`` in a STABLE order (sorted).

    The doc_id of a document is its position in this list, so a fixed order means the same
    corpus always gives the same index (also on another machine, and no matter how many
    workers build it). Only the *paths* are held in memory -- O(number of files), not O(bytes).
    """
    root = Path(root)
    if not root.is_dir():  # rglob on a missing directory would quietly yield nothing
        raise FileNotFoundError(f"corpus directory not found: {root}")
    return sorted(p for p in root.rglob("*.txt") if p.is_file())


def _iter_txt_tree(root: Path) -> Iterator[tuple[Path, str]]:
    for path in list_documents(root):
        yield path, read_document(path)  # the text is read lazily, one file per next()


def _iter_jsonl(path: Path) -> Iterator[tuple[Path, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):  # file object = lazy line iterator
            line = line.strip()
            if not line:
                continue
            try:
                obj = cast(object, json.loads(line))
            except json.JSONDecodeError:
                log.warning("%s:%d is not valid JSON; skipped", path, lineno)
                continue
            text = cast(dict[str, object], obj).get("text") if isinstance(obj, dict) else None
            if not isinstance(text, str):
                log.warning("%s:%d has no string 'text' field; skipped", path, lineno)
                continue
            yield path, text


def iter_documents(root: Path) -> Iterator[Document]:
    """Yield documents one by one.

    * ``root`` is a directory  -> every ``*.txt`` below it (one file = one document)
    * ``root`` is ``*.jsonl`` / ``*.jsonl.gz`` -> one JSON object per line,
      the document body is the ``"text"`` field

    This is a generator: calling it reads nothing; each ``next()`` reads one
    document. Memory use is one document, whatever the corpus size.
    """
    root = Path(root)
    if root.is_dir():
        source = _iter_txt_tree(root)
    elif root.is_file() and root.suffix in {".jsonl", ".gz"}:
        source = _iter_jsonl(root)
    else:
        raise FileNotFoundError(f"corpus not found or unsupported: {root}")
    for doc_id, (path, text) in enumerate(source):
        yield Document(doc_id, path, text)


def count_documents(root: Path) -> int | None:
    """Cheap document count for progress bars: number of ``*.txt`` files, or None if unknown
    (a ``.jsonl`` file would have to be read to be counted)."""
    root = Path(root)
    if root.is_dir():
        return len(list_documents(root))
    return None
