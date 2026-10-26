"""Persistence: two formats.

* ``pickle``  -- whole object graph in one call. Fast to write, but see the warning.
* ``binary``  -- our own format: magic + JSON header + raw ``array('I')`` buffers.
                 Safe to load (we only parse JSON and copy bytes), and compact.

    FINDEXBN | u64 header_len | header (JSON, utf-8) | per term: doc_ids, tfs[, positions]

Loading a ``binary`` file always yields ``storage="array"``.
"""

from __future__ import annotations

import json
import pickle
import struct
import sys
from array import array
from itertools import chain
from pathlib import Path

from findex.model import DocMeta, Index, PostingList

MAGIC = b"FINDEXBN"
FORMATS = ("binary", "pickle")
_ITEM = array("I").itemsize  # 4 on every mainstream platform


def save(index: Index, path: str | Path, fmt: str = "binary") -> int:
    """Write ``index`` to ``path``; return the file size in bytes."""
    path = Path(path)
    if fmt == "pickle":
        with path.open("wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif fmt == "binary":
        _save_binary(index, path)
    else:
        raise ValueError(f"unknown format {fmt!r}; choose from {FORMATS}")
    return path.stat().st_size


def load(path: str | Path) -> Index:
    """Load an index; the format is detected from the first bytes.

    SECURITY: ``pickle.load`` can execute ARBITRARY CODE embedded in the file
    (via ``__reduce__``). A pickle file downloaded from someone else is a program,
    not data: opening it is remote code execution. Only load pickles that YOU
    wrote on a machine you trust. The binary format has no such problem.
    """
    path = Path(path)
    with path.open("rb") as f:
        head = f.read(len(MAGIC))
        if head == MAGIC:
            return _load_binary(f)
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, Index):
        raise TypeError(f"{path} does not contain a findex Index")
    return obj


# ---------------------------------------------------------------------------------
def _arrays_of(index: Index, term: str) -> tuple[array, array, array | None]:
    p = index.postings[term]
    if isinstance(p, PostingList):
        return p.doc_ids, p.tfs, p.positions
    ids = array("I", (x.doc_id for x in p))
    tfs = array("I", (x.tf for x in p))
    pos = (
        array("I", chain.from_iterable(x.positions for x in p))
        if index.has_positions
        else None
    )
    return ids, tfs, pos


def _save_binary(index: Index, path: Path) -> None:
    terms = list(index.postings)
    metas = [index.doc_meta[d] for d in sorted(index.doc_meta)]
    header = {
        "version": 1,
        "byteorder": sys.byteorder,
        "has_positions": index.has_positions,
        "terms": terms,
        "counts": [len(index.postings[t]) for t in terms],
        "doc_ids": [m.doc_id for m in metas],
        "doc_paths": [m.path for m in metas],
        "doc_titles": [m.title for m in metas],
        "doc_lengths": [index.doc_lengths[m.doc_id] for m in metas],
    }
    raw = json.dumps(header, ensure_ascii=False).encode("utf-8")
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for term in terms:
            ids, tfs, pos = _arrays_of(index, term)
            f.write(ids.tobytes())
            f.write(tfs.tobytes())
            if index.has_positions:
                f.write(pos.tobytes())


def _take(buf: memoryview, offset: int, n: int, swap: bool) -> tuple[array, int]:
    a = array("I")
    end = offset + n * _ITEM
    if end > len(buf):
        raise ValueError("truncated or corrupt index file")
    a.frombytes(buf[offset:end])
    if swap:
        a.byteswap()
    return a, end


def _load_binary(f) -> Index:
    (hlen,) = struct.unpack("<Q", f.read(8))
    header = json.loads(f.read(hlen).decode("utf-8"))
    if header.get("version") != 1:
        raise ValueError("unsupported index version")
    blob = memoryview(f.read())
    swap = header["byteorder"] != sys.byteorder
    has_pos = header["has_positions"]

    postings: dict[str, PostingList] = {}
    off = 0
    for term, n in zip(header["terms"], header["counts"], strict=True):
        pl = PostingList(has_pos)
        pl.doc_ids, off = _take(blob, off, n, swap)
        pl.tfs, off = _take(blob, off, n, swap)
        if has_pos:
            pl.positions, off = _take(blob, off, sum(pl.tfs), swap)
        postings[term] = pl
    if off != len(blob):
        raise ValueError("trailing bytes in index file")

    ids = header["doc_ids"]
    meta = {
        d: DocMeta(d, p, t)
        for d, p, t in zip(ids, header["doc_paths"], header["doc_titles"], strict=True)
    }
    lengths = dict(zip(ids, header["doc_lengths"], strict=True))
    return Index("array", has_pos, postings, lengths, meta)
