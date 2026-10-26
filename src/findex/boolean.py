"""Boolean retrieval: two-pointer merge on sorted postings, plus a set-based alternative.

Query language (left to right, NO precedence -- Lab 3 adds a real parser):
    a b        implicit AND            a AND b    explicit AND
    a OR b     union                   a NOT b    a AND (NOT b)
    NOT b      complement (all docs without b)
    "a OR b c" means ((a OR b) AND c). The keywords are upper-case; ``or`` is a term.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from findex.model import Index
from findex.tokenize import tokenize

Ids = Sequence[int]


# ---- engine 1: two-pointer merge, O(len(a) + len(b)), no hashing --------------------
def merge_and(a: Ids, b: Ids) -> list[int]:
    out: list[int] = []
    i = j = 0
    la, lb = len(a), len(b)
    while i < la and j < lb:
        x, y = a[i], b[j]
        if x == y:
            out.append(x)
            i += 1
            j += 1
        elif x < y:
            i += 1
        else:
            j += 1
    return out


def merge_or(a: Ids, b: Ids) -> list[int]:
    out: list[int] = []
    i = j = 0
    la, lb = len(a), len(b)
    while i < la and j < lb:
        x, y = a[i], b[j]
        if x == y:
            out.append(x)
            i += 1
            j += 1
        elif x < y:
            out.append(x)
            i += 1
        else:
            out.append(y)
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def merge_andnot(a: Ids, b: Ids) -> list[int]:
    """a minus b."""
    out: list[int] = []
    i = j = 0
    la, lb = len(a), len(b)
    while i < la:
        x = a[i]
        while j < lb and b[j] < x:
            j += 1
        if j < lb and b[j] == x:
            j += 1
        else:
            out.append(x)
        i += 1
    return out


# ---- engine 2: sets (hash tables). Output is sorted so both engines are comparable ---
def set_and(a: Ids, b: Ids) -> list[int]:
    return sorted(set(a) & set(b))


def set_or(a: Ids, b: Ids) -> list[int]:
    return sorted(set(a) | set(b))


def set_andnot(a: Ids, b: Ids) -> list[int]:
    return sorted(set(a) - set(b))


Op = Callable[[Ids, Ids], list[int]]
ENGINES: dict[str, tuple[Op, Op, Op]] = {
    "merge": (merge_and, merge_or, merge_andnot),
    "set": (set_and, set_or, set_andnot),
}


def evaluate(query: str, index: Index, engine: str = "merge") -> list[int]:
    """Return the sorted doc_ids matching ``query``."""
    and_, or_, andnot = ENGINES[engine]
    everything = range(index.n_docs)

    def operand(raw: str) -> Ids | None:
        terms = list(tokenize(raw))  # same normalization as at index time
        if not terms:
            return None
        ids: Ids = index.doc_ids(terms[0])
        for t in terms[1:]:  # "a.b" tokenizes to two terms: treat as AND
            ids = and_(ids, index.doc_ids(t))
        return ids

    result: Ids | None = None
    op, negate = "AND", False
    for raw in query.split():
        if raw in ("AND", "OR"):
            op = raw
            continue
        if raw == "NOT":
            negate = True
            continue
        ids = operand(raw)
        if ids is None:
            continue
        if result is None:
            result = andnot(everything, ids) if negate else list(ids)
        elif negate:
            result = (
                andnot(result, ids)
                if op == "AND"
                else or_(result, andnot(everything, ids))
            )
        else:
            result = and_(result, ids) if op == "AND" else or_(result, ids)
        op, negate = "AND", False
    return list(result) if result is not None else []
