"""Query language: recursive-descent parser -> tree of operator-overloading nodes.

Grammar (AND binds tighter than OR; NOT tighter than AND; keywords are UPPER-case)::

    or_expr  -> and_expr ('OR' and_expr)*
    and_expr -> not_expr (['AND'] not_expr)*        # juxtaposition = implicit AND
    not_expr -> 'NOT' not_expr | atom
    atom     -> WORD | "quoted phrase" | '(' or_expr ')'

So ``a OR b c`` is ``Or(a, And(b, c))`` -- AND wins, like * over + in arithmetic.
"""

from __future__ import annotations

import re
from collections.abc import Collection

from findex.boolean import ENGINES, merge_and
from findex.model import Index, PostingList
from findex.tokenize import normalize, tokenize


class QuerySyntaxError(ValueError):
    pass


class PositionsRequired(ValueError):
    """Phrase queries need token positions: rebuild with ``--positions``."""


def _parts(node: Node, cls: type) -> tuple[Node, ...]:
    return node.children if isinstance(node, cls) else (node,)  # type: ignore[attr-defined]


class Node:
    """Immutable-by-convention query tree node.

    Operators build trees: ``Term('a') & (Term('b') | Term('c')) & ~Term('d')``.
    Chains are flattened, so ``a & b & c`` is ``And(a, b, c)`` -- the same tree
    the parser produces for ``a b c``.
    """

    __slots__ = ()

    def _key(self) -> tuple:
        raise NotImplementedError

    def evaluate(self, index: Index, engine: str = "merge") -> list[int]:
        raise NotImplementedError

    def terms(self) -> list[str]:
        """Terms that contribute to ranking (everything not under NOT)."""
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented  # not False: lets Python try the reflected operation
        return type(self) is type(other) and self._key() == other._key()

    def __hash__(self) -> int:  # defined together with __eq__: the hashing contract
        return hash((type(self).__name__, self._key()))

    def __and__(self, other: object) -> Node:
        if not isinstance(other, Node):
            return NotImplemented
        return And(*_parts(self, And), *_parts(other, And))

    def __or__(self, other: object) -> Node:
        if not isinstance(other, Node):
            return NotImplemented
        return Or(*_parts(self, Or), *_parts(other, Or))

    def __invert__(self) -> Node:
        return Not(self)


class Term(Node):
    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = normalize(text)  # same normalization as at index time

    def _key(self) -> tuple:
        return (self.text,)

    def __repr__(self) -> str:
        return f"Term({self.text!r})"

    def evaluate(self, index: Index, engine: str = "merge") -> list[int]:
        return list(index.doc_ids(self.text))

    def terms(self) -> list[str]:
        return [self.text]


class Phrase(Node):
    __slots__ = ("words",)

    def __init__(self, *words: str) -> None:
        self.words = tuple(normalize(w) for w in words)

    def _key(self) -> tuple:
        return self.words

    def __repr__(self) -> str:
        return f"Phrase({', '.join(map(repr, self.words))})"

    def terms(self) -> list[str]:
        return list(self.words)

    def evaluate(self, index: Index, engine: str = "merge") -> list[int]:
        words = self.words
        if not words:
            return []
        if len(words) == 1:
            return Term(words[0]).evaluate(index, engine)
        if not index.has_positions:
            raise PositionsRequired(
                "phrase queries need token positions: rebuild the index with --positions"
            )
        and_ = ENGINES[engine][0]
        candidates = list(index.doc_ids(words[0]))
        for w in words[1:]:
            candidates = and_(candidates, index.doc_ids(w))
        if not candidates:
            return []
        wanted = set(candidates)
        per_word = [_positions(index, w, wanted) for w in words]
        hits: list[int] = []
        for doc in candidates:
            cur = list(per_word[0][doc])
            for i, pos in enumerate(per_word[1:], start=1):
                # word i must sit exactly i places after word 0: shift and intersect
                cur = merge_and(cur, [p - i for p in pos[doc]])
                if not cur:
                    break
            if cur:
                hits.append(doc)
        return hits


def _positions(
    index: Index, term: str, wanted: Collection[int]
) -> dict[int, tuple[int, ...]]:
    postings = index[term]
    if isinstance(postings, PostingList):
        return postings.positions_by_doc(wanted)
    return {p.doc_id: p.positions for p in postings if p.doc_id in wanted}


class _NAry(Node):
    __slots__ = ("children",)

    def __init__(self, *children: Node) -> None:
        if not children:
            raise ValueError(f"{type(self).__name__} needs at least one child")
        self.children = children

    def _key(self) -> tuple:
        return self.children

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(map(repr, self.children))})"

    def terms(self) -> list[str]:
        return [t for c in self.children for t in c.terms()]


class And(_NAry):
    __slots__ = ()

    def evaluate(self, index: Index, engine: str = "merge") -> list[int]:
        and_, _, andnot = ENGINES[engine]
        positives = [c for c in self.children if not isinstance(c, Not)]
        negatives = [c.child for c in self.children if isinstance(c, Not)]  # type: ignore[attr-defined]
        if positives:
            lists = sorted((c.evaluate(index, engine) for c in positives), key=len)
            result = lists[0]  # smallest list first: cheapest intersections
            for other in lists[1:]:
                if not result:
                    break
                result = and_(result, other)
        else:
            result = list(range(index.num_docs))  # "NOT a NOT b": start from everything
        for neg in negatives:
            if not result:
                break
            result = andnot(result, neg.evaluate(index, engine))
        return result


class Or(_NAry):
    __slots__ = ()

    def evaluate(self, index: Index, engine: str = "merge") -> list[int]:
        or_ = ENGINES[engine][1]
        result: list[int] = []
        for c in self.children:
            result = or_(result, c.evaluate(index, engine))
        return result


class Not(Node):
    __slots__ = ("child",)

    def __init__(self, child: Node) -> None:
        self.child = child

    def _key(self) -> tuple:
        return (self.child,)

    def __repr__(self) -> str:
        return f"Not({self.child!r})"

    def __invert__(self) -> Node:  # ~~x is x
        return self.child

    def evaluate(self, index: Index, engine: str = "merge") -> list[int]:
        andnot = ENGINES[engine][2]
        return andnot(range(index.num_docs), self.child.evaluate(index, engine))

    def terms(self) -> list[str]:
        return []


# ---------------------------------------------------------------------------------
_LEX = re.compile(r'"([^"]*)"|(\()|(\))|([^\s()"]+)')


def _lex(query: str) -> list[tuple[str, str]]:
    if query.count('"') % 2:
        raise QuerySyntaxError("unterminated quote")
    out: list[tuple[str, str]] = []
    for m in _LEX.finditer(query):
        phrase, lpar, rpar, word = m.groups()
        if phrase is not None:
            out.append(("PHRASE", phrase))
        elif lpar:
            out.append(("LP", "("))
        elif rpar:
            out.append(("RP", ")"))
        elif word in ("AND", "OR", "NOT"):
            out.append((word, word))
        elif any(
            True for _ in tokenize(word)
        ):  # pure punctuation such as "&" is dropped
            out.append(("WORD", word))
    return out


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self.toks = tokens
        self.i = 0

    def peek(self) -> str | None:
        return self.toks[self.i][0] if self.i < len(self.toks) else None

    def take(self) -> tuple[str, str]:
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def parse(self) -> Node:
        if not self.toks:
            raise QuerySyntaxError("empty query")
        node = self.or_expr()
        if self.i < len(self.toks):
            raise QuerySyntaxError(f"unexpected {self.toks[self.i][1]!r}")
        return node

    def or_expr(self) -> Node:
        node = self.and_expr()
        while self.peek() == "OR":
            self.take()
            node = node | self.and_expr()
        return node

    def and_expr(self) -> Node:
        node = self.not_expr()
        while True:
            kind = self.peek()
            if kind == "AND":
                self.take()
                node = node & self.not_expr()
            elif kind in ("NOT", "WORD", "PHRASE", "LP"):  # implicit AND
                node = node & self.not_expr()
            else:
                return node

    def not_expr(self) -> Node:
        if self.peek() == "NOT":
            self.take()
            return ~self.not_expr()
        return self.atom()

    def atom(self) -> Node:
        kind = self.peek()
        if kind == "WORD":
            words = list(tokenize(self.take()[1]))
            node: Node = Term(words[0])
            for w in words[1:]:  # "state.of" -> two terms, ANDed
                node = node & Term(w)
            return node
        if kind == "PHRASE":
            words = list(tokenize(self.take()[1]))
            if not words:
                raise QuerySyntaxError("empty phrase")
            return Term(words[0]) if len(words) == 1 else Phrase(*words)
        if kind == "LP":
            self.take()
            if self.peek() == "RP":
                raise QuerySyntaxError("empty parentheses")
            node = self.or_expr()
            if self.peek() != "RP":
                raise QuerySyntaxError("missing ')'")
            self.take()
            return node
        if kind is None:
            raise QuerySyntaxError("unexpected end of query")
        raise QuerySyntaxError(f"unexpected {self.toks[self.i][1]!r}")


def parse(query: str) -> Node:
    """Parse ``query`` into a tree; raises ``QuerySyntaxError`` on bad input."""
    return _Parser(_lex(query)).parse()
