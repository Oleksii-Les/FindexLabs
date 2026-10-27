import pytest

from findex.model import STORAGES
from findex.query import (
    And,
    Not,
    Or,
    Phrase,
    PositionsRequired,
    QuerySyntaxError,
    Term,
    parse,
)

a, b, c, d = Term("a"), Term("b"), Term("c"), Term("d")


def test_precedence_is_a_tree_not_a_string():
    assert parse("a OR b c") == Or(a, And(b, c))  # AND binds tighter than OR
    assert parse("a b OR c") == Or(And(a, b), c)
    assert parse("(a OR b) c") == And(Or(a, b), c)


def test_implicit_and_explicit_and_flatten():
    assert parse("a b c") == parse("a AND b AND c") == And(a, b, c)
    assert parse("a OR b OR c") == Or(a, b, c)


def test_not_and_parentheses():
    assert parse("a NOT b") == And(a, Not(b))
    assert parse("NOT (a OR b)") == Not(Or(a, b))
    assert parse("NOT NOT a") == a  # double negation collapses (~~x is x)


def test_the_labs_example_query():
    tree = parse('python AND (async OR await) NOT java "event loop"')
    expected = (
        Term("python")
        & (Term("async") | Term("await"))
        & ~Term("java")
        & Phrase("event", "loop")
    )
    assert tree == expected
    assert tree == And(
        Term("python"),
        Or(Term("async"), Term("await")),
        Not(Term("java")),
        Phrase("event", "loop"),
    )


def test_operators_build_the_same_trees_as_the_parser():
    assert (a & (b | c) & ~d) == parse("a (b OR c) NOT d")
    assert (a | b | c) == Or(a, b, c)
    assert ~~a == a


def test_normalization_and_single_word_phrase():
    assert parse("PYTHON") == Term("python")
    assert parse('"Python"') == Term("python")
    assert parse("well-known") == Term("well-known")
    assert parse("x.y") == And(Term("x"), Term("y"))
    assert parse("a & b") == And(a, b)  # bare punctuation is ignored
    assert parse("a and b") == And(
        Term("a"), Term("and"), Term("b")
    )  # keywords are UPPER-case


def test_equality_and_hash_contract():
    assert a == Term("A") and hash(a) == hash(Term("A"))
    assert a != Phrase("a") and (a == "a") is False
    assert len({And(a, b), And(a, b), Or(a, b)}) == 2


@pytest.mark.parametrize(
    "bad", ["", "   ", "&", "(a", "a)", "a OR", "OR a", "()", '"abc', "NOT", '""']
)
def test_syntax_errors(bad):
    with pytest.raises(QuerySyntaxError):
        parse(bad)


def test_terms_exclude_negated_ones():
    assert parse('a (b OR c) NOT d "e f"').terms() == ["a", "b", "c", "e", "f"]


TEXTS = [
    "python async event loop",  # 0
    "python await java",  # 1
    "event loop in java and python",  # 2
    "loop event python await",  # 3 (words reversed)
    "nothing relevant here",  # 4
]


@pytest.mark.parametrize("engine", ["merge", "set"])
@pytest.mark.parametrize("storage", STORAGES)
def test_evaluate(mk, storage, engine):
    ix = mk(TEXTS, storage, positions=True)

    def ev(q):
        return parse(q).evaluate(ix, engine)

    assert ev("python") == [0, 1, 2, 3]
    assert ev("python AND (async OR await)") == [0, 1, 3]
    assert ev("python AND (async OR await) NOT java") == [0, 3]
    assert ev('"event loop"') == [0, 2]  # doc 3 has the words in the wrong order
    assert ev('python NOT "event loop"') == [1, 3]
    assert ev('"loop event"') == [3]
    assert ev("NOT python") == [4]
    assert ev("NOT python NOT java") == [4]
    assert ev("zebra") == [] and ev("python zebra") == []
    assert ev('"loop python"') == [] and ev('"no such phrase"') == []


def test_phrase_needs_positions(mk):
    ix = mk(TEXTS, "array", positions=False)
    assert parse("python await").evaluate(ix) == [1, 3]  # plain terms are fine
    with pytest.raises(PositionsRequired):
        parse('"event loop"').evaluate(ix)


def test_three_word_phrase_and_repeated_word(mk):
    ix = mk(["a b c d", "a b x c", "b a b a b", "x a a y"], positions=True)
    assert parse('"a b c"').evaluate(ix) == [0]
    assert parse('"b a b"').evaluate(ix) == [2]
    assert parse('"a a"').evaluate(ix) == [3]
