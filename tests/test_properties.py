"""Property-based tests (Hypothesis): state a law, let the computer hunt for counterexamples."""

from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from findex.boolean import ENGINES, merge_and, merge_andnot, merge_or
from findex.build import build_index
from findex.corpus import Document
from findex.model import Index, PostingList, StorageName
from findex.query import And, Node, Not, Or, Phrase, Term, parse
from findex.store import load, save
from findex.tokenize import tokenize

# ---- strategies -------------------------------------------------------------------------
WORDS = st.sampled_from(
    ["cat", "dog", "bird", "fish", "sea", "ship", "Cat", "DOG", "don't", "café"]
)
TEXTS = st.lists(st.lists(WORDS, max_size=12).map(" ".join), min_size=1, max_size=8)
ANY_TEXT = st.text(max_size=60)  # arbitrary Unicode, including nasty characters
SORTED_IDS = st.lists(st.integers(0, 60), unique=True, max_size=30).map(sorted)
TERM_NAMES = st.text(alphabet="abcdefgh", min_size=1, max_size=3)


def build(texts: list[str], storage: StorageName = "array", positions: bool = True) -> Index:
    docs = [Document(i, Path(f"{i}.txt"), t) for i, t in enumerate(texts)]
    return build_index(iter(docs), storage=storage, positions=positions)


def snapshot(index: Index) -> dict[str, list[tuple[int, int, tuple[int, ...]]]]:
    return {t: [(p.doc_id, p.tf, p.positions) for p in index[t]] for t in index}


# ---- 1. round trip: load(save(index)) == index ------------------------------------------
@settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(texts=TEXTS, positions=st.booleans(), fmt=st.sampled_from(["binary", "pickle"]))
def test_save_load_round_trip(tmp_path_factory, texts, positions, fmt):
    index = build(texts, storage="array", positions=positions)
    path = tmp_path_factory.mktemp("rt") / "i.idx"
    save(index, path, fmt)
    back = load(path)
    assert back == index  # Mapping equality: same terms, same postings
    assert snapshot(back) == snapshot(index)
    assert back.doc_lengths == index.doc_lengths and back.doc_meta == index.doc_meta
    assert back.has_positions == index.has_positions


# ---- 2. invariants of the index ---------------------------------------------------------
@given(texts=TEXTS, storage=st.sampled_from(["dataclass", "slots", "array"]), pos=st.booleans())
def test_postings_are_sorted_and_consistent(texts, storage, pos):
    index = build(texts, storage, pos)
    totals = dict.fromkeys(range(len(texts)), 0)
    for term in index:
        ids = [p.doc_id for p in index[term]]
        assert ids == sorted(set(ids)), "doc ids strictly increasing"
        for p in index[term]:
            assert p.tf >= 1
            totals[p.doc_id] += p.tf
            if pos:
                assert len(p.positions) == p.tf and list(p.positions) == sorted(p.positions)
    assert totals == index.doc_lengths  # every token is accounted for exactly once


@given(texts=TEXTS)
def test_all_storages_hold_the_same_information(texts):
    ref = snapshot(build(texts, "array", True))
    assert snapshot(build(texts, "slots", True)) == ref
    assert snapshot(build(texts, "dataclass", True)) == ref


# ---- 3. oracle: merge algorithms vs Python sets ----------------------------------------
@given(a=SORTED_IDS, b=SORTED_IDS)
def test_merge_matches_set_operations(a, b):
    assert merge_and(a, b) == sorted(set(a) & set(b))
    assert merge_or(a, b) == sorted(set(a) | set(b))
    assert merge_andnot(a, b) == sorted(set(a) - set(b))


@given(a=SORTED_IDS, b=SORTED_IDS)
def test_both_engines_agree(a, b):
    for merge_op, set_op in zip(ENGINES["merge"], ENGINES["set"], strict=True):
        assert merge_op(a, b) == set_op(a, b)


@given(a=SORTED_IDS, b=SORTED_IDS)
def test_boolean_algebra_laws(a, b):
    assert merge_and(a, b) == merge_and(b, a)
    assert merge_or(a, b) == merge_or(b, a)
    assert merge_and(a, a) == a and merge_or(a, a) == a
    # De Morgan inside a universe: universe - (a | b) == (universe - a) & (universe - b)
    u = list(range(61))
    assert merge_andnot(u, merge_or(a, b)) == merge_and(merge_andnot(u, a), merge_andnot(u, b))


# ---- 4. tokenizer laws ------------------------------------------------------------------
@given(text=ANY_TEXT)
def test_tokenizer_never_yields_empty_tokens(text):
    assert all(t for t in tokenize(text))


@given(text=ANY_TEXT)
def test_tokenizer_is_idempotent(text):
    once = list(tokenize(text))
    assert list(tokenize(" ".join(once))) == once


# ---- 5. query trees: render -> parse is the identity ------------------------------------
leaves = st.one_of(
    TERM_NAMES.map(Term),
    st.lists(TERM_NAMES, min_size=2, max_size=3).map(lambda ws: Phrase(*ws)),
)
trees = st.recursive(
    leaves,
    lambda inner: st.one_of(
        st.tuples(inner, inner).map(lambda t: t[0] & t[1]),
        st.tuples(inner, inner).map(lambda t: t[0] | t[1]),
        inner.map(lambda n: ~n),
    ),
    max_leaves=8,
)


@given(tree=trees)
def test_render_then_parse_gives_an_equal_tree(tree: Node):
    assert parse(str(tree)) == tree


@given(tree=trees)
def test_trees_are_well_formed(tree: Node):
    def check(n: Node) -> None:
        if isinstance(n, And | Or):
            assert len(n.children) >= 2
            assert all(not isinstance(c, type(n)) for c in n.children)  # always flattened
            for c in n.children:
                check(c)
        elif isinstance(n, Not):
            assert not isinstance(n.child, Not)  # ~~x collapses
            check(n.child)

    check(tree)


@given(texts=TEXTS)
def test_array_postings_behave_like_a_sequence(texts):
    index = build(texts, "array", True)
    for term in index:
        pl = index[term]
        assert isinstance(pl, PostingList)
        assert list(pl) == [pl[i] for i in range(len(pl))]
