from findex.build import build_index
from findex.corpus import iter_documents
from findex.engine import search
from findex.snippets import make_snippet

TEXT = (
    "Lorem ipsum dolor sit amet. " * 20
    + "The quick brown fox jumps over the lazy dog near the river. "
    + "Lorem ipsum dolor sit amet. " * 20
)


def test_window_and_highlight():
    s = make_snippet(TEXT, ["fox", "dog"], window=30)
    assert "**fox**" in s and "**dog**" in s
    assert s.startswith("...") and s.endswith("...")
    assert len(s) < 3 + 2 * 30 + 40 + 3 + 12  # window + the matched span + markers
    assert "\n" not in s


def test_prefers_window_with_more_distinct_terms():
    text = "alpha " + "pad " * 100 + "alpha beta " + "pad " * 100 + "alpha alpha alpha"
    s = make_snippet(text, ["alpha", "beta"], window=20)
    assert "**beta**" in s  # the window containing both terms wins over 3x alpha


def test_whole_tokens_only_and_case_insensitive():
    s = make_snippet("concatenate the Cat sat", ["cat"], window=40)
    assert "**Cat**" in s and "con**cat**" not in s


def test_apostrophe_variants_and_custom_marks():
    s = make_snippet("I don\u2019t know", ["don't"], window=20, marks=("<", ">"))
    assert "<don\u2019t>" in s


def test_fallback_to_normalized_text():
    s = make_snippet("Die STRASSE ist lang", ["strasse"], window=20)
    assert "**" in s


def test_no_match_returns_beginning():
    assert make_snippet("hello world", ["zebra"], window=5) == "hello worl"
    assert make_snippet("hello", []) == "hello"


def test_snippets_end_to_end(tmp_path):
    (tmp_path / "a.txt").write_text(
        "Title: Foxes\nThe quick brown fox jumps.\n" + "pad " * 200
    )
    (tmp_path / "b.txt").write_text("nothing here " * 5)
    ix = build_index(iter_documents(tmp_path), positions=False)
    (hit,) = search(ix, "fox", snippets=True)
    assert "**fox**" in hit.snippet and hit.title == "Foxes"
