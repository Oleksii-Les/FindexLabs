"""Lab 08: fragments, embeddings on disk, cosine ranking, RRF, the three modes, the web and CLI surface."""

from __future__ import annotations

import itertools
import json
import random
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from findex import semantic
from findex.build import build_index
from findex.cli import app as cli_app
from findex.corpus import Document, iter_documents
from findex.engine import SearchEngine
from findex.model import Index
from findex.query import PositionsRequired
from findex.semantic import (
    F32,
    LsaEmbedder,
    SemanticIndex,
    SemanticUnavailable,
    SentenceTransformerEmbedder,
    bag_of_words,
    build_embeddings,
    chunk_spans,
    fingerprint,
    informative_terms,
    make_reader,
    query_terms,
    rrf,
    search_mode,
)
from findex.store import save
from findex.web.app import create_app
from findex.web.deps import get_service, get_service_or_none
from findex.web.service import SearchService
from findex.web.settings import Settings


class LetterEmbedder:
    """A stand-in model: the vector is the letter histogram. Deterministic, no download, no scipy --
    it only has to be *an* Embedder (the protocol is the point), words with the same letters are 'close'."""

    name = "letters"

    def encode(self, texts: Sequence[str]) -> F32:
        out = np.zeros((len(texts), 26), dtype=np.float32)
        for r, text in enumerate(texts):
            for ch in text.lower():
                if "a" <= ch <= "z":
                    out[r, ord(ch) - 97] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


@pytest.fixture
def index(tiny_corpus: Path) -> Index:
    return build_index(iter_documents(tiny_corpus), storage="array", positions=True)


@pytest.fixture
def prefix(index: Index, tiny_corpus: Path, tmp_path: Path) -> Path:
    prefix = tmp_path / "emb"
    build_embeddings(
        index, prefix, make_reader(index, tiny_corpus), chunk_chars=60, embedder=LetterEmbedder()
    )
    return prefix


# ------------------------------------------------------------------------------- fragments
def test_fragments_cover_the_text_and_respect_the_limit():
    text = ("Paragraph one is here.\n\n" * 3) + ("word " * 200) + "\n\n" + "x" * 400
    spans = chunk_spans(text, 120)
    assert all(0 < e - s <= 120 for s, e in spans)
    assert all(a[1] <= b[0] for a, b in itertools.pairwise(spans))  # ordered, no overlap
    covered = "".join(text[s:e] for s, e in spans)
    assert covered.replace(" ", "").replace("\n", "") == text.replace(" ", "").replace("\n", "")
    assert chunk_spans("") == [] and chunk_spans("   \n\n  ") == []


def test_fragments_prefer_a_paragraph_break():
    text = "a" * 70 + "\n\n" + "b" * 70
    assert chunk_spans(text, 100)[0] == (0, 70)
    with pytest.raises(ValueError, match="50"):
        chunk_spans("text", 10)


# ------------------------------------------------------------------------------------ RRF
def test_rrf_scores_and_ties():
    fused = dict(rrf([[1, 2, 3], [3, 1]]))
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[3] == pytest.approx(1 / 63 + 1 / 61)
    assert fused[2] == pytest.approx(1 / 62)
    order = [d for d, _ in rrf([[1, 2, 3], [3, 1]])]
    assert order == [1, 3, 2]  # 1: 1/61+1/62 beats 3: 1/63+1/61
    assert [d for d, _ in rrf([[5], [9]])] == [
        9,
        5,
    ]  # equal scores: the larger doc id first, as in lab 3
    assert rrf([]) == [] and rrf([[], []]) == []


def test_bag_of_words_and_helpers(index: Index):
    assert bag_of_words("white whale") == "white OR whale"
    assert bag_of_words("whale") == "whale"
    assert (
        bag_of_words("whale AND NOT ship") == "whale AND NOT ship"
    )  # operators: as the user wrote it
    assert bag_of_words('"white whale" sea') == '"white whale" sea'
    assert bag_of_words('what is "this') == "what OR is OR this"  # prose with a stray quote
    assert query_terms("(whale") == ["whale"] and query_terms("whale NOT sea") == [
        "whale"
    ]  # negated words are not highlighted
    assert informative_terms(index, ["the", "ocean", "zzz"], max_df_ratio=0.5) == ["ocean"]


# ---------------------------------------------------------------------- the files on disk
def test_embeddings_roundtrip_and_cosine_ranking(index: Index, prefix: Path):
    meta = json.loads(prefix.with_name("emb.json").read_text())
    assert (
        meta["model"] == "letters" and meta["dim"] == 26 and meta["documents"] == 5
    )  # f_empty has no text
    matrix = np.load(prefix.with_name("emb.npy"))
    assert matrix.dtype == np.float32 and np.allclose(
        np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5
    )

    sem = SemanticIndex(
        np.load(prefix.with_name("emb.npy"), mmap_mode="r"), np.load(prefix.with_name("emb.chunks.npy")),
        LetterEmbedder(), meta,
    )  # fmt: skip
    assert len(sem) == 5 and isinstance(sem.matrix, np.memmap)  # mapped, not read into memory
    hits = sem.rank("whales live in the ocean and the sea", 3)
    assert next(h.doc_id for h in hits) == 2  # c_oceans.txt is that sentence
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert all(0 <= h.start < h.end for h in hits)
    assert sem.rank("12345 !!!", 3) == []  # no letters: a zero vector matches nothing


def test_best_fragment_points_into_the_document(index: Index, tiny_corpus: Path, tmp_path: Path):
    long_text = (
        "Dull filler about nothing. " * 12
        + "\n\nzebra zebra zebra quagga okapi\n\n"
        + "More filler words. " * 12
    )
    docs = [long_text, "completely different text about ships"]
    ix = build_index(
        iter([Document(i, Path(f"{i}.txt"), t) for i, t in enumerate(docs)]), storage="array"
    )
    build_embeddings(
        ix, tmp_path / "e", lambda d: docs[d], chunk_chars=100, embedder=LetterEmbedder()
    )
    sem = SemanticIndex.load(tmp_path / "e", ix, LetterEmbedder())
    hit = sem.rank("zebra quagga okapi", 1)[0]
    assert hit.doc_id == 0 and "zebra" in docs[0][hit.start : hit.end]


def test_load_checks_the_files_and_the_index(index: Index, prefix: Path, tmp_path: Path):
    assert SemanticIndex.load(prefix, index, LetterEmbedder()).model == "letters"
    other = build_index(
        iter_documents(tmp_path), storage="array"
    )  # an empty corpus: different documents
    meta = json.loads(prefix.with_name("emb.json").read_text())
    assert meta["fingerprint"] == fingerprint(index) != fingerprint(other)
    with pytest.raises(SemanticUnavailable, match="different index"):
        SemanticIndex.load(prefix, other, LetterEmbedder())
    with pytest.raises(SemanticUnavailable, match="missing"):
        SemanticIndex.load(tmp_path / "nope")
    meta["version"] = 99
    prefix.with_name("emb.json").write_text(json.dumps(meta))
    with pytest.raises(SemanticUnavailable, match="version"):
        SemanticIndex.load(prefix)
    meta.update(version=1, model="mystery")
    prefix.with_name("emb.json").write_text(json.dumps(meta))
    with pytest.raises(SemanticUnavailable, match="unknown model"):
        SemanticIndex.load(prefix)
    with pytest.raises(SemanticUnavailable, match="not 'letters'"):
        SemanticIndex.load(prefix, embedder=LetterEmbedder())


def test_nothing_to_embed_is_an_error(index: Index, tmp_path: Path):
    with pytest.raises(SemanticUnavailable, match="no document text"):
        build_embeddings(index, tmp_path / "x", lambda _d: None, embedder=LetterEmbedder())
    with pytest.raises(ValueError, match="unknown model"):
        build_embeddings(index, tmp_path / "x", lambda _d: "text " * 30, model="gpt")


# ------------------------------------------------------------------------------ LSA
def topical_corpus(n: int = 80, seed: int = 4) -> tuple[list[str], list[int]]:
    """Two topics. Every sea text says 'ship' OR 'vessel' (never both) next to shared sea words, so the
    two words never co-occur in a text but occur in the same *kind* of text -- what LSA captures."""
    rng = random.Random(seed)
    sea = ["sail", "ocean", "harbor", "captain", "wave", "anchor", "deck", "crew"]
    music = ["piano", "melody", "concert", "violin", "rhythm", "chord", "stage", "opera"]
    texts, topic = [], []
    for i in range(n):
        if i % 2 == 0:
            words = [rng.choice(["ship", "vessel"]), *rng.choices(sea, k=14)]
            topic.append(0)
        else:
            words = [rng.choice(["song", "tune"]), *rng.choices(music, k=14)]
            topic.append(1)
        rng.shuffle(words)
        texts.append(" ".join(words))
    return texts, topic


def test_lsa_finds_texts_that_do_not_contain_the_query_word(tmp_path: Path):
    pytest.importorskip("scipy")
    texts, topic = topical_corpus()
    model, matrix = LsaEmbedder.fit(texts, dim=3, min_df=2)  # about the number of topics
    assert matrix.shape == (len(texts), model.dim) and np.allclose(
        np.linalg.norm(matrix, axis=1), 1, atol=1e-5
    )
    sims = matrix @ model.encode(["vessel"])[0]
    ship_only = [i for i, t in enumerate(texts) if topic[i] == 0 and "vessel" not in t]
    music = [i for i, t in enumerate(texts) if topic[i] == 1]
    assert ship_only and music
    # a text that says "ship" (never "vessel") is closer to the query "vessel" than ANY music text
    assert sims[ship_only].min() > sims[music].max()
    assert model.encode(["qqqq unknownword"]).any() is np.False_  # nothing known -> a zero vector
    model.save(tmp_path / "m.npz")
    again = LsaEmbedder.load(tmp_path / "m.npz")
    assert np.allclose(again.encode(["vessel"]), model.encode(["vessel"]))


def test_lsa_needs_enough_text():
    pytest.importorskip("scipy")
    with pytest.raises(SemanticUnavailable, match="too small"):
        LsaEmbedder.fit(["one single document"], min_df=1)


# ------------------------------------------------------------- the modes (engine level)
def test_search_modes(index: Index, prefix: Path):
    engine = SearchEngine(index)
    sem = SemanticIndex.load(prefix, index, LetterEmbedder())
    lex = search_mode(engine, sem, "whale", "lexical", 5)
    assert [h.doc_id for h in lex] and lex[0].start is None
    s = search_mode(engine, sem, "whales and ocean", "semantic", 5)
    assert s and s[0].start is not None
    h = search_mode(engine, sem, "whales and ocean", "hybrid", 3)
    assert 0 < len(h) <= 3 and h[0].score == pytest.approx(max(x.score for x in h))
    # the lexical half of hybrid is an OR over the words: the AND of these words matches nothing at all
    assert engine.search("whale ocean python", 5, snippets=False) == []
    assert search_mode(engine, sem, "whale ocean python", "hybrid", 5)
    assert search_mode(engine, sem, "", "hybrid", 5) == []  # nothing to look for, in either half
    with pytest.raises(SemanticUnavailable, match="not configured"):
        search_mode(engine, None, "whale", "semantic", 5)
    assert search_mode(engine, None, "whale", "lexical", 5)  # lexical never needs embeddings


# ------------------------------------------------------------------------- minilm adapter
def test_minilm_adapter_without_and_with_the_package(monkeypatch: pytest.MonkeyPatch):
    def missing(_name: str) -> object:
        raise ImportError("no module")

    monkeypatch.setattr(semantic, "_optional", missing)
    with pytest.raises(SemanticUnavailable, match="--extra minilm"):
        SentenceTransformerEmbedder()

    class FakeModel:
        def __init__(self, name: str) -> None:
            self.name = name

        def encode(self, texts: list[str], **kw: object) -> np.ndarray:
            assert kw["normalize_embeddings"] is True
            return np.array([[3.0, 4.0]] * len(texts))

    class FakeModule:
        SentenceTransformer = FakeModel

    monkeypatch.setattr(semantic, "_optional", lambda _n: FakeModule)
    emb = SentenceTransformerEmbedder("some/model")
    assert np.allclose(emb.encode(["a", "b"]), [[0.6, 0.8], [0.6, 0.8]])


# ------------------------------------------------------------------------------- the web
@pytest.fixture
def web(index: Index, prefix: Path, tiny_corpus: Path) -> Callable[[bool], TestClient]:
    def make(with_semantic: bool) -> TestClient:
        service = SearchService(
            index,
            window=40,
            semantic=SemanticIndex.load(prefix, index, LetterEmbedder()) if with_semantic else None,
        )
        app: FastAPI = create_app(Settings(), setup_logging=False)
        app.dependency_overrides[get_service] = lambda: service
        app.dependency_overrides[get_service_or_none] = lambda: service
        return TestClient(app)

    return make


@pytest.mark.parametrize("mode", ["semantic", "hybrid"])
def test_api_modes(web: Callable[[bool], TestClient], mode: str):
    client = web(True)
    body = client.get("/search", params={"q": "whales in the ocean", "mode": mode, "k": 2}).json()
    assert body["mode"] == mode and len(body["hits"]) == 2 and body["total"] >= 2
    assert body["pages"] == -(-body["total"] // 2)
    page2 = client.get(
        "/search", params={"q": "whales in the ocean", "mode": mode, "k": 2, "page": 2}
    ).json()
    ids = [h["doc_id"] for h in body["hits"] + page2["hits"]]
    assert len(ids) == len(set(ids))  # pages do not overlap
    assert client.get("/stats").json()["semantic_model"] == "letters"
    assert body["hits"][0]["snippet"]  # the best fragment, not empty


def test_api_mode_without_embeddings_is_501_and_bad_mode_is_422(web: Callable[[bool], TestClient]):
    client = web(False)
    r = client.get("/search", params={"q": "whale", "mode": "semantic"})
    assert (
        r.status_code == 501 and "not configured" in r.json()["detail"] and r.json()["request_id"]
    )
    assert client.get("/search", params={"q": "whale", "mode": "lexical"}).status_code == 200
    assert client.get("/search", params={"q": "whale", "mode": "psychic"}).status_code == 422
    assert client.get("/stats").json()["semantic_model"] is None


def test_page_shows_the_mode_selector_only_when_it_works(web: Callable[[bool], TestClient]):
    with_sem = (
        web(True).get("/", params={"q": "whales and the ocean", "mode": "hybrid", "k": 2}).text
    )
    assert 'name="mode"' in with_sem and "RRF" in with_sem and "mode=hybrid" in with_sem
    plain = web(False).get("/", params={"q": "whale"}).text
    assert 'name="mode"' not in plain
    notice = web(False).get("/", params={"q": "whale", "mode": "semantic"})
    assert notice.status_code == 200 and "not configured" in notice.text


def test_service_survives_a_broken_embeddings_file(index: Index, tmp_path: Path, tiny_corpus: Path):
    path = tmp_path / "index.bin"
    save(index, path)
    svc = SearchService.from_path(path, embeddings=tmp_path / "missing")
    assert not svc.semantic_ready  # logged and ignored: keyword search still works
    assert svc.search("whale").total == 2


# -------------------------------------------------------------------------------- the CLI
def test_cli_embed_and_search_modes(index: Index, tiny_corpus: Path, tmp_path: Path):
    pytest.importorskip("scipy")
    path = tmp_path / "i.bin"
    save(index, path)
    emb = tmp_path / "emb"
    runner = CliRunner()
    r = runner.invoke(
        cli_app,
        ["embed", str(path), "--out", str(emb), "--dim", "4", "--corpus-root", str(tiny_corpus)],
    )
    assert r.exit_code == 0, r.output
    # tiny corpus: too few documents for min_df=3 may be refused; both outcomes must be a clean message
    if (tmp_path / "emb.json").exists():
        r = runner.invoke(
            cli_app,
            [
                "search",
                str(path),
                "whale",
                "--mode",
                "semantic",
                "--embeddings",
                str(emb),
                "--json",
            ],
        )
        assert r.exit_code == 0, r.output
    r = runner.invoke(cli_app, ["search", str(path), "whale", "--mode", "semantic"])
    assert r.exit_code == 1 and "--embeddings" in r.output
    r = runner.invoke(cli_app, ["search", str(path), "whale", "--mode", "psychic"])
    assert r.exit_code != 0


# ------------------------------------------------------------------------ the P@k table
def test_precision_table_for_systems(index: Index):
    from findex.evaluate import LabeledQuery, evaluate_systems, render_semantic_table

    queries: list[LabeledQuery] = [
        {"query": "whale", "relevant": ["whales"], "kind": "keyword"},
        {"query": "a big sea creature", "relevant": ["whales"], "kind": "paraphrase"},
    ]
    systems = {
        "perfect": lambda q, k: [0, 1, 2, 3, 4][:k],  # doc 0 (Whales) is first
        "wrong": lambda q, k: [4, 5][:k],
        "phrase": lambda q, k: (_ for _ in ()).throw(PositionsRequired("needs positions")),
    }
    rows = evaluate_systems(index, queries, systems, k=5)
    assert [r["perfect"] for _, r in rows] == [0.2, 0.2] and [r["wrong"] for _, r in rows] == [
        0.0,
        0.0,
    ]
    assert rows[0][1]["phrase"] is None
    table = render_semantic_table(rows, list(systems), 5)
    assert "mean, keyword" in table and "mean, paraphrase" in table and "n/a" in table
