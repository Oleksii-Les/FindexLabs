"""Semantic search: text -> vector, similarity = one matrix product, and Reciprocal Rank Fusion.

    findex embed index.bin --out emb --model lsa            # fragments -> vectors -> emb.npy (+ emb.json ...)
    findex search index.bin "keeping a program from crashing" --mode hybrid --embeddings emb

Pieces
------
* **Fragments.** A document is cut into pieces of about ``CHUNK_CHARS`` characters (at a paragraph break,
  else at a space); a long book would otherwise be one blurred vector. A document's semantic score is the
  score of its **best fragment**.
* **Embedders** (``Embedder`` protocol: ``encode(texts) -> float32 [n, dim]``, rows L2-normalised):
  ``LsaEmbedder`` (TF-IDF + truncated SVD = latent semantic analysis; no download, query time needs only
  NumPy) and ``SentenceTransformerEmbedder`` (a neural model such as all-MiniLM-L6-v2; needs the
  ``minilm`` extra and a download). They are interchangeable: the matrix on disk records which one made it.
* **Matrix.** All fragment vectors are rows of one float32 ``.npy`` file, opened with ``mmap_mode="r"``.
  Rows are unit vectors, so cosine similarity = dot product: ``sims = matrix @ query_vector`` is ONE call into
  BLAS for every fragment. The best fragment per document is ``np.maximum.reduceat`` over the (contiguous)
  fragments of each document.
* **RRF.** ``score(d) = sum over rankings of 1 / (60 + rank_of_d)``. It uses only ranks, so BM25 scores
  (unbounded) and cosines (in [-1, 1]) can be fused without making their scales comparable.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from findex.engine import SearchEngine
from findex.model import Index
from findex.query import And, QuerySyntaxError, Term, parse
from findex.scoring import Scorer
from findex.tokenize import tokenize_all
from findex.vector import top_k

F32 = NDArray[np.float32]
CHUNK_CHARS = 800
RRF_K = 60  # the constant of the original RRF paper (Cormack et al., 2009)
FORMAT_VERSION = 1


def _optional(name: str) -> Any:
    """Import an optional dependency by name; typed ``Any`` because scipy / sentence-transformers are
    not installed everywhere and do not ship complete type information."""
    return importlib.import_module(name)


class SemanticUnavailable(RuntimeError):
    """An embedding model or file cannot be used (missing package, wrong index, damaged file)."""


# ---------------------------------------------------------------------------------- fragments
def chunk_spans(text: str, max_chars: int = CHUNK_CHARS) -> list[tuple[int, int]]:
    """Character spans ``(start, end)`` of fragments of at most ``max_chars``; blank pieces are dropped."""
    if max_chars < 50:
        raise ValueError("fragments shorter than 50 characters carry no meaning")
    spans: list[tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n:
        j = min(i + max_chars, n)
        if j < n:  # cut at a paragraph break, else at a space, in the second half of the window
            low = i + max_chars // 2
            cut = text.rfind("\n\n", low, j)
            if cut == -1:
                cut = text.rfind(" ", low, j)
            if cut != -1:
                j = cut
        if text[i:j].strip():
            spans.append((i, j))
        i = j
    return spans


# ------------------------------------------------------------------------------------ embedders
class Embedder(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> F32:
        """Unit-length row vectors, shape ``(len(texts), dim)``."""
        ...


def _normalise(m: NDArray[np.floating[Any]]) -> F32:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.where(norms == 0, 1.0, norms)).astype(np.float32)


@dataclass
class LsaEmbedder:
    """TF-IDF -> truncated SVD ("latent semantic analysis"). Words that occur in the same kinds of
    fragments end up close together, so a query can match a fragment that shares NO word with it.

    Fitting needs ``scipy`` (sparse matrices); encoding needs NumPy only.
    """

    terms: list[str]
    idf: NDArray[np.float32]
    components: F32  # (dim, vocabulary)
    name: str = "lsa"

    def __post_init__(self) -> None:
        self.vocab = {t: i for i, t in enumerate(self.terms)}

    @property
    def dim(self) -> int:
        return int(self.components.shape[0])

    def _weights(self, text: str) -> tuple[list[int], list[float]]:
        counts = Counter(t for t in tokenize_all(text) if t in self.vocab)
        idx = [self.vocab[t] for t in counts]
        w = [
            (1.0 + math.log(c)) * float(self.idf[i])
            for i, c in zip(idx, counts.values(), strict=True)
        ]
        return idx, w

    def encode(self, texts: Sequence[str]) -> F32:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, text in enumerate(texts):
            idx, w = self._weights(text)
            if idx:
                out[r] = self.components[:, idx] @ np.array(w, dtype=np.float32)
        return _normalise(out)

    @classmethod
    def fit(
        cls,
        texts: Iterable[str],
        dim: int = 256,
        max_terms: int = 60_000,
        min_df: int = 3,
        seed: int = 0,
    ) -> tuple[LsaEmbedder, F32]:
        """Fit on ``texts``; returns the embedder and the (normalised) matrix of those texts."""
        try:
            sp = _optional("scipy.sparse")
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise SemanticUnavailable("fitting LSA needs scipy: uv sync --extra semantic") from exc
        docs = [Counter(tokenize_all(t)) for t in texts]
        df: Counter[str] = Counter()
        for c in docs:
            df.update(c.keys())
        kept = [t for t, n in df.most_common() if n >= min_df][:max_terms]
        if len(kept) < 2 or len(docs) < 2:
            raise SemanticUnavailable("corpus too small to fit a semantic model")
        vocab = {t: i for i, t in enumerate(kept)}
        idf = np.array(
            [math.log((1 + len(docs)) / (1 + df[t])) + 1.0 for t in kept], dtype=np.float32
        )
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for r, c in enumerate(docs):
            for t, n in c.items():
                if (j := vocab.get(t)) is not None:
                    rows.append(r)
                    cols.append(j)
                    vals.append((1.0 + math.log(n)) * float(idf[j]))
        raw: Any = sp.csr_matrix(
            (vals, (rows, cols)), shape=(len(docs), len(kept)), dtype=np.float32
        )
        norms = cast(NDArray[np.float64], np.sqrt(raw.multiply(raw).sum(axis=1)).A1)
        scale: Any = sp.diags(1.0 / np.where(norms == 0, 1.0, norms))
        x: Any = scale.dot(raw).tocsr()  # every row has length 1: cosine = dot product
        dim = max(2, min(dim, min(int(x.shape[0]), int(x.shape[1])) - 1))
        components = _randomized_svd(x, dim, seed)
        model = cls(kept, idf, components)
        projected: Any = x @ components.T
        return model, _normalise(cast(NDArray[np.float32], np.asarray(projected, dtype=np.float32)))

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path, terms=np.array(self.terms), idf=self.idf, components=self.components
        )

    @classmethod
    def load(cls, path: Path) -> LsaEmbedder:
        with np.load(path, allow_pickle=False) as z:  # no pickle: the file is data, not code
            return cls([str(t) for t in z["terms"]], z["idf"], z["components"])


def _randomized_svd(x: Any, dim: int, seed: int, oversample: int = 12, power: int = 2) -> F32:
    """Right singular vectors (dim x vocabulary) of a sparse matrix, by the randomized range finder
    (Halko, Martinsson, Tropp 2011): a few passes over ``x`` instead of a full SVD."""
    rng = np.random.default_rng(seed)
    y = x @ rng.standard_normal((x.shape[1], dim + oversample)).astype(np.float32)
    for _ in range(power):
        y, _r = np.linalg.qr(y)
        y = x @ (x.T @ y)
    q, _r = np.linalg.qr(y)
    b = np.asarray((x.T @ q).T)  # (dim + oversample, vocabulary), small
    _u, _s, vt = np.linalg.svd(b, full_matrices=False)
    return np.ascontiguousarray(vt[:dim], dtype=np.float32)


class SentenceTransformerEmbedder:
    """A neural sentence encoder (default: all-MiniLM-L6-v2, 384 dimensions, ~90 MB, CPU is enough)."""

    name = "minilm"

    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64
    ) -> None:
        try:
            st = _optional("sentence_transformers")
        except ImportError as exc:
            raise SemanticUnavailable("the minilm model needs: uv sync --extra minilm") from exc
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Any = st.SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> F32:
        vectors = self._model.encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )  # fmt: skip
        return _normalise(np.asarray(vectors, dtype=np.float32))


# ---------------------------------------------------------------------------- files on disk
def fingerprint(index: Index) -> str:
    """Identifies the corpus an embedding matrix belongs to (same documents, same order)."""
    h = hashlib.sha1(usedforsecurity=False)
    for d in sorted(index.doc_meta):
        h.update(f"{d}\0{index.doc_meta[d].path}\n".encode())
    return h.hexdigest()[:16]


def _paths(prefix: Path) -> dict[str, Path]:
    return {
        "matrix": prefix.with_name(prefix.name + ".npy"),
        "chunks": prefix.with_name(prefix.name + ".chunks.npy"),
        "meta": prefix.with_name(prefix.name + ".json"),
        "model": prefix.with_name(prefix.name + ".lsa.npz"),
    }


def build_embeddings(
    index: Index,
    prefix: Path,
    read: Callable[[int], str | None],
    *,
    model: str = "lsa",
    dim: int = 256,
    chunk_chars: int = CHUNK_CHARS,
    minilm_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedder: Embedder | None = None,
) -> dict[str, Any]:
    """Cut every document into fragments, embed them, write ``prefix.npy`` / ``.chunks.npy`` / ``.json``
    (and ``.lsa.npz``). ``read(doc_id)`` returns a document's text. Returns the metadata written."""
    texts: list[str] = []
    chunks: list[tuple[int, int, int]] = []
    for doc_id in sorted(index.doc_meta):
        text = read(doc_id)
        if not text:
            continue
        for start, end in chunk_spans(text, chunk_chars):
            texts.append(text[start:end])
            chunks.append((doc_id, start, end))
    if not texts:
        raise SemanticUnavailable(
            "no document text could be read (is the corpus where the index says?)"
        )
    files = _paths(prefix)
    if (
        embedder is not None
    ):  # an embedder made by the caller (a different model, or a stub in tests)
        matrix = embedder.encode(texts)
        model = embedder.name
    elif model == "lsa":
        lsa, matrix = LsaEmbedder.fit(texts, dim=dim)
        lsa.save(files["model"])
        embedder = lsa
    elif model == "minilm":
        embedder = SentenceTransformerEmbedder(minilm_name)
        matrix = embedder.encode(texts)
    else:
        raise ValueError(f"unknown model {model!r}; choose lsa or minilm")
    np.save(files["matrix"], matrix.astype(np.float32))
    np.save(files["chunks"], np.array(chunks, dtype=np.int32))
    meta: dict[str, Any] = {
        "version": FORMAT_VERSION, "model": embedder.name, "dim": int(matrix.shape[1]),
        "chunks": len(texts), "chunk_chars": chunk_chars, "documents": len({c[0] for c in chunks}),
        "fingerprint": fingerprint(index),
    }  # fmt: skip
    if model == "minilm":
        meta["model_name"] = minilm_name
    files["meta"].write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


# ---------------------------------------------------------------------------------------- query
@dataclass(frozen=True, slots=True)
class SemanticHit:
    doc_id: int
    score: float  # cosine similarity of the best fragment
    start: int  # that fragment's characters in the document text
    end: int


class SemanticIndex:
    """The matrix + the embedder that made it. ``rank(query, k)`` -> best documents by best fragment."""

    def __init__(
        self, matrix: F32, chunks: NDArray[np.int32], embedder: Embedder, meta: dict[str, Any]
    ) -> None:
        if matrix.shape[0] != chunks.shape[0]:
            raise SemanticUnavailable("embedding files do not match each other")
        self.matrix, self.chunks, self.embedder, self.meta = matrix, chunks, embedder, meta
        owner = chunks[:, 0]
        self._first = np.flatnonzero(
            np.diff(owner, prepend=-1) != 0
        )  # first fragment of each document
        self._docs = owner[self._first].astype(np.int32)
        self._last = np.append(self._first[1:], len(owner))

    @property
    def model(self) -> str:
        return str(self.meta["model"])

    def __len__(self) -> int:
        return len(self._docs)

    def rank(self, query: str, k: int) -> list[SemanticHit]:
        q = self.embedder.encode([query])[0]
        if not q.any():  # nothing of the query is known to the model (e.g. only unknown words)
            return []
        sims = np.asarray(self.matrix @ q, dtype=np.float32)  # every fragment at once
        per_doc = np.maximum.reduceat(sims, self._first).astype(np.float64)
        docs, scores = top_k(self._docs, per_doc, k)
        by_doc = {int(d): i for i, d in enumerate(self._docs)}
        hits: list[SemanticHit] = []
        for d, s in zip(docs.tolist(), scores.tolist(), strict=True):
            lo, hi = int(self._first[by_doc[d]]), int(self._last[by_doc[d]])
            best = lo + int(np.argmax(sims[lo:hi]))
            hits.append(SemanticHit(d, s, int(self.chunks[best, 1]), int(self.chunks[best, 2])))
        return hits

    @classmethod
    def load(
        cls, prefix: Path, index: Index | None = None, embedder: Embedder | None = None
    ) -> SemanticIndex:
        """Open the files of ``findex embed``. ``embedder`` overrides the model named in the metadata (it must
        be the one that made the matrix); by default ``lsa`` and ``minilm`` are recreated from their files."""
        files = _paths(Path(prefix))
        for key in ("meta", "matrix", "chunks"):
            if not files[key].is_file():
                raise SemanticUnavailable(f"missing {files[key]} (run: findex embed)")
        meta = cast(dict[str, Any], json.loads(files["meta"].read_text(encoding="utf-8")))
        if meta.get("version") != FORMAT_VERSION:
            raise SemanticUnavailable("unsupported embeddings version")
        if index is not None and meta.get("fingerprint") != fingerprint(index):
            raise SemanticUnavailable(
                "these embeddings were made for a different index; run findex embed again"
            )
        matrix = np.load(
            files["matrix"], mmap_mode="r", allow_pickle=False
        )  # mapped, not read into memory
        chunks = np.load(files["chunks"], allow_pickle=False)
        if embedder is not None:
            if embedder.name != meta["model"]:
                raise SemanticUnavailable(
                    f"embeddings were made by {meta['model']!r}, not {embedder.name!r}"
                )
        elif meta["model"] == "lsa":
            embedder = LsaEmbedder.load(files["model"])
        elif meta["model"] == "minilm":
            embedder = SentenceTransformerEmbedder(
                meta.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
            )
        else:
            raise SemanticUnavailable(f"unknown model {meta['model']!r}")
        return cls(matrix, chunks, embedder, meta)


# -------------------------------------------------------------------------------------- fusion
def rrf(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion of ranked lists of doc ids -> ``[(doc_id, score)]`` best first.

    A document missing from a list simply gets nothing from it. Ties: larger doc id first (as in lab 3).
    """
    score: defaultdict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            score[doc_id] += 1.0 / (k + rank)
    return sorted(score.items(), key=lambda t: (t[1], t[0]), reverse=True)


def bag_of_words(query: str) -> str:
    """For the lexical half of a hybrid search: plain words (an implicit AND) become an OR, so that one
    missing word does not empty the list; a query with operators or quotes is kept as the user wrote it.
    Text that is not valid query syntax (a stray quote in a sentence) is treated as plain words."""
    try:
        tree = parse(query)
    except QuerySyntaxError:
        return " OR ".join(dict.fromkeys(tokenize_all(query)))
    if isinstance(tree, And) and all(isinstance(c, Term) for c in tree.children):
        return " OR ".join(dict.fromkeys(tree.terms()))
    return query


def query_terms(query: str) -> list[str]:
    """The words of a query for highlighting: from the parser, or from plain tokenizing if it is prose."""
    try:
        return list(dict.fromkeys(parse(query).terms()))
    except QuerySyntaxError:
        return list(dict.fromkeys(tokenize_all(query)))


def informative_terms(index: Index, terms: Sequence[str], max_df_ratio: float = 0.05) -> list[str]:
    """Drop words found in more than ``max_df_ratio`` of the documents ("a", "in", "file"): marking them in the
    snippet of a semantic hit is noise. Rare words stay."""
    limit = max_df_ratio * index.num_docs
    return [t for t in terms if 0 < index.df(t) <= limit]


# ----------------------------------------------------------------------------------- the 3 modes
Mode = Literal["lexical", "semantic", "hybrid"]
MODES: tuple[Mode, ...] = ("lexical", "semantic", "hybrid")
DEPTH = 100  # how deep each of the two rankings goes before they are fused


@dataclass(frozen=True, slots=True)
class ModeHit:
    doc_id: int
    score: float
    start: int | None = None  # best semantic fragment, if the mode knows one
    end: int | None = None


def search_mode(
    engine: SearchEngine,
    semantic: SemanticIndex | None,
    query: str,
    mode: Mode,
    k: int,
    *,
    scorer: Scorer | None = None,
    depth: int = DEPTH,
) -> list[ModeHit]:
    """The one entry point of the three modes. ``lexical`` = lab 3/7 (BM25 or TF-IDF over the query
    language); ``semantic`` = cosine of the best fragment; ``hybrid`` = RRF of BM25 over the query's
    words (OR) and the semantic ranking."""
    if mode == "lexical":
        return [
            ModeHit(r.doc_id, r.score)
            for r in engine.search(query, k, scorer=scorer, snippets=False)
        ]
    if semantic is None:
        raise SemanticUnavailable("semantic search is not configured (no embeddings loaded)")
    if mode == "semantic":
        return [ModeHit(h.doc_id, h.score, h.start, h.end) for h in semantic.rank(query, k)]
    depth = max(depth, k)
    words = bag_of_words(query)
    lexical = engine.search(words, depth, scorer=scorer, snippets=False) if words else []
    neural = semantic.rank(query, depth)
    spans = {h.doc_id: (h.start, h.end) for h in neural}
    fused = rrf([[r.doc_id for r in lexical], [h.doc_id for h in neural]])[:k]
    return [ModeHit(d, s, *spans.get(d, (None, None))) for d, s in fused]


# ------------------------------------------------------------------------------- text helpers
def make_reader(index: Index, corpus_root: Path | None = None) -> Callable[[int], str | None]:
    """``doc_id -> text``: the path stored in the index, else ``corpus_root / <file name>``."""
    from findex.snippets import read_text

    def read(doc_id: int) -> str | None:
        path = Path(index.doc_meta[doc_id].path)
        text = read_text(path)
        if text is None and corpus_root is not None:
            text = read_text(corpus_root / path.name)
        return text

    return read


def fragment_snippet(
    text: str, start: int | None, end: int | None, terms: Sequence[str], *, window: int = 110,
    marks: tuple[str, str] = ("**", "**"),
) -> str:  # fmt: skip
    """Excerpt for a semantic hit: the best fragment, with the query's own words marked when it has any
    (a paraphrase usually has none: then the start of the fragment is shown)."""
    from findex.snippets import make_snippet

    piece = text if start is None or end is None else text[start:end]
    return make_snippet(piece, terms, window=window, marks=marks)
