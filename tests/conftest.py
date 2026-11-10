from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from findex.build import build_index
from findex.corpus import Document
from findex.model import Index, StorageName
from findex.store import save

# A tiny, hand-written corpus. File name == doc position, so tests can reason about ids
# without depending on directory listing order.
TINY_CORPUS: dict[str, str] = {
    "a_whales.txt": "Title: Whales\nThe white whale swam in the sea. The whale is white.",
    "b_ships.txt": "Title: Ships\nA ship sails the sea. Ships carry sailors across the ocean.",
    "c_oceans.txt": "Title: Oceans\nThe ocean is deep. Whales live in the ocean and the sea.",
    "d_cats.txt": "Title: Cats\nCats purr. A cat is not a whale. Cats hate the sea.",
    "e_python.txt": "Title: Python\nPython has an event loop. The event loop runs async code.",
    "f_empty.txt": "",
}


def make_index(
    texts: Sequence[str], storage: StorageName = "slots", positions: bool = True
) -> Index:
    docs = [Document(i, Path(f"{i}.txt"), t) for i, t in enumerate(texts)]
    return build_index(iter(docs), storage=storage, positions=positions)


@pytest.fixture
def mk() -> Callable[..., Index]:
    """Factory: ``mk(["text", ...], "array", positions=True)`` -> a built Index."""
    return make_index


@pytest.fixture(scope="session")
def tiny_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Directory with TINY_CORPUS on disk. Session scope: written once, never modified."""
    root = tmp_path_factory.mktemp("tiny_corpus")
    for name, text in TINY_CORPUS.items():
        (root / name).write_text(text, encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def tiny_index_file(tiny_corpus: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An index file (with positions) of the tiny corpus. Read-only for tests."""
    from findex.corpus import iter_documents

    index = build_index(iter_documents(tiny_corpus), storage="array", positions=True)
    out = tmp_path_factory.mktemp("tiny_index") / "tiny.bin"
    save(index, out, "binary")
    return out
