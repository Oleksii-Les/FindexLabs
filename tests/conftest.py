from pathlib import Path

import pytest

from findex.build import build_index
from findex.corpus import Document


def make_index(texts, storage="slots", positions=True):
    docs = [Document(i, Path(f"{i}.txt"), t) for i, t in enumerate(texts)]
    return build_index(iter(docs), storage=storage, positions=positions)


@pytest.fixture
def mk():
    return make_index
