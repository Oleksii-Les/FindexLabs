"""The deliberately GREEDY version — everything in lists. For the M4 comparison only."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from findex.corpus import iter_documents
from findex.stats import CorpusStats
from findex.tokenize import tokenize


def compute_stats_eager(root: Path, limit: int | None = None) -> CorpusStats:
    docs = list(iter_documents(root))  # 1) whole corpus text in RAM
    if limit is not None:
        docs = docs[:limit]
    token_lists = [
        list(tokenize(d.text)) for d in docs
    ]  # 2) every token as a str object
    all_tokens = [
        tok for toks in token_lists for tok in toks
    ]  # 3) a flat copy of pointers
    counts = Counter(all_tokens)
    return CorpusStats(len(docs), len(all_tokens), len(counts), counts)
