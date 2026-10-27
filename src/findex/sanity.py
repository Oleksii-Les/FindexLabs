"""The three ranking sanity checks of Lab 3, run on YOUR index.

Usage:  python -m findex.sanity index.bin
Prints Markdown ready for the README. Picks terms/documents automatically, so the
evidence comes from your corpus rather than from a toy example.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

from findex.engine import search
from findex.model import Index, Posting
from findex.scoring import BM25, TfIdf
from findex.store import open_index


def posting_of(index: Index, term: str, doc_id: int) -> Posting:
    return next(p for p in index[term] if p.doc_id == doc_id)


def check_rare_beats_common(index: Index, bm25: BM25) -> None:
    n = index.num_docs
    words = [t for t in index if t.isalpha()]
    rare_terms = sorted(t for t in words if 1 <= index.df(t) <= 3)
    common_terms = sorted(
        (t for t in words if index.df(t) < 0.9 * n), key=lambda t: -index.df(t)
    )
    print("### Check 1 -- a rare term outranks a common one\n")
    if not rare_terms or not common_terms:
        print("_not enough variety in the vocabulary to pick terms_\n")
        return
    # try several (rare, common) pairs until one has documents containing ONLY one of them
    pair = None
    for rare in rare_terms[:200]:
        for common in common_terms[:30]:
            d_rare, d_common = set(index.doc_ids(rare)), set(index.doc_ids(common))
            only_rare, only_common = (
                sorted(d_rare - d_common),
                sorted(d_common - d_rare),
            )
            if only_rare and only_common:
                pair = (rare, common, only_rare, only_common)
                break
        if pair:
            break
    if pair is None:
        print("_no (rare, common) pair with documents containing only one of them_\n")
        return
    rare, common, only_rare, only_common = pair
    a = only_rare[0]
    tf_a = posting_of(index, rare, a).tf
    # fairest opponent: contains ONLY the common term, with at least as many occurrences
    # (so the rare doc cannot win through tf) and the closest length
    cands = [d for d in only_common if posting_of(index, common, d).tf >= tf_a]
    pool = cands or only_common
    b = min(pool, key=lambda d: abs(index.doc_length(d) - index.doc_length(a)))
    tf_b = posting_of(index, common, b).tf
    ranking = {
        r.doc_id: i for i, r in enumerate(search(index, f"{rare} OR {common}", k=n), 1)
    }
    s_a = bm25.score(rare, posting_of(index, rare, a), index)
    s_b = bm25.score(common, posting_of(index, common, b), index)
    print(
        f"Query `{rare} OR {common}`. `{rare}` occurs in {index.df(rare)} docs, `{common}` in {index.df(common):,}.\n"
    )
    print("| Document | Contains | tf | length | BM25 score | Rank |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    print(
        f"| {Path(index.doc_meta[a].path).name} | only `{rare}` (rare) | {tf_a} | {index.doc_length(a):,} | {s_a:.3f} | {ranking[a]} |"
    )
    print(
        f"| {Path(index.doc_meta[b].path).name} | only `{common}` (common) | {tf_b} | {index.doc_length(b):,} | {s_b:.3f} | {ranking[b]} |"
    )
    print(
        f"\n{'PASS' if ranking[a] < ranking[b] else 'FAIL'}: the rare-term document ranks higher.\n"
    )


def check_saturation(index: Index, bm25: BM25, tfidf: TfIdf) -> None:
    n = index.num_docs
    print("### Check 2 -- the 20th repetition adds (almost) nothing under BM25\n")
    terms = sorted(
        (t for t in index if t.isalpha() and index.df(t) < 0.9 * n),
        key=lambda t: -index.df(t),
    )
    if not terms:
        print("_no suitable term_\n")
        return
    term = terms[0]
    docs = sorted(index.doc_ids(term), key=index.doc_length)
    doc = docs[len(docs) // 2]  # a median-length document that really contains the term
    print(
        f"Term `{term}`, document {Path(index.doc_meta[doc].path).name} (length {index.doc_length(doc):,}); the tf values are hypothetical, idf and length are real.\n"
    )
    print("| tf | BM25 | TF-IDF (1+ln tf) | TF-IDF (raw tf) |")
    print("| ---: | ---: | ---: | ---: |")
    raw = TfIdf(sublinear=False)
    for tf in (1, 2, 5, 10, 19, 20):
        p = Posting(doc, tf)
        print(
            f"| {tf} | {bm25.score(term, p, index):.4f} | {tfidf.score(term, p, index):.4f} | {raw.score(term, p, index):.4f} |"
        )

    def gain(s, a, b):
        return s.score(term, Posting(doc, b), index) - s.score(
            term, Posting(doc, a), index
        )

    g1, g20 = gain(bm25, 1, 2), gain(bm25, 19, 20)
    print(
        f"\nBM25: going 1->2 adds {g1:.4f}, going 19->20 adds {g20:.4f} ({g20 / g1:.1%} of the first gain)."
    )
    print(
        f"{'PASS' if g20 < 0.05 * g1 else 'FAIL'}: saturation (limit for tf->inf is {bm25.idf(term, index) * (bm25.k1 + 1):.3f}).\n"
    )


def check_length_normalization(index: Index, bm25: BM25, tfidf: TfIdf) -> None:
    n = index.num_docs
    print("### Check 3 -- a short document with one match beats a very long one\n")
    for term in sorted(
        (t for t in index if t.isalpha() and 2 <= index.df(t) < 0.9 * n),
        key=lambda t: -index.df(t),
    ):
        once = [p.doc_id for p in index[term] if p.tf == 1]
        if len(once) < 2:
            continue
        short = min(once, key=index.doc_length)
        long_ = max(once, key=index.doc_length)
        if index.doc_length(long_) < 2 * index.doc_length(short):
            continue
        ps, pl = posting_of(index, term, short), posting_of(index, term, long_)
        ranking = {r.doc_id: i for i, r in enumerate(search(index, term, k=n), 1)}
        print(f"Query `{term}`: both documents contain it exactly once (tf = 1).\n")
        print("| Document | length | BM25 | Rank (BM25) | TF-IDF |")
        print("| --- | ---: | ---: | ---: | ---: |")
        for label, d, p in (("short", short, ps), ("long", long_, pl)):
            print(
                f"| {Path(index.doc_meta[d].path).name} ({label}) | {index.doc_length(d):,} | {bm25.score(term, p, index):.4f} | {ranking[d]} | {tfidf.score(term, p, index):.4f} |"
            )
        ok = bm25.score(term, ps, index) > bm25.score(term, pl, index)
        print(
            f"\n{'PASS' if ok else 'FAIL'}: BM25 prefers the short document; TF-IDF (no length normalization) ties them.\n"
        )
        return
    print(
        "_no term found that occurs once in two documents of clearly different length_\n"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m findex.sanity")
    ap.add_argument("index", type=Path)
    args = ap.parse_args(argv)
    bm25, tfidf = BM25(), TfIdf()
    with open_index(args.index) as index:
        lens = list(index.doc_lengths.values())
        print(
            f"{index!r}; document length median {statistics.median(lens):,.0f}, avg {index.avg_doc_length:,.0f}\n"
        )
        check_rare_beats_common(index, bm25)
        check_saturation(index, bm25, tfidf)
        check_length_normalization(index, bm25, tfidf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
