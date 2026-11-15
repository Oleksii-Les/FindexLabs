"""Lab 05: what does it cost to move data between processes?

    uv run python experiments/pickle_tax.py data/gutenberg --chunks 8

Process pools talk through pipes, so every argument and every result is pickled. This script
measures, on YOUR corpus, the three things that cross the boundary in our design:

  * the PATHS going in (tiny),
  * the PARTIAL INDEXES coming back (the real tax),
  * and, for comparison, what sending the raw TEXTS instead would cost.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

from findex.corpus import list_documents, read_document
from findex.parallel import build_partial, merge, plan_chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    paths = list_documents(args.corpus)[: args.limit]
    sizes = [p.stat().st_size for p in paths]
    text_bytes = sum(sizes)
    print(f"{len(paths)} documents, {text_bytes / 2**20:.1f} MiB of text, {args.chunks} chunks\n")
    print(
        f"{'':14}{'build':>9}{'pickle':>9}{'unpickle':>10}{'merge':>8}{'payload':>10}{'vs text':>9}{'tax':>7}"
    )
    for positions in (False, True):
        t_build = t_dump = t_load = 0.0
        payload = 0
        parts = []
        for chunk in plan_chunks(paths, args.chunks, sizes):
            t = time.perf_counter()
            part = build_partial(chunk.paths, chunk.first_doc_id, positions=positions)
            t_build += time.perf_counter() - t
            t = time.perf_counter()
            blob = pickle.dumps(part, protocol=pickle.HIGHEST_PROTOCOL)
            t_dump += time.perf_counter() - t
            t = time.perf_counter()
            parts.append(pickle.loads(blob))
            t_load += time.perf_counter() - t
            payload += len(blob)
        t = time.perf_counter()
        merge(parts)
        t_merge = time.perf_counter() - t
        name = "with positions" if positions else "no positions"
        print(
            f"{name:14}{t_build:8.2f}s{t_dump:8.3f}s{t_load:9.3f}s{t_merge:7.3f}s"
            f"{payload / 2**20:8.1f}Mi{payload / text_bytes:8.2f}x{(t_dump + t_load) / t_build:7.1%}"
        )

    texts = [read_document(p) for p in paths]
    t = time.perf_counter()
    blob = pickle.dumps(texts, protocol=pickle.HIGHEST_PROTOCOL)
    t_dump = time.perf_counter() - t
    t = time.perf_counter()
    pickle.loads(blob)
    t_load = time.perf_counter() - t
    print(
        f"\nsending all TEXTS instead : {len(blob) / 2**20:.1f} MiB, dump {t_dump:.3f}s, load {t_load:.3f}s"
    )
    print(f"sending all PATHS (ours)  : {len(pickle.dumps(paths)) / 1024:.1f} KiB")
    print("(the texts would also have to be read, and held in memory, by the parent first)")


if __name__ == "__main__":
    main()
