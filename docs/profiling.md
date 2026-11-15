# Lab 08 — Where does the time go? (written BEFORE any optimization)

Corpus for every number below: 17 006 documents, 46.6 MB of English technical text (documentation and docstrings
found on the build machine, cut into ~4 KB documents: `scripts/sandbox_corpus.py`), index built with `--positions`:
171 496 terms, 50 MiB file. Machine: 2 vCPU Linux sandbox, Python 3.13. (Project Gutenberg was unreachable there;
the same commands run unchanged on your corpus — see the commands at the end.)

Tools: `cProfile` (exact call counts, but ~2.5x slower than real), `py-spy record -f flamegraph` (sampling, almost no
distortion; the SVGs are in this folder), Scalene (splits time into Python / native / system).
Honest timing, no profiler: **index 5.5 s**; **search 33 ms per query** after a **0.7 s load**.

Flame graphs (open in a browser, hover for numbers): [`flame-index.svg`](flame-index.svg),
[`flame-search.svg`](flame-search.svg). Raw cProfile output: `profile-index.txt`, `profile-search.txt`.

## `findex search` — the query `function OR value OR error` (15 439 matching postings per query)

Before the table: for a one-shot `findex search` the **load (0.7 s) is 20x the query (33 ms)**. That is a different
problem (what the server of lab 7 solves by loading once), and is not what this lab optimizes; it is native time
(`array.frombytes`, 2.8 % of Scalene's samples on a 60-query run, which includes the load).

| # | Where (cProfile, 100 queries, cumulative) | Share | Hypothesis | Python or native? (Scalene) |
| --- | --- | ---: | --- | --- |
| 1 | `scoring.py: BM25.score` + `idf`, called **once per posting** — 1.54 M calls, and `idf` (a `math.log` + `df()` + `num_docs`) is recomputed for every posting although it is the same for the whole term | 4.0 s of 7.9 s = **51 %** | Pure interpreter overhead: ~1.5 M Python function calls, attribute lookups and float boxing. The arithmetic itself is trivial. Expect a vector expression over the whole posting array to remove almost all of it. | Python (lines 61, 57: 1.1 + 0.5 % python vs 0.6 + 0.5 % native, the native part is `math.log`) |
| 2 | `PostingList.__iter__`: builds one frozen `Posting` object per posting (`Posting.__init__`, 1.0 M of them) | 1.6 s = **20 %** | Allocation + `__init__` of a dataclass per posting. Only needed to hand the scorer an object; with the arrays in NumPy no object is needed at all. | Python allocation (model.py:88: 1.3 % python, 3.3 % native = `tuple(slice)`/`array` slicing) |
| 3 | `heapq.nlargest` over `SearchResult` objects (one dataclass built per scored document, compared with the generated `order=True` `__lt__`) | 1.4 s = **18 %** | O(n log k) is right, but n here is ~10 000 candidates and each costs an object + Python-level comparisons. `np.argpartition` selects the top k in C in O(n) and never creates the objects. | Python (engine.py:61 1.0 % python) |

The three together are ~89 % of the query; `boolean.merge_or` (matching) is 5 % and snippets are negligible. So the
bound on what NumPy can buy for this query is about 1 / 0.11 ≈ 9x on the scoring part; matching and snippets stay.

## `findex index`

| # | Where (cProfile, cumulative) | Share | Hypothesis | Python or native? (Scalene) |
| --- | --- | ---: | --- | --- |
| 1 | `build_index` body: `for pos, tok in enumerate(tokenize(...)): occ[tok].append(pos)`, one dict lookup + list append per **token** (7.0 M) | 4.8 s of 13.2 s tottime = **37 %** | Interpreter-bound: a Python loop over every token. The only way out of it is fewer Python-level steps per token. | Python (build.py:69: 2.3 % python / 0.9 % native) |
| 2 | `tokenize`: a generator resumed 7.0 M times, `match.group()` 7.0 M calls | 3.8 s = **29 %** | Half is the regex (native), half is generator resumption + method call per token (Python). `re.findall` returns the whole list in one native call. | Mixed: tokenize.py:42 — 3.9 % python, 7.9 % native (the regex engine) |
| 3 | per-term bookkeeping: `PostingList.__init__` (171 k, three arrays each) + `array.append` ×5.2 M + `array.extend` ×2.6 M | 2.6 s = **20 %** | Many tiny native calls; cost is the call count, not the work. | Native calls, Python call overhead (build.py:80-83: 1.6 % native each) |

Scalene's "Python vs native" is a sampling estimate, rounded: treat it as direction, not as precise accounting. One
line in `threading.py` (a helper thread, 72 % of Scalene's total samples on the search run) is not findex code and is
left out of the numbers above.

## What I will change, and what I will not

* **Search:** hotspots 1–3 together are the target of the NumPy rewrite (`src/findex/vector.py`); the table of the
  speed-up is in the README.
* **Index:** only the tokenizer gets the cheap fix (hotspot 2: `findall` instead of a generator); hotspots 1 and 3 are the
  design of the one-pass builder and lab 5 already parallelizes it.
* **Not changed:** load time (native, and the lab-7 server pays it once).

## Reproduce

```powershell
uv run python scripts/sandbox_corpus.py --out data\sandbox          # or your Gutenberg folder
uv run findex index data\sandbox --out sandbox.bin --positions
uv run python experiments/profile_run.py index data\sandbox --positions --profile
uv run python experiments/profile_run.py search sandbox.bin "function OR value OR error" --profile
uv run py-spy record -f flamegraph -o docs\flame-search.svg -- python experiments\profile_run.py search sandbox.bin "function OR value OR error"
uv run scalene run --cpu-only --profile-all -o out.json experiments/profile_run.py --- search sandbox.bin "function OR value OR error"
```
(py-spy on Windows needs the console started normally; on Linux it may need `ptrace` permission.)
