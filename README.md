# findex

A small search engine built step by step (course: *Python — Build a Search Engine*).

- **Lab 01** — streaming corpus intake, tokenizer, corpus statistics in constant memory (tag `lab-01`)
- **Lab 02** — inverted index, Boolean search, persistence, memory study (tag `lab-02`) → see [Lab 02](#lab-02--the-inverted-index)
- **Lab 03** — `Index` as a Python object, TF-IDF / BM25 ranking, query language, snippets (tag `lab-03`) → see [Lab 03](#lab-03--ranking-and-the-object-model)

## Corpus


Project Gutenberg plain-text books, 150 books ≈ 90 MB, stored in `data/gutenberg/` (not committed).

```bash
uv sync
uv run python scripts/fetch_gutenberg.py --count 150   # ~50 MB; use 1000 for the full corpus
```

Any directory of `*.txt` files or a `*.jsonl` / `*.jsonl.gz` file (one object per line with a `"text"` field) works.

## Usage

```bash
uv run python -m findex.stats data/gutenberg                # full corpus
uv run python -m findex.stats data/gutenberg --limit 100    # first 100 documents
uv run python -m findex.benchmark data/gutenberg --limit 500  # eager vs lazy table
uv run pytest && uv run ruff check .
```

## Layout

| File | Role |
| --- | --- |
| `src/findex/corpus.py` | `iter_documents(root)` — generator yielding `Document(doc_id, path, text)` one at a time; bad UTF-8 is logged and replaced, not fatal |
| `src/findex/tokenize.py` | `tokenize(text)` — generator; NFC → casefold → `re.finditer` |
| `src/findex/stats.py` | single-pass pipeline `iter_documents → tokenize → Counter`, with time and peak memory (`tracemalloc`) |
| `src/findex/eager.py` | deliberately greedy list-based version (for the comparison only) |
| `src/findex/benchmark.py` | runs both versions and prints the Markdown table |

## Tokenizer rules

1. Normalize: NFC → `casefold()` → NFC (`Straße` ≡ `STRASSE`; `café` composed ≡ decomposed).
2. Token = run of Unicode letters/digits (`_` excluded), so Cyrillic etc. just work.
3. Apostrophe **between** word characters stays in the token (`don't`, `п'ять`); `’` and `ʼ` are mapped to `'`. Edge quotes are dropped.
4. Hyphen **between** word characters stays (`well-known`). Spaced dashes (` - `, `—`) split.
5. Digits are ordinary token characters (`mp3`, `2026`). `3.14` → `3`, `14` (no number-aware rules).
6. Everything else separates tokens.

## Eager vs lazy

Same corpus slice, same results (the benchmark asserts they agree).
Elapsed is measured in a run **without** `tracemalloc`; peak memory in a separate run **with** it.



| Version | Documents | Tokens | Peak memory | Elapsed |
| --- | ---: | ---: | ---: | ---: |
| eager (lists) | 150 | 15,094,967 | 1.0 GiB | 17.68 s |
| lazy (generators) | 150 | 15,094,967 | 95.8 MiB | 14.79 s |

Machine: Intel Core i5-11320H, 16 GB RAM, Windows 11, Python 3.12.3

### Where did the memory go?


The eager version keeps three things alive at once: the full text of every document (`list(iter_documents)`),
a list of lists holding every token as a separate `str` object (~50+ bytes each, plus 8 bytes per list slot),
and a flat copy of all token references. Memory therefore grows with corpus size (O(total tokens)).
The lazy version keeps only the current document, the regex match in flight, and the `Counter`,
which grows with the *vocabulary* (distinct terms), not with the number of tokens (O(vocabulary)).
It is not zero because the document text, the Counter and interned token strings still have to live somewhere.


---

# Lab 02 — The inverted index

## Quick start

```bash
uv run python -m findex.index data/gutenberg --out index.bin          # build + save (binary format)
uv run python -m findex.search index.bin "whale NOT ahab"             # load + search
uv run python -m findex.search index.bin "sea OR ocean" --engine set  # set-based engine
uv run python -m findex.bench_search index.bin                        # merge vs set table
uv run python -m findex.memstudy data/gutenberg                       # memory / file tables
```

Useful flags: `--limit N` (first N docs), `--positions` (store token offsets),
`--storage {dataclass,slots,array}`, `--format {binary,pickle}`, `--no-tracemalloc` (clean timings).
Both CLIs print elapsed time and peak memory. Note: `tracemalloc` slows code 3–5×, so timings printed
next to a peak are inflated; the tables below take times from separate runs without it.

## Design

| Module | Role |
| --- | --- |
| `model.py` | `Posting`, `DocMeta` (`@dataclass(frozen=True, slots=True)`, hashable), `PlainPosting` (baseline), `PostingList` (array storage), `Index` |
| `build.py` | `build_index(docs, storage=, positions=)` — one pass over the lazy stream; per-document `Counter`, flushed at the end of each document into a `defaultdict`. Doc ids arrive ascending, so every postings list is sorted for free |
| `boolean.py` | two-pointer merge (AND / OR / AND-NOT) and the set-based twin; `evaluate(query, index, engine)` |
| `store.py` | `save` / `load` in two formats: `pickle` and a custom `binary` one |
| `index.py`, `search.py` | the CLIs; `bench_search.py`, `memstudy.py` produce the tables below |

`model.py` is separate from the CLI on purpose: classes defined in a module run with `python -m` live in
`__main__`, and pickle would record them as `__main__.Posting`, which cannot be loaded by any other program.

`Index` holds `postings: dict[str, ...]`, `doc_lengths: dict[int, int]` (for BM25 in Lab 3) and
`doc_meta: dict[int, DocMeta]` (path + title; the title is the `Title:` header if present, else the first line).
The builder returns a plain `dict` (not the `defaultdict`): a lookup of an unknown term must not create entries,
and a `defaultdict` with a lambda factory cannot be pickled.

### Query language

Evaluated left to right, **no precedence** (a real parser arrives in Lab 3):
`a b` and `a AND b` = intersection, `a OR b` = union, `a NOT b` = `a` without `b`, `NOT b` = all documents without `b`.
`a OR b c` therefore means `(a OR b) AND c`. Keywords are upper-case. Terms go through the same normalization as indexing.

### Storage formats

* **pickle** — one call, any storage. **Security:** `pickle.load` can run arbitrary code stored in the file,
  so a pickle from someone else is a program, not data. `findex.search` prints a warning when it opens one.
* **binary** — `FINDEXBN` magic, a JSON header (terms, counts, doc metadata, doc lengths), then raw `array('I')`
  buffers per term (`doc_ids`, `tfs`, and optionally `positions`: the positions of posting *i* are the next `tf[i]` values).
  Loading only parses JSON and copies bytes, so it is safe, and always yields `array` storage.

## Results

Corpus: 150 documents, 585,554 terms, 1,494,103 postings. 
Machine: Intel Core i5-11320H, 16 GB RAM, Windows 11, Python 3.12.3

### Memory: three ways to store postings



| Postings representation | Peak (build) | Resident index | B/posting | Build | File (pickle) | Save | Load |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `list[Posting]`, plain `@dataclass` | 257.1 MiB | 228.3 MiB	 | 160.2 | 16.69 s | 42.8 MiB | 1.86 s | 1.11 s |
| `list[Posting]`, `slots=True` | 214.5 MiB | 182.7 MiB | 128.2 | 18.16 s | 34.2 MiB | 4.84 s | 6.02 s |
| `array('I')` pairs (no objects) | 226.8 MiB | 192.1 MiB | 134.8 | 16.70 s | 51.6 MiB | 5.15 s | 1.19 s |

**Where the bytes went.** 
A posting is two small integers, yet the list representations pay for an *object* around them.
Per posting, `list[Posting]` stores an 8-byte pointer in the list, the posting object itself
(56 B with `slots=True`; 48 B plus a separate instance `__dict__` for the plain class, although CPython 3.12
often keeps those attributes inline, so the gap to `slots` is smaller than older Python versions show),
and two `int` objects of 28 B each. (CPython caches ints in -5..256, so a tiny test corpus hides this cost: only
doc ids above 256 are separate objects.) The empty `positions` tuple is a shared singleton, so it costs nothing per posting.
`array('I')` removes all of that: 4 B for the doc id plus 4 B for the tf, with no per-element object and no
pointer, i.e. ~8 B/posting plus small per-term overhead (two arrays and a `PostingList` per term) and growth slack.
The "peak" is a little above "resident" because the per-document `Counter` and the `defaultdict` growth are live during the build.

### File formats



| Format | In-memory storage | File size | Save | Load |
| --- | --- | ---: | ---: | ---: |
| pickle | slots | 34.2 MiB | 3.45 s | 6.42 s |
| pickle | array | 51.6 MiB | 4.98 s | 1.14 s |
| binary | array | 24.8 MiB | 0.60 s | 2.79 s |

Why `pickle` of `slots` objects loads slowly: a frozen, slotted dataclass is rebuilt by a generated
`__setstate__` that sets every field through `object.__setattr__` — a Python-level call per field per posting —
while `array` buffers are restored with a single `memcpy`-like operation.

### Cost of token positions



| Storage | Positions | Resident index | Pickle file | vs. no positions (file) |
| --- | --- | ---: | ---: | ---: |
| slots | off | 182.7 MiB | 34.2 MiB | 1.0x |
| slots | on | 756.9 MiB | 94.4 MiB | 2.8x |
| array | off | 192.1 MiB | 51.6 MiB | 1.0x |
| array | on | 302.1 MiB | 117.6 MiB | 2.3x |

### Boolean search: merge vs set



| Op | Case | len(a) | len(b) | merge | set | winner |
| --- | --- | ---: | ---: | ---: | ---: | --- |
|AND |	common & common |	150 |	150 |	12.5 µs |	6.1 µs | set | x2.1 |
|AND |	common & rare |	150 |	1 |	5.8 µs |	1.7 µs | set | x3.4 |
|AND |	rare & rare |	1 |	1 |	0.2 µs |	0.4 µs |	merge | x2.2 |
|OR |	common and common |	150 |	150 |	12.7 µs |	5.3 µs |	set | x2.4 |
|OR |	common and rare |	150 |	1 |	7.7 µs |	3.3 µs |	set | x2.3 |
|NOT |	common - common |	150 |	150 |	14.1 µs |	3.6 µs |	set | x3.9 |
|NOT |	common - rare |	150 |	1 |	12.8 µs |	3.3 µs |	set | x3.9 |

Both engines receive identical pre-extracted id lists and return sorted lists (the benchmark asserts they agree).
Merge is O(|a|+|b|) with no extra memory but runs as a bytecode loop; `set` also costs O(|a|+|b|)
(it hashes every element) but the work happens in C. 
`set` almost always outperforms `merge`,except for short lists, because `set` uses C (which is runs very fast), whereas `merge` uses a loop that iterates through bytes; therefore, `set` wins with long lists. In the case of very short lists, `merge` may turn out to be faster, since `set` spends time creating collections, whereas `merge` does practically nothing.
Merge remains the design to keep: it streams, needs no hashing, works on compressed postings and
extends to phrase queries (Lab 3).


---

# Lab 03 — Ranking and the object model

## Quick start

Phrase queries need token positions, so build the index with `--positions`
(`--storage array` keeps it compact). Indexes from Lab 02 are not compatible: rebuild.

```bash
uv run python -m findex.index data/gutenberg --out index.bin --storage array --positions
uv run python -m findex.search index.bin 'whale AND (sea OR ocean) NOT ahab "white whale"' -v --repeat 2
uv run python -m findex.search index.bin "whale" --scorer tfidf --top 5
uv run python -m findex.sanity index.bin                          # the three ranking checks
uv run python -m findex.evaluate index.bin eval/queries.json --show   # precision@5
```

`-v` prints the `@timed` lines and `cache miss` / `cache HIT`. In Python:

```python
from findex.store import open_index
from findex.engine import SearchEngine
from findex.scoring import BM25

with open_index("index.bin") as ix:  # closed again on exit, even after an exception
    print(len(ix), "whale" in ix, ix["whale"][0], ix)  # Index(terms=..., docs=...)
    for hit in SearchEngine(ix, BM25(k1=1.2)).search('"white whale" NOT ahab', k=5):
        print(hit, hit.snippet)
```

## What changed

| Module | Role |
| --- | --- |
| `model.py` | `Index(Mapping)`: `len`, `in`, `ix[term]` (`KeyError`), iteration, `repr`; `num_docs`, `avg_doc_length` (`cached_property`), `doc_length`, `df`, `close()`. `PostingList` is now a `Sequence[Posting]` view over the arrays |
| `store.py` | `open_index(path)` — generator-style context manager; `load` is `@timed` |
| `scoring.py` | `Scorer` protocol, `TfIdf`, `BM25(k1=1.5, b=0.75)` (frozen dataclasses with `__call__`), `SearchResult` (`order=True`) |
| `query.py` | recursive-descent parser → `Term` / `Phrase` / `And` / `Or` / `Not` with `&`, `\|`, `~` and `evaluate(index)` |
| `engine.py` | `search()` (`@timed`, `heapq.nlargest`) and `SearchEngine` with an `lru_cache`d `query → doc ids` |
| `snippets.py` | ±80-character window around the densest cluster of query terms, matches highlighted |
| `decorators.py` | `timed` (`functools.wraps`; works bare and with arguments) |
| `sanity.py`, `evaluate.py` | produce the evidence tables below |

Design notes:

* **Scorer = protocol, not base class.** Anything with `score(term, posting, index) -> float` works
  (`isinstance(x, Scorer)` is a structural check); the tests plug in a scorer that inherits from nothing.
* **TF-IDF** = `(1 + ln tf) · ln(N / df)`. **BM25** = `idf · tf(k1+1) / (tf + k1(1 − b + b·dl/avgdl))`, with Lucene's
  `idf = ln(1 + (N − df + 0.5)/(df + 0.5))`, which never goes negative.
* **`SearchResult` fields start with `score`** because `order=True` compares fields in definition order;
  `title` and `snippet` are `compare=False`. Keyword construction: `SearchResult(score=…, doc_id=…, title=…)`.
* **Grammar:** NOT binds tighter than AND, AND tighter than OR; juxtaposition is an implicit AND, so
  `a OR b c` parses to `Or(a, And(b, c))`. Keywords are UPPER-case (`and` is an ordinary word). This replaces Lab 02's
  left-to-right evaluation (`boolean.evaluate` is kept only for the Lab 02 tests).
* **Phrase** = intersect the doc lists, then for word *i* shift its positions by −*i* and merge-intersect with word 0's.
* **Cache:** `lru_cache` wraps a bound method *per engine instance*, so the cache does not pin every engine alive
  through `self`; it stores tuples, which are hashable and cannot be mutated by callers.
* **Snippets** need the original text, so they work for `.txt` corpora (path stored in `DocMeta`); documents that live
  inside a `.jsonl` get no snippet.

## Evidence

Corpus: Index(terms=585_554, docs=150, storage='array', positions=True); document length median 66,752, avg 100,633
Machine: Intel Core i5-11320H, 16 GB RAM, Windows 11, Python 3.12.3

### Ranking sanity checks

### Check 1 -- a rare term outranks a common one

Query `aage OR whole`. `aage` occurs in 1 docs, `whole` in 130.

| Document | Contains | tf | length | BM25 score | Rank |
| --- | --- | ---: | ---: | ---: | ---: |
| 14.txt | only `aage` (rare) | 1 | 282,878 | 2.541 | 1 |
| 17.txt | only `whole` (common) | 72 | 285,849 | 0.348 | 53 |

PASS: the rare-term document ranks higher.

### Check 2 -- the 20th repetition adds (almost) nothing under BM25

Term `own`, document 146.txt (length 70,092); the tf values are hypothetical, idf and length are real.

| tf | BM25 | TF-IDF (1+ln tf) | TF-IDF (raw tf) |
| ---: | ---: | ---: | ---: |
| 1 | 0.1340 | 0.1128 | 0.1128 |
| 2 | 0.1832 | 0.1910 | 0.2256 |
| 5 | 0.2349 | 0.2943 | 0.5640 |
| 10 | 0.2593 | 0.3725 | 1.1280 |
| 19 | 0.2727 | 0.4449 | 2.1431 |
| 20 | 0.2734 | 0.4507 | 2.2559 |

BM25: going 1->2 adds 0.0492, going 19->20 adds 0.0008 (1.6% of the first gain).
PASS: saturation (limit for tf->inf is 0.289).

### Check 3 -- a short document with one match beats a very long one

Query `own`: both documents contain it exactly once (tf = 1).

| Document | length | BM25 | Rank (BM25) | TF-IDF |
| --- | ---: | ---: | ---: | ---: |
| 8.txt (short) | 3,875 | 0.2040 | 130 | 0.1128 |
| 136.txt (long) | 11,304 | 0.1927 | 133 | 0.1128 |

PASS: BM25 prefers the short document; TF-IDF (no length normalization) ties them.

1. **Rare term above common term** — a document with a single `rare` outranks one with 24 `common`, same length, because idf(rare) ≫ idf(common)" -->
2. **20th repetition adds almost nothing** — BM25: going 1->2 adds 0.0492, going 19->20 adds 0.0008 (1.6% of the first gain) why: contribution tends to idf·(k1+1)
3. **Short document above very long one** — BM25 prefers the short document; TF-IDF ties them because it has no length normalization (b = 0 behaves the same)

### precision@5, TF-IDF vs BM25

10 queries, labeled by hand **before** looking at any ranking (`eval/queries.json`; format in `eval/queries.example.json`).



| # | Query | Relevant | P@5 TF-IDF | P@5 BM25 |
| ---: | --- | ---: | ---: | ---: |
| 1 | `another class of servants` | 1 | 0.20 | 0.20 |
| 2 | `devising of idols` | 1 | 0.20 | 0.20 |
| 3 | `To think he may be lying dead there` | 1 | 0.00 | 0.00 |
| | **mean** | | **0.13** | **0.13** |

In this "match" we have a tie

### `@timed` and cache traces



```text
findex.timing: load took 2642.82 ms
findex.engine: match 'whale sea' -> 19 docs [cache miss]
findex.timing: search took 0.76 ms
findex.engine: match 'whale sea' -> 19 docs [cache HIT]
findex.timing: search took 0.65 ms
findex.timing: search took 276.09 ms
Index(terms=585_554, docs=150, storage='array', positions=True)
query: 'whale sea'  scorer=bm25  ->  19 matching documents
```

### Query with brackets and a phrase (demo)


Index(terms=585_554, docs=150, storage='array', positions=True)
query: 'whale AND ocean NOT ahab "white whale"'  scorer=bm25  ->  0 matching documents


load + search: 9.034 s (inflated by tracemalloc), peak memory 383.0 MiB; query part 3.50 ms
