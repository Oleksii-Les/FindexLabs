# All measurements, labs 1–8

One place for every table of the project. Each section says **how to regenerate it**. Numbers marked ✍️ come from the
author's own machine and corpus (Project Gutenberg, Windows 11, Intel i5-11320H, 4 cores / 8 threads, 16 GB) and are
pasted from the per-lab notes in [lab-notes.md](lab-notes.md); everything else was measured in a 2-vCPU Linux sandbox on a
17 006-document test corpus and is marked 🧪. Do not compare a ✍️ number with a 🧪 one.

| Lab | Table | Status |
| --- | --- | --- |
| 1 | eager vs lazy intake | ✍️ paste from lab-notes |
| 2 | memory of three posting representations; file formats; cost of positions; merge vs set | ✍️ paste |
| 3 | precision@5, TF-IDF vs BM25 | ✍️ paste |
| 5 | parallel indexing: serial / threads / processes, GIL on and off | ✍️ paste (`docs/speedup.png`) |
| 6 | crawler: concurrency 1 / 5 / 20; blocking the event loop | 🧪 below |
| 7 | `def` vs `async def`; the `oha` table | 🧪 below (+ deployed row ✍️) |
| 8 | profile, NumPy scorer, service under load, semantic precision | 🧪 below |

## Lab 1 — eager vs lazy intake ✍️
`uv run python -m findex.benchmark data/gutenberg --limit N`

| Version | Documents | Tokens | Peak memory | Elapsed |
| --- | ---: | ---: | ---: | ---: |
| eager (lists) | … | … | … | … |
| lazy (generators) | … | … | … | … |

## Lab 2 — the inverted index ✍️
`uv run python -m findex.memstudy …`, `uv run python -m findex.bench_search …` (see lab-notes.md for the exact commands)

| Postings representation | Peak (build) | Resident index | B/posting | Build | File (pickle) | Save | Load |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `list[Posting]`, plain `@dataclass` | … | … | … | … | … | … | … |
| `list[Posting]`, `slots=True` | … | … | … | … | … | … | … |
| `array('I')` pairs (no objects) | … | … | … | … | … | … | … |

| Format | In-memory storage | File size | Save | Load |
| --- | --- | ---: | ---: | ---: |
| pickle | slots | … | … | … |
| pickle | array | … | … | … |
| binary | array | … | … | … |

| Storage | Positions | Resident index | Pickle file | vs. no positions |
| --- | --- | ---: | ---: | ---: |
| array | off / on | … | … | 1.0x / …x |

## Lab 3 — ranking ✍️
`uv run python -m findex.evaluate index.bin eval/queries.json --show`

| # | Query | Relevant | P@5 TF-IDF | P@5 BM25 |
| ---: | --- | ---: | ---: | ---: |
| 1–10 | … | | | |
| | **mean** | | | |

## Lab 5 — parallel indexing ✍️
`uv run python -m findex.parbench run …` then `report` (see lab-notes.md)

| Interpreter | Executor | Workers | Wall, s | CPU, s | Peak RSS | Speed-up |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 3.12 [GIL on] | serial | 1 | … | … | … | 1.00x |
| 3.12 [GIL on] | threads | 2 / 4 / 8 | … | … | … | … |
| 3.12 [GIL on] | processes | 2 / 4 / 8 | … | … | … | … |
| 3.13t [GIL off] | threads | 2 / 4 / 8 | … | … | … | … |

## Lab 6 — the crawler 🧪
`uv run python -m findex.crawlbench run --pages 200`; `uv run python experiments/loop_blocking.py`

| Setup | Wall, s | Pages/s | Errors | Speed-up |
| --- | ---: | ---: | ---: | ---: |
| concurrency 1 | 6.80 | 29.4 | 0 | 1.00x |
| concurrency 5 | 1.44 | 139.3 | 0 | 4.74x |
| concurrency 20 | 0.57 | 349.2 | 0 | 11.87x |
| concurrency 20, 2 per host, 50 ms pause | 10.08 | 19.8 | 0 | 0.67x |

| Where 20 ms of work per page runs | Pages/s | Heartbeat, worst delay |
| --- | ---: | ---: |
| nowhere | 180 | 6 ms |
| inline in the loop | 43 | **158 ms** |
| `asyncio.to_thread` | 42 | 36 ms |
| process pool | 44 | 6 ms |

## Lab 7 — the web service 🧪
`uv run python experiments/def_vs_async.py --index index.bin`; `bash experiments/oha_local.sh index.bin 15s 20`
(588-document demo corpus, 20 simultaneous requests)

| Route | 20 requests take | req/s | search p50 | `/health` p50 | `/health` p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `def` | 677 ms | 30 | 643 ms | 9.8 ms | 32.9 ms |
| `async def`, work inline | 494 ms | 40 | 284 ms | **234.7 ms** | **575.0 ms** |
| `async def` + thread pool | 662 ms | 30 | 624 ms | 10.2 ms | 34.0 ms |

| `oha`, 20 connections, 15 s | req/s | p50 | p95 | p99 |
| --- | ---: | ---: | ---: | ---: |
| 1 worker, `async def` | 133 | 134 ms | 209 ms | 345 ms |
| 1 worker, `def` | 109 | 176 ms | 279 ms | 354 ms |
| 4 workers | 251 | 61 ms | 178 ms | 238 ms |
| deployed service ✍️ | … | … | … | … |

## Lab 8 — profile, NumPy, semantic search 🧪
Corpus: [scripts/sandbox_corpus.py](../scripts/sandbox_corpus.py) → 17 006 documents, 46.6 MB, 171 496 terms, index with positions 50 MiB.

**Where the time went** (`experiments/profile_run.py`; details and hypotheses in [profiling.md](profiling.md))

| | Before | After |
| --- | ---: | ---: |
| `findex index`, wall | 5.5 s | 4.9 s (eager `findall` tokenizer) |
| one query, 3 common terms, with snippets | 33 ms | **5.6 ms** |
| load of the index file | 0.70 s | 0.70 s (unchanged; native `frombytes`) |
| top of the search profile | scorer + `Posting` objects + heap (89 %) | Boolean `merge_or` (77 %) |

Flame graphs: [before](flame-search.svg), [after](flame-search-after.svg), [index](flame-index.svg).

**Scoring only, k = 10, median of 30** (`experiments/numpy_table.py index.bin`) — [README](../README.md#what-made-it-faster-lab-8)

| Query | Postings | Lab 3 | NumPy | Speed-up |
| --- | ---: | ---: | ---: | ---: |
| frequent term | 17 006 | 34.14 ms | 0.616 ms | 55.4x |
| mid-frequency term | 340 | 0.742 ms | 0.064 ms | 11.6x |
| 3 terms (frequent + mid + rare) | 17 349 | 35.0 ms | 0.649 ms | 53.9x |
| 3 common terms | 15 439 | 28.7 ms | 0.491 ms | 58.4x |
| term in 3 documents | 3 | 0.017 ms | 0.037 ms | 0.5x |
| term in 1 document | 1 | 0.011 ms | 0.030 ms | 0.4x |

**The service** (`bash experiments/oha_modes.sh index.bin emb 15s 20`, 1 worker, cache off)

| Row | req/s | p50 | p95 | p99 |
| --- | ---: | ---: | ---: | ---: |
| lexical, loop of lab 3 | 28 | 746 ms | 1 015 ms | 1 133 ms |
| lexical, NumPy | 210 | 94 ms | 133 ms | 164 ms |
| semantic (LSA) | 137 | 145 ms | 208 ms | 259 ms |
| hybrid | 73 | 276 ms | 376 ms | 441 ms |

**Embeddings** (`findex embed`, LSA, 256 dimensions): 73 320 fragments of ≤ 800 characters, matrix 75 MB (`float32`,
memory-mapped), fitted and written in 16 s.

**Precision@5** (`python -m findex.evaluate … --embeddings emb`; 15 queries in `eval/sandbox_queries.json`)

| Kind | BM25 | BM25 (OR) | semantic | hybrid |
| --- | ---: | ---: | ---: | ---: |
| keyword (10) | 0.30 | 0.30 | 0.44 | 0.32 |
| paraphrase (5) | 0.00 | 0.08 | 0.04 | 0.04 |

Per-query rows: run the command above. Read the caveats in the README before quoting these.
