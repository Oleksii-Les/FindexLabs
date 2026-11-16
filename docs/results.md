# All measurements, labs 1–8

One place for every table of the project. Each section says **how to regenerate it**. Numbers marked ✍️ come from the
author's own machine and corpus (Project Gutenberg, Windows 11, Intel i5-11320H, 4 cores / 8 threads, 16 GB) and are
pasted from the per-lab notes in [lab-notes.md](lab-notes.md); everything else was measured in a 2-vCPU Linux sandbox on a
17 006-document test corpus and is marked 🧪. Do not compare a ✍️ number with a 🧪 one.



## Lab 1 — eager vs lazy intake ✍️
`uv run python -m findex.benchmark data/gutenberg --limit N`

| Version | Documents | Tokens | Peak memory | Elapsed |
| --- | ---: | ---: | ---: | ---: |
| eager (lists) | 150 | 15,094,967 | 1.0 GiB | 17.68 s |
| lazy (generators) | 150 | 15,094,967 | 95.8 MiB | 14.79 s |

## Lab 2 — the inverted index ✍️
`uv run python -m findex.memstudy …`, `uv run python -m findex.bench_search …` (see lab-notes.md for the exact commands)

| Postings representation | Peak (build) | Resident index | B/posting | Build | File (pickle) | Save | Load |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `list[Posting]`, plain `@dataclass` | 257.1 MiB | 228.3 MiB	 | 160.2 | 16.69 s | 42.8 MiB | 1.86 s | 1.11 s |
| `list[Posting]`, `slots=True` | 214.5 MiB | 182.7 MiB | 128.2 | 18.16 s | 34.2 MiB | 4.84 s | 6.02 s |
| `array('I')` pairs (no objects) | 226.8 MiB | 192.1 MiB | 134.8 | 16.70 s | 51.6 MiB | 5.15 s | 1.19 s |

| Format | In-memory storage | File size | Save | Load |
| --- | --- | ---: | ---: | ---: |
| pickle | slots | 34.2 MiB | 3.45 s | 6.42 s |
| pickle | array | 51.6 MiB | 4.98 s | 1.14 s |
| binary | array | 24.8 MiB | 0.60 s | 2.79 s |

| Storage | Positions | Resident index | Pickle file | vs. no positions |
| --- | --- | ---: | ---: | ---: |
| slots | off | 182.7 MiB | 34.2 MiB | 1.0x |
| slots | on | 756.9 MiB | 94.4 MiB | 2.8x |
| array | off | 192.1 MiB | 51.6 MiB | 1.0x |
| array | on | 302.1 MiB | 117.6 MiB | 2.3x |

## Lab 3 — ranking ✍️
`uv run python -m findex.evaluate index.bin eval/queries.json --show`

| # | Query | Relevant | P@5 TF-IDF | P@5 BM25 |
| ---: | --- | ---: | ---: | ---: |
| 1 | `another class of servants` | 1 | 0.20 | 0.20 |
| 2 | `devising of idols` | 1 | 0.20 | 0.20 |
| 3 | `To think he may be lying dead there` | 1 | 0.00 | 0.00 |
| | **mean** | | **0.13** | **0.13** |

## Lab 5 — parallel indexing ✍️
`uv run python -m findex.parbench run …` then `report` (see lab-notes.md)

| Interpreter | Executor | Workers | Wall, s | CPU, s | Peak RSS | Speed-up |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 3.12 [py 3.12.15, GIL on] | serial | 1 | 15.86 | 15.75-16.16 | 15.75 | 255 MiB | 0.14 | 1.00x |
| 3.12 [py 3.12.15, GIL on] | threads | 1 | 15.95 | 15.79-19.00 | 15.92 | 286 MiB | 0.19 | 0.99x |
| 3.12 [py 3.12.15, GIL on] | threads | 2 | 16.39 | 16.16-20.46 | 16.39 | 316 MiB | 0.26 | 0.97x |
| 3.12 [py 3.12.15, GIL on] | threads | 4 | 16.62 | 16.48-20.09 | 16.73 | 347 MiB | 0.33 | 0.95x |
| 3.12 [py 3.12.15, GIL on] | threads | 8 | 17.04 | 16.62-17.14 | 17.00 | 383 MiB | 0.40 | 0.93x |
| 3.12 [py 3.12.15, GIL on] | processes | 1 | 20.65 | 20.64-20.81 | 18.98 | 497 MiB | 0.22 | 0.77x |
| 3.12 [py 3.12.15, GIL on] | processes | 2 | 12.31 | 12.23-12.46 | 21.39 | 438 MiB | 0.30 | 1.29x |
| 3.12 [py 3.12.15, GIL on] | processes | 4 | 8.63 | 8.53-8.95 | 27.58 | 421 MiB | 0.40 | 1.84x |
| 3.12 [py 3.12.15, GIL on] | processes | 8 | 8.36 | 8.32-8.45 | 44.56 | 460 MiB | 0.50 | 1.90x |
| 3.13t [py 3.13.16t, GIL OFF] | serial | 1 | 15.79 | 15.78-15.83 | 15.72 | 274 MiB | 0.20 | 1.00x |
| 3.13t [py 3.13.16t, GIL OFF] | threads | 1 | 16.38 | 16.38-16.39 | 16.33 | 324 MiB | 0.35 | 0.96x |
| 3.13t [py 3.13.16t, GIL OFF] | threads | 2 | 9.95 | 9.85-10.02 | 17.92 | 342 MiB | 0.45 | 1.59x |
| 3.13t [py 3.13.16t, GIL OFF] | threads | 4 | 7.48 | 7.41-7.51 | 21.47 | 415 MiB | 0.53 | 2.11x |
| 3.13t [py 3.13.16t, GIL OFF] | threads | 8 | 7.53 | 7.48-7.58 | 34.09 | 503 MiB | 0.64 | 2.10x |

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
