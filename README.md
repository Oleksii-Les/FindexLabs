# findex

A small search engine built step by step (course: *Python — Build a Search Engine*).
**Lab 01** — streaming corpus intake, tokenizer, and corpus statistics in constant memory.

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

|      Version      | Documents |   Tokens   | Peak memory | Elapsed |
| ----------------- | --------: | -------:   | ----------: | ------: |
|   eager (lists)   |    150    | 15,094,967 |   1.0 GiB   | 17.68 s |
| lazy (generators) |    150    | 15,094,967 |   95.8 MiB  | 14.79 s |

Machine: Intel Core i5-11320H, 16 GB RAM, Windows 11, Python 3.12.3

### Where did the memory go?

The eager version keeps three things alive at once: the full text of every document (`list(iter_documents)`),
a list of lists holding every token as a separate `str` object (~50+ bytes each, plus 8 bytes per list slot),
and a flat copy of all token references. Memory therefore grows with corpus size (O(total tokens)).
The lazy version keeps only the current document, the regex match in flight, and the `Counter`,
which grows with the *vocabulary* (distinct terms), not with the number of tokens (O(vocabulary)).
It is not zero because the document text, the Counter and interned token strings still have to live somewhere.
