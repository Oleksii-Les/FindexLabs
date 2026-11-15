# Lab 04 — A real tool

findex is now an installable, typed, tested command-line program with CI and a tagged release.

## Install and use

```bash
# from the GitHub Release (wheel attached to v0.4.0) -- no clone needed:
uv tool install https://github.com/YOUR_GITHUB_USER/findex/releases/download/v0.4.0/findex-0.4.0-py3-none-any.whl
# or from a clone / a local wheel:
uv tool install .                      # puts `findex` on your PATH
uv build && uv tool install dist/findex-0.4.0-py3-none-any.whl

findex --help
findex index data/gutenberg --out index.bin --positions          # rich progress bar (stderr)
findex search index.bin 'python AND (async OR await) NOT java "event loop"'   # rich table
findex search index.bin "whale" --json | jq .title               # JSON lines on stdout
findex stats index.bin --top 10
findex -vv search index.bin "whale" --repeat 2                   # timings + cache hit on stderr
```

(On Windows PowerShell write a phrase as `'... \"white whale\"'`.)

| Command | What it does |
| --- | --- |
| `findex index CORPUS --out FILE [--positions] [--limit N] [--storage …] [--format …]` | build and save an index; progress bar and summary |
| `findex search INDEX "QUERY" [-k 10] [--scorer bm25\|tfidf] [--engine merge\|set] [--json] [--no-snippets] [--repeat N]` | ranked results as a table with highlighted snippets, or JSON lines |
| `findex stats INDEX [--top N] [--json]` | documents, tokens, vocabulary, postings, document lengths, most frequent terms |
| `findex -v / -vv …` | INFO / DEBUG logs on **stderr** (`-v` is also accepted after the subcommand) |

Contract: **stdout carries data, stderr carries diagnostics** (progress bar, logs, errors), so
`findex search … --json | jq` is never polluted. Expected failures — missing index or corpus, malformed or
damaged index file, bad query, phrase on an index built without `--positions` — print **one line**
`error: …` and exit with status 1, never a traceback. Usage errors (unknown option, invalid `--scorer`) use
status 2. Logging is configured in one place (`logging_setup.configure_logging`, called by the CLI entry point);
library modules only call `logging.getLogger(__name__)`; `@timed` logs at DEBUG.

## Typing

`pyright` in **strict** mode with **0 errors** (`[tool.pyright] typeCheckingMode = "strict"`,
`reportUnnecessaryTypeIgnoreComment = true`, so stale ignores fail the build).

* `Scorer` is a `typing.Protocol`; `TfIdf` and `BM25` satisfy it without inheriting from it
  (pyright checks the call sites; the test suite also plugs in a third scorer that inherits from nothing).
  `PostingView` is a second Protocol, so `Posting` (frozen, slots) and `PlainPosting` (plain dataclass) are
  interchangeable wherever a posting is read.
* Generators and pipelines: `Iterator[...]` as the return type of generator functions, `Iterable[...]` for what
  functions accept (`build_index(docs: Iterable[Document])`).
* `Literal` for closed string sets: `EngineName = Literal["merge", "set"]`, `ScorerName`, `StorageName`,
  `FormatName` — typer turns them into validated choices in `--help`.
* `@timed` is typed with PEP 695 / `ParamSpec` (`def timed[**P, R]`), so a decorated function keeps its exact
  signature for the type checker.
* `TypedDict` for JSON-shaped data (`LabeledQuery`, `IndexStats`).
* **`# type: ignore` count: 1** — `PostingList.__hash__ = None` (marking a container of mutable arrays
  unhashable; pyright rejects assigning `None` to the inherited method; the reason is in a comment next to it).

## Tests

```bash
uv run pytest                          # everything
uv run pytest -m "not slow"            # development loop (~3 s)
uv run pytest --cov=findex             # with coverage (fails under 90 %)
```

**173 tests** (of which 11 are Hypothesis properties and 2 are marked `slow`: the wheel build and the memory study).

| Area | Tests |
| --- | --- |
| tokenizer | `parametrize` tables incl. Unicode (Cyrillic, ß, Greek final sigma, CJK, NFC/NFD forms) |
| index build | tiny on-disk corpus and `mk([...])` factory in `conftest.py`; all three storages must agree |
| merge algorithms | hand-written cases + Hypothesis oracle against Python sets |
| query parser | trees are compared, not strings (`parse("a OR b c") == Or(a, And(b, c))`) |
| scorers | the three sanity checks of Lab 03 as tests (rare > common, saturation, length normalization) |
| save / load | both formats, with and without positions, damaged and foreign files |
| CLI | `typer.testing.CliRunner`: every subcommand, `--json` purity, error lines, exit codes, `-v/-vv` |
| packaging | builds the wheel and checks its entry point and metadata (`slow`) |

Hypothesis properties (all in `tests/test_properties.py`):

1. `load(save(index)) == index` — both formats, with/without positions;
2. postings are strictly sorted by doc id, `tf ≥ 1`, positions sorted and `len(positions) == tf`, and the `tf`s of a
   document add up to its length;
3. all three storage representations hold identical information;
4. `merge_and/or/andnot` equal the `set` operations; both engines agree; commutativity, idempotence, De Morgan;
5. the tokenizer never yields an empty token and is idempotent on its own output;
6. `parse(str(tree)) == tree` for randomly generated query trees, and trees are always flattened / `~~x` collapsed;
7. array postings behave like a `Sequence` (iteration equals indexing).

### Coverage

**95.5 %** line + branch coverage of `src/findex` (`pytest --cov=findex`; CI fails below 90 %).

Deliberately **not** covered (and why):

* `__main__.py` — one line, exercised only through a subprocess, which coverage does not follow.
* The fallback branches of `sanity.py` ("no suitable term in this corpus"), whose purpose is to cope with
  corpora the test data does not resemble; the three checks themselves *are* tested.
* Defensive branches that need a hostile environment: byte-swapping a big-endian index file, `OSError` on a
  full disk, unreadable files.
* The animated parts of `rich` (spinner and colour on a real terminal): tests run without a TTY, so they check
  that the progress bar goes to stderr, not how it looks.
* `scripts/fetch_gutenberg.py` — downloads from the network, lives outside the package.
* Timing numbers of the measurement tools are not asserted (they are machine-dependent); only that the tools run
  and print their tables.

## CI and releases

* `.github/workflows/ci.yml` — on every push and pull request, Python 3.12 and 3.13:
  `uv sync --locked` → `ruff check` → `ruff format --check` → `pyright` → `pytest --cov`.
* `.github/workflows/release.yml` — pushing a tag `v*` builds the wheel, installs it into an isolated tool
  environment, runs `findex --help`, and attaches `dist/*` to a GitHub Release.

```bash
git tag v0.4.0 && git push origin v0.4.0      # the workflow creates the release
# or by hand:
uv build && gh release create v0.4.0 dist/*.whl dist/*.tar.gz --title "findex v0.4.0" --generate-notes
```

<!-- TODO: after the tag is pushed, check that the Release page shows findex-0.4.0-py3-none-any.whl -->

Verified from a clean environment: `uv tool install findex-0.4.0-py3-none-any.whl` → `findex --help` works
(also with `UV_TOOL_DIR` pointing at an empty directory, i.e. nothing from the repository on the path).

## Evidence

<!-- TODO: add yours:
  1. the output / a screenshot of `findex search index.bin '…' ` on YOUR corpus (the rich table);
  2. the last lines of `uv run pytest --cov=findex` (test count and coverage on your machine);
  3. a screenshot of the green CI run. -->
