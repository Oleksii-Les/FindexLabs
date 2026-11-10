"""The ``findex`` command: ``findex index | search | stats``.

Rules this module follows:

* **stdout is for data, stderr is for diagnostics.** Progress bars, log lines and error
  messages go to stderr; results (table, ``--json`` lines) go to stdout, so
  ``findex search idx "q" --json | jq`` works.
* **Expected failures** (missing file, bad query, corrupt index) print ONE line
  ``error: ...`` and exit with status 1 -- never a traceback. Anything else is a bug and
  is allowed to crash loudly.
* **Logging is configured here, once** (``configure_logging``); library modules only
  call ``logging.getLogger(__name__)``.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Generator, Iterable, Iterator
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from itertools import islice
from pathlib import Path
from typing import Annotated, NoReturn, TypedDict

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from findex.boolean import EngineName
from findex.build import build_index
from findex.corpus import Document, count_documents, iter_documents
from findex.engine import SearchEngine
from findex.logging_setup import configure_logging
from findex.model import Index, StorageName
from findex.query import PositionsRequired, QuerySyntaxError
from findex.scoring import ScorerName, make_scorer
from findex.store import FormatName, IndexFormatError, detect_format, open_index, save

log = logging.getLogger(__name__)

app = typer.Typer(
    name="findex",
    help="findex: index a text corpus and search it with BM25 ranking.",
    no_args_is_help=True,
    add_completion=False,
)

# Private-use control characters mark highlighted spans inside snippets; they are turned
# into rich styles for the table (and into ``**bold**`` for ``--json``).
_HI_ON, _HI_OFF = "\x02", "\x03"

IndexArg = Annotated[
    Path, typer.Argument(metavar="INDEX", help="Index file made by `findex index`.")
]
# -v is accepted on the root command (`findex -v search ...`) AND after the subcommand
# (`findex search ... -v`); the larger of the two counts wins.
VerboseOpt = Annotated[
    int, typer.Option("--verbose", "-v", count=True, help="-v: info, -vv: debug and timings.")
]


def apply_verbosity(ctx: typer.Context, local: int) -> None:
    configure_logging(max(int(ctx.obj or 0), local))


def fail(message: str, code: int = 1) -> NoReturn:
    """Report an expected error as one line on stderr and leave with a non-zero status."""
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code)


@contextmanager
def expected_errors() -> Generator[None]:
    """Turn the errors a user can cause into a one-line message (see module docstring)."""
    try:
        yield
    except (QuerySyntaxError, PositionsRequired, IndexFormatError) as exc:
        fail(str(exc))
    except OSError as exc:  # missing/unreadable files, permission problems, full disk
        fail((exc.strerror and f"{exc.strerror}: {exc.filename}") or str(exc))


def require_file(path: Path, what: str = "index file") -> None:
    if not path.is_file():
        fail(f"{what} not found: {path}")


def _version_callback(value: bool) -> None:
    if value:
        try:
            typer.echo(f"findex {version('findex')}")
        except PackageNotFoundError:  # running from a source tree that is not installed
            typer.echo("findex (not installed)")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose", "-v", count=True, help="-v: progress info, -vv: debug and timings."
        ),
    ] = 0,
    version_flag: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Root command: the only place that configures logging."""
    del version_flag  # handled by the eager callback
    ctx.obj = verbose  # remembered for the subcommands
    configure_logging(verbose)


# --------------------------------------------------------------------------- index
def _tracked(docs: Iterable[Document], progress: Progress, task: TaskID) -> Iterator[Document]:
    """Pass documents through while advancing the bar: still lazy, one document at a time."""
    for doc in docs:
        yield doc
        progress.advance(task)


@app.command("index")
def index_cmd(
    ctx: typer.Context,
    corpus: Annotated[
        Path,
        typer.Argument(metavar="CORPUS", help="Directory of .txt files, or a .jsonl[.gz] file."),
    ],
    out: Annotated[Path, typer.Option("--out", "-o", help="Where to write the index.")] = Path(
        "index.bin"
    ),
    positions: Annotated[
        bool, typer.Option("--positions", help="Store token positions (needed for phrases).")
    ] = False,
    limit: Annotated[
        int | None, typer.Option("--limit", min=1, help="Only the first N documents.")
    ] = None,
    storage: Annotated[
        StorageName, typer.Option("--storage", help="In-memory postings representation.")
    ] = "array",
    fmt: Annotated[FormatName, typer.Option("--format", help="File format.")] = "binary",
    verbose: VerboseOpt = 0,
) -> None:
    """Build an inverted index from CORPUS and save it."""
    apply_verbosity(ctx, verbose)
    if not corpus.exists():
        fail(f"corpus not found: {corpus}")
    with expected_errors():
        total = count_documents(corpus)
        if total == 0:
            fail(f"no documents found in {corpus}")
        if limit is not None:
            total = limit if total is None else min(total, limit)
        docs: Iterable[Document] = iter_documents(corpus)
        if limit is not None:
            docs = islice(docs, limit)

        t0 = time.perf_counter()
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]indexing[/bold]"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=Console(stderr=True),  # the bar is diagnostics: stderr
        )
        with progress:
            task = progress.add_task("index", total=total)
            index = build_index(
                _tracked(docs, progress, task), storage=storage, positions=positions
            )
        built = time.perf_counter() - t0
        if index.num_docs == 0:
            fail(f"no documents found in {corpus}")
        log.info("built %r in %.2f s", index, built)
        size = save(index, out, fmt)

    table = Table(title="Index built", show_header=False, title_justify="left")
    table.add_column(style="bold")
    table.add_column(justify="right")
    for label, value in (
        ("documents", f"{index.num_docs:,}"),
        ("terms", f"{len(index):,}"),
        ("postings", f"{index.n_postings():,}"),
        ("positions", "yes" if positions else "no"),
        ("storage / format", f"{storage} / {fmt}"),
        ("file", f"{out}  ({size / 1024 / 1024:.1f} MiB)"),
        ("build time", f"{built:.2f} s"),
    ):
        table.add_row(label, value)
    Console().print(table)


# -------------------------------------------------------------------------- search
def _highlighted(snippet: str) -> Text:
    """Snippet with HI marks -> rich Text with the matches in bold yellow."""
    text = Text()
    pos = 0
    while True:
        start = snippet.find(_HI_ON, pos)
        if start < 0:
            text.append(snippet[pos:].replace(_HI_OFF, ""))
            return text
        end = snippet.find(_HI_OFF, start)
        if end < 0:
            text.append(snippet[pos:])
            return text
        text.append(snippet[pos:start])
        text.append(snippet[start + 1 : end], style="bold yellow")
        pos = end + 1


@app.command("search")
def search_cmd(
    ctx: typer.Context,
    index_path: IndexArg,
    query: Annotated[
        str,
        typer.Argument(
            metavar="QUERY", help='Query: python AND (async OR await) NOT java "event loop"'
        ),
    ],
    k: Annotated[int, typer.Option("--k", "-k", min=1, help="How many hits to show.")] = 10,
    scorer: Annotated[ScorerName, typer.Option("--scorer", help="Ranking function.")] = "bm25",
    engine: Annotated[
        EngineName, typer.Option("--engine", help="Set-operation algorithm.")
    ] = "merge",
    json_out: Annotated[
        bool, typer.Option("--json", help="One JSON object per hit on stdout (JSON lines).")
    ] = False,
    snippets: Annotated[bool, typer.Option("--snippets/--no-snippets")] = True,
    window: Annotated[int, typer.Option(min=10, help="Snippet half-width in characters.")] = 80,
    repeat: Annotated[
        int, typer.Option(min=1, help="Run the query N times (shows the cache hit with -v).")
    ] = 1,
    verbose: VerboseOpt = 0,
) -> None:
    """Search INDEX for QUERY and print the best K documents."""
    apply_verbosity(ctx, verbose)
    require_file(index_path)
    with expected_errors():
        if detect_format(index_path) == "pickle":
            log.warning("%s is a pickle file: loading it can run arbitrary code", index_path)
        with open_index(index_path) as index:
            eng = SearchEngine(index, make_scorer(scorer), engine=engine)
            marks = ("**", "**") if json_out else (_HI_ON, _HI_OFF)
            for _ in range(repeat - 1):  # warm-up runs: the 2nd one is served from the cache
                eng.search(query, k, snippets=False)
            hits = eng.search(query, k, snippets=snippets, window=window, marks=marks)
            matched = len(eng.match(query))
            if json_out:
                for rank, r in enumerate(hits, start=1):
                    row = {
                        "rank": rank,
                        "doc_id": r.doc_id,
                        "score": round(r.score, 6),
                        "title": r.title,
                        "path": index.doc_meta[r.doc_id].path,
                        "snippet": r.snippet,
                    }
                    typer.echo(json.dumps(row, ensure_ascii=False))
                return
            if not hits:
                typer.echo("No documents match.")
                return
            table = Table(
                title=f"{query}",
                caption=f"{matched:,} matching documents, showing {len(hits)}  ({scorer})",
                title_justify="left",
                caption_justify="left",
                show_lines=True,
            )
            table.add_column("#", justify="right", style="dim")
            table.add_column("score", justify="right")
            table.add_column("document", overflow="fold")
            table.add_column("snippet", overflow="fold", ratio=1)
            for rank, r in enumerate(hits, start=1):
                name = Path(index.doc_meta[r.doc_id].path).name
                document = Text.assemble((r.title, "bold"), "\n", (name, "dim"))
                table.add_row(str(rank), f"{r.score:.3f}", document, _highlighted(r.snippet))
            Console().print(table)


# --------------------------------------------------------------------------- stats
@app.command("stats")
def stats_cmd(
    ctx: typer.Context,
    index_path: IndexArg,
    top: Annotated[int, typer.Option("--top", min=0, help="How many most frequent terms.")] = 20,
    json_out: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
    verbose: VerboseOpt = 0,
) -> None:
    """Corpus statistics of an index: size, vocabulary, document lengths, top terms."""
    apply_verbosity(ctx, verbose)
    require_file(index_path)
    with expected_errors(), open_index(index_path) as index:
        data = _stats(index, top)
        if json_out:
            typer.echo(json.dumps(data, ensure_ascii=False))
            return
        table = Table(title="Index statistics", show_header=False, title_justify="left")
        table.add_column(style="bold")
        table.add_column(justify="right")
        table.add_row("documents", f"{data['documents']:,}")
        table.add_row("tokens", f"{data['tokens']:,}")
        table.add_row("vocabulary", f"{data['vocabulary']:,}")
        table.add_row("postings", f"{data['postings']:,}")
        table.add_row("avg document length", f"{data['avg_doc_length']:,.1f}")
        table.add_row(
            "shortest / longest", f"{data['min_doc_length']:,} / {data['max_doc_length']:,}"
        )
        table.add_row(
            "storage / positions", f"{index.storage} / {'yes' if index.has_positions else 'no'}"
        )
        console = Console()
        console.print(table)
        if top:
            terms = Table(title=f"Top {top} terms", title_justify="left")
            terms.add_column("#", justify="right", style="dim")
            terms.add_column("term")
            terms.add_column("occurrences", justify="right")
            terms.add_column("documents", justify="right")
            for rank, row in enumerate(data["top_terms"], start=1):
                terms.add_row(str(rank), row["term"], f"{row['count']:,}", f"{row['df']:,}")
            console.print(terms)


class TopTerm(TypedDict):
    term: str
    count: int
    df: int


class IndexStats(TypedDict):
    documents: int
    tokens: int
    vocabulary: int
    postings: int
    avg_doc_length: float
    min_doc_length: int
    max_doc_length: int
    top_terms: list[TopTerm]


def _stats(index: Index, top: int) -> IndexStats:
    lengths = list(index.doc_lengths.values())
    return {
        "documents": index.num_docs,
        "tokens": sum(lengths),
        "vocabulary": len(index),
        "postings": index.n_postings(),
        "avg_doc_length": index.avg_doc_length,
        "min_doc_length": min(lengths, default=0),
        "max_doc_length": max(lengths, default=0),
        "top_terms": [
            {"term": t, "count": c, "df": index.df(t)}
            for t, c in index.term_frequencies().most_common(top)
        ],
    }


if __name__ == "__main__":
    app(prog_name="findex")
