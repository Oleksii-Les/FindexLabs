"""Build a mid-size text corpus from documentation already on this machine (no network).

    python scripts/sandbox_corpus.py --out data/sandbox --chunk 4000

Used for the Lab 08 measurements where Project Gutenberg is unreachable. Every .md/.rst/.txt
file and every Python module docstring found under the given roots is cut into pieces of about
``--chunk`` characters at paragraph boundaries, one piece = one document. English technical
prose, not novels: the numbers it produces are *its* numbers, rerun on your corpus.
"""

from __future__ import annotations

import argparse
import ast
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
ROOTS = ["/usr/lib/node_modules", "/usr/share/doc", "/usr/local/lib/python3.13", "/usr/lib/python3",
         "/root/.cargo", "/usr/share/man", sys.prefix]  # fmt: skip


def texts(roots: list[str]):
    seen: set[Path] = set()
    for root in roots:
        for p in sorted(Path(root).rglob("*")):
            if p in seen or not p.is_file() or p.stat().st_size > 2_000_000:
                continue
            seen.add(p)
            try:
                if p.suffix in {".md", ".rst", ".txt"}:
                    yield p, p.read_text(encoding="utf-8", errors="ignore")
                elif p.suffix == ".py":
                    tree = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
                    docs = [d for n in ast.walk(tree)
                            if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                            and (d := ast.get_docstring(n))]  # fmt: skip
                    if docs:
                        yield p, "\n\n".join(docs)
            except (OSError, SyntaxError, ValueError, RecursionError):
                continue


def pieces(text: str, size: int):
    buf = ""
    for para in text.split("\n\n"):
        if buf and len(buf) + len(para) > size:
            yield buf
            buf = ""
        buf += para + "\n\n"
    if len(buf.strip()) > 200:
        yield buf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/sandbox"))
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--roots", nargs="*", default=ROOTS)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    n = total = 0
    for path, text in texts(args.roots):
        for piece in pieces(text, args.chunk):
            piece = piece.strip()
            if len(piece) < 200:
                continue
            title = f"{path.parent.name}/{path.stem} #{n}"
            (args.out / f"{n:06d}.txt").write_text(f"Title: {title}\n\n{piece}\n", encoding="utf-8")
            n += 1
            total += len(piece)
    print(f"{n} documents, {total / 1e6:.1f} MB -> {args.out}")


if __name__ == "__main__":
    main()
