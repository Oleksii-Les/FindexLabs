"""Download plain-text books from Project Gutenberg into data/gutenberg/.

Usage: uv run python scripts/fetch_gutenberg.py --count 1000
Polite: sequential, 1 s pause. Skips books that are missing or already present.
Roughly 1 000 books is ~400 MB; use --count 150 to get ~50 MB.
"""

import argparse
import time
import urllib.error
import urllib.request
from pathlib import Path

URLS = (
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=150)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("data/gutenberg"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    got, book_id = 0, args.start
    while got < args.count:
        dest = args.out / f"{book_id}.txt"
        if dest.exists():
            got += 1
        else:
            for tmpl in URLS:
                try:
                    req = urllib.request.Request(
                        tmpl.format(id=book_id),
                        headers={"User-Agent": "findex-lab/0.1"},
                    )
                    with urllib.request.urlopen(req, timeout=30) as r:
                        dest.write_bytes(r.read())
                    got += 1
                    print(f"[{got}/{args.count}] {dest}")
                    time.sleep(1)
                    break
                except (urllib.error.URLError, TimeoutError):
                    continue
        book_id += 1


if __name__ == "__main__":
    main()
