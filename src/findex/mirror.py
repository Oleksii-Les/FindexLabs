"""A tiny local web site to crawl: the "local mirror" the lab allows.

    python -m findex.mirror --pages 400 --latency-ms 30            # prints "LISTENING <port>"
    python -m findex.mirror --pages 400 --faults --port 8765       # adds a 429, a 503, a hang, ...

Why it exists: a crawler benchmark against someone else's server measures their server and
their mood, and is impolite. This one is yours: deterministic pages, a fixed per-request latency
(that is the *network wait* the crawler is supposed to overlap), a ``robots.txt``, and optional
faults for the failure-handling section of the lab.

The site (``/`` is page 0):

* ``/page/<i>``  -- i < ``--pages``. Page i links to children 5i+1..5i+5 (so the whole site is
  reachable from ``/``), to a few random pages, and to the pages that exercise the crawler:
  a trailing-slash and a ``#fragment`` / unsorted-query duplicate of another page (canonicalization),
  a ``/private/..`` page (disallowed by robots.txt) and an off-site link (must not be followed);
* ``--faults`` adds ``/flaky/429`` (429 + ``Retry-After: 1`` once, then 200), ``/flaky/503``
  (503 twice, then 200), ``/slow`` (answers after 5 s: a timeout), ``/redirect-loop`` (302 to itself),
  ``/missing`` (404), ``/file.pdf`` (not HTML).

HTTP/1.1 with keep-alive, one asyncio server, no framework.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import random
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

log = logging.getLogger(__name__)

_SYLLABLES = ("ka", "to", "ri", "me", "su", "lo", "na", "pe", "di", "vo", "an", "el", "or", "is")
_TOPICS = ("whale", "ocean", "engine", "python", "garden", "market", "library", "river", "signal")


def _vocabulary(size: int = 2500) -> list[str]:
    rng = random.Random(1)
    return ["".join(rng.choices(_SYLLABLES, k=rng.randint(2, 4))) for _ in range(size)]


_VOCAB = _vocabulary()
_WEIGHTS = [1 / (rank + 1) for rank in range(len(_VOCAB))]  # Zipf-like: a few common words


@dataclass(frozen=True, slots=True)
class MirrorConfig:
    pages: int = 400
    latency_ms: float = 30.0  # simulated network + server time per request
    words: int = 300  # words of text per page
    faults: bool = False
    crawl_delay: float | None = None  # advertise "Crawl-delay: N" in robots.txt
    corpus: Path | None = None  # optional directory of .txt books to take the text from


class Mirror:
    """Request handler state: which fault URLs have been hit how many times."""

    def __init__(self, config: MirrorConfig) -> None:
        self.cfg = config
        self.hits: Counter[str] = Counter()  # path -> requests seen (for tests and the demo)
        self._books: list[str] = []
        if config.corpus is not None:
            for p in sorted(config.corpus.rglob("*.txt"))[:20]:
                self._books.append(p.read_text(encoding="utf-8", errors="replace"))

    # ------------------------------------------------------------------ content
    def _text(self, i: int) -> str:
        if self._books:
            book = self._books[i % len(self._books)]
            start = (i // len(self._books)) * 2000 % max(1, len(book) - 2000)
            return book[start : start + 2000].replace("<", " ").replace("&", " ")
        rng = random.Random(i * 7919 + 13)
        words = rng.choices(_VOCAB, weights=_WEIGHTS, k=self.cfg.words)
        words[rng.randrange(len(words))] = _TOPICS[i % len(_TOPICS)]
        return " ".join(words)

    def _page(self, i: int) -> str:
        n = self.cfg.pages
        rng = random.Random(i)
        links = [5 * i + k for k in range(1, 6) if 5 * i + k < n]
        links += [rng.randrange(n) for _ in range(3)]
        hrefs = [f"/page/{j}" for j in links]
        j = (i + 1) % n
        hrefs += [
            f"/page/{j}/",  # same page, trailing slash
            f"/page/{j}#comments",  # same page, fragment
            f"/page/{j}?b=2&a=1",  # query variants are different pages for the server ...
            f"/page/{j}?a=1&b=2",  # ... but the same canonical URL for the crawler
            "/private/secret",  # disallowed by robots.txt
            "http://example.org/elsewhere",  # another host: never followed
            "mailto:nobody@example.org",
        ]
        if i == 0 and self.cfg.faults:
            hrefs += [
                "/flaky/429",
                "/flaky/503",
                "/slow",
                "/redirect-loop",
                "/missing",
                "/file.pdf",
            ]
        anchors = "\n".join(f'<li><a href="{h}">link</a></li>' for h in hrefs)
        return (
            f"<!doctype html><html><head><meta charset='utf-8'><title>Page {i} about "
            f"{_TOPICS[i % len(_TOPICS)]}</title><style>p{{margin:0}}</style></head><body>"
            f"<h1>Page {i}</h1><script>var tracking = 'ignored';</script><p>{self._text(i)}</p>"
            f"<ul>{anchors}</ul></body></html>"
        )

    # ------------------------------------------------------------------ routing
    async def respond(self, target: str) -> tuple[int, dict[str, str], bytes]:
        url = urlsplit(target)
        path = url.path.rstrip("/") or "/"
        self.hits[path] += 1
        lat = self.cfg.latency_ms / 1000
        if path != "/robots.txt" and lat > 0:
            await asyncio.sleep(lat * random.uniform(0.8, 1.2))
        html = {"Content-Type": "text/html; charset=utf-8"}

        if path == "/robots.txt":
            body = "User-agent: *\nDisallow: /private/\n"
            if self.cfg.crawl_delay is not None:
                body += f"Crawl-delay: {self.cfg.crawl_delay}\n"
            return 200, {"Content-Type": "text/plain"}, body.encode()
        if path == "/":
            return 200, html, self._page(0).encode()
        if path.startswith("/page/"):
            tail = path.removeprefix("/page/")
            if tail.isdigit() and int(tail) < self.cfg.pages:
                return 200, html, self._page(int(tail)).encode()
        if path.startswith("/private/"):
            return 200, html, b"<html><title>secret</title><body>not for robots</body></html>"
        if self.cfg.faults:
            if path == "/flaky/429":
                if self.hits[path] == 1:
                    return 429, {"Retry-After": "1"}, b"slow down"
                return 200, html, b"<html><title>flaky 429</title><body>ok now</body></html>"
            if path == "/flaky/503":
                if self.hits[path] <= 2:
                    return 503, {}, b"try later"
                return 200, html, b"<html><title>flaky 503</title><body>recovered</body></html>"
            if path == "/slow":
                await asyncio.sleep(5)
                return 200, html, b"<html><title>slow</title><body>finally</body></html>"
            if path == "/redirect-loop":
                return 302, {"Location": "/redirect-loop"}, b""
            if path == "/file.pdf":
                return 200, {"Content-Type": "application/pdf"}, b"%PDF-1.4 not html"
        return 404, html, b"<html><title>404</title><body>not found</body></html>"

    # ------------------------------------------------------------------ HTTP/1.1
    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                try:
                    head = await reader.readuntil(b"\r\n\r\n")
                except (asyncio.IncompleteReadError, ConnectionError):
                    return
                lines = head.decode("latin-1").split("\r\n")
                method, _, rest = lines[0].partition(" ")
                target = rest.rsplit(" ", 1)[0]
                close = any(ln.lower().replace(" ", "") == "connection:close" for ln in lines[1:])
                if method != "GET":
                    status, headers, body = 405, {"Allow": "GET"}, b""
                else:
                    status, headers, body = await self.respond(target)
                reason = {200: "OK", 302: "Found", 404: "Not Found", 405: "Method Not Allowed",
                          429: "Too Many Requests", 503: "Service Unavailable"}.get(status, "OK")  # fmt: skip
                out = [f"HTTP/1.1 {status} {reason}", f"Content-Length: {len(body)}"]
                out += [f"{k}: {v}" for k, v in headers.items()]
                out.append("Connection: close" if close else "Connection: keep-alive")
                writer.write(("\r\n".join(out) + "\r\n\r\n").encode("latin-1") + body)
                await writer.drain()
                if close:
                    return
        except (ConnectionError, asyncio.CancelledError):
            return
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError, OSError):
                await writer.wait_closed()


async def start(
    config: MirrorConfig, host: str = "127.0.0.1", port: int = 0
) -> tuple[asyncio.Server, Mirror, int]:
    """Start listening; returns (server, handler state, actual port)."""
    mirror = Mirror(config)
    server = await asyncio.start_server(mirror.handle, host, port, backlog=256)
    actual = server.sockets[0].getsockname()[1]
    return server, mirror, int(actual)


def run_in_thread(config: MirrorConfig) -> tuple[str, Mirror, Callable[[], None]]:
    """Serve in a background thread (own event loop). Returns (base_url, mirror, stop).

    For tests and demos only: the crawler's loop and the server's loop are different loops in
    different threads, so the server's work never delays the crawler's loop.
    """
    ready = threading.Event()
    box: dict[str, object] = {}

    def target() -> None:
        async def main() -> None:
            server, mirror, port = await start(config)
            stop = asyncio.Event()
            box.update(mirror=mirror, port=port, loop=asyncio.get_running_loop(), stop=stop)
            ready.set()
            await stop.wait()
            server.close()
            await server.wait_closed()

        asyncio.run(main())

    thread = threading.Thread(target=target, name="mirror-server", daemon=True)
    thread.start()
    if not ready.wait(10):
        raise RuntimeError("mirror server did not start")
    loop = box["loop"]
    stop_event = box["stop"]
    assert isinstance(loop, asyncio.AbstractEventLoop) and isinstance(stop_event, asyncio.Event)

    def stop() -> None:
        loop.call_soon_threadsafe(stop_event.set)
        thread.join(5)

    mirror = box["mirror"]
    assert isinstance(mirror, Mirror)
    return f"http://127.0.0.1:{box['port']}", mirror, stop


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - runs a server until Ctrl+C
    ap = argparse.ArgumentParser(description="Serve a small local site for crawler experiments.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0, help="0 = pick a free port")
    ap.add_argument("--pages", type=int, default=400)
    ap.add_argument("--latency-ms", type=float, default=30.0)
    ap.add_argument("--words", type=int, default=300)
    ap.add_argument("--faults", action="store_true", help="add 429/503/slow/redirect-loop URLs")
    ap.add_argument("--crawl-delay", type=float, default=None)
    ap.add_argument("--corpus", type=Path, default=None, help="directory of .txt files for text")
    args = ap.parse_args(argv)
    cfg = MirrorConfig(args.pages, args.latency_ms, args.words, args.faults, args.crawl_delay,
                       args.corpus)  # fmt: skip

    async def serve() -> None:
        server, _, port = await start(cfg, args.host, args.port)
        print(f"LISTENING {port}", flush=True)  # the benchmark reads this line
        async with server:
            await server.serve_forever()

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve())


if __name__ == "__main__":
    main()
