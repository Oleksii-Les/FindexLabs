"""An asynchronous, polite web crawler (Lab 6).

One thread, one event loop, many requests in flight: the crawler spends nearly all its time
*waiting for the network*, and while one coroutine waits (``await``) the loop runs another.

Layers, bottom to top:

* :func:`canonicalize`, :func:`parse_html` -- pure functions (URL normal form; text and links);
* :func:`fetch` -- ONE URL: ``asyncio.timeout``, status / bytes / time, bounded retries with
  exponential backoff + jitter, ``Retry-After`` honoured, only 429 / 5xx / timeouts retried;
* :class:`RobotsCache` -- ``robots.txt`` per host, fetched once, cached;
* :class:`HostLimiter` -- the global request semaphore, a semaphore per host and a minimum pause
  between request starts to the same host;
* :func:`crawl` -- an *async generator*: URL queue + workers inside a ``TaskGroup``; yields
  :class:`Page` objects one by one, so nothing but the frontier is held in memory.

Rules this module lives by (they are the point of the lab):

* **Never block the loop.** Everything that waits is ``await``-ed. Parsing a page is a few
  milliseconds of C code (selectolax); the CPU-heavy indexing happens *after* the crawl, outside
  the loop (``findex index crawl.jsonl``). See ``experiments/loop_blocking.py``.
* **Be polite.** Honest User-Agent with a contact address, ``robots.txt`` (including
  ``Crawl-delay``), a ceiling on pages, per-host concurrency and spacing, never leave the seed hosts.
"""

from __future__ import annotations

import asyncio
import contextlib
import email.utils
import logging
import posixpath
import random
import time
import urllib.robotparser
from collections import Counter
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final
from urllib.parse import parse_qsl, quote, unquote, urlencode, urljoin, urlsplit, urlunsplit

import httpx
from selectolax.lexbor import LexborHTMLParser

log = logging.getLogger(__name__)
access_log = logging.getLogger(__name__ + ".access")  # one line per URL, goes to crawl.log

RETRYABLE_STATUSES: Final = frozenset({429}) | frozenset(range(500, 600))
HTML_TYPES: Final = ("text/html", "application/xhtml+xml")
DEFAULT_PORTS: Final = {"http": 80, "https": 443}


# ====================================================================== URL normal form
def _remove_dot_segments(path: str) -> str:
    """Resolve ``.``/``..``, collapse ``//``, drop the trailing slash; the root stays ``/``."""
    if not path:
        return "/"
    rest = posixpath.normpath(path).lstrip("/")  # normpath also strips a trailing slash
    return "/" + rest if rest else "/"


def canonicalize(url: str, base: str | None = None) -> str | None:
    """Return the canonical form of ``url`` (joined to ``base`` first), or ``None`` if it is not
    a crawlable http(s) URL.

    Two spellings of the same page must give the same string, because ``seen`` is a set of
    strings. The rules: scheme and host in lower case; default port dropped; credentials dropped;
    ``#fragment`` dropped; dot-segments resolved, repeated slashes collapsed and the trailing
    slash removed (``/a/`` == ``/a``; the root stays ``/``); query parameters sorted by name,
    empty query removed. Percent-escapes are re-encoded uniformly.
    """
    try:
        joined = urljoin(base, url.strip()) if base else url.strip()
        parts = urlsplit(joined)
        scheme = parts.scheme.lower()
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:  # e.g. an invalid IPv6 literal or port
        return None
    if scheme not in DEFAULT_PORTS or not host:
        return None
    netloc = f"[{host}]" if ":" in host else host
    if port is not None and port != DEFAULT_PORTS[scheme]:
        netloc += f":{port}"
    path = quote(_remove_dot_segments(unquote(parts.path)), safe="/:@!$&'()*+,;=-._~")
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, netloc, path, query, ""))


def host_of(url: str) -> str:
    """``host[:port]`` of a canonical URL: the unit politeness is applied to."""
    return urlsplit(url).netloc


# ============================================================================ HTML
@dataclass(frozen=True, slots=True)
class ParsedPage:
    title: str
    text: str
    links: list[str]  # canonical absolute URLs, in document order, without duplicates


def parse_html(body: bytes, base_url: str) -> ParsedPage:
    """Title, visible text and outgoing links of an HTML document.

    ``selectolax`` (Lexbor) is a C parser: a typical page takes about a millisecond, which is
    why parsing may stay on the event loop. Malformed HTML does not raise; it is repaired the way
    a browser would.
    """
    tree = LexborHTMLParser(body)
    title_node = tree.css_first("title")
    title = " ".join(title_node.text(strip=True).split()) if title_node else ""
    links: dict[str, None] = {}  # dict as an ordered set
    for node in tree.css("a[href]"):
        href = node.attributes.get("href")
        if href and (canon := canonicalize(href, base_url)):
            links.setdefault(canon)
    tree.strip_tags(["script", "style", "noscript", "template", "head"])
    root = tree.body
    text = " ".join(root.text(separator=" ", strip=True).split()) if root else ""
    return ParsedPage(title, text, list(links))


# ======================================================================== fetching
def parse_retry_after(value: str | None, now: datetime | None = None) -> float | None:
    """``Retry-After`` is either a number of seconds or an HTTP date. ``None`` if absent/garbage."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - (now or datetime.now(UTC))).total_seconds())


@dataclass(slots=True)
class FetchResult:
    """What happened to one URL (all attempts together)."""

    url: str
    final_url: str  # after redirects
    status: int | None  # last HTTP status, None if no response at all
    nbytes: int  # body bytes received by the last attempt
    elapsed: float  # seconds for ALL attempts including backoff sleeps
    attempts: int
    content_type: str = ""
    body: bytes = b""
    error: str | None = None  # None means: a response was received and it was a success (2xx)
    truncated: bool = False
    retry_after_header: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.status is not None and 200 <= self.status < 300


class OffSiteRedirect(Exception):
    """Raised by the request hook when a redirect leads to a host we may not visit."""


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    retries: int = 3  # retries AFTER the first attempt: at most 4 requests per URL
    backoff_base: float = 0.5  # seconds; the cap for attempt k is base * 2**k
    backoff_max: float = 20.0
    max_retry_after: float = 60.0  # a longer Retry-After is not waited for: give up instead


def backoff_delay(attempt: int, policy: RetryPolicy, rng: random.Random) -> float:
    """Exponential backoff with FULL jitter: uniform(0, min(max, base * 2**attempt)).

    The exponent spreads load over time when a server is struggling; the jitter keeps many
    clients (or many workers of ours) from retrying in lockstep.
    """
    return rng.uniform(0.0, min(policy.backoff_max, policy.backoff_base * 2**attempt))


Limiter = Callable[[str], AbstractAsyncContextManager[None]]


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    policy: RetryPolicy | None = None,
    max_bytes: int = 2 * 1024 * 1024,
    limiter: Limiter | None = None,
    rng: random.Random | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_retry: Callable[[], None] | None = None,
) -> FetchResult:
    """GET ``url`` and say what happened. Never raises for network or HTTP problems.

    * every attempt runs under ``asyncio.timeout(timeout)`` (headers AND body);
    * retried: 429, any 5xx, timeouts -- at most ``policy.retries`` times, sleeping
      ``max(Retry-After, jittered backoff)`` between attempts; NOT retried: other 4xx, redirect
      loops, connection errors, and a ``Retry-After`` longer than ``policy.max_retry_after``;
    * ``limiter(url)`` is an async context manager entered around each attempt *only* (so the
      backoff sleep does not hold a request slot);
    * the body is read in chunks and cut at ``max_bytes`` (``truncated``).
    """
    policy = policy or RetryPolicy()
    rng = rng or random.Random()
    t0 = time.perf_counter()
    result = FetchResult(url, url, None, 0, 0.0, 0)
    for attempt in range(policy.retries + 1):
        result.attempts = attempt + 1
        retry_after: float | None = None
        result.error = None
        try:
            async with contextlib.AsyncExitStack() as stack:
                if limiter is not None:
                    await stack.enter_async_context(limiter(url))
                async with asyncio.timeout(timeout):
                    await _one_attempt(client, url, result, max_bytes)
            if result.status is not None and not 200 <= result.status < 300:
                result.error = f"HTTP {result.status}"
            if result.status in RETRYABLE_STATUSES:
                retry_after = parse_retry_after(result.retry_after_header)
        except TimeoutError:
            result.status, result.error = None, "timeout"
        except httpx.TimeoutException:
            result.status, result.error = None, "timeout"
        except httpx.TooManyRedirects:
            result.status, result.error = None, "too many redirects"
            break  # a loop will not get better by waiting
        except OffSiteRedirect as exc:
            result.status, result.error = None, str(exc)
            break
        except httpx.HTTPError as exc:  # connection refused / reset / DNS / protocol errors
            result.status, result.error = None, f"{type(exc).__name__}: {exc}"
            break
        if result.error is None:
            break
        retryable = result.status in RETRYABLE_STATUSES or result.error == "timeout"
        if not retryable or attempt == policy.retries:
            break
        wait = backoff_delay(attempt, policy, rng)
        if retry_after is not None:
            if retry_after > policy.max_retry_after:
                log.warning("%s: Retry-After %.0f s is too long, giving up", url, retry_after)
                break
            wait = max(wait, retry_after)
        log.warning(
            "retry %d/%d for %s after %s (Retry-After=%s): sleeping %.2f s",
            attempt + 1,
            policy.retries,
            url,
            result.error,
            "-" if retry_after is None else f"{retry_after:.0f}",
            wait,
        )
        if on_retry:
            on_retry()
        await sleep(wait)
    result.elapsed = time.perf_counter() - t0
    return result


async def _one_attempt(
    client: httpx.AsyncClient, url: str, result: FetchResult, max_bytes: int
) -> None:
    """One GET; fills ``result`` (status, final_url, body ...). The body is read only for 2xx."""
    async with client.stream("GET", url) as response:
        result.status = response.status_code
        result.final_url = str(response.url)
        result.content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        result.retry_after_header = response.headers.get("retry-after")
        chunks: list[bytes] = []
        received = 0
        result.truncated = False
        if 200 <= response.status_code < 300:
            async for chunk in response.aiter_bytes():
                chunks.append(chunk)
                received += len(chunk)
                if received > max_bytes:
                    result.truncated = True
                    break
        result.nbytes = min(received, max_bytes)
        result.body = b"".join(chunks)[:max_bytes]


# ====================================================================== robots.txt
@dataclass(slots=True)
class _RobotsEntry:
    parser: urllib.robotparser.RobotFileParser | None  # None: no rules, see allow_all
    allow_all: bool


class RobotsCache:
    """``robots.txt`` per host (scheme + host + port), fetched at most once and cached.

    Outcomes follow RFC 9309: a 2xx file is parsed; **4xx** (no file) means everything is allowed;
    **5xx, 429, timeouts and network errors** mean "unreachable" and everything is *disallowed* --
    when in doubt, the polite answer is no. Concurrent callers for the same host wait on one
    lock instead of fetching the file several times.
    """

    MAX_BYTES: Final = 512 * 1024

    def __init__(
        self,
        client: httpx.AsyncClient,
        user_agent: str,
        *,
        limiter: Limiter | None = None,
        timeout: float = 10.0,
        policy: RetryPolicy | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._client = client
        self._ua = user_agent
        self._limiter = limiter
        self._timeout = timeout
        self._policy = policy or RetryPolicy(retries=2)
        self._rng = rng
        self._entries: dict[str, _RobotsEntry] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self.fetches: Counter[str] = Counter()  # robots.txt requests per host (for tests / stats)

    async def _entry(self, url: str) -> _RobotsEntry:
        parts = urlsplit(url)
        key = f"{parts.scheme}://{parts.netloc}"
        if key in self._entries:
            return self._entries[key]
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key not in self._entries:  # somebody else may have filled it while we waited
                self._entries[key] = await self._load(key)
        return self._entries[key]

    async def _load(self, key: str) -> _RobotsEntry:
        robots_url = f"{key}/robots.txt"
        self.fetches[urlsplit(key).netloc] += 1
        res = await fetch(
            self._client,
            robots_url,
            timeout=self._timeout,
            policy=self._policy,
            max_bytes=self.MAX_BYTES,
            limiter=self._limiter,
            rng=self._rng,
        )
        if res.ok:
            parser = urllib.robotparser.RobotFileParser()
            parser.parse(res.body.decode("utf-8", errors="replace").splitlines())
            log.info("robots.txt of %s loaded (%d bytes)", key, res.nbytes)
            return _RobotsEntry(parser, allow_all=False)
        if res.status is not None and 400 <= res.status < 500 and res.status != 429:
            log.info("no robots.txt at %s (HTTP %d): everything is allowed", key, res.status)
            return _RobotsEntry(None, allow_all=True)
        log.warning(
            "robots.txt of %s is unreachable (%s): treating the host as disallowed", key, res.error
        )
        return _RobotsEntry(None, allow_all=False)

    async def allowed(self, url: str) -> bool:
        entry = await self._entry(url)
        if entry.allow_all:
            return True
        if entry.parser is None:
            return False
        return entry.parser.can_fetch(self._ua, url)

    async def crawl_delay(self, url: str) -> float | None:
        """The ``Crawl-delay`` the host asks for, in seconds, if any."""
        entry = await self._entry(url)
        if entry.parser is None:
            return None
        delay = entry.parser.crawl_delay(self._ua)
        return None if delay is None else float(delay)


# ====================================================================== politeness
@dataclass(slots=True)
class _HostState:
    sem: asyncio.Semaphore
    delay: float
    next_slot: float = 0.0


class HostLimiter:
    """``async with limiter(url):`` -- the three politeness limits in one place.

    1. ``per_host`` simultaneous requests to one host (a semaphore per host);
    2. a minimum ``delay`` between the *starts* of requests to one host (a slot clock);
    3. ``concurrency`` simultaneous requests in total (ONE semaphore for everything).

    The global semaphore is taken last and held only while the request is really in flight, so a
    worker waiting for its turn at a slow host never blocks requests to other hosts.
    """

    def __init__(
        self,
        concurrency: int,
        per_host: int,
        delay: float,
        on_flight: Callable[[int], None] | None = None,
    ) -> None:
        self._global = asyncio.Semaphore(concurrency)
        self._per_host = per_host
        self._delay = delay
        self._hosts: dict[str, _HostState] = {}
        self._on_flight = on_flight
        self.in_flight = 0

    def _state(self, host: str) -> _HostState:
        state = self._hosts.get(host)
        if state is None:
            state = self._hosts[host] = _HostState(asyncio.Semaphore(self._per_host), self._delay)
        return state

    def raise_delay(self, host: str, delay: float) -> None:
        """Apply a ``Crawl-delay`` from robots.txt (never lowers the configured delay)."""
        state = self._state(host)
        state.delay = max(state.delay, delay)

    @contextlib.asynccontextmanager
    async def __call__(self, url: str) -> AsyncGenerator[None]:
        state = self._state(host_of(url))
        async with state.sem:
            loop = asyncio.get_running_loop()
            now = loop.time()
            start = max(now, state.next_slot)  # no await between read and write: atomic
            state.next_slot = start + state.delay
            if start > now:
                await asyncio.sleep(start - now)
            async with self._global:
                self.in_flight += 1
                if self._on_flight:
                    self._on_flight(self.in_flight)
                try:
                    yield
                finally:
                    self.in_flight -= 1
                    if self._on_flight:
                        self._on_flight(self.in_flight)


# ============================================================================ crawl
@dataclass(frozen=True, slots=True)
class CrawlConfig:
    seeds: tuple[str, ...]
    user_agent: str  # honest, with a contact address: "findex-crawler/0.6 (+mailto:me@example.org)"
    max_pages: int = 200  # the ceiling: the crawl stops here whatever the site offers
    concurrency: int = 5  # simultaneous requests, all hosts together
    per_host: int = 2  # simultaneous requests to one host
    delay: float = 0.1  # minimum seconds between request starts to one host
    timeout: float = 10.0
    max_depth: int | None = None
    max_bytes: int = 2 * 1024 * 1024
    policy: RetryPolicy = field(default_factory=RetryPolicy)
    workers: int | None = None  # default: 2 x concurrency (some are always waiting for a turn)
    allowed_hosts: frozenset[str] | None = None  # default: the hosts of the seeds
    rng_seed: int | None = None  # fixes the backoff jitter (tests)

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("at least one seed URL is required")
        if self.max_pages < 1 or self.concurrency < 1 or self.per_host < 1:
            raise ValueError("max_pages, concurrency and per_host must be >= 1")
        if self.delay < 0 or self.timeout <= 0:
            raise ValueError("delay must be >= 0 and timeout > 0")
        if not self.user_agent.strip():
            raise ValueError("a User-Agent is required")


@dataclass(slots=True)
class CrawlStats:
    """Live counters, read by the progress display while the crawl runs."""

    started: float = field(default_factory=time.perf_counter)
    finished: float | None = None
    pages: int = 0  # pages delivered to the consumer
    errors: int = 0  # URLs that failed for good (after retries)
    retries: int = 0
    in_flight: int = 0  # requests waiting for / receiving a response right now
    queue_depth: int = 0  # URLs discovered but not yet taken by a worker
    robots_skipped: int = 0
    other_skipped: int = 0  # not HTML/text, left the allowed hosts, budget reached
    bytes: int = 0
    statuses: Counter[str] = field(default_factory=lambda: Counter[str]())

    @property
    def elapsed(self) -> float:
        return (self.finished or time.perf_counter()) - self.started

    @property
    def pages_per_second(self) -> float:
        return self.pages / self.elapsed if self.elapsed > 0 else 0.0


@dataclass(frozen=True, slots=True)
class Page:
    url: str
    final_url: str
    title: str
    text: str
    status: int
    depth: int
    nbytes: int
    elapsed: float
    attempts: int
    fetched_at: str  # ISO 8601, UTC


class _Done:
    """Sentinel: the frontier is exhausted."""


@dataclass(slots=True)
class _Failure:
    error: Exception


class _Run:
    """The state of one crawl: frontier, ``seen`` set, budget and the worker coroutines."""

    def __init__(
        self,
        config: CrawlConfig,
        client: httpx.AsyncClient,
        stats: CrawlStats,
        hosts: frozenset[str],
    ) -> None:
        self.cfg = config
        self.client = client
        self.stats = stats
        self.hosts = hosts
        self.queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
        self.out: asyncio.Queue[Page | _Done | _Failure] = asyncio.Queue(
            maxsize=max(4, 2 * config.concurrency)  # back-pressure on a slow consumer
        )
        self.seen: set[str] = set()
        self.seen_limit = max(1000, config.max_pages * 50)  # the frontier must not grow forever
        self.reserved = 0  # pages fetched or being fetched: never above max_pages
        self.delivered = 0  # pages handed to the consumer
        self.budget = asyncio.Condition()  # signalled when a reservation is released or used up
        self.rng = random.Random(config.rng_seed)
        self.limiter = HostLimiter(config.concurrency, config.per_host, config.delay, self._flight)
        self.robots = RobotsCache(
            client,
            config.user_agent,
            limiter=self.limiter,
            timeout=config.timeout,
            policy=RetryPolicy(retries=2, backoff_base=config.policy.backoff_base),
            rng=self.rng,
        )

    def _flight(self, n: int) -> None:
        self.stats.in_flight = n

    def enqueue(self, url: str, depth: int) -> None:
        if url in self.seen or len(self.seen) >= self.seen_limit or host_of(url) not in self.hosts:
            return
        self.seen.add(url)
        self.queue.put_nowait((url, depth))
        self.stats.queue_depth = self.queue.qsize()

    async def worker(self) -> None:
        while True:
            url, depth = await self.queue.get()
            self.stats.queue_depth = self.queue.qsize()
            try:
                await self.visit(url, depth)
            finally:
                self.queue.task_done()

    async def _reserve(self) -> bool:
        """Claim one page of the budget, waiting while the budget is fully reserved.

        Waiting (not dropping the URL) matters: a reserved fetch may still fail, and then its
        slot goes to a waiting URL. False means the budget is really used up.
        """
        cfg = self.cfg
        async with self.budget:
            await self.budget.wait_for(
                lambda: self.reserved < cfg.max_pages or self.delivered >= cfg.max_pages
            )
            if self.delivered >= cfg.max_pages:
                return False
            self.reserved += 1
            return True

    async def _release(self, delivered: bool) -> None:
        async with self.budget:
            if delivered:
                self.delivered += 1
            else:
                self.reserved -= 1  # a failed or skipped URL does not use up the page budget
            self.budget.notify_all()

    async def visit(self, url: str, depth: int) -> None:
        cfg, stats = self.cfg, self.stats
        if self.delivered >= cfg.max_pages:
            return
        if not await self.robots.allowed(url):
            stats.robots_skipped += 1
            access_log.info("SKIP %s reason=robots.txt depth=%d", url, depth)
            return
        delay = await self.robots.crawl_delay(url)
        if delay:
            self.limiter.raise_delay(host_of(url), delay)
        if not await self._reserve():
            return
        delivered = False
        try:
            res = await fetch(
                self.client,
                url,
                timeout=cfg.timeout,
                policy=cfg.policy,
                max_bytes=cfg.max_bytes,
                limiter=self.limiter,
                rng=self.rng,
                on_retry=self._count_retry,
            )
            stats.statuses[str(res.status) if res.status else (res.error or "error")] += 1
            if not res.ok:
                stats.errors += 1
                access_log.warning(
                    'FAIL %s error="%s" status=%s attempts=%d time=%.3fs depth=%d',
                    url, res.error, res.status, res.attempts, res.elapsed, depth,
                )  # fmt: skip
                return
            final = canonicalize(res.final_url)
            if final is None or host_of(final) not in self.hosts:
                stats.other_skipped += 1
                access_log.info("SKIP %s reason=redirected off-site to %s", url, res.final_url)
                return
            is_html = res.content_type in HTML_TYPES
            if not is_html and res.content_type != "text/plain":
                stats.other_skipped += 1
                access_log.info("SKIP %s reason=content-type %r", url, res.content_type)
                return
            try:
                if is_html:
                    parsed = parse_html(res.body, final)
                else:
                    parsed = ParsedPage("", res.body.decode("utf-8", errors="replace"), [])
            except Exception as exc:  # hostile HTML is expected in the wild; a bug elsewhere is not
                stats.errors += 1
                access_log.warning("FAIL %s error=%r (parse) depth=%d", url, exc, depth)
                return
            if cfg.max_depth is None or depth < cfg.max_depth:
                for link in parsed.links:
                    if self.delivered < cfg.max_pages:
                        self.enqueue(link, depth + 1)
            access_log.info(
                "GET %s status=%d bytes=%d time=%.3fs attempts=%d depth=%d",
                url, res.status, res.nbytes, res.elapsed, res.attempts, depth,
            )  # fmt: skip
            page = Page(
                url=url,
                final_url=final,
                title=parsed.title,
                text=parsed.text,
                status=res.status or 0,
                depth=depth,
                nbytes=res.nbytes,
                elapsed=res.elapsed,
                attempts=res.attempts,
                fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
            stats.pages += 1
            stats.bytes += res.nbytes
            delivered = True
            await self.out.put(page)
        finally:
            await asyncio.shield(self._release(delivered))

    def _count_retry(self) -> None:
        self.stats.retries += 1

    async def produce(self) -> None:
        """Run the workers in a TaskGroup until the frontier is empty, then say so."""
        try:
            async with asyncio.TaskGroup() as tg:
                n = self.cfg.workers or 2 * self.cfg.concurrency
                workers = [tg.create_task(self.worker(), name=f"worker-{i}") for i in range(n)]
                await self.queue.join()  # every discovered URL has been handled
                for task in workers:
                    task.cancel()  # idle workers wait on queue.get(): stop them
        except Exception as exc:  # incl. ExceptionGroup: hand it to the consumer
            await self.out.put(_Failure(exc))
            return
        await self.out.put(_Done())


def make_client(
    config: CrawlConfig,
    hosts: frozenset[str],
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    """The one HTTP client of a crawl: honest User-Agent, bounded redirects and connections,
    and a hook that refuses to follow a redirect to a host we were not allowed to visit."""

    async def refuse_off_site(request: httpx.Request) -> None:
        canon = canonicalize(str(request.url))
        if canon is None or host_of(canon) not in hosts:
            raise OffSiteRedirect(f"redirect to {request.url.host} leaves the allowed hosts")

    return httpx.AsyncClient(
        headers={
            "User-Agent": config.user_agent,
            "Accept": "text/html,application/xhtml+xml;q=0.9,text/plain;q=0.5",
        },
        follow_redirects=True,
        max_redirects=5,
        timeout=httpx.Timeout(config.timeout * 2),  # asyncio.timeout in fetch() is the real limit
        limits=httpx.Limits(
            max_connections=config.concurrency, max_keepalive_connections=config.concurrency
        ),
        event_hooks={"request": [refuse_off_site]},
        transport=transport,
    )


async def crawl(
    config: CrawlConfig,
    *,
    stats: CrawlStats | None = None,
    client: httpx.AsyncClient | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncGenerator[Page]:
    """Crawl from ``config.seeds`` and yield pages as they arrive (arrival order, not BFS order).

    An async generator: nothing runs until the first ``async for`` step, and pages are handed over
    one at a time, so the memory held is the frontier and the ``seen`` set, not the pages.
    Stops after ``config.max_pages`` pages or when the frontier is empty; then the workers are
    cancelled and the HTTP client closed. Use it as::

        async with contextlib.aclosing(crawl(config)) as pages:
            async for page in pages: ...

    ``aclosing`` guarantees the clean-up even if you leave the loop early.
    A bug in a worker (anything but a network / HTTP problem) is not swallowed: it surfaces
    here as an ``ExceptionGroup`` and the crawl stops.
    """
    stats = stats if stats is not None else CrawlStats()
    seeds = [c for s in config.seeds if (c := canonicalize(s)) is not None]
    if not seeds:
        raise ValueError(f"no valid http(s) seed URL in {config.seeds!r}")
    hosts = config.allowed_hosts or frozenset(host_of(s) for s in seeds)
    own_client = client is None
    http = client if client is not None else make_client(config, hosts, transport)
    run = _Run(config, http, stats, hosts)
    producer: asyncio.Task[None] | None = None
    try:
        for seed in seeds:
            run.enqueue(seed, 0)
        # The TaskGroup lives in `produce()`, not around the `yield` below: a cancellation caused
        # by a failing worker must never be delivered into the consumer's own code.
        producer = asyncio.create_task(run.produce(), name="crawl-producer")
        emitted = 0
        while emitted < config.max_pages:
            item = await run.out.get()
            if isinstance(item, _Done):
                break
            if isinstance(item, _Failure):
                raise item.error
            emitted += 1
            yield item
    finally:
        if producer is not None:
            producer.cancel()  # cancels the TaskGroup, hence every worker
            await asyncio.gather(producer, return_exceptions=True)
        if own_client:
            await http.aclose()
        stats.finished = time.perf_counter()
        stats.queue_depth = 0
        stats.in_flight = 0
