"""Lab 6: the crawler, tested without any live network (``httpx.MockTransport`` only)."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import random
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st

from findex.crawl_run import SlowCallbackCounter
from findex.crawler import (
    CrawlConfig,
    CrawlStats,
    HostLimiter,
    OffSiteRedirect,
    RetryPolicy,
    RobotsCache,
    backoff_delay,
    canonicalize,
    crawl,
    fetch,
    make_client,
    parse_html,
    parse_retry_after,
)

UA = "findex-test/1.0 (+mailto:tests@example.org)"
FAST = RetryPolicy(retries=3, backoff_base=0.001, backoff_max=0.01)

Handler = Callable[[httpx.Request], Awaitable[httpx.Response] | httpx.Response]


def run[T](coro: Awaitable[T]) -> T:
    async def wrapper() -> T:
        return await coro

    return asyncio.run(wrapper())


def client_for(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=True, max_redirects=5
    )


async def no_sleep(_seconds: float) -> None:
    return None


# ------------------------------------------------------------------------ canonicalize
@pytest.mark.parametrize(
    ("url", "base", "expected"),
    [
        ("HTTP://Example.COM/Path", None, "http://example.com/Path"),  # host lower, path kept
        ("http://example.com", None, "http://example.com/"),
        ("http://example.com:80/x", None, "http://example.com/x"),
        ("https://example.com:443/x", None, "https://example.com/x"),
        ("http://example.com:8080/x", None, "http://example.com:8080/x"),
        ("http://example.com/a/#frag", None, "http://example.com/a"),
        ("http://example.com/a#one", None, "http://example.com/a"),
        ("http://example.com/a//b/./c/../d/", None, "http://example.com/a/b/d"),
        ("http://example.com/?b=2&a=1&c=", None, "http://example.com/?a=1&b=2&c="),
        ("http://example.com/p?", None, "http://example.com/p"),
        ("http://user:secret@example.com/x", None, "http://example.com/x"),
        ("http://[::1]:8000/x", None, "http://[::1]:8000/x"),
        ("page.html", "http://h/dir/index.html", "http://h/dir/page.html"),
        ("../up", "http://h/a/b/c", "http://h/a/up"),
        ("/abs?z=1&y=2", "http://h/dir/", "http://h/abs?y=2&z=1"),
        ("//other.org/x", "https://h/", "https://other.org/x"),
        ("#top", "http://h/page", "http://h/page"),
        ("/café", "http://h/", "http://h/caf%C3%A9"),
    ],
)
def test_canonicalize(url: str, base: str | None, expected: str):
    assert canonicalize(url, base) == expected


@pytest.mark.parametrize(
    "url",
    [
        "mailto:a@b.c",
        "javascript:void(0)",
        "ftp://h/file",
        "tel:123",
        "",
        "http://",
        "data:text/x,1",
    ],
)
def test_canonicalize_rejects_non_http(url: str):
    assert canonicalize(url) is None


def test_every_spelling_of_a_page_has_one_canonical_form():
    spellings = [
        "http://Example.com/a/b?x=1&y=2",
        "http://example.com:80/a/b/?y=2&x=1#top",
        "HTTP://EXAMPLE.COM/a/./b/?x=1&y=2",
        "http://example.com//a//b?y=2&x=1",
    ]
    assert len({canonicalize(s) for s in spellings}) == 1


_hosts = st.from_regex(r"[a-z][a-z0-9]{0,8}(\.[a-z]{2,4})?", fullmatch=True)
_segments = st.lists(st.sampled_from(["a", "b", "..", ".", "", "x y", "café", "A"]), max_size=5)
_queries = st.lists(
    st.tuples(st.sampled_from(["a", "b", "q"]), st.sampled_from(["", "1", "x y"])), max_size=4
)


@given(
    scheme=st.sampled_from(["http", "https", "HTTP"]),
    host=_hosts,
    segs=_segments,
    query=_queries,
    frag=st.sampled_from(["", "#f"]),
)
def test_canonicalize_is_idempotent(
    scheme: str, host: str, segs: list[str], query: list[tuple[str, str]], frag: str
):
    q = "&".join(f"{k}={v}" for k, v in query)
    url = f"{scheme}://{host}/{'/'.join(segs)}{'?' + q if q else ''}{frag}"
    once = canonicalize(url)
    assert once is not None
    assert canonicalize(once) == once
    assert "#" not in once


# ----------------------------------------------------------------------------- parsing
HTML = b"""<!doctype html><html><head><title> My   page </title>
<style>body{color:red}</style><script>var hidden = 1;</script></head>
<body><h1>Heading</h1><p>Some <b>bold</b> text.</p>
<a href="/one">1</a> <a href="two?b=2&a=1#x">2</a> <a href="/one/">dup</a>
<a href="mailto:x@y.z">mail</a> <a href="http://other.org/z">out</a> <a>no href</a>
<noscript>enable js</noscript></body></html>"""


def test_parse_html_title_text_and_links():
    page = parse_html(HTML, "http://site.org/dir/index.html")
    assert page.title == "My page"
    assert "Heading" in page.text and "Some bold text." in page.text
    assert "hidden" not in page.text and "color:red" not in page.text
    assert "enable js" not in page.text
    assert page.links == [
        "http://site.org/one",  # "/one/" is a duplicate of it
        "http://site.org/dir/two?a=1&b=2",
        "http://other.org/z",
    ]


def test_parse_html_survives_garbage():
    page = parse_html(b"\xff\xfe<<<>>><p>unclosed <a href=/q>x", "http://h/")
    assert page.links == ["http://h/q"] and "unclosed" in page.text
    assert parse_html(b"", "http://h/").links == []


# --------------------------------------------------------------------------- retry-after
def test_parse_retry_after():
    now = datetime(2015, 10, 21, 7, 27, 50, tzinfo=UTC)
    assert parse_retry_after("7") == 7.0
    assert parse_retry_after(" 120 ") == 120.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("soon") is None
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT", now) == pytest.approx(10.0)
    assert parse_retry_after("Wed, 21 Oct 2015 07:00:00 GMT", now) == 0.0  # in the past


def test_backoff_is_exponential_and_jittered():
    policy = RetryPolicy(backoff_base=0.5, backoff_max=3.0)

    class Top(random.Random):  # always picks the top of the range: shows the cap
        def uniform(self, a: float, b: float) -> float:
            return b

    caps = [backoff_delay(k, policy, Top()) for k in range(5)]
    assert caps == [0.5, 1.0, 2.0, 3.0, 3.0]  # doubles, then stops at backoff_max
    rng = random.Random(1)
    draws = {backoff_delay(2, policy, rng) for _ in range(50)}
    assert len(draws) > 40 and all(0 <= d <= 2.0 for d in draws)  # jitter: many different values


# ------------------------------------------------------------------------------ fetch
def test_fetch_success_records_status_bytes_and_time():
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, html="<p>hello</p>")

    async def go():
        async with client_for(handler) as client:
            return await fetch(client, "http://h/x")

    res = run(go())
    assert res.ok and res.status == 200 and res.attempts == 1
    assert res.nbytes == len(b"<p>hello</p>") and res.body == b"<p>hello</p>"
    assert res.content_type == "text/html" and res.elapsed > 0 and res.error is None


def test_fetch_retries_429_and_honours_retry_after():
    calls = 0

    def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "3"})
        return httpx.Response(200, text="ok")

    sleeps: list[float] = []
    retried: list[int] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    async def go():
        async with client_for(handler) as client:
            return await fetch(
                client,
                "http://h/",
                policy=FAST,
                sleep=fake_sleep,
                on_retry=lambda: retried.append(1),
            )

    res = run(go())
    assert res.ok and res.attempts == 2 and retried == [1]
    assert sleeps == [3.0]  # max(tiny jittered backoff, Retry-After)


@pytest.mark.parametrize("status", [500, 502, 503, 504, 599])
def test_fetch_retries_server_errors_at_most_three_times(status: int):
    calls = 0

    def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status)

    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    async def go():
        async with client_for(handler) as client:
            return await fetch(client, "http://h/", policy=FAST, sleep=fake_sleep)

    res = run(go())
    assert calls == 4 and res.attempts == 4 and len(sleeps) == 3  # 1 try + 3 retries
    assert not res.ok and res.status == status and res.error == f"HTTP {status}"


def test_fetch_backoff_sleeps_grow_exponentially():
    class Top(random.Random):
        def uniform(self, a: float, b: float) -> float:
            return b

    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    async def go():
        async with client_for(lambda _r: httpx.Response(503)) as client:
            policy = RetryPolicy(retries=3, backoff_base=0.5, backoff_max=30)
            return await fetch(client, "http://h/", policy=policy, rng=Top(), sleep=fake_sleep)

    run(go())
    assert sleeps == [0.5, 1.0, 2.0]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410, 418])
def test_fetch_never_retries_other_4xx(status: int):
    calls = 0

    def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status)

    async def go():
        async with client_for(handler) as client:
            return await fetch(client, "http://h/", policy=FAST, sleep=no_sleep)

    res = run(go())
    assert calls == 1 and res.attempts == 1 and res.status == status and not res.ok


def test_fetch_times_out_and_retries_a_timeout():
    calls = 0

    async def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(5)  # cancelled by asyncio.timeout long before this ends
        return httpx.Response(200, text="late but fine")

    async def go():
        async with client_for(handler) as client:
            return await fetch(client, "http://h/", timeout=0.05, policy=FAST, sleep=no_sleep)

    t0 = time.perf_counter()
    res = run(go())
    assert time.perf_counter() - t0 < 2
    assert res.ok and res.attempts == 2


def test_fetch_gives_up_after_repeated_timeouts():
    async def handler(_req: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200)

    async def go():
        async with client_for(handler) as client:
            policy = RetryPolicy(retries=2, backoff_base=0.001)
            return await fetch(client, "http://h/", timeout=0.02, policy=policy, sleep=no_sleep)

    res = run(go())
    assert not res.ok and res.error == "timeout" and res.attempts == 3 and res.status is None


def test_fetch_treats_httpx_timeouts_like_timeouts():
    calls = 0

    def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("slow server")

    async def go():
        async with client_for(handler) as client:
            return await fetch(client, "http://h/", policy=FAST, sleep=no_sleep)

    res = run(go())
    assert calls == 4 and res.error == "timeout"


def test_fetch_does_not_wait_for_a_huge_retry_after():
    calls = 0

    def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, headers={"Retry-After": "86400"})

    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    async def go():
        async with client_for(handler) as client:
            return await fetch(client, "http://h/", policy=FAST, sleep=fake_sleep)

    res = run(go())
    assert calls == 1 and sleeps == [] and res.status == 429 and not res.ok


def test_fetch_redirect_loop_is_one_failure_not_a_retry_storm():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(302, headers={"Location": str(request.url)})

    async def go():
        async with client_for(handler) as client:
            return await fetch(client, "http://h/loop", policy=FAST, sleep=no_sleep)

    res = run(go())
    assert res.error == "too many redirects" and res.attempts == 1 and 5 <= calls <= 7


def test_fetch_connection_errors_are_reported_not_raised_and_not_retried():
    def handler(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async def go():
        async with client_for(handler) as client:
            return await fetch(client, "http://h/", policy=FAST, sleep=no_sleep)

    res = run(go())
    assert res.attempts == 1 and res.status is None
    assert res.error is not None and res.error.startswith("ConnectError")


def test_fetch_cuts_the_body_at_max_bytes():
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 10_000)

    async def go():
        async with client_for(handler) as client:
            return await fetch(client, "http://h/", max_bytes=1000)

    res = run(go())
    assert res.truncated and res.nbytes == 1000 and len(res.body) == 1000 and res.ok


def test_fetch_limiter_wraps_every_attempt_but_not_the_backoff_sleep():
    active = 0
    entered = 0
    held_during_sleep: list[int] = []

    @contextlib.asynccontextmanager
    async def limiter(_url: str):
        nonlocal active, entered
        active += 1
        entered += 1
        try:
            yield
        finally:
            active -= 1

    async def fake_sleep(_s: float) -> None:
        held_during_sleep.append(active)

    async def go():
        async with client_for(lambda _r: httpx.Response(503)) as client:
            return await fetch(client, "http://h/", policy=FAST, limiter=limiter, sleep=fake_sleep)

    res = run(go())
    assert entered == res.attempts == 4
    assert held_during_sleep == [0, 0, 0]  # a request slot is never held while sleeping


def test_off_site_redirect_is_refused_before_the_request_is_made():
    seen_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_hosts.append(request.url.host)
        if request.url.host == "good.example":
            return httpx.Response(302, headers={"Location": "http://evil.example/x"})
        return httpx.Response(200, text="should never be fetched")

    config = CrawlConfig(("http://good.example/",), UA)

    async def go():
        client = make_client(config, frozenset({"good.example"}), httpx.MockTransport(handler))
        async with client:
            return await fetch(client, "http://good.example/", policy=FAST)

    res = run(go())
    assert seen_hosts == ["good.example"]
    assert res.error is not None and "leaves the allowed hosts" in res.error and res.attempts == 1
    assert issubclass(OffSiteRedirect, Exception)


# ------------------------------------------------------------------------- robots.txt
def robots_client(
    files: dict[str, tuple[int, str]], log: list[str] | None = None
) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/robots.txt"
        if log is not None:
            log.append(request.url.host)
        status, body = files.get(request.url.host, (404, ""))
        return httpx.Response(status, text=body)

    return client_for(handler)


ROBOTS = (
    "User-agent: findex-test\nDisallow: /secret\nCrawl-delay: 2\n\nUser-agent: *\nDisallow: /\n"
)


def test_robots_rules_apply_to_our_user_agent_token():
    async def go():
        async with robots_client({"a.example": (200, ROBOTS)}) as client:
            cache = RobotsCache(client, UA, policy=FAST)
            return (
                await cache.allowed("http://a.example/open"),
                await cache.allowed("http://a.example/secret/page"),
                await cache.crawl_delay("http://a.example/open"),
            )

    assert run(go()) == (True, False, 2.0)


def test_robots_is_fetched_once_per_host_even_for_concurrent_callers():
    log: list[str] = []

    async def go():
        files = {"a.example": (200, "User-agent: *\nDisallow: /x\n"), "b.example": (404, "")}
        async with robots_client(files, log) as client:
            cache = RobotsCache(client, UA, policy=FAST)
            urls = [f"http://a.example/p{i}" for i in range(20)] + ["http://b.example/q"] * 5
            urls += ["http://a.example/x/1", "http://a.example:8080/other-port"]
            results = await asyncio.gather(*(cache.allowed(u) for u in urls))
            return cache, results

    cache, results = run(go())
    assert sorted(log) == ["a.example", "a.example", "b.example"]  # a.example:8080 is its own host
    assert cache.fetches == Counter({"a.example": 1, "b.example": 1, "a.example:8080": 1})
    assert results[:20] == [True] * 20 and results[20:25] == [True] * 5
    assert results[25] is False  # /x is disallowed on a.example


@pytest.mark.parametrize("status", [401, 403, 404, 410])
def test_robots_4xx_means_everything_is_allowed(status: int):
    async def go():
        async with robots_client({"h": (status, "")}) as client:
            return await RobotsCache(client, UA, policy=FAST).allowed("http://h/anything")

    assert run(go()) is True


@pytest.mark.parametrize("status", [500, 503])
def test_robots_5xx_means_nothing_is_allowed(status: int):
    async def go():
        async with robots_client({"h": (status, "")}) as client:
            return await RobotsCache(client, UA, policy=FAST).allowed("http://h/anything")

    assert run(go()) is False


def test_robots_unreachable_host_is_disallowed():
    def handler(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    async def go():
        async with client_for(handler) as client:
            return await RobotsCache(client, UA, policy=FAST).allowed("http://h/x")

    assert run(go()) is False


# ----------------------------------------------------------------------- HostLimiter
def test_limiter_caps_requests_per_host_and_in_total():
    per_host: Counter[str] = Counter()
    peak_host: Counter[str] = Counter()
    total = 0
    peak_total = 0

    async def one(limiter: HostLimiter, url: str) -> None:
        nonlocal total, peak_total
        async with limiter(url):
            host = url.split("/")[2]
            per_host[host] += 1
            total += 1
            peak_host[host] = max(peak_host[host], per_host[host])
            peak_total = max(peak_total, total)
            await asyncio.sleep(0.01)
            per_host[host] -= 1
            total -= 1

    async def go():
        limiter = HostLimiter(concurrency=3, per_host=2, delay=0.0)
        urls = [f"http://{h}.example/{i}" for h in "abc" for i in range(8)]
        await asyncio.gather(*(one(limiter, u) for u in urls))

    run(go())
    assert max(peak_host.values()) == 2  # per-host semaphore
    assert peak_total == 3  # global semaphore: 3 hosts could do 6, only 3 may


def test_limiter_spaces_request_starts_to_one_host():
    starts: list[float] = []

    async def one(limiter: HostLimiter) -> None:
        async with limiter("http://h.example/x"):
            starts.append(asyncio.get_running_loop().time())

    async def go():
        limiter = HostLimiter(concurrency=10, per_host=10, delay=0.05)
        await asyncio.gather(*(one(limiter) for _ in range(5)))

    run(go())
    gaps = [b - a for a, b in itertools.pairwise(starts)]
    assert all(g >= 0.045 for g in gaps), gaps


def test_limiter_raise_delay_never_lowers_the_delay():
    async def go() -> tuple[float, float]:
        limiter = HostLimiter(2, 2, delay=0.2)
        limiter.raise_delay("h", 0.05)
        first = limiter._state("h").delay  # pyright: ignore[reportPrivateUsage]
        limiter.raise_delay("h", 1.0)
        return first, limiter._state("h").delay  # pyright: ignore[reportPrivateUsage]

    assert run(go()) == (0.2, 1.0)


# ---------------------------------------------------------------------------- crawl()
def page(title: str, *links: str, extra: str = "") -> str:
    anchors = "".join(f'<a href="{href}">x</a>' for href in links)
    return f"<html><head><title>{title}</title></head><body><p>{title} body {extra}</p>{anchors}</body></html>"


class Site:
    """A fake web site served by MockTransport; records every request."""

    def __init__(self, pages: dict[str, Any], robots: str | None = None) -> None:
        self.pages = pages
        self.robots = robots
        self.requests: list[tuple[str, str, float]] = []  # host, path+query, time
        self.in_flight = 0
        self.peak = 0
        self.closed = False
        self.latency = 0.0
        outer = self

        class Transport(httpx.MockTransport):
            async def aclose(self) -> None:
                outer.closed = True

        self.transport = Transport(self.handle)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.raw_path.decode()
        self.requests.append((request.url.host, path, asyncio.get_running_loop().time()))
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            if self.latency:
                await asyncio.sleep(self.latency)
            if path == "/robots.txt":
                if self.robots is None:
                    return httpx.Response(404)
                return httpx.Response(200, text=self.robots)
            entry = self.pages.get(path)
            if callable(entry):
                return entry()  # type: ignore[no-any-return]
            if entry is None:
                return httpx.Response(404, text="nope")
            if isinstance(entry, httpx.Response):
                return entry
            return httpx.Response(200, headers={"content-type": "text/html"}, text=entry)
        finally:
            self.in_flight -= 1

    def paths(self, *, with_robots: bool = False) -> list[str]:
        return [p for _, p, _ in self.requests if with_robots or p != "/robots.txt"]


def collect(site: Site, config: CrawlConfig, stats: CrawlStats | None = None):
    async def go():
        urls = []
        async with contextlib.aclosing(crawl(config, transport=site.transport, stats=stats)) as it:
            async for p in it:
                urls.append(p)
        return urls

    return run(go())


def cfg(**kw: Any) -> CrawlConfig:
    base: dict[str, Any] = {
        "seeds": ("http://site.example/",),
        "user_agent": UA,
        "delay": 0.0,
        "per_host": 5,
        "concurrency": 5,
        "policy": FAST,
        "timeout": 5.0,
    }
    return CrawlConfig(**(base | kw))


def small_site() -> Site:
    return Site(
        {
            "/": page(
                "Home",
                "/a", "/a/", "/a#section",  # three spellings of one page
                "/b?y=2&x=1", "/b?x=1&y=2",  # two spellings of one page
                "/private/p",  # robots.txt says no
                "http://other.example/z",  # another host
                "mailto:me@example.org",
                "/file.pdf", "/missing",
            ),
            "/a": page("Page A", "/c", "/"),
            "/b?x=1&y=2": page("Page B", "/"),
            "/c": page("Page C"),
            "/private/p": page("Secret"),
            "/file.pdf": httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF"),
        },
        robots="User-agent: *\nDisallow: /private/\n",
    )  # fmt: skip


def test_crawl_fetches_each_canonical_url_exactly_once_and_respects_the_rules():
    site = small_site()
    stats = CrawlStats()
    pages = collect(site, cfg(), stats)

    assert sorted(p.url for p in pages) == [
        "http://site.example/",
        "http://site.example/a",
        "http://site.example/b?x=1&y=2",
        "http://site.example/c",
    ]
    assert {p.title for p in pages} == {"Home", "Page A", "Page B", "Page C"}
    fetched = site.paths()
    assert sorted(fetched) == sorted(
        ["/", "/a", "/b?x=1&y=2", "/c", "/file.pdf", "/missing"]
    )  # no duplicates, nothing disallowed, nothing off-site
    assert "/private/p" not in fetched
    assert all(host == "site.example" for host, _, _ in site.requests)
    assert site.paths(with_robots=True).count("/robots.txt") == 1
    assert (stats.pages, stats.errors, stats.robots_skipped, stats.other_skipped) == (4, 1, 1, 1)
    assert stats.queue_depth == 0 and stats.in_flight == 0 and stats.finished is not None
    assert stats.statuses["200"] == 5 and stats.statuses["404"] == 1


def test_crawl_stops_at_max_pages_and_fetches_no_more_than_the_ceiling():
    pages = {"/": page("p0", "/p1", "/p2", "/p3")}
    for i in range(1, 60):
        pages[f"/p{i}"] = page(f"p{i}", f"/p{i + 1}", f"/p{i + 2}", f"/p{i + 3}")
    site = Site(pages)
    site.latency = 0.001
    got = collect(site, cfg(max_pages=7))
    assert len(got) == 7
    assert len(site.paths()) == 7  # not one request more than the ceiling


def test_crawl_ends_when_the_site_is_exhausted_before_the_ceiling():
    got = collect(small_site(), cfg(max_pages=1000))
    assert len(got) == 4


def test_crawl_respects_max_depth():
    pages = {
        "/": page("d0", "/d1"),
        "/d1": page("d1", "/d2"),
        "/d2": page("d2", "/d3"),
        "/d3": page("d3"),
    }
    got = collect(Site(pages), cfg(max_depth=1))
    assert sorted(p.depth for p in got) == [0, 1]


def test_crawl_retries_a_flaky_page_and_counts_the_retries():
    hits = 0

    def flaky() -> httpx.Response:
        nonlocal hits
        hits += 1
        if hits <= 2:
            return httpx.Response(503)
        return httpx.Response(200, headers={"content-type": "text/html"}, text=page("Flaky"))

    site = Site({"/": page("Home", "/flaky"), "/flaky": flaky})
    stats = CrawlStats()
    got = collect(site, cfg(), stats)
    assert {p.title for p in got} == {"Home", "Flaky"}
    assert stats.retries == 2 and stats.errors == 0 and hits == 3
    assert next(p for p in got if p.title == "Flaky").attempts == 3


def test_crawl_keeps_going_after_a_permanent_failure():
    site = Site({"/": page("Home", "/gone", "/ok"), "/ok": page("OK")})
    stats = CrawlStats()
    got = collect(site, cfg(), stats)
    assert {p.title for p in got} == {"Home", "OK"} and stats.errors == 1


def test_a_failed_fetch_gives_its_budget_slot_to_another_url():
    site = Site({"/": page("Home", "/bad", "/g1", "/g2"), "/g1": page("g1"), "/g2": page("g2")})
    got = collect(site, cfg(max_pages=3, concurrency=1, per_host=1))
    assert len(got) == 3  # /bad failed, the ceiling was still reached with good pages


def test_crawl_accepts_plain_text_pages_and_skips_other_types():
    site = Site(
        {
            "/": page("Home", "/notes.txt", "/pic.png"),
            "/notes.txt": httpx.Response(200, headers={"content-type": "text/plain"}, text="hello plain"),
            "/pic.png": httpx.Response(200, headers={"content-type": "image/png"}, content=b"\x89PNG"),
        }
    )  # fmt: skip
    got = collect(site, cfg())
    assert {p.url.rsplit("/", 1)[-1] for p in got} == {"", "notes.txt"}
    assert next(p for p in got if p.url.endswith("notes.txt")).text == "hello plain"


def test_crawl_never_follows_a_redirect_to_another_host():
    site = Site(
        {
            "/": page("Home", "/go"),
            "/go": httpx.Response(302, headers={"location": "http://evil.example/x"}),
        }
    )
    stats = CrawlStats()
    got = collect(site, cfg(), stats)
    assert [p.title for p in got] == ["Home"] and stats.errors == 1
    assert all(host == "site.example" for host, _, _ in site.requests)


def test_crawl_caps_simultaneous_requests():
    n = 40
    site = Site(
        {"/": page("home", *[f"/p{i}" for i in range(n)])} | {f"/p{i}": page("x") for i in range(n)}
    )
    site.latency = 0.01
    collect(site, cfg(concurrency=3, per_host=3, max_pages=n))
    assert 2 <= site.peak <= 3  # really concurrent, never above the cap

    site2 = Site(
        {"/": page("home", *[f"/p{i}" for i in range(10)])}
        | {f"/p{i}": page("x") for i in range(10)}
    )
    site2.latency = 0.005
    collect(site2, cfg(concurrency=8, per_host=1))
    assert site2.peak == 1  # the per-host limit wins over the global one


def test_crawl_spaces_requests_to_the_same_host():
    site = Site(
        {"/": page("home", "/a", "/b", "/c"), "/a": page("a"), "/b": page("b"), "/c": page("c")}
    )
    collect(site, cfg(delay=0.05))
    times = sorted(t for _, _, t in site.requests)
    gaps = [b - a for a, b in itertools.pairwise(times)]
    assert len(times) >= 5 and all(g >= 0.045 for g in gaps), gaps  # robots.txt counts too


@pytest.mark.slow
def test_crawl_obeys_the_crawl_delay_from_robots_txt():
    site = Site(
        {"/": page("home", "/a"), "/a": page("a")}, robots="User-agent: *\nCrawl-delay: 1\n"
    )
    collect(site, cfg(delay=0.0))
    page_times = [t for _, p, t in site.requests if p != "/robots.txt"]
    assert len(page_times) == 2 and page_times[1] - page_times[0] >= 0.95


def test_leaving_the_loop_early_cancels_workers_and_closes_the_client():
    pages = {"/": page("home", *[f"/p{i}" for i in range(30)])} | {
        f"/p{i}": page("x") for i in range(30)
    }
    site = Site(pages)
    site.latency = 0.002

    async def go() -> tuple[int, set[asyncio.Task[Any]]]:
        got = 0
        async with contextlib.aclosing(crawl(cfg(max_pages=30), transport=site.transport)) as it:
            async for _ in it:
                got += 1
                if got == 3:
                    break
        return got, asyncio.all_tasks() - {asyncio.current_task()}

    got, leftovers = run(go())
    assert got == 3 and leftovers == set()  # no worker is left running
    assert site.closed  # the client (and its transport) was closed


def test_the_client_is_closed_after_a_normal_finish_too():
    site = small_site()
    collect(site, cfg())
    assert site.closed


def test_a_bug_in_a_worker_is_not_swallowed():
    def boom() -> httpx.Response:
        raise RuntimeError("bug in the transport")

    site = Site({"/": page("home", "/boom"), "/boom": boom})

    async def go() -> None:
        async with contextlib.aclosing(crawl(cfg(), transport=site.transport)) as it:
            async for _ in it:
                pass

    with pytest.raises(ExceptionGroup) as info:
        run(go())
    assert any(isinstance(e, RuntimeError) for e in info.value.exceptions)
    assert site.closed


def test_config_validation():
    with pytest.raises(ValueError, match="seed"):
        CrawlConfig((), UA)
    with pytest.raises(ValueError, match="max_pages"):
        CrawlConfig(("http://h/",), UA, max_pages=0)
    with pytest.raises(ValueError, match="User-Agent"):
        CrawlConfig(("http://h/",), "  ")
    with pytest.raises(ValueError, match="no valid"):
        collect(Site({}), cfg(seeds=("mailto:x@y.z",)))


# ------------------------------------------------------- the loop must never be blocked
def test_debug_mode_reports_no_blocked_callbacks_for_a_whole_crawl():
    pages = {"/": page("home", *[f"/p{i}" for i in range(30)])}
    pages |= {f"/p{i}": page(f"p{i}", extra="word " * 500) for i in range(30)}
    site = Site(pages)
    site.latency = 0.002
    counter = SlowCallbackCounter()
    logger = logging.getLogger("asyncio")
    logger.addHandler(counter)

    async def main() -> int:
        asyncio.get_running_loop().slow_callback_duration = 0.2
        async with contextlib.aclosing(crawl(cfg(max_pages=31), transport=site.transport)) as it:
            return len([p async for p in it])

    try:
        assert asyncio.run(main(), debug=True) == 31
    finally:
        logger.removeHandler(counter)
    assert counter.count == 0


def test_the_detector_does_catch_a_blocking_call():
    """Negative control: a ``time.sleep`` inside a coroutine IS reported, so the zero above means something."""
    counter = SlowCallbackCounter()
    logger = logging.getLogger("asyncio")
    logger.addHandler(counter)

    async def main() -> None:
        asyncio.get_running_loop().slow_callback_duration = 0.2
        time.sleep(0.3)  # the sin: no await, the whole loop stands still
        await asyncio.sleep(0)

    try:
        asyncio.run(main(), debug=True)
    finally:
        logger.removeHandler(counter)
    assert counter.count >= 1
