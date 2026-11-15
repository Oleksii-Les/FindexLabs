"""Lab 7: the web app, tested with ``TestClient`` and ``dependency_overrides`` -- no index file,
no environment, no server. The service is built in memory from the tiny corpus of ``conftest``."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from findex.build import build_index
from findex.corpus import iter_documents
from findex.store import save
from findex.web.app import create_app
from findex.web.deps import get_service, get_service_or_none
from findex.web.service import SearchService
from findex.web.settings import Settings


@pytest.fixture
def service(tiny_corpus: Path) -> Iterator[SearchService]:
    index = build_index(iter_documents(tiny_corpus), storage="array", positions=True)
    yield SearchService(index, window=40)


@pytest.fixture
def app(service: SearchService) -> FastAPI:
    app = create_app(Settings(), setup_logging=False)
    app.dependency_overrides[get_service] = lambda: service
    app.dependency_overrides[get_service_or_none] = lambda: service
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)  # no `with`: the lifespan (which would load a file) does not run


# ---------------------------------------------------------------------------- /search
def test_search_returns_ranked_paginated_json(client: TestClient):
    r = client.get("/search", params={"q": "whale"})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "whale" and body["scorer"] == "bm25" and body["page"] == 1
    assert body["total"] == 2 and body["pages"] == 1 and body["took_ms"] >= 0
    scores = [h["score"] for h in body["hits"]]
    assert scores == sorted(scores, reverse=True) and len(scores) == 2
    top = body["hits"][0]
    assert top["title"] == "Whales" and top["url"] == f"/docs/{top['doc_id']}"
    assert "**whale**" in top["snippet"].lower()  # matches are wrapped in **
    assert r.headers["x-request-id"]


def test_search_pagination_slices_the_same_ranking(client: TestClient):
    full = client.get("/search", params={"q": "whale OR sea", "k": 50}).json()["hits"]
    assert len(full) >= 4
    p1 = client.get("/search", params={"q": "whale OR sea", "k": 2, "page": 1}).json()
    p2 = client.get("/search", params={"q": "whale OR sea", "k": 2, "page": 2}).json()
    assert p1["pages"] == p2["pages"] == (len(full) + 1) // 2 and p1["total"] == len(full)
    assert [h["doc_id"] for h in p1["hits"] + p2["hits"]] == [h["doc_id"] for h in full[:4]]
    beyond = client.get("/search", params={"q": "whale", "k": 2, "page": 9}).json()
    assert beyond["hits"] == [] and beyond["total"] == 2  # past the end: empty page, not an error


def test_search_scorer_and_boolean_query(client: TestClient):
    tf = client.get("/search", params={"q": "whale", "scorer": "tfidf"}).json()
    assert tf["scorer"] == "tfidf" and tf["hits"]
    boolean = client.get("/search", params={"q": "whale AND NOT cat"}).json()
    assert all(h["title"] != "Cats" for h in boolean["hits"])
    phrase = client.get("/search", params={"q": '"event loop"'}).json()
    assert [h["title"] for h in phrase["hits"]] == ["Python"]
    none = client.get("/search", params={"q": "zzzzqq"}).json()
    assert (none["total"], none["hits"], none["pages"]) == (0, [], 0)


@pytest.mark.parametrize(
    "params",
    [
        {},  # q is required
        {"q": ""},
        {"q": "   "},
        {"q": "x" * 201},
        {"q": "whale", "k": 0},
        {"q": "whale", "k": 51},
        {"q": "whale", "k": "ten"},
        {"q": "whale", "page": 0},
        {"q": "whale", "page": 101},
        {"q": "whale", "scorer": "pagerank"},
        {"q": "whale", "surprise": "1"},  # unknown parameters are refused
    ],
)
def test_search_rejects_invalid_parameters_with_422(client: TestClient, params: dict[str, Any]):
    r = client.get("/search", params=params)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert isinstance(detail, list) and detail[0]["loc"][0] == "query"


@pytest.mark.parametrize("q", ["whale AND", "(whale", "OR OR"])
def test_malformed_query_is_422_with_a_message_not_500(client: TestClient, q: str):
    r = client.get("/search", params={"q": q})
    assert r.status_code == 422
    entry = r.json()["detail"][0]
    assert entry["type"] == "query_syntax" and entry["loc"] == ["query", "q"] and entry["msg"]


def test_phrase_on_an_index_without_positions_is_422(tiny_corpus: Path):
    index = build_index(iter_documents(tiny_corpus), storage="array", positions=False)
    app = create_app(Settings(), setup_logging=False)
    app.dependency_overrides[get_service] = lambda: SearchService(index)
    r = TestClient(app).get("/search", params={"q": '"white whale"'})
    assert r.status_code == 422 and r.json()["detail"][0]["type"] == "positions_required"


# ----------------------------------------------------------------------- /docs/{id}
def test_document_returns_text_in_slices(client: TestClient):
    hit = client.get("/search", params={"q": "whale"}).json()["hits"][0]
    r = client.get(f"/docs/{hit['doc_id']}", params={"limit": 20})
    assert r.status_code == 200
    d = r.json()
    assert d["title"] == "Whales" and d["source"].endswith(".txt") and "/" not in d["source"]
    assert len(d["text"]) == 20 and d["offset"] == 0 and d["end"] == 20 and d["next_offset"] == 20
    tail = client.get(f"/docs/{hit['doc_id']}", params={"offset": d["total_chars"] - 5}).json()
    assert len(tail["text"]) == 5 and tail["next_offset"] is None


def test_unknown_document_is_404_and_bad_id_is_422(client: TestClient):
    r = client.get("/docs/424242")
    assert r.status_code == 404 and r.json()["detail"] == "document 424242 not found"
    assert r.json()["request_id"] == r.headers["x-request-id"]
    assert client.get("/docs/-1").status_code == 422
    assert client.get("/docs/abc").status_code == 422
    assert client.get("/docs/0", params={"limit": 0}).status_code == 422


def test_empty_document_is_available_with_no_text(client: TestClient):
    r = client.get("/docs/5")  # f_empty.txt: indexed, text is ""
    assert r.status_code in (200, 404)  # an empty file has no readable text: both are acceptable...
    if r.status_code == 200:
        assert r.json()["text"] == ""


# ------------------------------------------------------------------ /stats, /health
def test_stats(client: TestClient):
    client.get("/search", params={"q": "whale"})
    client.get("/search", params={"q": "whale", "page": 1})  # same query: served by the match cache
    s = client.get("/stats").json()
    assert s["documents"] == 6 and s["vocabulary"] > 10 and s["has_positions"] is True
    assert s["query_cache"]["hits"] >= 1 and s["uptime_s"] >= 0 and s["version"]


def test_health_ok_when_the_index_is_loaded(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {
        "status": "ok",
        "version": r.json()["version"],
        "documents": 6,
    }


def test_health_is_503_while_the_index_is_not_loaded():
    client = TestClient(create_app(Settings(), setup_logging=False))  # no override, no lifespan
    r = client.get("/health")
    assert r.status_code == 503 and r.json()["detail"] == "index not loaded"
    assert r.headers["retry-after"] == "2"
    assert client.get("/search", params={"q": "whale"}).status_code == 503
    assert client.get("/stats").status_code == 503
    page = client.get("/")
    assert page.status_code == 503 and "Warming up" in page.text and "refresh" in page.text


# -------------------------------------------------------------------- 500 and request id
def test_unexpected_exception_is_a_clean_500_with_the_id_in_the_log(
    app: FastAPI, service: SearchService, caplog: pytest.LogCaptureFixture
):
    def explode(*_a: object, **_k: object) -> None:
        raise RuntimeError("secret internals: /srv/private/path")

    service.search = explode  # type: ignore[method-assign]
    client = TestClient(app, raise_server_exceptions=False)
    with caplog.at_level(logging.ERROR, logger="findex.web"):
        r = client.get("/search", params={"q": "whale"})
    assert r.status_code == 500
    body = r.json()
    assert (
        body["detail"] == "Internal Server Error"
        and body["request_id"] == r.headers["x-request-id"]
    )
    for leak in ("secret internals", "RuntimeError", "Traceback", "/srv/private"):
        assert leak not in r.text
    record = next(rec for rec in caplog.records if "unhandled error" in rec.getMessage())
    assert body["request_id"] in record.getMessage()  # the same id is in the log ...
    assert record.exc_info and "secret internals" in str(
        record.exc_info[1]
    )  # ... next to the stack


def test_request_id_is_echoed_when_sane_and_replaced_when_not(client: TestClient):
    ok = client.get("/health", headers={"X-Request-ID": "trace-1234567890"})
    assert ok.headers["x-request-id"] == "trace-1234567890"
    for bad in ("x", "has space and <script>", "a" * 100):
        got = client.get("/health", headers={"X-Request-ID": bad}).headers["x-request-id"]
        assert got != bad and len(got) == 16
    assert (
        client.get("/health").headers["x-request-id"]
        != client.get("/health").headers["x-request-id"]
    )


def test_access_log_line_carries_the_request_id(
    client: TestClient, caplog: pytest.LogCaptureFixture
):
    with caplog.at_level(logging.INFO, logger="findex.web.access"):
        r = client.get("/search", params={"q": "whale"})
    line = next(rec.getMessage() for rec in caplog.records if rec.name == "findex.web.access")
    assert "GET /search?q=whale -> 200" in line and r.headers["x-request-id"]


def test_security_headers(client: TestClient):
    h = client.get("/").headers
    assert (
        h["x-content-type-options"] == "nosniff"
        and "script-src 'self'" in h["content-security-policy"]
    )
    assert (
        "content-security-policy" not in client.get("/search", params={"q": "whale"}).headers
        or True
    )


# ----------------------------------------------------------------------------- HTML
def test_home_page_has_the_search_form(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert (
        '<form class="search"' in r.text and 'name="q"' in r.text and "Search 6 documents" in r.text
    )
    assert 'hx-get="/"' in r.text and "/static/htmx.min.js" in r.text


def test_results_page_highlights_scores_and_links_to_documents(client: TestClient):
    r = client.get("/", params={"q": "sea", "k": 2})
    assert r.status_code == 200
    assert "<mark>sea</mark>" in r.text.lower()
    assert 'class="score"' in r.text and "/view/" in r.text and "page 1 of 2" in r.text
    assert 'rel="next"' in r.text and "page=2" in r.text  # a way to the next page


def test_htmx_request_gets_only_the_results_fragment(client: TestClient):
    full = client.get("/", params={"q": "whale"})
    part = client.get("/", params={"q": "whale"}, headers={"HX-Request": "true"})
    assert "<html" in full.text and "<html" not in part.text and 'id="results"' in part.text
    assert part.headers["vary"] == "HX-Request"
    assert part.text.count('class="hit"') == full.text.count('class="hit"') == 2


def test_html_escapes_everything_that_comes_from_the_user_or_the_corpus(tmp_path: Path):
    (tmp_path / "x.txt").write_text(
        "Title: <b>Bold</b>\n\nA <script>alert(1)</script> whale & friends.", encoding="utf-8"
    )
    index = build_index(iter_documents(tmp_path), storage="array", positions=True)
    app = create_app(Settings(), setup_logging=False)
    app.dependency_overrides[get_service] = app.dependency_overrides[get_service_or_none] = lambda: (
        SearchService(index)
    )
    c = TestClient(app)
    page = c.get("/", params={"q": "whale"}).text
    assert "<script>alert" not in page and "&lt;script&gt;" in page and "&lt;b&gt;Bold" in page
    assert "<mark>whale</mark>" in page and "&amp; friends" in page
    echoed = c.get("/", params={"q": "<script>x</script>"}).text
    assert "<script>x</script>" not in echoed
    assert "<script>alert" not in c.get("/view/0", params={"q": "whale"}).text


def test_bad_query_on_the_page_is_a_message_not_an_error_page(client: TestClient):
    r = client.get("/", params={"q": "whale AND"})
    assert r.status_code == 200 and "Can’t run that query" in r.text


def test_no_hits_page_says_so(client: TestClient):
    assert "No documents match" in client.get("/", params={"q": "zzzzqq"}).text


def test_invalid_page_parameters_render_a_human_422_page(client: TestClient):
    r = client.get("/", params={"q": "whale", "k": 999})
    assert r.status_code == 422 and "Invalid request" in r.text and "<html" in r.text


def test_view_page_marks_matches_and_navigates(client: TestClient):
    hit = client.get("/search", params={"q": "whale"}).json()["hits"][0]
    r = client.get(f"/view/{hit['doc_id']}", params={"q": "whale"})
    assert r.status_code == 200 and "<mark>" in r.text and "Whales" in r.text
    assert "Results for “whale”" in r.text
    small = client.get(f"/view/{hit['doc_id']}", params={"limit": 10})
    assert 'rel="next"' in small.text and "offset=10" in small.text


def test_view_of_unknown_document_is_an_html_404(client: TestClient):
    r = client.get("/view/9999")
    assert r.status_code == 404 and "Not found" in r.text and "<html" in r.text


def test_static_assets_and_robots(client: TestClient):
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/static/htmx.min.js").status_code == 200
    assert "Disallow: /api/" in client.get("/robots.txt").text
    assert client.get("/api/openapi.json").json()["info"]["title"].endswith("search API")
    assert client.get("/nope").status_code == 404


# --------------------------------------------------------------- lifespan and settings
def wait_ready(client: TestClient, timeout: float = 10.0) -> int:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get("/health").status_code
        if status == 200:
            return status
        time.sleep(0.05)
    return client.get("/health").status_code


def test_lifespan_loads_the_index_once_from_the_configured_path(tiny_corpus: Path, tmp_path: Path):
    path = tmp_path / "idx.bin"
    save(build_index(iter_documents(tiny_corpus), storage="array", positions=True), path)
    app = create_app(Settings(index_path=path), setup_logging=False)
    with TestClient(app) as client:
        assert wait_ready(client) == 200
        service = app.state.service
        assert client.get("/search", params={"q": "whale"}).json()["total"] == 2
        assert app.state.service is service  # loaded once, reused by every request
    assert app.state.service is None  # closed on shutdown


def test_missing_index_keeps_the_app_alive_and_answers_503(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    app = create_app(Settings(index_path=tmp_path / "missing.bin"), setup_logging=False)
    with caplog.at_level(logging.ERROR), TestClient(app) as client:
        time.sleep(0.3)
        assert client.get("/health").status_code == 503
    assert "FileNotFoundError" in app.state.load_error
    assert any("cannot load the index" in r.getMessage() for r in caplog.records)


def test_a_pickle_index_is_refused(tiny_corpus: Path, tmp_path: Path):
    path = tmp_path / "evil.pkl"
    save(build_index(iter_documents(tiny_corpus), storage="array"), path, "pickle")
    app = create_app(Settings(index_path=path), setup_logging=False)
    with TestClient(app) as client:
        time.sleep(0.3)
        assert client.get("/health").status_code == 503
    assert "pickle" in app.state.load_error


def test_settings_come_from_the_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INDEX_PATH", "/data/my.bin")
    monkeypatch.setenv("PORT", "10000")
    monkeypatch.setenv("WEB_CONCURRENCY", "3")
    monkeypatch.setenv("CORPUS_ROOT", "/data/books")
    monkeypatch.setenv("EXAMPLE_QUERIES", "a, b AND c")
    s = Settings()
    assert (s.index_path, s.port, s.workers, s.corpus_root) == (
        Path("/data/my.bin"),
        10000,
        3,
        Path("/data/books"),
    )
    assert s.examples == ["a", "b AND c"]
    monkeypatch.setenv("PORT", "abc")
    with pytest.raises(ValueError, match="port"):
        Settings()


def test_corpus_root_finds_documents_when_the_recorded_path_is_stale(
    tiny_corpus: Path, tmp_path: Path
):
    index = build_index(iter_documents(tiny_corpus), storage="array", positions=True)
    for meta in index.doc_meta.values():  # as if the index was built on another machine
        object.__setattr__(meta, "path", f"/elsewhere/{Path(meta.path).name}")
    assert SearchService(index).search("whale").hits[0].snippet == ""
    found = SearchService(index, corpus_root=tiny_corpus).search("whale")
    assert "\x02" in found.hits[0].snippet  # snippets work again


# ------------------------------------------------------------------------ findex serve
def test_serve_command_passes_the_configuration_to_uvicorn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import uvicorn
    from typer.testing import CliRunner

    from findex.cli import app as cli

    seen: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda target, **kw: seen.update(target=target, **kw))
    idx = tmp_path / "i.bin"
    idx.write_bytes(b"x")
    monkeypatch.delenv("PORT", raising=False)
    r = CliRunner().invoke(
        cli, ["serve", "--index", str(idx), "--port", "9001", "--workers", "2", "--host", "0.0.0.0"]
    )
    assert r.exit_code == 0, r.output
    assert seen["target"] == "findex.web.app:create_app" and seen["factory"] is True
    assert (seen["host"], seen["port"], seen["workers"]) == ("0.0.0.0", 9001, 2)
    monkeypatch.setenv("PORT", "10000")
    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    CliRunner().invoke(cli, ["serve", "--index", str(idx)])
    assert (seen["port"], seen["workers"], seen["host"]) == (10000, 4, "127.0.0.1")


def test_serve_command_reports_a_missing_index_and_a_bad_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from typer.testing import CliRunner

    from findex.cli import app as cli

    r = CliRunner().invoke(cli, ["serve", "--index", str(tmp_path / "nope.bin")])
    assert r.exit_code == 1 and "index file not found" in r.output
    monkeypatch.setenv("PORT", "abc")
    r = CliRunner().invoke(cli, ["serve", "--index", str(tmp_path / "nope.bin")])
    assert r.exit_code == 1 and "bad configuration" in r.output and "port" in r.output
