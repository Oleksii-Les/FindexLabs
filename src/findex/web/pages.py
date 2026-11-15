"""HTML routes: ``/`` (search + results) and ``/view/{doc_id}`` (read a document).

With htmx the page asks for ``/`` again on submit / paging and swaps only the results block
(``HX-Request`` header -> ``_results.html``); without JavaScript the same links and the same form
simply load the whole page. ``Vary: HX-Request`` keeps caches from mixing the two.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, Response

from findex.query import PositionsRequired, QuerySyntaxError
from findex.web.deps import AppSettings, MaybeService
from findex.web.schemas import PageParams, ViewParams
from findex.web.service import DocumentNotFound, DocumentUnavailable
from findex.web.templating import render

router = APIRouter(include_in_schema=False)


@router.get("/", response_class=HTMLResponse)
def home(
    request: Request,
    params: Annotated[PageParams, Query()],
    service: MaybeService,
    settings: AppSettings,
) -> Response:
    if service is None:
        return render(request, "loading.html", {"site": settings.site_name}, 503)
    result = None
    error: str | None = None
    if params.q:
        try:
            result = service.search(params.q, k=params.k, scorer=params.scorer, page=params.page)
        except (QuerySyntaxError, PositionsRequired) as exc:
            error = str(exc)  # a human typed this: show the reason on the page, status 200
    partial = request.headers.get("hx-request") == "true"
    context = {
        "site": settings.site_name,
        "stats": service.stats(),
        "examples": [q for q in settings.examples if service.index.has_positions or '"' not in q],
        "q": params.q, "k": params.k, "scorer": params.scorer, "page": params.page,
        "result": result, "error": error,
    }  # fmt: skip
    response = render(request, "_results.html" if partial else "index.html", context)
    response.headers["Vary"] = "HX-Request"
    return response


@router.get("/view/{doc_id}", response_class=HTMLResponse)
def view(
    request: Request,
    doc_id: Annotated[int, Path(ge=0)],
    params: Annotated[ViewParams, Query()],
    service: MaybeService,
    settings: AppSettings,
) -> Response:
    if service is None:
        return render(request, "loading.html", {"site": settings.site_name}, 503)
    query: str | None = params.q or None
    try:
        doc = service.document(doc_id, offset=params.offset, limit=params.limit, query=query)
    except (QuerySyntaxError, PositionsRequired):
        doc = service.document(doc_id, offset=params.offset, limit=params.limit)  # no highlighting
    except (DocumentNotFound, DocumentUnavailable):
        raise HTTPException(status_code=404, detail=f"document {doc_id} not found") from None
    return render(
        request, "doc.html",
        {"site": settings.site_name, "doc": doc, "q": params.q, "limit": params.limit}, 
    )  # fmt: skip


@router.get("/robots.txt", response_class=PlainTextResponse)
def robots() -> str:
    return "User-agent: *\nAllow: /\nDisallow: /api/\nDisallow: /docs/\n"
