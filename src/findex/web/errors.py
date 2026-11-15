"""Error handling: validation (422), HTTP errors (404, 503, ...) and the catch-all 500.

The rule for a 500: **the client gets no internals**. The body is a fixed message plus the
request id; the exception with its stack goes to the log, tagged with the same id, so
"it broke, request id 3f2a..." in a bug report leads straight to the stack trace.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from findex.query import PositionsRequired, QuerySyntaxError
from findex.web.middleware import request_id_of
from findex.web.templating import render

log = logging.getLogger(__name__)


def is_page(request: Request) -> bool:
    """Routes that answer with HTML (and so get an HTML error page, not JSON)."""
    path = request.url.path
    return path == "/" or path.startswith("/view/")


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(QuerySyntaxError)
    @app.exception_handler(PositionsRequired)
    async def bad_query(request: Request, exc: Exception) -> Response:
        # Same shape as FastAPI's own 422, so a client handles one format.
        kind = "positions_required" if isinstance(exc, PositionsRequired) else "query_syntax"
        body = {"detail": [{"type": kind, "loc": ["query", "q"], "msg": str(exc)}],
                "request_id": request_id_of(request.scope)}  # fmt: skip
        return JSONResponse(body, status_code=422)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> Response:
        if is_page(request):
            msgs = [f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in exc.errors()]
            return render(
                request, "error.html", {"title": "Invalid request", "messages": msgs}, 422
            )
        return await request_validation_exception_handler(request, exc)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> Response:
        rid = request_id_of(request.scope)
        headers = dict(exc.headers or {})
        if is_page(request):
            title = {404: "Not found", 503: "Starting up"}.get(exc.status_code, "Error")
            resp = render(
                request,
                "error.html",
                {"title": title, "messages": [str(exc.detail)]},
                exc.status_code,
            )
            for k, v in headers.items():
                resp.headers[k] = v
            return resp
        return JSONResponse(
            {"detail": exc.detail, "request_id": rid}, status_code=exc.status_code, headers=headers
        )

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> Response:
        rid = request_id_of(request.scope)
        # The stack goes to the log (with the id); the client sees nothing of it.
        log.error(
            "unhandled error request_id=%s on %s %s", rid, request.method, request.url.path,
            exc_info=exc,
        )  # fmt: skip
        headers = {
            "X-Request-ID": rid
        }  # the middleware cannot add it: this handler runs outside it
        if is_page(request):
            resp = render(request, "error.html", {"title": "Something went wrong",
                          "messages": ["Our side. Quote this id if you report it:"], "request_id": rid}, 500)  # fmt: skip
            resp.headers["X-Request-ID"] = rid
            return resp
        return JSONResponse(
            {"detail": "Internal Server Error", "request_id": rid}, status_code=500, headers=headers
        )
