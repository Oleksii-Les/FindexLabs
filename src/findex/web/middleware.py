"""A pure-ASGI middleware: request id, one access-log line, security headers.

Written against the raw ASGI interface (``scope``, ``receive``, ``send``) instead of
``BaseHTTPMiddleware``: no extra task and no body streaming layer per request, and context
variables set here stay visible in the route (``BaseHTTPMiddleware`` is known to lose them).
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import MutableMapping
from typing import Any

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from findex.web.logs import request_id_var

log = logging.getLogger("findex.web.access")

_VALID_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
# Swagger UI loads its JS from a CDN, so the strict policy is not applied to /api/docs.
_CSP = (
    "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
    "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)
_QUIET = {"/health"}  # platform probes every few seconds: DEBUG, not INFO


def pick_request_id(scope: Scope) -> str:
    """Reuse the caller's ``X-Request-ID`` if it is sane (so ids follow a request across
    services), otherwise make one. Never trust it blindly: it ends up in logs and headers."""
    for key, value in scope["headers"]:
        if key == b"x-request-id":
            candidate = value.decode("latin-1")
            if _VALID_ID.match(candidate):
                return candidate
    return uuid.uuid4().hex[:16]


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        rid = pick_request_id(scope)
        scope["request_id"] = rid  # the 500 handler runs outside this middleware: it reads it here
        token = request_id_var.set(rid)
        started = time.perf_counter()
        status = 500  # what the client gets if the app raises before sending anything
        path: str = scope["path"]

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = rid
                headers["X-Content-Type-Options"] = "nosniff"
                headers["Referrer-Policy"] = "same-origin"
                if not path.startswith("/api/"):
                    headers["Content-Security-Policy"] = _CSP
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            ms = (time.perf_counter() - started) * 1000
            level = logging.DEBUG if path in _QUIET and status < 400 else logging.INFO
            query = scope.get("query_string", b"").decode("latin-1")
            log.log(
                level, "%s %s%s -> %d in %.1f ms",
                scope["method"], path, f"?{query}" if query else "", status, ms,
            )  # fmt: skip
            request_id_var.reset(token)


def request_id_of(scope: MutableMapping[str, Any]) -> str:
    return str(scope.get("request_id", "-"))
