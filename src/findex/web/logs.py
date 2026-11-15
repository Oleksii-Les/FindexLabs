"""Logging for the service: one line per event on stderr, each tagged with the request id.

The id lives in a ``ContextVar`` set by the middleware; a logging filter copies it onto every
record, so a line logged deep inside the search code (even in a worker thread: FastAPI copies the
context into it) can be matched with the access line and with the ``X-Request-ID`` the client saw.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def configure_service_logging(level: str = "INFO") -> None:
    """Replace the root handler with a stderr handler that prints the request id.
    Platforms (Render, Fly.io, Railway) collect stderr/stdout into their log panel."""
    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s")
    )
    logging.basicConfig(level=level, handlers=[handler], force=True)
    logging.getLogger("uvicorn.access").disabled = True  # the middleware logs requests itself
