"""The web face of findex: a FastAPI application (``findex.web.app:create_app``).

* ``settings``   -- configuration from the environment (Pydantic ``BaseSettings``);
* ``service``    -- the search logic over an open ``Index`` (plain dataclasses, no web types);
* ``schemas``    -- Pydantic models, used ONLY on the way in (query parameters) and on the way out (JSON);
* ``deps``       -- ``Depends`` providers (settings, service) -- tests replace them with ``dependency_overrides``;
* ``api``        -- JSON routes; ``pages`` -- HTML routes (Jinja2 + htmx);
* ``middleware`` -- request id, access log, security headers; ``errors`` -- 422 / 500 handlers.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("findex")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0+dev"
