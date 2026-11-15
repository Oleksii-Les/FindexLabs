"""Jinja2 environment shared by the HTML routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape

from findex.web import __version__
from findex.web.service import MARK_OFF, MARK_ON

PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))  # autoescape is ON for .html


def highlight_filter(text: str) -> Markup:
    """Escape FIRST, then turn the service's private marks into <mark>: text from the corpus
    can contain anything (``<script>``), the marks are the only HTML we create."""
    return Markup(str(escape(text)).replace(MARK_ON, "<mark>").replace(MARK_OFF, "</mark>"))


def number_filter(value: int | float) -> str:
    return f"{value:,.0f}".replace(",", " ")  # narrow no-break space: "1 234 567"


templates.env.filters["highlight"] = highlight_filter
templates.env.filters["number"] = number_filter


def render(
    request: Request, name: str, context: dict[str, Any], status_code: int = 200
) -> HTMLResponse:
    return templates.TemplateResponse(
        request, name, {"version": __version__, **context}, status_code=status_code
    )
