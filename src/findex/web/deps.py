"""``Depends`` providers. A route asks for ``Service`` / ``AppSettings``; it neither knows nor
cares where they come from. Tests replace these functions through
``app.dependency_overrides[get_service] = lambda: fake`` -- no index file, no environment."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request

from findex.web.service import SearchService
from findex.web.settings import Settings


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_service_or_none(request: Request) -> SearchService | None:
    """The loaded service, or None while the index is still loading (or failed to load)."""
    service: SearchService | None = request.app.state.service
    return service


def get_service(
    service: Annotated[SearchService | None, Depends(get_service_or_none)],
) -> SearchService:
    if service is None:
        raise HTTPException(
            status_code=503, detail="the index is not loaded yet", headers={"Retry-After": "2"}
        )
    return service


AppSettings = Annotated[Settings, Depends(get_settings)]
Service = Annotated[SearchService, Depends(get_service)]
MaybeService = Annotated[SearchService | None, Depends(get_service_or_none)]
