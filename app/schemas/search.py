from __future__ import annotations

from app.schemas.common import BaseSchema


class SearchResult(BaseSchema):
    type: str
    id: str
    title: str
    subtitle: str | None = None


class SearchResponse(BaseSchema):
    query: str
    results: list[SearchResult]
