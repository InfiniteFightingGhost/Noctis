from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.core.settings import get_settings


def require_api_key(
    api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    settings = get_settings()
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )


def require_admin_key(
    admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
) -> None:
    settings = get_settings()
    if admin_key != settings.admin_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin key"
        )
