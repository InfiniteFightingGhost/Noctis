from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.core.metrics import MODEL_RELOAD_COUNT
from app.core.security import require_admin_key
from app.schemas.models import ModelReloadResponse


router = APIRouter(tags=["models"], dependencies=[Depends(require_admin_key)])


@router.post("/models/reload", response_model=ModelReloadResponse)
def reload_model(request: Request) -> ModelReloadResponse:
    registry = request.app.state.model_registry
    loaded = registry.reload()
    MODEL_RELOAD_COUNT.inc()
    return ModelReloadResponse(model_version=loaded.version, reloaded=True)
