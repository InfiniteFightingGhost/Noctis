from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.auth.dependencies import require_scopes
from app.db.session import run_with_db_retry
from app.experiments.service import get_model, list_experiments, list_models
from app.schemas.experiments import ExperimentResponse, ExperimentsResponse
from app.schemas.models import ModelVersionResponse, ModelVersionsResponse
from app.tenants.context import TenantContext, get_tenant_context


router = APIRouter(tags=["experiments", "models"], dependencies=[Depends(require_scopes("read"))])


@router.get("/experiments", response_model=ExperimentsResponse)
def experiments(
    tenant: TenantContext = Depends(get_tenant_context),
) -> ExperimentsResponse:
    def _op(session):
        return list_experiments(session, tenant_id=tenant.id)

    payload = run_with_db_retry(_op, operation_name="list_experiments")
    return ExperimentsResponse(experiments=[ExperimentResponse(**exp) for exp in payload])


@router.get("/models", response_model=ModelVersionsResponse)
def models(
    tenant: TenantContext = Depends(get_tenant_context),
) -> ModelVersionsResponse:
    payload = run_with_db_retry(list_models, operation_name="list_models")
    return ModelVersionsResponse(models=[ModelVersionResponse(**model) for model in payload])


@router.get("/models/{version}", response_model=ModelVersionResponse)
def model_detail(
    version: str,
    tenant: TenantContext = Depends(get_tenant_context),
) -> ModelVersionResponse:
    def _op(session):
        return get_model(session, version)

    payload = run_with_db_retry(_op, operation_name="get_model")
    if payload is None:
        raise HTTPException(status_code=404, detail="Model version not found")
    return ModelVersionResponse(**payload)
