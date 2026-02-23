from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth.dependencies import require_scopes
from app.db.session import run_with_db_retry
from app.feature_store.service import (
    get_feature_schema_by_version,
    list_feature_schemas,
)
from app.schemas.feature_schemas import (
    FeatureSchemaResponse,
    FeatureSchemasResponse,
    FeatureSchemaFeatureResponse,
)
from app.tenants.context import TenantContext, get_tenant_context


router = APIRouter(
    tags=["feature-schemas"], dependencies=[Depends(require_scopes("read"))]
)


@router.get("/feature-schemas", response_model=FeatureSchemasResponse)
def list_schemas(
    tenant: TenantContext = Depends(get_tenant_context),
) -> FeatureSchemasResponse:
    schemas = run_with_db_retry(
        lambda session: list_feature_schemas(session),
        operation_name="feature_schemas_list",
    )
    return FeatureSchemasResponse(schemas=[_serialize(schema) for schema in schemas])


@router.get("/feature-schemas/{version}", response_model=FeatureSchemaResponse)
def get_schema(
    version: str,
    tenant: TenantContext = Depends(get_tenant_context),
) -> FeatureSchemaResponse:
    schema = run_with_db_retry(
        lambda session: get_feature_schema_by_version(session, version),
        operation_name="feature_schema_get",
    )
    if schema is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Feature schema not found"
        )
    return _serialize(schema)


def _serialize(schema) -> FeatureSchemaResponse:
    features = [
        FeatureSchemaFeatureResponse(
            name=feature.name,
            dtype=feature.dtype,
            allowed_range=feature.allowed_range,
            description=feature.description,
            introduced_in_version=feature.introduced_in_version,
            deprecated_in_version=feature.deprecated_in_version,
            position=feature.position,
        )
        for feature in schema.features
    ]
    return FeatureSchemaResponse(
        id=schema.id,
        version=schema.version,
        hash=schema.hash,
        description=schema.description,
        is_active=schema.is_active,
        created_at=schema.created_at,
        features=features,
    )
