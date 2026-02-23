from __future__ import annotations

from fastapi import APIRouter, Depends

from app.auth.dependencies import require_admin
from app.db.models import User
from app.db.session import run_with_db_retry
from app.schemas.users import UserCreate, UserResponse
from app.tenants.context import TenantContext, get_tenant_context


router = APIRouter(tags=["users"], dependencies=[Depends(require_admin)])


@router.post("/users", response_model=UserResponse)
def create_user(
    payload: UserCreate,
    tenant: TenantContext = Depends(get_tenant_context),
) -> UserResponse:
    def _op(session):
        user = User(
            tenant_id=tenant.id,
            name=payload.name,
            external_id=payload.external_id,
        )
        session.add(user)
        session.flush()
        session.refresh(user)
        return user

    user = run_with_db_retry(_op, commit=True, operation_name="create_user")
    return UserResponse.model_validate(user)
