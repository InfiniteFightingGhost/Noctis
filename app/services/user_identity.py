from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.auth.context import AuthContext
from app.db.models import User


def auth_external_id(auth: AuthContext) -> str:
    return f"auth:{auth.client_id}"


def get_domain_user_for_auth(
    session: Session,
    *,
    tenant_id: uuid.UUID,
    auth: AuthContext,
) -> User | None:
    if auth.principal_type != "user":
        return None
    return (
        session.query(User)
        .filter(User.tenant_id == tenant_id)
        .filter(User.external_id == auth_external_id(auth))
        .one_or_none()
    )


def resolve_or_create_domain_user_for_auth(
    session: Session,
    *,
    tenant_id: uuid.UUID,
    auth: AuthContext,
) -> User:
    user = get_domain_user_for_auth(session, tenant_id=tenant_id, auth=auth)
    if user:
        return user
    if auth.principal_type != "user":
        raise ValueError("User principal required")
    user = User(
        tenant_id=tenant_id,
        name=auth.client_name,
        external_id=auth_external_id(auth),
    )
    session.add(user)
    session.flush()
    session.refresh(user)
    return user
