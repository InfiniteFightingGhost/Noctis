from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.db.models import AuthUser


def get_auth_user_by_email(session: Session, email: str) -> AuthUser | None:
    return session.query(AuthUser).filter(AuthUser.email == email).first()


def get_auth_user_by_id(session: Session, user_id: uuid.UUID) -> AuthUser | None:
    return session.get(AuthUser, user_id)


def create_auth_user(session: Session, email: str, password_hash: str) -> AuthUser:
    now = datetime.now(timezone.utc)
    user = AuthUser(
        email=email,
        password_hash=password_hash,
        created_at=now,
        updated_at=now,
    )
    session.add(user)
    session.flush()
    session.refresh(user)
    return user
