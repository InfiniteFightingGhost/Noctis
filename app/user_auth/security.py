from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Request
from jwt import InvalidTokenError

from app.core.settings import get_settings
from app.user_auth.errors import (
    InvalidUserSchemeError,
    InvalidUserTokenError,
    MissingUserTokenError,
)


@dataclass(frozen=True)
class UserTokenClaims:
    subject: uuid.UUID
    email: str


def hash_password(password: str) -> str:
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def issue_access_token(*, user_id: uuid.UUID, email: str) -> tuple[str, int]:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    exp = now + timedelta(seconds=settings.auth_access_token_ttl_seconds)
    payload = {
        "sub": str(user_id),
        "email": email,
        "type": "access",
        "iss": settings.auth_jwt_issuer,
        "aud": settings.auth_jwt_audience,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }
    token = jwt.encode(
        payload,
        settings.auth_jwt_secret,
        algorithm=settings.auth_jwt_algorithm,
    )
    return token, settings.auth_access_token_ttl_seconds


def verify_access_token(token: str) -> UserTokenClaims:
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.auth_jwt_secret,
            algorithms=[settings.auth_jwt_algorithm],
            audience=settings.auth_jwt_audience,
            issuer=settings.auth_jwt_issuer,
        )
    except InvalidTokenError as exc:
        raise InvalidUserTokenError() from exc

    subject = payload.get("sub")
    email = payload.get("email")
    token_type = payload.get("type")
    if not subject or not email or token_type != "access":
        raise InvalidUserTokenError()
    try:
        user_id = uuid.UUID(str(subject))
    except ValueError as exc:
        raise InvalidUserTokenError() from exc
    return UserTokenClaims(subject=user_id, email=str(email))


def get_user_token_claims(request: Request) -> UserTokenClaims:
    auth_header = request.headers.get(get_settings().auth_header)
    if not auth_header:
        raise MissingUserTokenError()
    if not auth_header.startswith("Bearer "):
        raise InvalidUserSchemeError()
    token = auth_header.replace("Bearer ", "", 1).strip()
    return verify_access_token(token)
