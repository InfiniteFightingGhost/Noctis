from __future__ import annotations

from sqlalchemy.exc import IntegrityError

from app.db.models import AuthUser
from app.db.session import run_with_db_retry
from app.user_auth import repository
from app.user_auth.errors import (
    InvalidCredentialsError,
    InvalidUserTokenError,
    UserAlreadyExistsError,
)
from app.user_auth.schemas import (
    AuthResponse,
    AuthTokenResponse,
    AuthUserResponse,
    LoginRequest,
    RegisterRequest,
)
from app.user_auth.security import (
    UserTokenClaims,
    hash_password,
    issue_access_token,
    verify_password,
)


def register(payload: RegisterRequest) -> AuthResponse:
    username = _normalize_username(payload.username)
    email = _normalize_email(payload.email)

    existing = run_with_db_retry(
        lambda session: repository.get_auth_user_by_email(session, email),
        operation_name="auth_user_lookup_email",
    )
    if existing is not None:
        raise UserAlreadyExistsError()

    password_hash = hash_password(payload.password)
    try:
        user = run_with_db_retry(
            lambda session: repository.create_auth_user(session, username, email, password_hash),
            commit=True,
            operation_name="auth_user_register",
        )
    except IntegrityError as exc:
        raise UserAlreadyExistsError() from exc
    return _build_auth_response(user)


def login(payload: LoginRequest) -> AuthResponse:
    email = _normalize_email(payload.email)
    user = run_with_db_retry(
        lambda session: repository.get_auth_user_by_email(session, email),
        operation_name="auth_user_login_lookup",
    )
    if user is None:
        raise InvalidCredentialsError()
    if not verify_password(payload.password, user.password_hash):
        raise InvalidCredentialsError()
    return _build_auth_response(user)


def get_me(claims: UserTokenClaims) -> AuthUserResponse:
    user = run_with_db_retry(
        lambda session: repository.get_auth_user_by_id(session, claims.subject),
        operation_name="auth_user_me_lookup",
    )
    if user is None or user.email != claims.email:
        raise InvalidUserTokenError()
    return AuthUserResponse.model_validate(user)


def _build_auth_response(user: AuthUser) -> AuthResponse:
    token, ttl = issue_access_token(user_id=user.id, email=user.email)
    token_response = AuthTokenResponse(access_token=token, expires_in=ttl)
    return AuthResponse(
        access_token=token_response.access_token,
        token_type=token_response.token_type,
        expires_in=token_response.expires_in,
        user=AuthUserResponse.model_validate(user),
    )


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _normalize_username(username: str) -> str:
    return username.strip()
