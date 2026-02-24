from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.user_auth.schemas import AuthResponse, AuthUserResponse, LoginRequest, RegisterRequest
from app.user_auth.security import UserTokenClaims, get_user_token_claims
from app.user_auth.service import get_me, login, register


router = APIRouter(tags=["user-auth"])


@router.post("/auth/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register_user(payload: RegisterRequest) -> AuthResponse:
    return register(payload)


@router.post("/auth/login", response_model=AuthResponse)
def login_user(payload: LoginRequest) -> AuthResponse:
    return login(payload)


@router.get("/auth/me", response_model=AuthUserResponse)
def get_current_user(claims: UserTokenClaims = Depends(get_user_token_claims)) -> AuthUserResponse:
    return get_me(claims)
