from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import Field

from app.schemas.common import BaseSchema


class RegisterRequest(BaseSchema):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseSchema):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=128)


class AuthUserResponse(BaseSchema):
    id: uuid.UUID
    email: str
    created_at: datetime
    updated_at: datetime


class AuthTokenResponse(BaseSchema):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class AuthResponse(AuthTokenResponse):
    user: AuthUserResponse
