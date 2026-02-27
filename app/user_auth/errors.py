from __future__ import annotations

from app.utils.errors import AppError


class UserAlreadyExistsError(AppError):
    def __init__(self, message: str = "Email already registered") -> None:
        super().__init__(
            code="user_exists",
            message=message,
            classification="client",
            status_code=409,
        )


class InvalidCredentialsError(AppError):
    def __init__(self) -> None:
        super().__init__(
            code="invalid_credentials",
            message="Invalid credentials",
            classification="client",
            status_code=401,
        )


class MissingUserTokenError(AppError):
    def __init__(self) -> None:
        super().__init__(
            code="missing_token",
            message="Missing Authorization header",
            classification="client",
            status_code=401,
        )


class InvalidUserSchemeError(AppError):
    def __init__(self) -> None:
        super().__init__(
            code="invalid_scheme",
            message="Invalid Authorization scheme",
            classification="client",
            status_code=401,
        )


class InvalidUserTokenError(AppError):
    def __init__(self) -> None:
        super().__init__(
            code="invalid_token",
            message="Invalid or expired token",
            classification="client",
            status_code=401,
        )
