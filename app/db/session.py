from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
from functools import lru_cache
from typing import Callable, TypeVar, cast

from sqlalchemy import create_engine
from sqlalchemy.exc import DBAPIError, OperationalError, SQLAlchemyError
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.metrics import DB_CIRCUIT_OPEN, DB_COMMIT_LATENCY, DB_RETRY_COUNT
from app.resilience.faults import get_fault
from app.utils.errors import CircuitBreakerOpenError

from app.core.settings import get_settings


@lru_cache
def _get_engine_cached(database_url: str, pool_size: int, max_overflow: int) -> Engine:
    return create_engine(
        database_url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,
    )


@lru_cache
def _get_sessionmaker(database_url: str, pool_size: int, max_overflow: int) -> sessionmaker:
    engine = _get_engine_cached(database_url, pool_size, max_overflow)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False)


def get_engine() -> Engine:
    settings = get_settings()
    return _get_engine_cached(
        settings.database_url, settings.db_pool_size, settings.db_max_overflow
    )


def SessionLocal() -> Session:
    settings = get_settings()
    maker = _get_sessionmaker(
        settings.database_url, settings.db_pool_size, settings.db_max_overflow
    )
    return maker()


T = TypeVar("T")


@dataclass
class DbRetryPolicy:
    max_attempts: int
    base_delay_seconds: float
    max_delay_seconds: float


class DbCircuitBreaker:
    def __init__(self, failure_threshold: int, recovery_seconds: int) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_seconds = recovery_seconds
        self._failure_count = 0
        self._opened_until: datetime | None = None
        self._lock = Lock()

    def allow_request(self) -> bool:
        with self._lock:
            if self._opened_until is None:
                return True
            if datetime.now(timezone.utc) >= self._opened_until:
                self._opened_until = None
                self._failure_count = 0
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._opened_until = None

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            if self._failure_count >= self._failure_threshold:
                self._opened_until = datetime.now(timezone.utc) + timedelta(
                    seconds=self._recovery_seconds
                )
                DB_CIRCUIT_OPEN.inc()

    @property
    def state(self) -> str:
        with self._lock:
            return "open" if self._opened_until else "closed"


_circuit_breaker: DbCircuitBreaker | None = None


def get_circuit_breaker() -> DbCircuitBreaker:
    global _circuit_breaker
    if _circuit_breaker is None:
        settings = get_settings()
        _circuit_breaker = DbCircuitBreaker(
            failure_threshold=settings.db_circuit_failure_threshold,
            recovery_seconds=settings.db_circuit_recovery_seconds,
        )
    return _circuit_breaker


def _is_transient_db_error(exc: BaseException) -> bool:
    if isinstance(exc, OperationalError):
        return True
    if isinstance(exc, DBAPIError) and getattr(exc, "connection_invalidated", False):
        return True
    return False


def run_with_db_retry(
    operation: Callable[[Session], T],
    *,
    commit: bool = False,
    operation_name: str = "db_operation",
) -> T:
    settings = get_settings()
    breaker = get_circuit_breaker()
    if not breaker.allow_request():
        raise CircuitBreakerOpenError()
    policy = DbRetryPolicy(
        max_attempts=settings.db_retry_max_attempts,
        base_delay_seconds=settings.db_retry_base_delay_seconds,
        max_delay_seconds=settings.db_retry_max_delay_seconds,
    )
    last_error: SQLAlchemyError | None = None
    for attempt in range(1, policy.max_attempts + 1):
        fault = get_fault("db_latency_ms")
        if fault:
            latency_ms = int(fault.params.get("latency_ms") or 0)
            if latency_ms > 0:
                time.sleep(latency_ms / 1000.0)
        session = SessionLocal()
        try:
            if get_fault("db_down"):
                raise OperationalError("fault injected", None, cast(BaseException, None))
            result = operation(session)
            if commit:
                commit_start = time.perf_counter()
                session.commit()
                DB_COMMIT_LATENCY.observe(time.perf_counter() - commit_start)
            breaker.record_success()
            return result
        except SQLAlchemyError as exc:
            session.rollback()
            breaker.record_failure()
            last_error = exc
            if not _is_transient_db_error(exc) or attempt >= policy.max_attempts:
                raise
            DB_RETRY_COUNT.labels(operation=operation_name).inc()
            delay = min(
                policy.max_delay_seconds,
                policy.base_delay_seconds * (2 ** (attempt - 1)),
            )
            delay *= 1 + random.uniform(-0.1, 0.1)
            time.sleep(delay)
        finally:
            session.close()
    if last_error:
        raise last_error
    raise RuntimeError("DB retry failed without exception")
