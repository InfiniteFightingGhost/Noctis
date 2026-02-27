from __future__ import annotations

from datetime import datetime, timezone
import uuid

from fastapi import APIRouter, Depends

from app.auth.dependencies import require_scopes
from app.db.models import Routine, RoutineStep
from app.db.session import run_with_db_retry
from app.schemas.routines import (
    RoutineCurrentResponse,
    RoutineCurrentUpdate,
    RoutineResponse,
    RoutineStepResponse,
)
from app.tenants.context import TenantContext, get_tenant_context


router = APIRouter(tags=["routines"])


@router.get(
    "/routines",
    response_model=list[RoutineResponse],
    dependencies=[Depends(require_scopes("read"))],
)
def list_routines(
    tenant: TenantContext = Depends(get_tenant_context),
) -> list[RoutineResponse]:
    def _op(session):
        return (
            session.query(Routine)
            .filter(Routine.tenant_id == tenant.id)
            .order_by(Routine.created_at.desc())
            .all()
        )

    routines = run_with_db_retry(_op, operation_name="routines_list")
    return [RoutineResponse.model_validate(routine) for routine in routines]


@router.get(
    "/routines/current",
    response_model=RoutineCurrentResponse,
    dependencies=[Depends(require_scopes("read"))],
)
def get_current_routine(
    tenant: TenantContext = Depends(get_tenant_context),
) -> RoutineCurrentResponse:
    def _op(session):
        routine = _get_or_create_routine(session, tenant.id)
        _ = list(routine.steps)
        return routine

    routine = run_with_db_retry(_op, commit=True, operation_name="routine_current_get")
    return _to_current_response(routine)


@router.put(
    "/routines/current",
    response_model=RoutineCurrentResponse,
    dependencies=[Depends(require_scopes("ingest"))],
)
def update_current_routine(
    payload: RoutineCurrentUpdate,
    tenant: TenantContext = Depends(get_tenant_context),
) -> RoutineCurrentResponse:
    def _op(session):
        routine = _get_or_create_routine(session, tenant.id)
        if payload.title is not None:
            routine.name = payload.title
        if payload.steps is not None:
            session.query(RoutineStep).filter(RoutineStep.routine_id == routine.id).delete()
            for position, step in enumerate(payload.steps):
                session.add(
                    RoutineStep(
                        id=step.id or uuid.uuid4(),
                        tenant_id=tenant.id,
                        routine_id=routine.id,
                        title=step.title,
                        duration_minutes=step.duration_minutes,
                        emoji=step.emoji,
                        position=position,
                    )
                )
        routine.updated_at = datetime.now(timezone.utc)
        session.add(routine)
        session.flush()
        session.refresh(routine)
        _ = list(routine.steps)
        return routine

    routine = run_with_db_retry(_op, commit=True, operation_name="routine_current_update")
    return _to_current_response(routine)


def _get_or_create_routine(session, tenant_id) -> Routine:
    routine = (
        session.query(Routine)
        .filter(Routine.tenant_id == tenant_id)
        .filter(Routine.status == "active")
        .order_by(Routine.updated_at.desc())
        .first()
    )
    if routine:
        return routine
    now = datetime.now(timezone.utc)
    routine = Routine(
        tenant_id=tenant_id,
        name="Tonight's routine",
        description=None,
        status="active",
        created_at=now,
        updated_at=now,
    )
    session.add(routine)
    session.flush()
    session.refresh(routine)
    return routine


def _to_current_response(routine: Routine) -> RoutineCurrentResponse:
    steps = [
        RoutineStepResponse(
            id=step.id,
            title=step.title,
            duration_minutes=step.duration_minutes,
            emoji=step.emoji,
        )
        for step in routine.steps
    ]
    total_minutes = sum(step.duration_minutes for step in routine.steps)
    return RoutineCurrentResponse(
        id=routine.id,
        title=routine.name,
        total_minutes=total_minutes,
        steps=steps,
        updated_at=routine.updated_at,
    )
