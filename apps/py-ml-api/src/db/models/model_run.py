from typing import Any, Optional

from sqlalchemy import (
    DateTime,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base
from .epoch_prediction_30s import EpochPrediction30s


class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[Any] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),  # requires pgcrypto extension
    )

    model_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    git_sha: Mapped[Optional[str]] = mapped_column(String(64), index=True)

    params: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default="{}"
    )

    created_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    predictions: Mapped[list["EpochPrediction30s"]] = relationship(
        back_populates="model_run"
    )
