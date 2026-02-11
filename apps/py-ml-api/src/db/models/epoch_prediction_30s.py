from __future__ import annotations

from typing import Any, Optional, List

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    SmallInteger,
    CheckConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base
from .recording import Recording
from .model_run import ModelRun


class EpochPrediction30s(Base):
    """
    Hypertable: model predictions per epoch (stage + confidence + optional probs).
    PK: (recording_id, ts_start, model_run_id)
    """

    __tablename__ = "epoch_predictions_30s"

    recording_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("recordings.id", ondelete="CASCADE"), primary_key=True
    )
    ts_start: Mapped[Any] = mapped_column(DateTime(timezone=True), primary_key=True)
    model_run_id: Mapped[Any] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("model_runs.id", ondelete="CASCADE"),
        primary_key=True,
    )

    epoch_s: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default="30"
    )

    stage: Mapped[int] = mapped_column(SmallInteger, nullable=False)  # 0..4
    confidence: Mapped[float] = mapped_column(Float, nullable=False)  # 0..1

    # optional: 5-way softmax, store if you need calibration/UX
    probs: Mapped[Optional[List[float]]] = mapped_column(ARRAY(Float))

    created_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    recording: Mapped["Recording"] = relationship(back_populates="epoch_predictions")
    model_run: Mapped["ModelRun"] = relationship(back_populates="predictions")

    __table_args__ = (
        CheckConstraint("stage BETWEEN 0 AND 4", name="stage"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="confidence"),
        CheckConstraint(
            "probs IS NULL OR array_length(probs, 1) = 5",
            name="probs_len_5",
        ),
        Index("ix_epoch_predictions_30s_recording_ts_desc", "recording_id", "ts_start"),
        Index("ix_epoch_predictions_30s_model_run", "model_run_id"),
    )
