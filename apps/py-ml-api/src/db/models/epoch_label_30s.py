from __future__ import annotations

from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    SmallInteger,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base
from .recording import Recording


class EpochLabel30s(Base):
    """
    Hypertable: ground truth labels per epoch.
    PK: (recording_id, ts_start)
    """

    __tablename__ = "epoch_labels_30s"

    recording_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("recordings.id", ondelete="CASCADE"), primary_key=True
    )
    ts_start: Mapped[Any] = mapped_column(DateTime(timezone=True), primary_key=True)
    epoch_s: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default="30"
    )

    stage: Mapped[int] = mapped_column(SmallInteger, nullable=False)  # 0..4
    source: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="edf"
    )

    inserted_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    recording: Mapped["Recording"] = relationship(back_populates="epoch_labels")

    __table_args__ = (
        CheckConstraint("stage BETWEEN 0 AND 4", name="stage"),
        Index("ix_epoch_labels_30s_recording_ts_desc", "recording_id", "ts_start"),
    )
