from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    SmallInteger,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base
from .recording import Recording


class DeviceMetric1s(Base):
    """
    Hypertable: raw-ish device metrics at 1 Hz (or close).
    PK: (recording_id, ts)
    """

    __tablename__ = "device_metrics_1s"

    recording_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("recordings.id", ondelete="CASCADE"), primary_key=True
    )
    ts: Mapped[Any] = mapped_column(DateTime(timezone=True), primary_key=True)

    hr_bpm: Mapped[Optional[int]] = mapped_column(SmallInteger)
    rr_brpm: Mapped[Optional[int]] = mapped_column(SmallInteger)

    movement: Mapped[Optional[int]] = mapped_column(SmallInteger)  # arbitrary units
    in_bed: Mapped[Optional[bool]] = mapped_column(Boolean)

    confidence: Mapped[Optional[float]] = mapped_column(Float)  # 0..1
    flags: Mapped[Optional[int]] = mapped_column(SmallInteger)  # 0..255

    inserted_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    recording: Mapped["Recording"] = relationship(back_populates="metrics_1s")

    __table_args__ = (
        CheckConstraint(
            "flags IS NULL OR (flags BETWEEN 0 AND 255)", name="flags_range"
        ),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="confidence_range",
        ),
        Index("ix_device_metrics_1s_recording_ts_desc", "recording_id", "ts"),
    )
