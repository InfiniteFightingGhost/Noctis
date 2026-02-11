from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    SmallInteger,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base
from .recording import Recording


class EpochFeature30s(Base):
    """
    Hypertable: EpochQ12 features per 30s epoch.
    PK: (recording_id, ts_start)
    """

    __tablename__ = "epoch_features_30s"

    recording_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("recordings.id", ondelete="CASCADE"), primary_key=True
    )
    ts_start: Mapped[Any] = mapped_column(DateTime(timezone=True), primary_key=True)
    epoch_s: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default="30"
    )

    # EpochQ12 packed fields (stored as SMALLINT with constraints)
    in_bed_pct: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    hr_mean: Mapped[Optional[int]] = mapped_column(SmallInteger)
    hr_std: Mapped[Optional[int]] = mapped_column(SmallInteger)
    dhr: Mapped[Optional[int]] = mapped_column(SmallInteger)

    rr_mean: Mapped[Optional[int]] = mapped_column(SmallInteger)
    rr_std: Mapped[Optional[int]] = mapped_column(SmallInteger)
    drr: Mapped[Optional[int]] = mapped_column(SmallInteger)

    large_move_pct: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    minor_move_pct: Mapped[int] = mapped_column(SmallInteger, nullable=False)

    turnovers_delta: Mapped[Optional[int]] = mapped_column(SmallInteger)
    flags: Mapped[int] = mapped_column(SmallInteger, nullable=False)

    inserted_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    recording: Mapped["Recording"] = relationship(back_populates="epoch_features")

    __table_args__ = (
        CheckConstraint("in_bed_pct BETWEEN 0 AND 100", name="in_bed_pct"),
        CheckConstraint("large_move_pct BETWEEN 0 AND 100", name="large_move_pct"),
        CheckConstraint("minor_move_pct BETWEEN 0 AND 100", name="minor_move_pct"),
        CheckConstraint("flags BETWEEN 0 AND 255", name="flags"),
        CheckConstraint(
            "turnovers_delta IS NULL OR (turnovers_delta BETWEEN 0 AND 255)",
            name="turnovers_delta",
        ),
        CheckConstraint(
            "hr_mean IS NULL OR (hr_mean BETWEEN 0 AND 255)", name="hr_mean"
        ),
        CheckConstraint("hr_std IS NULL OR (hr_std BETWEEN 0 AND 255)", name="hr_std"),
        CheckConstraint("dhr IS NULL OR (dhr BETWEEN -128 AND 127)", name="dhr"),
        CheckConstraint(
            "rr_mean IS NULL OR (rr_mean BETWEEN 0 AND 255)", name="rr_mean"
        ),
        CheckConstraint("rr_std IS NULL OR (rr_std BETWEEN 0 AND 255)", name="rr_std"),
        CheckConstraint("drr IS NULL OR (drr BETWEEN -128 AND 127)", name="drr"),
        Index("ix_epoch_features_30s_recording_ts_desc", "recording_id", "ts_start"),
    )
