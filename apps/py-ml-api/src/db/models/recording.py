from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    from .device import Device
from .device_metrics import DeviceMetric1s
from .epoch_features_30s import EpochFeature30s
from .epoch_label_30s import EpochLabel30s
from .epoch_prediction_30s import EpochPrediction30s


class Recording(Base):
    __tablename__ = "recordings"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    device_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("devices.id", ondelete="SET NULL"), index=True
    )

    # optional dimension
    user_id: Mapped[Optional[int]] = mapped_column(BigInteger, index=True)

    start_ts: Mapped[Any] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    end_ts: Mapped[Optional[Any]] = mapped_column(DateTime(timezone=True), index=True)

    # 'device' | 'edf' | 'h5'
    source: Mapped[str] = mapped_column(String(16), nullable=False, index=True)

    # pointer to object storage / filesystem; store the big EDF/H5 outside DB
    raw_uri: Mapped[Optional[str]] = mapped_column(Text)

    meta: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default="{}"
    )

    created_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    device: Mapped[Optional["Device"]] = relationship(
        "Device", back_populates="recordings"
    )

    # Timeseries tables (hypertables)
    metrics_1s: Mapped[list["DeviceMetric1s"]] = relationship(
        back_populates="recording", cascade="all, delete-orphan"
    )
    epoch_features: Mapped[list["EpochFeature30s"]] = relationship(
        back_populates="recording", cascade="all, delete-orphan"
    )
    epoch_labels: Mapped[list["EpochLabel30s"]] = relationship(
        back_populates="recording", cascade="all, delete-orphan"
    )
    epoch_predictions: Mapped[list["EpochPrediction30s"]] = relationship(
        back_populates="recording", cascade="all, delete-orphan"
    )
