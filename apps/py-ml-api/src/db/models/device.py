from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    from .recording import Recording


class Device(Base):
    __tablename__ = "devices"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    serial: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    model: Mapped[Optional[str]] = mapped_column(String(64))
    created_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    recording: Mapped["Recording"] = relationship(
        "Recording", back_populates="metrics_1s"
    )
