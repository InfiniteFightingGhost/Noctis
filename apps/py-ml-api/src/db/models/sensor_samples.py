from sqlalchemy import Float, Column, TIMESTAMP, String
from db.models.base import Base


class SensorSamples(Base):
    __tablename__ = "sensor_samples"

    ts = Column(TIMESTAMP(timezone=True), primary_key=True, nullable=False)
    session_id = Column(String, primary_key=True)
    value = Column(Float, nullable=False)
