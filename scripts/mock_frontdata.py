from __future__ import annotations

import sys
from pathlib import Path
import uuid
from datetime import datetime, timedelta, timezone

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.db.models import Tenant, User, Device, Recording, Epoch, Prediction
from app.db.session import run_with_db_retry

# Use the same IDs as in apps/web-app/src/api/mockApiClient.ts for consistency
MOCK_USER_ID = uuid.UUID("018f4a1b-6e3d-7b5c-8d7e-8f9000000001")  # Adjusted to valid UUID
MOCK_TENANT_ID = uuid.UUID("018f4a1b-6e3d-7b5c-8d7e-8f9000000000")
MOCK_DEVICE_ID = uuid.UUID("018f4a1b-6e3d-7b5c-8d7e-8f9000000002")


def mock_data():
    def _op(session):
        # 1. Ensure Tenant
        tenant = session.query(Tenant).filter(Tenant.id == MOCK_TENANT_ID).first()
        if not tenant:
            tenant = Tenant(id=MOCK_TENANT_ID, name="Mock Tenant")
            session.add(tenant)
            session.flush()

        # 2. Ensure User
        user = session.query(User).filter(User.id == MOCK_USER_ID).first()
        if not user:
            user = User(
                id=MOCK_USER_ID,
                tenant_id=tenant.id,
                name="Sample User",
                external_id="sample-user-ext",
            )
            session.add(user)
            session.flush()

        # 3. Ensure Device
        device = session.query(Device).filter(Device.id == MOCK_DEVICE_ID).first()
        if not device:
            device = Device(
                id=MOCK_DEVICE_ID,
                tenant_id=tenant.id,
                user_id=user.id,
                name="Noctis Halo Mock",
                external_id="noctis-halo-mock-001",
            )
            session.add(device)
            session.flush()

        # 4. Create Recording for last night
        now = datetime.now(timezone.utc)
        # Previous night: 10 PM yesterday to 6 AM today
        yesterday_10pm = (now - timedelta(days=1)).replace(
            hour=22, minute=0, second=0, microsecond=0
        )
        today_6am = now.replace(hour=6, minute=0, second=0, microsecond=0)

        # Ensure today_6am is actually after yesterday_10pm (in case script runs late at night)
        if today_6am <= yesterday_10pm:
            today_6am += timedelta(days=1)

        recording = Recording(
            tenant_id=tenant.id,
            device_id=device.id,
            started_at=yesterday_10pm,
            ended_at=today_6am,
            timezone="UTC",
        )
        session.add(recording)
        session.flush()

        # 5. Create Epochs and Predictions (every 30s)
        total_seconds = int((today_6am - yesterday_10pm).total_seconds())
        num_epochs = total_seconds // 30

        print(f"Generating {num_epochs} epochs for recording {recording.id}...")

        for i in range(num_epochs):
            ts = yesterday_10pm + timedelta(seconds=i * 30)

            # Realistic sleep cycle logic
            # ~90 minute cycles (180 epochs)
            pos_in_cycle = i % 180
            if pos_in_cycle < 10:
                stage = "W"
            elif pos_in_cycle < 100:
                stage = "N2"
            elif pos_in_cycle < 140:
                stage = "N3"
            else:
                stage = "R"

            # More REM later in the night
            if i > num_epochs * 0.6 and stage == "N2" and pos_in_cycle > 80:
                stage = "R"

            epoch = Epoch(
                tenant_id=tenant.id,
                recording_id=recording.id,
                epoch_index=i,
                epoch_start_ts=ts,
                feature_schema_version="v1",
                features_payload={"hr_mean": 60, "rr_mean": 14},
            )
            session.add(epoch)

            prediction = Prediction(
                tenant_id=tenant.id,
                recording_id=recording.id,
                window_start_ts=ts,
                window_end_ts=ts + timedelta(seconds=30),
                model_version="mock-v1",
                feature_schema_version="v1",
                predicted_stage=stage,
                probabilities={"W": 0.05, "N1": 0.05, "N2": 0.4, "N3": 0.3, "R": 0.2},
                confidence=0.85,
            )
            session.add(prediction)

            if i % 200 == 0:
                session.flush()

        print(f"Successfully created recording {recording.id} starting at {yesterday_10pm}")
        return recording.id

    return run_with_db_retry(_op, commit=True)


if __name__ == "__main__":
    try:
        mock_data()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
