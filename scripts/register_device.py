from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
import uuid
from typing import Any
from urllib import error, request

import jwt
from sqlalchemy.exc import SQLAlchemyError

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.settings import get_settings
from app.db.models import ServiceClient, ServiceClientKey
from app.db.session import run_with_db_retry


@dataclass(frozen=True)
class ServiceAuthConfig:
    service_client_id: str
    tenant_id: str
    key_id: str
    key_secret: str
    issuer: str = "noctis"
    audience: str = "noctis-services"
    token_ttl_seconds: int = 3600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register or find a device by external id")
    parser.add_argument("--name", required=True, help="Display name for the device")
    parser.add_argument("--external-id", required=True, help="Hardware external id")
    parser.add_argument("--base-url", default=os.getenv("NOCTIS_BASE_URL", "http://localhost:8000"))
    parser.add_argument(
        "--token-ttl-seconds",
        type=int,
        default=int(os.getenv("NOCTIS_TOKEN_TTL_SECONDS", "3600")),
        help="JWT lifetime in seconds",
    )
    return parser.parse_args()


def load_auth_config(token_ttl_seconds: int) -> ServiceAuthConfig:
    env_values = {
        "service_client_id": os.getenv("NOCTIS_SERVICE_CLIENT_ID", "").strip(),
        "tenant_id": os.getenv("NOCTIS_TENANT_ID", "").strip(),
        "key_id": os.getenv("NOCTIS_SERVICE_KEY_ID", "").strip(),
        "key_secret": os.getenv("NOCTIS_SERVICE_KEY_SECRET", "").strip(),
    }
    if all(env_values.values()):
        return ServiceAuthConfig(
            service_client_id=env_values["service_client_id"],
            tenant_id=env_values["tenant_id"],
            key_id=env_values["key_id"],
            key_secret=env_values["key_secret"],
            issuer=os.getenv("NOCTIS_JWT_ISSUER", "noctis"),
            audience=os.getenv("NOCTIS_JWT_AUDIENCE", "noctis-services"),
            token_ttl_seconds=token_ttl_seconds,
        )

    return bootstrap_auth_config(token_ttl_seconds=token_ttl_seconds)


def bootstrap_auth_config(token_ttl_seconds: int) -> ServiceAuthConfig:
    settings = _resolve_settings_for_host()
    tenant_id = os.getenv("NOCTIS_TENANT_ID", settings.default_tenant_id).strip()
    client_name = os.getenv("NOCTIS_SERVICE_CLIENT_NAME", "local-device-registrar").strip()
    key_id = os.getenv("NOCTIS_SERVICE_KEY_ID", "local-device-registrar-kid").strip()
    secret = os.getenv("NOCTIS_SERVICE_KEY_SECRET", settings.jwt_secret).strip()

    if not secret:
        raise SystemExit(
            "Unable to bootstrap auth: set NOCTIS_SERVICE_KEY_SECRET or JWT_SECRET in .env"
        )

    tenant_uuid = uuid.UUID(tenant_id)

    def _op(session):
        client = (
            session.query(ServiceClient).filter(ServiceClient.name == client_name).one_or_none()
        )
        if client is None:
            client = ServiceClient(
                tenant_id=tenant_uuid,
                name=client_name,
                role="ingest",
                status="active",
            )
            session.add(client)
            session.flush()
        elif client.tenant_id != tenant_uuid:
            raise RuntimeError(
                f"Service client '{client_name}' belongs to tenant {client.tenant_id}, expected {tenant_uuid}"
            )
        else:
            client.role = "ingest"
            client.status = "active"
            session.add(client)

        key = (
            session.query(ServiceClientKey)
            .filter(ServiceClientKey.client_id == client.id)
            .filter(ServiceClientKey.key_id == key_id)
            .one_or_none()
        )
        if key is None:
            key = ServiceClientKey(
                client_id=client.id,
                key_id=key_id,
                secret=secret,
                status="active",
            )
            session.add(key)
            session.flush()
        else:
            key.status = "active"
            if not key.secret:
                key.secret = secret
            session.add(key)

        return {
            "service_client_id": str(client.id),
            "tenant_id": str(client.tenant_id),
            "key_id": key.key_id,
            "key_secret": key.secret,
        }

    try:
        data = run_with_db_retry(_op, commit=True, operation_name="bootstrap_device_register_auth")
    except SQLAlchemyError as exc:
        raise SystemExit(
            "Unable to bootstrap service auth from the database. "
            "Ensure DB is reachable from this shell (or export NOCTIS_SERVICE_* credentials). "
            f"Details: {exc.__class__.__name__}"
        )
    key_secret = data.get("key_secret")
    if not isinstance(key_secret, str) or not key_secret:
        raise SystemExit(
            "Service key has no secret. Set NOCTIS_SERVICE_KEY_SECRET and rerun, or update service_client_keys.secret."
        )

    return ServiceAuthConfig(
        service_client_id=data["service_client_id"],
        tenant_id=data["tenant_id"],
        key_id=data["key_id"],
        key_secret=key_secret,
        issuer=os.getenv("NOCTIS_JWT_ISSUER", settings.jwt_issuer),
        audience=os.getenv("NOCTIS_JWT_AUDIENCE", settings.jwt_audience),
        token_ttl_seconds=token_ttl_seconds,
    )


def mint_service_token(config: ServiceAuthConfig) -> str:
    now = int(time.time())
    claims = {
        "sub": config.service_client_id,
        "tenant_id": config.tenant_id,
        "iss": config.issuer,
        "aud": config.audience,
        "iat": now,
        "exp": now + config.token_ttl_seconds,
    }
    return jwt.encode(claims, config.key_secret, algorithm="HS256", headers={"kid": config.key_id})


def post_json(url: str, token: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with request.urlopen(req, timeout=10) as response:
        data = response.read().decode("utf-8")
        return json.loads(data) if data else {}


def get_devices(url: str, token: str) -> list[dict[str, Any]]:
    req = request.Request(
        url,
        method="GET",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with request.urlopen(req, timeout=10) as response:
        data = response.read().decode("utf-8")
        payload = json.loads(data) if data else []
        if isinstance(payload, list):
            return payload
        return []


def parse_error_message(exc: error.HTTPError) -> str:
    try:
        payload = json.loads(exc.read().decode("utf-8"))
    except Exception:
        return f"HTTP {exc.code}"

    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str):
            return detail
        message = payload.get("message")
        if isinstance(message, str):
            return message
    return f"HTTP {exc.code}"


def main() -> None:
    args = parse_args()
    auth = load_auth_config(token_ttl_seconds=args.token_ttl_seconds)
    token = mint_service_token(auth)

    base_url = args.base_url.rstrip("/")
    devices_url = f"{base_url}/v1/devices"

    try:
        created = post_json(
            devices_url,
            token,
            {
                "name": args.name.strip(),
                "external_id": args.external_id.strip(),
            },
        )
        print(
            f"CREATED id={created.get('id')} external_id={created.get('external_id')} name={created.get('name')}"
        )
        return
    except error.HTTPError as exc:
        if exc.code != 409:
            raise SystemExit(f"Failed to create device: {parse_error_message(exc)}")

    try:
        devices = get_devices(devices_url, token)
    except error.HTTPError as exc:
        raise SystemExit(f"Device exists but lookup failed: {parse_error_message(exc)}")

    existing = next(
        (item for item in devices if item.get("external_id") == args.external_id.strip()), None
    )
    if existing:
        print(
            f"EXISTS id={existing.get('id')} external_id={existing.get('external_id')} name={existing.get('name')}"
        )
        return

    raise SystemExit("Device create returned conflict, but no matching external_id was found.")


def _resolve_settings_for_host():
    settings = get_settings()

    if "@timescaledb:" in settings.database_url and not Path("/.dockerenv").exists():
        os.environ["DATABASE_URL"] = settings.database_url.replace("@timescaledb:", "@localhost:")
        settings = get_settings()

    return settings


if __name__ == "__main__":
    main()
