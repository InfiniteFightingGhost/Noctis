from __future__ import annotations

from dataclasses import dataclass
import uuid


@dataclass(frozen=True)
class AuthContext:
    client_id: uuid.UUID
    client_name: str
    role: str
    tenant_id: uuid.UUID
    scopes: set[str]
    key_id: str
