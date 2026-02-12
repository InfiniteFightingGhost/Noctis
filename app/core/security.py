from __future__ import annotations

from app.auth.dependencies import require_admin, require_scopes

# Backwards-compat aliases for legacy imports
require_api_key = require_scopes("ingest")
require_admin_key = require_admin
