from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from app.utils.request_id import get_request_id


class JsonFormatter(logging.Formatter):
    _RESERVED_FIELDS = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "asctime",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": get_request_id(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key in self._RESERVED_FIELDS or key.startswith("_"):
                continue
            payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(level: str) -> None:
    logging.basicConfig(level=level)
    root = logging.getLogger()
    for handler in root.handlers:
        handler.setFormatter(JsonFormatter())
