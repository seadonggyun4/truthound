"""Lossless JSON-safe rich values; unsupported values are errors, never reprs."""

from __future__ import annotations

import base64
import math
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum

from truthound.mcp.policy import MCPError


def encode(value, *, depth: int = 0):
    if depth > 32:
        raise MCPError("unsupported_value")
    if isinstance(value, Enum):
        return encode(value.value, depth=depth + 1)
    if type(value) is int and abs(value) > 2**53 - 1:
        return {"$type": "integer", "value": str(value)}
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if math.isfinite(value):
            return value
        return {
            "$type": "float",
            "value": "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf"),
        }
    if isinstance(value, Decimal):
        return {"$type": "decimal", "value": str(value)}
    if isinstance(value, (datetime, date, time)):
        return {"$type": type(value).__name__, "value": value.isoformat()}
    if isinstance(value, timedelta):
        return {
            "$type": "duration",
            "days": value.days,
            "seconds": value.seconds,
            "microseconds": value.microseconds,
        }
    if isinstance(value, bytes):
        return {"$type": "binary", "base64": base64.b64encode(value).decode("ascii")}
    # Tag containers too, so user dictionaries cannot impersonate rich values.
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise MCPError("unsupported_value")
        return {
            "$type": "object",
            "entries": {key: encode(item, depth=depth + 1) for key, item in value.items()},
        }
    if isinstance(value, tuple):
        return {"$type": "tuple", "items": [encode(item, depth=depth + 1) for item in value]}
    if isinstance(value, list):
        return [encode(item, depth=depth + 1) for item in value]
    raise MCPError("unsupported_value")
