"""Data-only model values: exact containers, no imports or executable payloads."""

import base64
import math
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum


def unique_object(pairs):
    """Reject ambiguous JSON objects before a model mutates its learned state."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate model JSON key")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("Nonfinite model JSON constant")


def encode(value, depth=0):
    if depth > 64:
        raise ValueError("Model state nesting exceeds 64")
    if isinstance(value, Enum):
        return encode(value.value, depth + 1)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {"type": "float", "value": str(value)}
    if isinstance(value, (datetime, date, time, Decimal)):
        return {
            "type": type(value).__name__,
            "value": value.isoformat() if hasattr(value, "isoformat") else str(value),
        }
    if isinstance(value, timedelta):
        return {"type": "timedelta", "value": [value.days, value.seconds, value.microseconds]}
    if isinstance(value, bytes):
        return {"type": "bytes", "value": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {
            "type": "dict",
            "value": [[encode(k, depth + 1), encode(v, depth + 1)] for k, v in value.items()],
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": "tuple" if isinstance(value, tuple) else "list",
            "value": [encode(v, depth + 1) for v in value],
        }
    raise TypeError(f"Unsupported model state value: {type(value).__name__}")


def decode(value, depth=0):
    if depth > 64:
        raise ValueError("Model state nesting exceeds 64")
    if type(value) is float and not math.isfinite(value):
        raise ValueError("Nonfinite model values require a tagged representation")
    if value is None or type(value) in (str, bool, int, float):
        return value
    if not isinstance(value, dict) or set(value) != {"type", "value"}:
        raise ValueError("Invalid model state value")
    kind, payload = value["type"], value["value"]
    if type(kind) is not str:
        raise ValueError("Invalid model state type")
    if kind == "dict":
        if type(payload) is not list or any(
            type(pair) is not list or len(pair) != 2 for pair in payload
        ):
            raise ValueError("Invalid model state dictionary")
        pairs = [(decode(k, depth + 1), decode(v, depth + 1)) for k, v in payload]
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("Duplicate model state key")
        return result
    if kind in {"list", "tuple"}:
        if type(payload) is not list:
            raise ValueError("Invalid model state sequence")
        result = [decode(v, depth + 1) for v in payload]
        return tuple(result) if kind == "tuple" else result
    if kind == "float" and type(payload) is str and payload in {"nan", "inf", "-inf"}:
        return float(payload)
    if kind == "bytes":
        if type(payload) is not str:
            raise ValueError("Invalid model state bytes")
        return base64.b64decode(payload, validate=True)
    if kind == "timedelta":
        if (
            type(payload) is not list
            or len(payload) != 3
            or any(type(item) is not int for item in payload)
        ):
            raise ValueError("Invalid model state duration")
        return timedelta(days=payload[0], seconds=payload[1], microseconds=payload[2])
    constructors = {
        "datetime": datetime.fromisoformat,
        "date": date.fromisoformat,
        "time": time.fromisoformat,
        "Decimal": Decimal,
    }
    if kind in constructors:
        if type(payload) is not str:
            raise ValueError("Invalid model state scalar")
        return constructors[kind](payload)
    raise ValueError("Unknown model state value type")
