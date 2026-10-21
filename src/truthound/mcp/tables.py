"""Lossless bounded table binding over native Polars values and Arrow snapshots."""

from __future__ import annotations

from truthound.mcp.encoding import encode
from truthound.mcp.policy import MCPError


def fixed_slots(spec, depth=0):
    """Bound tiny datatype declarations that imply enormous dense null arrays."""
    if depth > 16:
        raise MCPError("resource_exhausted")
    if spec.kind in {"Array", "List"}:
        slots = fixed_slots(spec.inner, depth + 1) * (spec.width if spec.kind == "Array" else 1)
    elif spec.kind == "Struct":
        slots = sum(fixed_slots(child, depth + 1) for child in spec.fields.values())
    else:
        slots = 1
    # This is an untrusted inline representation budget, not a Core algorithm
    # limit or an OS memory sandbox. Native approved objects use a separate port.
    if slots > 524288:
        raise MCPError("resource_exhausted")
    return max(1, slots)


def _value_slots(spec, value):
    minimum = fixed_slots(spec)
    if value is None:
        return minimum
    if spec.kind in {"Array", "List"} and isinstance(value, list):
        return max(minimum, sum(_value_slots(spec.inner, item) for item in value))
    if spec.kind == "Struct" and isinstance(value, dict):
        entries = value.get("entries")
        if isinstance(entries, dict):
            return max(
                minimum,
                sum(_value_slots(child, entries.get(name)) for name, child in spec.fields.items()),
            )
    return minimum


def datatype(spec, depth=0):
    import polars as pl

    if depth > 16:
        raise MCPError("resource_exhausted")
    if depth == 0:
        fixed_slots(spec)
    simple = {
        name: getattr(pl, name)
        for name in (
            "Null",
            "Boolean",
            "String",
            "Binary",
            "Int8",
            "Int16",
            "Int32",
            "Int64",
            "Int128",
            "UInt8",
            "UInt16",
            "UInt32",
            "UInt64",
            "Float32",
            "Float64",
            "Date",
            "Time",
            "Categorical",
        )
    }
    if spec.kind in simple:
        return simple[spec.kind]
    if spec.kind == "Datetime":
        return pl.Datetime(spec.unit or "us", time_zone=spec.time_zone)
    if spec.kind == "Duration":
        return pl.Duration(spec.unit or "us")
    if spec.kind == "Decimal":
        return pl.Decimal(spec.precision, spec.scale or 0)
    if spec.kind == "List":
        return pl.List(datatype(spec.inner, depth + 1))
    if spec.kind == "Array":
        return pl.Array(datatype(spec.inner, depth + 1), spec.width)
    if spec.kind == "Struct":
        return pl.Struct({name: datatype(child, depth + 1) for name, child in spec.fields.items()})
    if spec.kind == "Enum":
        return pl.Enum(spec.categories)
    raise MCPError("unsupported_value")


def decode(value, depth=0):
    """Decode the closed tagged-value grammar. No classes are imported by name."""
    import base64
    from datetime import date, datetime, time, timedelta
    from decimal import Decimal, InvalidOperation

    if depth > 32:
        raise MCPError("resource_exhausted")
    if isinstance(value, list):
        return [decode(item, depth + 1) for item in value]
    if not isinstance(value, dict):
        return value
    kind = value.get("$type")
    try:
        if (
            kind == "object"
            and set(value) == {"$type", "entries"}
            and isinstance(value["entries"], dict)
        ):
            return {key: decode(item, depth + 1) for key, item in value["entries"].items()}
        if (
            kind == "tuple"
            and set(value) == {"$type", "items"}
            and isinstance(value["items"], list)
        ):
            return tuple(decode(item, depth + 1) for item in value["items"])
        if (
            kind in {"integer", "decimal", "float", "date", "datetime", "time"}
            and set(value) == {"$type", "value"}
            and isinstance(value["value"], str)
        ):
            item = value["value"]
            if kind == "integer":
                if len(item) > 128 or not item.lstrip("-").isdigit():
                    raise ValueError()
                return int(item)
            if kind == "decimal":
                if len(item) > 128:
                    raise ValueError()
                return Decimal(item)
            if kind == "float":
                return {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf")}[item]
            return {
                "date": date.fromisoformat,
                "datetime": datetime.fromisoformat,
                "time": time.fromisoformat,
            }[kind](item)
        if kind == "duration" and set(value) == {"$type", "days", "seconds", "microseconds"}:
            if any(type(value[key]) is not int for key in ("days", "seconds", "microseconds")):
                raise ValueError()
            return timedelta(
                days=value["days"], seconds=value["seconds"], microseconds=value["microseconds"]
            )
        if (
            kind == "binary"
            and set(value) == {"$type", "base64"}
            and isinstance(value["base64"], str)
        ):
            return base64.b64decode(value["base64"], validate=True)
        if kind == "temporal" and set(value) == {"$type", "kind", "unit", "time_zone", "value"}:
            if value["kind"] not in {"Datetime", "Duration", "Time", "Date"} or value[
                "unit"
            ] not in {"ns", "us", "ms", "d"}:
                raise ValueError()
            item = value["value"]
            if not isinstance(item, str) or len(item) > 32 or not item.lstrip("-").isdigit():
                raise ValueError()
            return int(item)
    except (ValueError, TypeError, KeyError, OverflowError, InvalidOperation):
        raise MCPError("invalid_arguments") from None
    raise MCPError("invalid_arguments")


def inline_table(columns, *, maximum_bytes=1024**3):
    import polars as pl

    # Check before constructing any native dtype/Series. Even a single null
    # can materialize a large fixed-size nested Array buffer.
    required = sum(
        sum(_value_slots(column.dtype, value) for value in column.values) * 16 for column in columns
    )
    # Zero-row declarations still require a safe native datatype span.
    for column in columns:
        fixed_slots(column.dtype)
    if required > maximum_bytes:
        raise MCPError("resource_exhausted")
    series = []
    for column in columns:
        dtype = datatype(column.dtype)
        values = [decode_column(value, column.dtype) for value in column.values]
        series.append(pl.Series(column.name, values, dtype=dtype, strict=True))
    return pl.DataFrame(series)


def decode_column(value, dtype, depth=0):
    if depth > 32:
        raise MCPError("invalid_arguments")
    if value is None:
        return None
    if isinstance(value, dict) and value.get("$type") == "temporal":
        unit = "d" if dtype.kind == "Date" else "ns" if dtype.kind == "Time" else dtype.unit or "us"
        if (
            value.get("kind") != dtype.kind
            or value.get("unit") != unit
            or value.get("time_zone") != dtype.time_zone
        ):
            raise MCPError("invalid_arguments")
    if dtype.kind in {"List", "Array"} and isinstance(value, list):
        return [decode_column(item, dtype.inner, depth + 1) for item in value]
    if dtype.kind == "Struct" and isinstance(value, dict) and value.get("$type") == "object":
        if (
            set(value) != {"$type", "entries"}
            or not isinstance(value["entries"], dict)
            or set(value["entries"]) != set(dtype.fields)
        ):
            raise MCPError("invalid_arguments")
        return {
            key: decode_column(item, dtype.fields[key], depth + 1)
            for key, item in value["entries"].items()
        }
    return decode(value)


def column_values(series, depth=0):
    import polars as pl

    if depth > 32:
        raise MCPError("unsupported_value")
    dtype = series.dtype
    kind = dtype.base_type()
    if kind in {pl.Datetime, pl.Duration, pl.Time, pl.Date}:
        unit = "d" if kind == pl.Date else ("ns" if kind == pl.Time else dtype.time_unit)
        return [
            None
            if value is None
            else {
                "$type": "temporal",
                "kind": str(kind).split("(")[0],
                "unit": unit,
                "time_zone": getattr(dtype, "time_zone", None),
                "value": str(value),
            }
            for value in series.cast(pl.Int64).to_list()
        ]
    if kind in {pl.List, pl.Array}:
        return [None if item is None else column_values(item, depth + 1) for item in series]
    if kind == pl.Struct:
        fields = {
            field.name: column_values(series.struct.field(field.name), depth + 1)
            for field in dtype.fields
        }
        return [
            None
            if null
            else {
                "$type": "object",
                "entries": {name: values[index] for name, values in fields.items()},
            }
            for index, null in enumerate(series.is_null().to_list())
        ]
    if kind == pl.Object:
        raise MCPError("unsupported_value")
    return [encode(value, depth=depth) for value in series.to_list()]


def page(frame, offset: int, limit: int) -> list:
    bounded = frame.slice(offset, limit)
    columns = {name: column_values(bounded[name]) for name in bounded.columns}
    return [
        {"$type": "object", "entries": {name: values[index] for name, values in columns.items()}}
        for index in range(bounded.height)
    ]
