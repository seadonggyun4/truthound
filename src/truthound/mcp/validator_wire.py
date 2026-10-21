"""Closed registry constructor transport; never evaluates model-supplied code.

Only the checked-in Core contract selects constructors. Rich inputs use explicit
frames, expressions and datatype grammars, rather than Python object decoding.
"""

from __future__ import annotations

import importlib
import json
import math
from functools import lru_cache
from pathlib import Path

from truthound.mcp.policy import MCPError
from truthound.mcp.source_identity import class_identity


@lru_cache(maxsize=1)
def contracts():
    return json.loads(
        Path(__file__).with_name("validator_contracts.json").read_text(encoding="utf-8")
    )


def spec(name):
    try:
        document = contracts()
        row = document["validators"][name]
        return {
            **row,
            "fields": {**document["common_fields"], **row["fields"]},
            "source_ast_sha256": {
                target: document["source_ast_sha256"][target] for target in row["source_types"]
            },
        }
    except KeyError:
        raise MCPError("unknown_validator") from None


def record_fields(shape):
    return contracts()["record_types"][shape["target"]]


def record_definitions(shape):
    """Only transitively referenced records, so unrelated domains cannot bloat discovery."""
    found = {}

    def walk(value):
        if isinstance(value, dict):
            if value.get("kind") == "record" and value["target"] not in found:
                found[value["target"]] = record_fields(value)
                walk(found[value["target"]])
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(shape)
    return found


@lru_cache(maxsize=256)
def _known_type(target):
    # Target is exclusively from checked-in metadata, never the input document.
    if not target.startswith("truthound."):
        raise MCPError("contract_drift")
    module, name = target.rsplit(".", 1)
    cls = getattr(importlib.import_module(module), name)
    for parent in cls.__mro__:
        if parent.__module__.startswith("truthound."):
            reference = f"{parent.__module__}.{parent.__qualname__}"
            expected = contracts()["source_ast_sha256"].get(reference)
            if expected is None or class_identity(parent) != expected:
                raise MCPError("contract_drift")
    return cls


@lru_cache(maxsize=512)
def constructor(name):
    from truthound.validators import get_validator

    row = spec(name)
    # Resolve the pinned concrete module first. Optional import failures must
    # remain missing_dependency, not an unknown-registry/configuration error.
    cls = _known_type(row["target"])
    if get_validator(name) is not cls:
        raise MCPError("contract_drift")
    if f"{cls.__module__}.{cls.__qualname__}" != row["target"]:
        raise MCPError("contract_drift")
    for parent in cls.__mro__:
        key = f"{parent.__module__}.{parent.__qualname__}"
        if key in row["source_ast_sha256"]:
            digest = class_identity(parent)
            if digest != row["source_ast_sha256"][key]:
                raise MCPError("contract_drift")
    return cls


def _json(value, depth=0):
    if depth > 24:
        raise MCPError("invalid_arguments")
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, list) and len(value) <= 10000:
        return [_json(v, depth + 1) for v in value]
    if isinstance(value, dict) and len(value) <= 1000 and all(type(k) is str for k in value):
        return {k: _json(v, depth + 1) for k, v in value.items()}
    raise MCPError("invalid_arguments")


def expression(value, *, build=False, depth=0):
    """Finite Polars expression vocabulary. No getattr, SQL or deserialization."""
    if depth > 16 or not isinstance(value, dict) or "op" not in value:
        raise MCPError("invalid_arguments")
    operation = value["op"]
    unary = {
        "not",
        "is_null",
        "is_not_null",
        "abs",
        "sum",
        "mean",
        "min",
        "max",
        "count",
        "n_unique",
    }
    binary = {"eq", "ne", "lt", "le", "gt", "ge", "add", "sub", "mul", "div", "and", "or"}
    keys = (
        {"op", "name"}
        if operation == "column"
        else {"op", "value"}
        if operation == "literal"
        else {"op", "arg"}
        if operation in unary
        else {"op", "left", "right"}
        if operation in binary
        else set()
    )
    if set(value) != keys:
        raise MCPError("invalid_arguments")
    if operation == "column":
        if not isinstance(value["name"], str) or not value["name"] or len(value["name"]) > 512:
            raise MCPError("invalid_arguments")
        if build:
            import polars as pl

            return pl.col(value["name"])
    elif operation == "literal":
        item = _json(value["value"])
        if isinstance(item, dict):
            raise MCPError("invalid_arguments")
        if build:
            import polars as pl

            return pl.lit(item)
    elif operation in unary:
        arg = expression(value["arg"], build=build, depth=depth + 1)
        if build:
            return {
                "not": lambda: ~arg,
                "is_null": arg.is_null,
                "is_not_null": arg.is_not_null,
                "abs": arg.abs,
                "sum": arg.sum,
                "mean": arg.mean,
                "min": arg.min,
                "max": arg.max,
                "count": arg.count,
                "n_unique": arg.n_unique,
            }[operation]()
    elif operation in binary:
        left = expression(value["left"], build=build, depth=depth + 1)
        right = expression(value["right"], build=build, depth=depth + 1)
        if build:
            import operator

            return {
                "eq": operator.eq,
                "ne": operator.ne,
                "lt": operator.lt,
                "le": operator.le,
                "gt": operator.gt,
                "ge": operator.ge,
                "add": operator.add,
                "sub": operator.sub,
                "mul": operator.mul,
                "div": operator.truediv,
                "and": operator.and_,
                "or": operator.or_,
            }[operation](left, right)
    return value


def convert(shape, value, *, frames=None, build=False, depth=0, trusted=False, callbacks=None):
    if depth > 24:
        raise MCPError("invalid_arguments")
    kind = shape["kind"]

    def nested(child, item):
        return convert(
            child,
            item,
            frames=frames,
            build=build,
            depth=depth + 1,
            trusted=trusted,
            callbacks=callbacks,
        )

    if kind == "union":
        errors = []
        for child in shape["items"]:
            try:
                return nested(child, value)
            except MCPError as exc:
                errors.append(exc.code)
        raise MCPError(
            "operator_configuration_required"
            if "operator_configuration_required" in errors
            else "invalid_arguments"
        )
    if kind == "blocked":
        raise MCPError("operator_configuration_required")
    if kind == "callback":
        if (
            not trusted
            or not isinstance(value, dict)
            or set(value) != {"operator_capability"}
            or not isinstance(value["operator_capability"], str)
            or value["operator_capability"] not in (callbacks or {})
        ):
            raise MCPError("operator_configuration_required")
        if build and not callable(callbacks[value["operator_capability"]]):
            raise MCPError("invalid_binding")
        return callbacks[value["operator_capability"]] if build else value
    if kind == "json":
        return _json(value)
    if kind == "null" and value is None:
        return None
    if kind in ("str", "int", "float", "bool"):
        valid = type(value) is {"str": str, "int": int, "float": float, "bool": bool}[kind]
        if kind == "float" and type(value) is int:
            valid = True
        if valid and (kind != "float" or math.isfinite(value)):
            return value
    elif kind in ("literal", "enum"):
        if any(type(value) is type(option) and value == option for option in shape["values"]):
            return _known_type(shape["target"])(value) if build and kind == "enum" else value
    elif kind in ("list", "set", "tuple", "tuple_list"):
        if isinstance(value, list) and len(value) <= 10000:
            if kind == "tuple":
                if len(value) != len(shape["items"]):
                    raise MCPError("invalid_arguments")
                output = [
                    nested(child, item) for child, item in zip(shape["items"], value, strict=True)
                ]
            else:
                output = [nested(shape["item"], item) for item in value]
            if not build:
                return output
            return (
                tuple(output)
                if kind in {"tuple", "tuple_list"}
                else set(output)
                if kind == "set"
                else output
            )
    elif kind == "map":
        if shape["key"]["kind"] == "str" and isinstance(value, dict) and len(value) <= 1000:
            return {nested(shape["key"], k): nested(shape["value"], v) for k, v in value.items()}
        if (
            shape["key"]["kind"] != "str"
            and isinstance(value, dict)
            and set(value) == {"entries"}
            and isinstance(value["entries"], list)
            and len(value["entries"]) <= 1000
        ):
            pairs = []
            for pair in value["entries"]:
                if not isinstance(pair, dict) or set(pair) != {"key", "value"}:
                    raise MCPError("invalid_arguments")
                pairs.append(
                    (nested(shape["key"], pair["key"]), nested(shape["value"], pair["value"]))
                )
            if not build:
                return value
            result = dict(pairs)
            if len(result) != len(pairs):
                raise MCPError("invalid_arguments")
            return result
    elif kind == "record":
        result = fields(
            record_fields(shape),
            value,
            frames=frames,
            build=build,
            depth=depth + 1,
            trusted=trusted,
            callbacks=callbacks,
        )
        return _known_type(shape["target"])(**result) if build else result
    elif kind == "frame":
        from truthound.mcp.session_contracts import AssetInput

        try:
            handle = AssetInput.model_validate(value).asset_id
        except ValueError:
            raise MCPError("invalid_arguments") from None
        if frames is not None and handle not in frames:
            raise MCPError("invalid_handle")
        if not build:
            return value
        import polars as pl

        frame = pl.read_ipc(frames[handle], memory_map=False)
        return frame.lazy() if shape["lazy"] else frame
    elif kind == "consistency_rule":
        if (
            not isinstance(value, dict)
            or not {"when", "then"} <= value.keys()
            or value.keys() - {"when", "then", "description"}
        ):
            raise MCPError("invalid_arguments")
        if "description" in value and (
            type(value["description"]) is not str or len(value["description"]) > 4096
        ):
            raise MCPError("invalid_arguments")
        return {
            **value,
            "when": expression(value["when"], build=build),
            "then": expression(value["then"], build=build),
        }
    elif kind == "expression":
        return expression(value, build=build)
    elif kind == "dtype":
        from truthound.mcp.session_contracts import DType
        from truthound.mcp.tables import datatype

        try:
            dtype = DType.model_validate(value)
        except ValueError:
            raise MCPError("invalid_arguments") from None
        return datatype(dtype) if build else value
    elif kind == "temporal":
        from datetime import date, datetime, timedelta

        if isinstance(value, dict) and set(value) == {"value"}:
            raw = value["value"]
            try:
                output = (
                    timedelta(microseconds=raw)
                    if shape["type"] == "timedelta" and type(raw) is int
                    else {"date": date, "datetime": datetime}[shape["type"]].fromisoformat(raw)
                )
            except (KeyError, TypeError, ValueError, OverflowError):
                raise MCPError("invalid_arguments") from None
            return output if build else value
    raise MCPError("invalid_arguments")


def fields(
    shapes,
    values,
    *,
    frames=None,
    build=False,
    depth=0,
    partial=False,
    trusted=False,
    callbacks=None,
):
    if not isinstance(values, dict) or set(values) - set(shapes):
        raise MCPError("invalid_arguments")
    if not partial and any(row["required"] and name not in values for name, row in shapes.items()):
        raise MCPError("invalid_arguments")
    return {
        name: convert(
            shapes[name]["type"],
            value,
            frames=frames,
            build=build,
            depth=depth + 1,
            trusted=trusted,
            callbacks=callbacks,
        )
        for name, value in values.items()
    }


def settings(
    name, values, *, frames=None, build=False, partial=False, trusted=False, callbacks=None
):
    row = spec(name)
    _json(values)
    if not trusted and name == "json_schema" and "schema" in values:
        from truthound.mcp.schema_guard import validate_json_schema

        validate_json_schema(values["schema"])
    # These inputs can escape the admitted-data model or execute arbitrary
    # Python. They need a separate trusted operator binding, not string tricks.
    forbidden = {
        "query",
        "compare_query",
        "geojson_path",
        "geojson_file",
        "matcher",
        "row_validator",
    }
    if not trusted and any(key in forbidden and value is not None for key, value in values.items()):
        raise MCPError("operator_configuration_required")
    for key, value in values.items():
        if not trusted and "pattern" in key and value is not None:
            # Arbitrary Python regex is not a safe transport grammar. The
            # concrete regular-expression binding is separately audited.
            raise MCPError("operator_configuration_required")
    return fields(
        row["fields"],
        values,
        frames=frames,
        build=build,
        partial=partial,
        trusted=trusted,
        callbacks=callbacks,
    )


def references(name, values, *, kind_wanted, trusted=False, callbacks=None, partial=False):
    """Type-directed references; data dictionaries never impersonate capabilities."""
    found = set()

    def walk(shape, value):
        kind = shape["kind"]
        if (
            kind == kind_wanted == "frame"
            and isinstance(value, dict)
            and set(value) == {"asset_id"}
        ):
            found.add(value["asset_id"])
        elif kind == kind_wanted == "callback":
            found.add(value["operator_capability"])
        elif kind == "union":
            for item in shape["items"]:
                try:
                    convert(item, value, trusted=trusted, callbacks=callbacks)
                except MCPError:
                    continue
                walk(item, value)
                break
        elif kind in {"list", "set", "tuple_list"}:
            for item in value:
                walk(shape["item"], item)
        elif kind == "tuple":
            for child, item in zip(shape["items"], value, strict=True):
                walk(child, item)
        elif kind == "map":
            if shape["key"]["kind"] == "str":
                for item in value.values():
                    walk(shape["value"], item)
            else:
                for item in value["entries"]:
                    walk(shape["key"], item["key"])
                    walk(shape["value"], item["value"])
        elif kind == "record":
            for key, item in value.items():
                walk(record_fields(shape)[key]["type"], item)

    settings(name, values, trusted=trusted, callbacks=callbacks, partial=partial)
    for key, value in values.items():
        walk(spec(name)["fields"][key]["type"], value)
    return found


def frame_handles(name, values, **kwargs):
    return references(name, values, kind_wanted="frame", **kwargs)


def callback_handles(name, values, **kwargs):
    return references(name, values, kind_wanted="callback", **kwargs)
