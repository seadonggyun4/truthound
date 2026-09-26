"""Bounded strict JSON loading and explicit, versioned canonical projections."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from enum import Enum

from ._contract import (
    MAX_BYTES,
    MAX_DEPTH,
    MAX_ITEMS,
    MAX_NODES,
    ModelError,
    convert,
    require,
    text,
)
from .models import (
    CatalogSnapshot,
    DesignRevision,
    Document,
    Identifier,
    LogicalName,
    StandardRevision,
    State,
    TypeDescriptor,
)
from .normalization import (
    POLICY_VERSION,
    normalize_identifier,
    normalize_logical_name,
    normalize_type,
)

ROOTS = {"STANDARD": StandardRevision, "DESIGN": DesignRevision, "CATALOG": CatalogSnapshot}
DIGEST_PROFILE = "truthound.schema-audit.semantic/1"


class _Pairs(list):
    pass


def _reject_number(_value):
    raise ModelError("non_integer_number")


def _depth_preflight(data: bytes) -> None:
    depth, quoted, escaped = 0, False, False
    for c in data:
        if quoted:
            if escaped:
                escaped = False
            elif c == 92:
                escaped = True
            elif c == 34:
                quoted = False
        elif c == 34:
            quoted = True
        elif c in (91, 123):
            depth += 1
            require(depth <= MAX_DEPTH, "depth_limit")
        elif c in (93, 125):
            depth -= 1


def _plain(value, pointer: str, budget: list[int]):
    budget[0] -= 1
    require(budget[0] >= 0, "node_limit", pointer)
    if isinstance(value, _Pairs):
        result = {}
        require(len(value) <= MAX_ITEMS, "item_limit", pointer)
        for key, item in value:
            text(key, pointer)
            require(key not in result, "duplicate_key", pointer)
            # Input keys can contain secrets. Only numeric positions appear here;
            # typed decoding later supplies known contract field names.
            result[key] = _plain(item, f"{pointer}/{len(result)}", budget)
        return result
    if type(value) is list:
        require(len(value) <= MAX_ITEMS, "item_limit", pointer)
        return [_plain(v, f"{pointer}/{i}", budget) for i, v in enumerate(value)]
    if type(value) is str:
        text(value, pointer)
    return value


def load_document(data: bytes) -> Document:
    """Accept UTF-8 bytes only. No coercion, duplicate keys, floats or unknown fields."""
    require(type(data) is bytes, "bytes_required")
    require(len(data) <= MAX_BYTES, "byte_limit")
    _depth_preflight(data)
    try:
        parsed = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_Pairs,
            parse_float=_reject_number,
            parse_constant=_reject_number,
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        if isinstance(exc, ModelError):
            raise
        raise ModelError("invalid_json") from None
    obj = _plain(parsed, "", [MAX_NODES])
    require(type(obj) is dict, "object_required")
    kind = obj.get("kind")
    require(type(kind) is str and kind in ROOTS, "invalid_kind", "/kind")
    return convert(obj, ROOTS[kind], "", wire=True)


def _wire(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {f.name: _wire(getattr(value, f.name)) for f in fields(value)}
    if type(value) is tuple:
        return [_wire(item) for item in value]
    return value


def to_dict(model: Document) -> dict:
    require(type(model) in ROOTS.values(), "invalid_document")
    return _wire(model)


def dump_document(model: Document) -> bytes:
    result = json.dumps(
        to_dict(model), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    require(len(result) <= MAX_BYTES, "byte_limit")
    return result


def raw_digest(data: bytes) -> str:
    require(type(data) is bytes, "bytes_required")
    require(len(data) <= MAX_BYTES, "byte_limit")
    return hashlib.sha256(data).hexdigest()


_UNORDERED = {"words", "terms", "domains", "tables", "columns", "constraints"}
_PROVENANCE = {"id", "revision", "sources", "captured_at", "adapter_version", "binding_ref"}


def _semantic(value, dialect, *, root=False, field=""):
    if isinstance(value, Identifier):
        key = normalize_identifier(value, dialect)
        return {"normalized": key.value} if key.state is State.VALUE else _wire(value)
    if isinstance(value, TypeDescriptor):
        result = _wire(value)
        key = normalize_type(value, dialect)
        if key.state is State.VALUE:
            result["native_type"] = key.value
        return result
    if isinstance(value, LogicalName):
        result = _wire(value)
        if value.name.state is State.VALUE:
            result["name"]["value"] = normalize_logical_name(value.name.value)
        return result
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            f.name: _semantic(getattr(value, f.name), dialect, field=f.name)
            for f in fields(value)
            if not (root and f.name in _PROVENANCE)
        }
    if type(value) is tuple:
        if field in _UNORDERED:
            value = sorted(value, key=lambda item: item.id)
        elif field == "coverage":
            value = sorted(value, key=lambda item: item.facet)
        elif field == "scope":
            value = sorted(value, key=lambda item: item.text)
        return [_semantic(item, dialect) for item in value]
    if type(value) is str and field == "logical_name":
        return normalize_logical_name(value)
    return value


def semantic_bytes(model: Document) -> bytes:
    """Versioned domain projection, deliberately NOT RFC 8785/JCS.

    Identity of referenced objects and ordered key/word IDs is preserved.
    Revision envelope identity and source acquisition metadata are excluded.
    """
    require(type(model) in ROOTS.values(), "invalid_document")
    result = {
        "profile": DIGEST_PROFILE,
        "normalization": POLICY_VERSION,
        "model": _semantic(model, model.dialect, root=True),
    }
    return json.dumps(
        result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def semantic_digest(model: Document) -> str:
    return hashlib.sha256(semantic_bytes(model)).hexdigest()
