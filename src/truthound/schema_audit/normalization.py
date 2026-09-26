"""Versioned, conservative normalizers; never parse or execute SQL expressions."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol

from ._contract import ModelError, require, text
from .models import (
    Attribute,
    Dialect,
    Identifier,
    NameOrigin,
    Reason,
    State,
    TypeDescriptor,
    known,
    unknown,
)

POLICY_VERSION = "pg17-ascii-nfc32-logical-v1"


def normalize_logical_name(value: str) -> str:
    require(type(value) is str, "invalid_type")
    text(value, "")
    # NFC is for logical dictionary labels only, never database identifiers.
    # Pin the standard-library database shared by all supported Python versions.
    return unicodedata.ucd_3_2_0.normalize("NFC", value)


class DialectNormalizer(Protocol):
    def identifier(self, value: Identifier) -> Attribute: ...

    def type_name(self, value: TypeDescriptor) -> Attribute: ...


_ALIASES = {
    "int2": "smallint",
    "smallint": "smallint",
    "int4": "integer",
    "int": "integer",
    "integer": "integer",
    "int8": "bigint",
    "bigint": "bigint",
    "decimal": "numeric",
    "numeric": "numeric",
    "bool": "boolean",
    "boolean": "boolean",
    "float4": "real",
    "real": "real",
    "float8": "double precision",
    "double precision": "double precision",
    "varchar": "character varying",
    "character varying": "character varying",
    "bpchar": "character",
    "character": "character",
    "text": "text",
    "bytea": "bytea",
    "date": "date",
    "uuid": "uuid",
    "json": "json",
    "jsonb": "jsonb",
    "timestamp": "timestamp",
    "timestamp without time zone": "timestamp",
    "timestamptz": "timestamp with time zone",
    "timestamp with time zone": "timestamp with time zone",
    "time": "time",
    "time without time zone": "time",
    "timetz": "time with time zone",
    "time with time zone": "time with time zone",
}


@dataclass(frozen=True)
class PostgreSQL17Normalizer:
    """UTF-8 identifiers <=63 bytes; ASCII unquoted folding only.

    native_type is a resolved base-type token, not SQL such as numeric(10,2).
    Unknown/custom types retain their original descriptor outside this key.
    """

    def identifier(self, value: Identifier) -> Attribute:
        require(type(value) is Identifier, "invalid_model")
        if len(value.text.encode("utf-8")) > 63:
            return unknown(Reason.UNSUPPORTED)
        if value.origin is NameOrigin.UNQUOTED:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", value.text):
                return unknown(Reason.UNSUPPORTED)
            return known(value.text.lower())
        return known(value.text)

    def type_name(self, value: TypeDescriptor) -> Attribute:
        require(type(value) is TypeDescriptor, "invalid_model")
        if value.type_schema != known("pg_catalog"):
            return unknown(Reason.UNSUPPORTED)
        if (
            value.domain_ref.state is not State.ABSENT
            or value.enum_values_ref.state is not State.ABSENT
        ):
            return unknown(Reason.UNSUPPORTED)
        name = _ALIASES.get(value.native_type)
        if name is None:
            return unknown(Reason.UNSUPPORTED)
        if name == "numeric":
            if value.precision.state is State.VALUE and value.precision.value > 1000:
                return unknown(Reason.UNSUPPORTED)
            if value.scale.state is State.VALUE and not -1000 <= value.scale.value <= 1000:
                return unknown(Reason.UNSUPPORTED)
        if name.startswith(("time", "timestamp")):
            expected = "with time zone" in name
            if value.timezone.state is State.VALUE and value.timezone.value != expected:
                return unknown(Reason.INCOMPLETE)
        return known(name)


@dataclass(frozen=True)
class PostgreSQL14Normalizer(PostgreSQL17Normalizer):
    """Shared name rules, with pre-PG15 numeric scale restrictions."""

    def type_name(self, value: TypeDescriptor) -> Attribute:
        result = super().type_name(value)
        if (
            result == known("numeric")
            and value.scale.state is State.VALUE
            and (
                value.scale.value < 0
                or (
                    value.precision.state is State.VALUE
                    and value.scale.value > value.precision.value
                )
            )
        ):
            return unknown(Reason.UNSUPPORTED)
        return result


@dataclass(frozen=True)
class NormalizationRegistry:
    """Explicit immutable injection point, not a mutable global plugin loader."""

    entries: tuple[tuple[str, int, DialectNormalizer], ...] = (
        ("postgresql", 14, PostgreSQL14Normalizer()),
        ("postgresql", 17, PostgreSQL17Normalizer()),
    )

    def __post_init__(self) -> None:
        require(type(self.entries) is tuple, "immutable_array_required")
        keys = []
        for entry in self.entries:
            require(type(entry) is tuple and len(entry) == 3, "invalid_registry_entry")
            name, major, handler = entry
            Dialect(name=name, major=major)
            require(
                callable(getattr(handler, "identifier", None))
                and callable(getattr(handler, "type_name", None)),
                "invalid_normalizer",
            )
            keys.append((name, major))
        require(len(keys) == len(set(keys)), "duplicate_dialect")

    def resolve(self, dialect: Dialect) -> DialectNormalizer:
        require(type(dialect) is Dialect, "invalid_model")
        for name, major, handler in self.entries:
            if (name, major) == (dialect.name, dialect.major):
                return handler
        raise ModelError("unsupported_dialect")


def normalize_identifier(value: Identifier, dialect: Dialect) -> Attribute:
    try:
        return NormalizationRegistry().resolve(dialect).identifier(value)
    except ModelError as exc:
        if exc.code != "unsupported_dialect":
            raise
        return unknown(Reason.UNSUPPORTED)


def normalize_type(value: TypeDescriptor, dialect: Dialect) -> Attribute:
    """Return only the normalized base name; all original facets remain authoritative."""
    try:
        return NormalizationRegistry().resolve(dialect).type_name(value)
    except ModelError as exc:
        if exc.code != "unsupported_dialect":
            raise
        return unknown(Reason.UNSUPPORTED)
