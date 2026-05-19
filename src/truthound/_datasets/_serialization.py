"""Serialization helpers for private dataset repository contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, TypeVar


class DatasetArtifactContractError(ValueError):
    """Raised when a dataset artifact violates the bootstrap contract."""


class DatasetArtifactVersionError(DatasetArtifactContractError):
    """Raised when a dataset artifact or policy version is unsupported."""


EnumT = TypeVar("EnumT", bound=StrEnum)


def utc_now() -> datetime:
    return datetime.now(UTC)


def parse_datetime(value: datetime | str | None, *, field_name: str) -> datetime:
    if value is None:
        return utc_now()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise DatasetArtifactContractError(
            f"Invalid dataset artifact {field_name} value: {value!r}"
        ) from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def require_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetArtifactContractError(f"Dataset artifact {field_name} must be a mapping")
    return dict(value)


def require_field(data: Mapping[str, Any], field_name: str) -> Any:
    if field_name not in data:
        raise DatasetArtifactContractError(
            f"Dataset artifact envelope missing required field: {field_name}"
        )
    return data[field_name]


def require_non_empty_str(value: Any, *, field_name: str) -> str:
    text = str(value)
    if not text:
        raise DatasetArtifactContractError(f"Dataset artifact {field_name} must be non-empty")
    return text


def require_non_negative_int(value: Any, *, field_name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise DatasetArtifactContractError(
            f"Dataset artifact {field_name} must be an integer"
        ) from exc
    if number < 0:
        raise DatasetArtifactContractError(f"Dataset artifact {field_name} must be non-negative")
    return number


def normalize_str_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items: Iterable[Any] = (value,)
    elif isinstance(value, Iterable):
        items = value
    else:
        raise DatasetArtifactContractError(
            f"Dataset artifact {field_name} must be a string sequence"
        )

    normalized = tuple(str(item) for item in items)
    if any(not item for item in normalized):
        raise DatasetArtifactContractError(
            f"Dataset artifact {field_name} cannot contain empty values"
        )
    return normalized


def normalize_mapping_tuple(
    value: Any,
    *,
    field_name: str,
) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, (Mapping, str)):
        raise DatasetArtifactContractError(
            f"Dataset artifact {field_name} must be a sequence of mappings"
        )
    if not isinstance(value, Iterable):
        raise DatasetArtifactContractError(
            f"Dataset artifact {field_name} must be a sequence of mappings"
        )
    return tuple(require_mapping(item, field_name=f"{field_name}[]") for item in value)


def coerce_enum(enum_cls: type[EnumT], value: Any, *, field_name: str) -> EnumT:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value))
    except ValueError as exc:
        raise DatasetArtifactContractError(
            f"Dataset artifact {field_name} has unsupported value: {value!r}"
        ) from exc


def normalize_enum_tuple(
    enum_cls: type[EnumT],
    value: Any,
    *,
    field_name: str,
) -> tuple[EnumT, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, enum_cls)):
        values: Iterable[Any] = (value,)
    elif isinstance(value, Iterable):
        values = value
    else:
        raise DatasetArtifactContractError(
            f"Dataset artifact {field_name} must be an enum sequence"
        )
    return tuple(coerce_enum(enum_cls, item, field_name=field_name) for item in values)


__all__ = [
    "DatasetArtifactContractError",
    "DatasetArtifactVersionError",
    "coerce_enum",
    "normalize_enum_tuple",
    "normalize_mapping_tuple",
    "normalize_str_tuple",
    "parse_datetime",
    "require_field",
    "require_mapping",
    "require_non_empty_str",
    "require_non_negative_int",
    "utc_now",
]
