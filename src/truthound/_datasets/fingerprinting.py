"""Deterministic MVP fingerprinting for private dataset repository contracts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import polars as pl

from truthound._datasets._serialization import (
    DatasetArtifactContractError,
    normalize_str_tuple,
    require_mapping,
    require_non_negative_int,
)
from truthound._datasets.contracts import (
    TRUTHOUND_DATASET_FINGERPRINT_POLICY_VERSION,
    TRUTHOUND_DATASET_SAMPLING_POLICY_VERSION,
)
from truthound._datasets.primitives import DatasetFingerprint


@dataclass(frozen=True)
class DatasetFingerprintOptions:
    """Options controlling deterministic MVP fingerprint generation."""

    sample_size: int = 128
    stable_key_columns: tuple[str, ...] = ()
    include_content_checksum: bool = False

    def __post_init__(self) -> None:
        sample_size = require_non_negative_int(self.sample_size, field_name="sample_size")
        if sample_size == 0:
            raise DatasetArtifactContractError(
                "Dataset artifact sample_size must be greater than zero"
            )
        object.__setattr__(self, "sample_size", sample_size)
        object.__setattr__(
            self,
            "stable_key_columns",
            normalize_str_tuple(self.stable_key_columns, field_name="stable_key_columns"),
        )


def normalize_dataset_input(data: Any) -> pl.DataFrame:
    """Normalize MVP-supported dataset inputs into an eager Polars DataFrame."""

    if isinstance(data, pl.DataFrame):
        return data.clone()
    if isinstance(data, pl.LazyFrame):
        return data.collect()
    if isinstance(data, (str, Path)):
        return _read_local_dataset(Path(data))
    if isinstance(data, Mapping):
        return pl.DataFrame(dict(data))
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        return pl.DataFrame(data)
    if hasattr(data, "columns") and hasattr(data, "to_dict"):
        try:
            return pl.from_pandas(data)
        except Exception as exc:
            raise DatasetArtifactContractError(
                "Dataset artifact input could not be normalized from pandas-like data"
            ) from exc
    raise DatasetArtifactContractError(
        f"Unsupported dataset input type for fingerprinting: {type(data).__name__}"
    )


def fingerprint_dataset(
    data: Any,
    *,
    sample_size: int = 128,
    stable_key_columns: Sequence[str] = (),
    include_content_checksum: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> DatasetFingerprint:
    """Create a deterministic summary fingerprint for an MVP-supported dataset."""

    options = DatasetFingerprintOptions(
        sample_size=sample_size,
        stable_key_columns=tuple(stable_key_columns),
        include_content_checksum=include_content_checksum,
    )
    frame = normalize_dataset_input(data)
    schema_items = _schema_items(frame)
    row_count = frame.height
    profile = _profile_frame(frame)
    profile_hash_payload = {
        **profile,
        "columns": sorted(profile["columns"], key=lambda column: column["name"]),
    }
    null_profile = {
        column["name"]: {
            "null_count": column["null_count"],
            "null_ratio": column["null_ratio"],
        }
        for column in profile["columns"]
    }
    row_hashes = _row_hashes(frame)
    sampled_row_hash = _sampled_row_digest(
        frame,
        row_hashes=row_hashes,
        options=options,
    )
    content_checksum = _digest(row_hashes) if include_content_checksum else None
    profile_hash = _digest(profile_hash_payload)
    metadata_payload = {
        "engine": "truthound._datasets.fingerprinting",
        "schema": schema_items,
        "profile": profile,
        "profile_hash": profile_hash,
        "stable_key_columns": list(options.stable_key_columns),
        "sample_size": options.sample_size,
        "content_checksum_included": include_content_checksum,
    }
    if metadata:
        metadata_payload["user_metadata"] = require_mapping(metadata, field_name="metadata")

    return DatasetFingerprint(
        schema_hash=_digest(sorted(schema_items, key=lambda item: item["name"])),
        column_list_hash=_digest([item["name"] for item in schema_items]),
        row_count=row_count,
        null_profile_hash=_digest(null_profile),
        sampled_row_hash=sampled_row_hash,
        content_checksum=content_checksum,
        fingerprint_policy_version=TRUTHOUND_DATASET_FINGERPRINT_POLICY_VERSION,
        sampling_policy_version=TRUTHOUND_DATASET_SAMPLING_POLICY_VERSION,
        metadata=metadata_payload,
    )


def _read_local_dataset(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise DatasetArtifactContractError(f"Dataset artifact input path does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pl.read_ndjson(path)
    if suffix == ".json":
        return pl.read_json(path)
    raise DatasetArtifactContractError(
        f"Unsupported local dataset input suffix for fingerprinting: {suffix}"
    )


def _schema_items(frame: pl.DataFrame) -> list[dict[str, Any]]:
    schema = frame.schema
    return [
        {"name": name, "dtype": str(dtype), "position": index}
        for index, (name, dtype) in enumerate(schema.items())
    ]


def _profile_frame(frame: pl.DataFrame) -> dict[str, Any]:
    row_count = frame.height
    columns = [
        _profile_column(frame.get_column(name), row_count=row_count) for name in frame.columns
    ]
    return {
        "row_count": row_count,
        "column_count": frame.width,
        "columns": columns,
    }


def _profile_column(series: pl.Series, *, row_count: int) -> dict[str, Any]:
    null_count = series.null_count()
    non_null = series.drop_nulls()
    profile: dict[str, Any] = {
        "name": series.name,
        "dtype": str(series.dtype),
        "null_count": null_count,
        "null_ratio": _safe_ratio(null_count, row_count),
        "distinct_count": series.n_unique(),
    }
    if len(non_null) == 0:
        return profile
    if _is_string_like(series.dtype):
        lengths = non_null.cast(pl.String).str.len_chars()
        profile["min_length"] = int(lengths.min())
        profile["max_length"] = int(lengths.max())
        return profile
    if (
        _is_numeric_like(series.dtype)
        or _is_temporal_like(series.dtype)
        or str(series.dtype) == "Boolean"
    ):
        profile["min"] = _stable_json_value(non_null.min())
        profile["max"] = _stable_json_value(non_null.max())
    return profile


def _is_string_like(dtype: pl.DataType) -> bool:
    return str(dtype) in {"String", "Utf8", "Categorical", "Enum"}


def _is_numeric_like(dtype: pl.DataType) -> bool:
    try:
        return bool(dtype.is_numeric())
    except AttributeError:
        return str(dtype) in {
            "Int8",
            "Int16",
            "Int32",
            "Int64",
            "UInt8",
            "UInt16",
            "UInt32",
            "UInt64",
            "Float32",
            "Float64",
        }


def _is_temporal_like(dtype: pl.DataType) -> bool:
    try:
        return bool(dtype.is_temporal())
    except AttributeError:
        return str(dtype).startswith(("Date", "Datetime", "Time"))


def _safe_ratio(value: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(value / total, 12)


def _row_hashes(frame: pl.DataFrame) -> list[str]:
    return [_digest(row) for row in frame.to_dicts()]


def _sampled_row_digest(
    frame: pl.DataFrame,
    *,
    row_hashes: list[str],
    options: DatasetFingerprintOptions,
) -> str:
    if not row_hashes:
        return _digest([])
    if options.stable_key_columns and all(
        column in frame.columns for column in options.stable_key_columns
    ):
        rows = frame.to_dicts()
        ordered_rows = sorted(
            rows,
            key=lambda row: tuple(
                _stable_json_value(row.get(column)) for column in options.stable_key_columns
            ),
        )
        sampled_hashes = [_digest(row) for row in ordered_rows[: options.sample_size]]
    else:
        sampled_hashes = sorted(row_hashes)[: options.sample_size]
    return _digest(
        {
            "sampling_policy_version": TRUTHOUND_DATASET_SAMPLING_POLICY_VERSION,
            "stable_key_columns": list(options.stable_key_columns),
            "row_hashes": sampled_hashes,
        }
    )


def _digest(value: Any) -> str:
    payload = _stable_json_value(value)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"sha256:{hashlib.sha256(encoded.encode()).hexdigest()}"


def _stable_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _stable_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_json_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_stable_json_value(item) for item in value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if hasattr(value, "item"):
        try:
            return _stable_json_value(value.item())
        except Exception:
            pass
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


__all__ = [
    "DatasetFingerprintOptions",
    "fingerprint_dataset",
    "normalize_dataset_input",
]
