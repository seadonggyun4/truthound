"""Deterministic MVP diffing for private dataset repository contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from truthound._datasets._serialization import (
    DatasetArtifactContractError,
    require_mapping,
)
from truthound._datasets.fingerprinting import fingerprint_dataset
from truthound._datasets.primitives import (
    DatasetDiff,
    DatasetDiffCategory,
    DatasetFingerprint,
)


def diff_datasets(
    source: Any,
    target: Any,
    *,
    source_snapshot_id: str,
    target_snapshot_id: str,
    sample_size: int = 128,
    stable_key_columns: Sequence[str] = (),
    include_content_checksum: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> DatasetDiff:
    """Compare two MVP-supported datasets through deterministic fingerprints.

    This intentionally produces a summary-level diff. It does not promise row-level
    diffs, conflict resolution, or merge policy decisions.
    """

    source_fingerprint = fingerprint_dataset(
        source,
        sample_size=sample_size,
        stable_key_columns=stable_key_columns,
        include_content_checksum=include_content_checksum,
    )
    target_fingerprint = fingerprint_dataset(
        target,
        sample_size=sample_size,
        stable_key_columns=stable_key_columns,
        include_content_checksum=include_content_checksum,
    )
    return diff_fingerprints(
        source_fingerprint,
        target_fingerprint,
        source_snapshot_id=source_snapshot_id,
        target_snapshot_id=target_snapshot_id,
        metadata=metadata,
    )


def diff_fingerprints(
    source: DatasetFingerprint,
    target: DatasetFingerprint,
    *,
    source_snapshot_id: str,
    target_snapshot_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> DatasetDiff:
    """Create a deterministic summary diff between two dataset fingerprints."""

    source.validate()
    target.validate()

    schema_detail = _schema_diff(source, target)
    profile_detail = _profile_diff(source, target)
    categories = _categories(source, target, schema_detail, profile_detail)
    summary = {
        "changed": bool(categories),
        "category_count": len(categories),
        "categories": [category.value for category in categories],
        "source_row_count": source.row_count,
        "target_row_count": target.row_count,
        "row_count_delta": target.row_count - source.row_count,
        "schema_added_count": len(schema_detail["added"]),
        "schema_removed_count": len(schema_detail["removed"]),
        "schema_type_changed_count": len(schema_detail["type_changed"]),
        "scope": "summary_fingerprint_mvp",
        "row_level_diff_available": False,
    }
    details = {
        "schema": schema_detail,
        "row_count": {
            "changed": source.row_count != target.row_count,
            "source": source.row_count,
            "target": target.row_count,
            "delta": target.row_count - source.row_count,
        },
        "null_profile": {
            "changed": source.null_profile_hash != target.null_profile_hash,
            "source_hash": source.null_profile_hash,
            "target_hash": target.null_profile_hash,
        },
        "profile": profile_detail,
        "sampled_row_digest": {
            "changed": source.sampled_row_hash != target.sampled_row_hash,
            "source_hash": source.sampled_row_hash,
            "target_hash": target.sampled_row_hash,
        },
        "content_checksum": {
            "available": source.content_checksum is not None
            and target.content_checksum is not None,
            "changed": (
                source.content_checksum is not None
                and target.content_checksum is not None
                and source.content_checksum != target.content_checksum
            ),
            "source_hash": source.content_checksum,
            "target_hash": target.content_checksum,
        },
        "fingerprints": {
            "source": _fingerprint_signature(source),
            "target": _fingerprint_signature(target),
        },
    }
    diff_metadata: dict[str, Any] = {
        "engine": "truthound._datasets.diffing",
        "comparison_level": "summary_fingerprint",
        "row_level_diff_available": False,
    }
    if metadata is not None:
        diff_metadata["user_metadata"] = require_mapping(metadata, field_name="metadata")

    return DatasetDiff(
        source_snapshot_id=source_snapshot_id,
        target_snapshot_id=target_snapshot_id,
        categories=tuple(categories),
        summary=summary,
        details=details,
        metadata=diff_metadata,
    )


def _categories(
    source: DatasetFingerprint,
    target: DatasetFingerprint,
    schema_detail: Mapping[str, Any],
    profile_detail: Mapping[str, Any],
) -> list[DatasetDiffCategory]:
    categories: list[DatasetDiffCategory] = []
    if schema_detail["added"]:
        categories.append(DatasetDiffCategory.SCHEMA_ADDED)
    if schema_detail["removed"]:
        categories.append(DatasetDiffCategory.SCHEMA_REMOVED)
    if schema_detail["type_changed"] or schema_detail["column_order_changed"]:
        categories.append(DatasetDiffCategory.SCHEMA_CHANGED)
    if source.row_count != target.row_count:
        categories.append(DatasetDiffCategory.ROW_COUNT_CHANGED)
    if source.null_profile_hash != target.null_profile_hash:
        categories.append(DatasetDiffCategory.NULL_PROFILE_CHANGED)
    if profile_detail["changed"]:
        categories.append(DatasetDiffCategory.PROFILE_CHANGED)
    if source.sampled_row_hash != target.sampled_row_hash:
        categories.append(DatasetDiffCategory.SAMPLE_CHANGED)
    if not categories and _fingerprint_signature(source) != _fingerprint_signature(target):
        categories.append(DatasetDiffCategory.UNKNOWN_CHANGED)
    return categories


def _schema_diff(
    source: DatasetFingerprint,
    target: DatasetFingerprint,
) -> dict[str, Any]:
    source_items = _schema_items(source)
    target_items = _schema_items(target)
    source_by_name = {item["name"]: item for item in source_items}
    target_by_name = {item["name"]: item for item in target_items}
    source_names = [item["name"] for item in source_items]
    common_names = [name for name in source_names if name in target_by_name]

    added = [
        {"name": item["name"], "dtype": item["dtype"]}
        for item in target_items
        if item["name"] not in source_by_name
    ]
    removed = [
        {"name": item["name"], "dtype": item["dtype"]}
        for item in source_items
        if item["name"] not in target_by_name
    ]
    type_changed = [
        {
            "name": name,
            "source_dtype": source_by_name[name]["dtype"],
            "target_dtype": target_by_name[name]["dtype"],
        }
        for name in common_names
        if source_by_name[name]["dtype"] != target_by_name[name]["dtype"]
    ]
    column_order_changed = (
        source.column_list_hash != target.column_list_hash
        and not added
        and not removed
        and not type_changed
    )
    return {
        "added": added,
        "removed": removed,
        "type_changed": type_changed,
        "column_order_changed": column_order_changed,
        "source_column_count": len(source_items),
        "target_column_count": len(target_items),
    }


def _profile_diff(
    source: DatasetFingerprint,
    target: DatasetFingerprint,
) -> dict[str, Any]:
    source_columns = _profile_columns(source)
    target_columns = _profile_columns(target)
    changed_columns = sorted(
        name
        for name in set(source_columns) & set(target_columns)
        if source_columns[name] != target_columns[name]
    )
    added_columns = sorted(set(target_columns) - set(source_columns))
    removed_columns = sorted(set(source_columns) - set(target_columns))
    return {
        "changed": _profile_hash(source) != _profile_hash(target),
        "source_hash": _profile_hash(source),
        "target_hash": _profile_hash(target),
        "changed_columns": changed_columns,
        "added_columns": added_columns,
        "removed_columns": removed_columns,
    }


def _schema_items(fingerprint: DatasetFingerprint) -> list[dict[str, Any]]:
    metadata = require_mapping(fingerprint.metadata, field_name="fingerprint.metadata")
    raw_schema = metadata.get("schema", ())
    if raw_schema is None:
        return []
    if not isinstance(raw_schema, Sequence) or isinstance(raw_schema, (str, bytes)):
        raise DatasetArtifactContractError("Dataset fingerprint metadata.schema must be a sequence")
    items: list[dict[str, Any]] = []
    for item in raw_schema:
        mapping = require_mapping(item, field_name="fingerprint.metadata.schema[]")
        items.append(
            {
                "name": str(mapping.get("name", "")),
                "dtype": str(mapping.get("dtype", "")),
                "position": int(mapping.get("position", len(items))),
            }
        )
    return sorted(items, key=lambda item: item["position"])


def _profile_columns(fingerprint: DatasetFingerprint) -> dict[str, dict[str, Any]]:
    metadata = require_mapping(fingerprint.metadata, field_name="fingerprint.metadata")
    profile_value = metadata.get("profile", {})
    if profile_value is None:
        return {}
    profile = require_mapping(profile_value, field_name="metadata.profile")
    raw_columns = profile.get("columns", ())
    if raw_columns is None:
        return {}
    if not isinstance(raw_columns, Sequence) or isinstance(raw_columns, (str, bytes)):
        raise DatasetArtifactContractError(
            "Dataset fingerprint metadata.profile.columns must be a sequence"
        )
    columns: dict[str, dict[str, Any]] = {}
    for item in raw_columns:
        mapping = require_mapping(item, field_name="metadata.profile.columns[]")
        name = str(mapping.get("name", ""))
        if name:
            columns[name] = dict(mapping)
    return columns


def _profile_hash(fingerprint: DatasetFingerprint) -> str | None:
    metadata = require_mapping(fingerprint.metadata, field_name="fingerprint.metadata")
    profile_hash = metadata.get("profile_hash")
    return str(profile_hash) if profile_hash is not None else None


def _fingerprint_signature(fingerprint: DatasetFingerprint) -> dict[str, Any]:
    return {
        "schema_hash": fingerprint.schema_hash,
        "column_list_hash": fingerprint.column_list_hash,
        "row_count": fingerprint.row_count,
        "null_profile_hash": fingerprint.null_profile_hash,
        "sampled_row_hash": fingerprint.sampled_row_hash,
        "content_checksum": fingerprint.content_checksum,
    }


__all__ = [
    "diff_datasets",
    "diff_fingerprints",
]
