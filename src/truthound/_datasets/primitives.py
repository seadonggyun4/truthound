"""Typed private dataset repository contract primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Self

from truthound._datasets._serialization import (
    DatasetArtifactContractError,
    coerce_enum,
    normalize_enum_tuple,
    normalize_mapping_tuple,
    normalize_str_tuple,
    parse_datetime,
    require_field,
    require_mapping,
    require_non_empty_str,
    require_non_negative_int,
    utc_now,
)
from truthound._datasets.contracts import (
    TRUTHOUND_DATASET_FINGERPRINT_POLICY_VERSION,
    TRUTHOUND_DATASET_SAMPLING_POLICY_VERSION,
    DatasetArtifactEnvelope,
)
from truthound._datasets.redaction import assert_dataset_artifact_safe

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime


class DatasetArtifactType(StrEnum):
    ASSET_MANIFEST = "dataset_asset_manifest"
    SNAPSHOT_MANIFEST = "dataset_snapshot_manifest"
    FINGERPRINT = "dataset_fingerprint"
    DIFF = "dataset_diff"
    QUALITY_GATE_RESULT = "quality_gate_result"
    SNAPSHOT_BUNDLE = "dataset_snapshot_bundle"
    DIFF_BUNDLE = "dataset_diff_bundle"
    QUALITY_GATE_BUNDLE = "quality_gate_bundle"
    EVIDENCE_INPUT_PAYLOAD = "dataset_evidence_input_payload"


class DatasetAssetType(StrEnum):
    FILE_BUNDLE = "file_bundle"
    TABLE_SNAPSHOT = "table_snapshot"
    OBJECT_PREFIX = "object_prefix"
    RAG_DOCUMENT_BUNDLE = "rag_document_bundle"
    ML_EVAL_DATASET = "ml_eval_dataset"
    GENERIC_DATASET = "generic_dataset"


class DatasetDiffCategory(StrEnum):
    SCHEMA_ADDED = "schema_added"
    SCHEMA_REMOVED = "schema_removed"
    SCHEMA_CHANGED = "schema_changed"
    ROW_COUNT_CHANGED = "row_count_changed"
    NULL_PROFILE_CHANGED = "null_profile_changed"
    SAMPLE_CHANGED = "sample_changed"
    PROFILE_CHANGED = "profile_changed"
    UNKNOWN_CHANGED = "unknown_changed"


class QualityGateStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    ERROR = "error"


class QualityGateType(StrEnum):
    UPLOAD = "upload"
    BRANCH = "branch"
    MERGE = "merge"
    RELEASE = "release"
    ROLLBACK = "rollback"


def _metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    return require_mapping(value, field_name="metadata")


def _envelope_payload(
    envelope: DatasetArtifactEnvelope,
    *,
    artifact_type: DatasetArtifactType,
) -> dict[str, Any]:
    if envelope.artifact_type != artifact_type.value:
        raise DatasetArtifactContractError(
            "Dataset artifact envelope type mismatch: "
            f"expected {artifact_type.value!r}, got {envelope.artifact_type!r}"
        )
    return require_mapping(envelope.payload, field_name="payload")


@dataclass(frozen=True)
class DatasetAssetManifest:
    asset_id: str
    asset_name: str
    asset_type: DatasetAssetType | str
    logical_path: str
    source_kind: str
    schema_ref: str
    content_ref: str
    created_at: datetime = field(default_factory=utc_now)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "asset_id",
            require_non_empty_str(self.asset_id, field_name="asset_id"),
        )
        object.__setattr__(
            self,
            "asset_name",
            require_non_empty_str(self.asset_name, field_name="asset_name"),
        )
        object.__setattr__(
            self,
            "asset_type",
            coerce_enum(DatasetAssetType, self.asset_type, field_name="asset_type"),
        )
        object.__setattr__(
            self,
            "logical_path",
            require_non_empty_str(self.logical_path, field_name="logical_path"),
        )
        object.__setattr__(
            self,
            "source_kind",
            require_non_empty_str(self.source_kind, field_name="source_kind"),
        )
        object.__setattr__(
            self,
            "schema_ref",
            require_non_empty_str(self.schema_ref, field_name="schema_ref"),
        )
        object.__setattr__(
            self,
            "content_ref",
            require_non_empty_str(self.content_ref, field_name="content_ref"),
        )
        object.__setattr__(
            self,
            "created_at",
            parse_datetime(self.created_at, field_name="created_at"),
        )
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        self.validate()

    def validate(self) -> None:
        assert_dataset_artifact_safe(self.to_dict(redaction_safe=False), label="asset manifest")

    def to_dict(self, *, redaction_safe: bool = True) -> dict[str, Any]:
        payload = {
            "asset_id": self.asset_id,
            "asset_name": self.asset_name,
            "asset_type": self.asset_type.value,
            "logical_path": self.logical_path,
            "source_kind": self.source_kind,
            "schema_ref": self.schema_ref,
            "content_ref": self.content_ref,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }
        if redaction_safe:
            assert_dataset_artifact_safe(payload, label="asset manifest")
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        payload = require_mapping(data, field_name="asset manifest")
        return cls(
            asset_id=require_non_empty_str(
                require_field(payload, "asset_id"), field_name="asset_id"
            ),
            asset_name=require_non_empty_str(
                require_field(payload, "asset_name"), field_name="asset_name"
            ),
            asset_type=require_field(payload, "asset_type"),
            logical_path=require_non_empty_str(
                require_field(payload, "logical_path"), field_name="logical_path"
            ),
            source_kind=require_non_empty_str(
                require_field(payload, "source_kind"), field_name="source_kind"
            ),
            schema_ref=require_non_empty_str(
                require_field(payload, "schema_ref"), field_name="schema_ref"
            ),
            content_ref=require_non_empty_str(
                require_field(payload, "content_ref"), field_name="content_ref"
            ),
            created_at=parse_datetime(
                require_field(payload, "created_at"), field_name="created_at"
            ),
            metadata=require_mapping(payload.get("metadata", {}), field_name="metadata"),
        )

    def to_envelope(self) -> DatasetArtifactEnvelope:
        return DatasetArtifactEnvelope(
            artifact_type=DatasetArtifactType.ASSET_MANIFEST.value,
            payload=self.to_dict(),
        )

    @classmethod
    def from_envelope(cls, envelope: DatasetArtifactEnvelope) -> Self:
        return cls.from_dict(
            _envelope_payload(envelope, artifact_type=DatasetArtifactType.ASSET_MANIFEST)
        )


@dataclass(frozen=True)
class DatasetSnapshotManifest:
    snapshot_id: str
    asset_id: str
    parent_snapshot_id: str | None
    fingerprint: str
    schema_fingerprint: str
    profile_fingerprint: str
    row_count: int
    column_count: int
    size_bytes: int
    created_at: datetime
    created_by: str
    validation_refs: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "snapshot_id", require_non_empty_str(self.snapshot_id, field_name="snapshot_id")
        )
        object.__setattr__(
            self, "asset_id", require_non_empty_str(self.asset_id, field_name="asset_id")
        )
        if self.parent_snapshot_id is not None:
            object.__setattr__(
                self,
                "parent_snapshot_id",
                require_non_empty_str(self.parent_snapshot_id, field_name="parent_snapshot_id"),
            )
        object.__setattr__(
            self, "fingerprint", require_non_empty_str(self.fingerprint, field_name="fingerprint")
        )
        object.__setattr__(
            self,
            "schema_fingerprint",
            require_non_empty_str(self.schema_fingerprint, field_name="schema_fingerprint"),
        )
        object.__setattr__(
            self,
            "profile_fingerprint",
            require_non_empty_str(self.profile_fingerprint, field_name="profile_fingerprint"),
        )
        object.__setattr__(
            self, "row_count", require_non_negative_int(self.row_count, field_name="row_count")
        )
        object.__setattr__(
            self,
            "column_count",
            require_non_negative_int(self.column_count, field_name="column_count"),
        )
        object.__setattr__(
            self, "size_bytes", require_non_negative_int(self.size_bytes, field_name="size_bytes")
        )
        object.__setattr__(
            self, "created_at", parse_datetime(self.created_at, field_name="created_at")
        )
        object.__setattr__(
            self, "created_by", require_non_empty_str(self.created_by, field_name="created_by")
        )
        object.__setattr__(
            self,
            "validation_refs",
            normalize_str_tuple(self.validation_refs, field_name="validation_refs"),
        )
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        self.validate()

    def validate(self) -> None:
        assert_dataset_artifact_safe(self.to_dict(redaction_safe=False), label="snapshot manifest")

    def to_dict(self, *, redaction_safe: bool = True) -> dict[str, Any]:
        payload = {
            "snapshot_id": self.snapshot_id,
            "asset_id": self.asset_id,
            "parent_snapshot_id": self.parent_snapshot_id,
            "fingerprint": self.fingerprint,
            "schema_fingerprint": self.schema_fingerprint,
            "profile_fingerprint": self.profile_fingerprint,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "validation_refs": list(self.validation_refs),
            "metadata": dict(self.metadata),
        }
        if redaction_safe:
            assert_dataset_artifact_safe(payload, label="snapshot manifest")
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        payload = require_mapping(data, field_name="snapshot manifest")
        parent = payload.get("parent_snapshot_id")
        return cls(
            snapshot_id=require_non_empty_str(
                require_field(payload, "snapshot_id"), field_name="snapshot_id"
            ),
            asset_id=require_non_empty_str(
                require_field(payload, "asset_id"), field_name="asset_id"
            ),
            parent_snapshot_id=str(parent) if parent is not None else None,
            fingerprint=require_non_empty_str(
                require_field(payload, "fingerprint"), field_name="fingerprint"
            ),
            schema_fingerprint=require_non_empty_str(
                require_field(payload, "schema_fingerprint"), field_name="schema_fingerprint"
            ),
            profile_fingerprint=require_non_empty_str(
                require_field(payload, "profile_fingerprint"), field_name="profile_fingerprint"
            ),
            row_count=require_non_negative_int(
                require_field(payload, "row_count"), field_name="row_count"
            ),
            column_count=require_non_negative_int(
                require_field(payload, "column_count"), field_name="column_count"
            ),
            size_bytes=require_non_negative_int(
                require_field(payload, "size_bytes"), field_name="size_bytes"
            ),
            created_at=parse_datetime(
                require_field(payload, "created_at"), field_name="created_at"
            ),
            created_by=require_non_empty_str(
                require_field(payload, "created_by"), field_name="created_by"
            ),
            validation_refs=normalize_str_tuple(
                require_field(payload, "validation_refs"), field_name="validation_refs"
            ),
            metadata=require_mapping(payload.get("metadata", {}), field_name="metadata"),
        )

    def to_envelope(self) -> DatasetArtifactEnvelope:
        return DatasetArtifactEnvelope(
            artifact_type=DatasetArtifactType.SNAPSHOT_MANIFEST.value,
            payload=self.to_dict(),
        )

    @classmethod
    def from_envelope(cls, envelope: DatasetArtifactEnvelope) -> Self:
        return cls.from_dict(
            _envelope_payload(envelope, artifact_type=DatasetArtifactType.SNAPSHOT_MANIFEST)
        )


@dataclass(frozen=True)
class DatasetFingerprint:
    schema_hash: str
    column_list_hash: str
    row_count: int
    null_profile_hash: str
    sampled_row_hash: str
    content_checksum: str | None = None
    fingerprint_policy_version: str = TRUTHOUND_DATASET_FINGERPRINT_POLICY_VERSION
    sampling_policy_version: str = TRUTHOUND_DATASET_SAMPLING_POLICY_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "schema_hash", require_non_empty_str(self.schema_hash, field_name="schema_hash")
        )
        object.__setattr__(
            self,
            "column_list_hash",
            require_non_empty_str(self.column_list_hash, field_name="column_list_hash"),
        )
        object.__setattr__(
            self, "row_count", require_non_negative_int(self.row_count, field_name="row_count")
        )
        object.__setattr__(
            self,
            "null_profile_hash",
            require_non_empty_str(self.null_profile_hash, field_name="null_profile_hash"),
        )
        object.__setattr__(
            self,
            "sampled_row_hash",
            require_non_empty_str(self.sampled_row_hash, field_name="sampled_row_hash"),
        )
        if self.content_checksum is not None:
            object.__setattr__(
                self,
                "content_checksum",
                require_non_empty_str(self.content_checksum, field_name="content_checksum"),
            )
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        self.validate()

    def validate(self) -> None:
        if self.fingerprint_policy_version != TRUTHOUND_DATASET_FINGERPRINT_POLICY_VERSION:
            raise DatasetArtifactContractError(
                f"Dataset fingerprint policy version mismatch: {self.fingerprint_policy_version!r}"
            )
        if self.sampling_policy_version != TRUTHOUND_DATASET_SAMPLING_POLICY_VERSION:
            raise DatasetArtifactContractError(
                f"Dataset sampling policy version mismatch: {self.sampling_policy_version!r}"
            )
        assert_dataset_artifact_safe(
            self.to_dict(redaction_safe=False), label="dataset fingerprint"
        )

    def to_dict(self, *, redaction_safe: bool = True) -> dict[str, Any]:
        payload = {
            "schema_hash": self.schema_hash,
            "column_list_hash": self.column_list_hash,
            "row_count": self.row_count,
            "null_profile_hash": self.null_profile_hash,
            "sampled_row_hash": self.sampled_row_hash,
            "content_checksum": self.content_checksum,
            "fingerprint_policy_version": self.fingerprint_policy_version,
            "sampling_policy_version": self.sampling_policy_version,
            "metadata": dict(self.metadata),
        }
        if redaction_safe:
            assert_dataset_artifact_safe(payload, label="dataset fingerprint")
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        payload = require_mapping(data, field_name="dataset fingerprint")
        checksum = payload.get("content_checksum")
        return cls(
            schema_hash=require_non_empty_str(
                require_field(payload, "schema_hash"), field_name="schema_hash"
            ),
            column_list_hash=require_non_empty_str(
                require_field(payload, "column_list_hash"), field_name="column_list_hash"
            ),
            row_count=require_non_negative_int(
                require_field(payload, "row_count"), field_name="row_count"
            ),
            null_profile_hash=require_non_empty_str(
                require_field(payload, "null_profile_hash"), field_name="null_profile_hash"
            ),
            sampled_row_hash=require_non_empty_str(
                require_field(payload, "sampled_row_hash"), field_name="sampled_row_hash"
            ),
            content_checksum=str(checksum) if checksum is not None else None,
            fingerprint_policy_version=str(require_field(payload, "fingerprint_policy_version")),
            sampling_policy_version=str(require_field(payload, "sampling_policy_version")),
            metadata=require_mapping(payload.get("metadata", {}), field_name="metadata"),
        )

    def to_envelope(self) -> DatasetArtifactEnvelope:
        return DatasetArtifactEnvelope(
            artifact_type=DatasetArtifactType.FINGERPRINT.value,
            payload=self.to_dict(),
            fingerprint_policy_version=self.fingerprint_policy_version,
            sampling_policy_version=self.sampling_policy_version,
        )

    @classmethod
    def from_envelope(cls, envelope: DatasetArtifactEnvelope) -> Self:
        return cls.from_dict(
            _envelope_payload(envelope, artifact_type=DatasetArtifactType.FINGERPRINT)
        )


@dataclass(frozen=True)
class DatasetDiff:
    source_snapshot_id: str
    target_snapshot_id: str
    categories: tuple[DatasetDiffCategory | str, ...]
    summary: Mapping[str, Any]
    details: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_snapshot_id",
            require_non_empty_str(self.source_snapshot_id, field_name="source_snapshot_id"),
        )
        object.__setattr__(
            self,
            "target_snapshot_id",
            require_non_empty_str(self.target_snapshot_id, field_name="target_snapshot_id"),
        )
        object.__setattr__(
            self,
            "categories",
            normalize_enum_tuple(DatasetDiffCategory, self.categories, field_name="categories"),
        )
        object.__setattr__(self, "summary", require_mapping(self.summary, field_name="summary"))
        object.__setattr__(self, "details", require_mapping(self.details, field_name="details"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        self.validate()

    def validate(self) -> None:
        assert_dataset_artifact_safe(self.to_dict(redaction_safe=False), label="dataset diff")

    def to_dict(self, *, redaction_safe: bool = True) -> dict[str, Any]:
        payload = {
            "source_snapshot_id": self.source_snapshot_id,
            "target_snapshot_id": self.target_snapshot_id,
            "categories": [category.value for category in self.categories],
            "summary": dict(self.summary),
            "details": dict(self.details),
            "metadata": dict(self.metadata),
        }
        if redaction_safe:
            assert_dataset_artifact_safe(payload, label="dataset diff")
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        payload = require_mapping(data, field_name="dataset diff")
        return cls(
            source_snapshot_id=require_non_empty_str(
                require_field(payload, "source_snapshot_id"), field_name="source_snapshot_id"
            ),
            target_snapshot_id=require_non_empty_str(
                require_field(payload, "target_snapshot_id"), field_name="target_snapshot_id"
            ),
            categories=normalize_enum_tuple(
                DatasetDiffCategory, require_field(payload, "categories"), field_name="categories"
            ),
            summary=require_mapping(require_field(payload, "summary"), field_name="summary"),
            details=require_mapping(require_field(payload, "details"), field_name="details"),
            metadata=require_mapping(payload.get("metadata", {}), field_name="metadata"),
        )

    def to_envelope(self) -> DatasetArtifactEnvelope:
        return DatasetArtifactEnvelope(
            artifact_type=DatasetArtifactType.DIFF.value,
            payload=self.to_dict(),
        )

    @classmethod
    def from_envelope(cls, envelope: DatasetArtifactEnvelope) -> Self:
        return cls.from_dict(_envelope_payload(envelope, artifact_type=DatasetArtifactType.DIFF))


@dataclass(frozen=True)
class QualityGateResult:
    quality_gate_id: str
    gate_type: QualityGateType | str
    status: QualityGateStatus | str
    asset_id: str
    snapshot_id: str
    suite_ref: str
    run_ref: str
    summary: Mapping[str, Any]
    blocking_failures: tuple[Mapping[str, Any], ...]
    warnings: tuple[Mapping[str, Any], ...]
    created_at: datetime = field(default_factory=utc_now)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "quality_gate_id",
            require_non_empty_str(self.quality_gate_id, field_name="quality_gate_id"),
        )
        object.__setattr__(
            self, "gate_type", coerce_enum(QualityGateType, self.gate_type, field_name="gate_type")
        )
        object.__setattr__(
            self, "status", coerce_enum(QualityGateStatus, self.status, field_name="status")
        )
        object.__setattr__(
            self, "asset_id", require_non_empty_str(self.asset_id, field_name="asset_id")
        )
        object.__setattr__(
            self, "snapshot_id", require_non_empty_str(self.snapshot_id, field_name="snapshot_id")
        )
        object.__setattr__(
            self, "suite_ref", require_non_empty_str(self.suite_ref, field_name="suite_ref")
        )
        object.__setattr__(
            self, "run_ref", require_non_empty_str(self.run_ref, field_name="run_ref")
        )
        object.__setattr__(self, "summary", require_mapping(self.summary, field_name="summary"))
        object.__setattr__(
            self,
            "blocking_failures",
            normalize_mapping_tuple(self.blocking_failures, field_name="blocking_failures"),
        )
        object.__setattr__(
            self,
            "warnings",
            normalize_mapping_tuple(self.warnings, field_name="warnings"),
        )
        object.__setattr__(
            self, "created_at", parse_datetime(self.created_at, field_name="created_at")
        )
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        self.validate()

    def validate(self) -> None:
        assert_dataset_artifact_safe(
            self.to_dict(redaction_safe=False), label="quality gate result"
        )

    def to_dict(self, *, redaction_safe: bool = True) -> dict[str, Any]:
        payload = {
            "quality_gate_id": self.quality_gate_id,
            "gate_type": self.gate_type.value,
            "status": self.status.value,
            "asset_id": self.asset_id,
            "snapshot_id": self.snapshot_id,
            "suite_ref": self.suite_ref,
            "run_ref": self.run_ref,
            "summary": dict(self.summary),
            "blocking_failures": [dict(item) for item in self.blocking_failures],
            "warnings": [dict(item) for item in self.warnings],
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }
        if redaction_safe:
            assert_dataset_artifact_safe(payload, label="quality gate result")
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        payload = require_mapping(data, field_name="quality gate result")
        return cls(
            quality_gate_id=require_non_empty_str(
                require_field(payload, "quality_gate_id"), field_name="quality_gate_id"
            ),
            gate_type=require_field(payload, "gate_type"),
            status=require_field(payload, "status"),
            asset_id=require_non_empty_str(
                require_field(payload, "asset_id"), field_name="asset_id"
            ),
            snapshot_id=require_non_empty_str(
                require_field(payload, "snapshot_id"), field_name="snapshot_id"
            ),
            suite_ref=require_non_empty_str(
                require_field(payload, "suite_ref"), field_name="suite_ref"
            ),
            run_ref=require_non_empty_str(require_field(payload, "run_ref"), field_name="run_ref"),
            summary=require_mapping(require_field(payload, "summary"), field_name="summary"),
            blocking_failures=normalize_mapping_tuple(
                require_field(payload, "blocking_failures"),
                field_name="blocking_failures",
            ),
            warnings=normalize_mapping_tuple(
                require_field(payload, "warnings"), field_name="warnings"
            ),
            created_at=parse_datetime(
                require_field(payload, "created_at"), field_name="created_at"
            ),
            metadata=require_mapping(payload.get("metadata", {}), field_name="metadata"),
        )

    def to_envelope(self) -> DatasetArtifactEnvelope:
        return DatasetArtifactEnvelope(
            artifact_type=DatasetArtifactType.QUALITY_GATE_RESULT.value,
            payload=self.to_dict(),
        )

    @classmethod
    def from_envelope(cls, envelope: DatasetArtifactEnvelope) -> Self:
        return cls.from_dict(
            _envelope_payload(envelope, artifact_type=DatasetArtifactType.QUALITY_GATE_RESULT)
        )


__all__ = [
    "DatasetArtifactType",
    "DatasetAssetManifest",
    "DatasetAssetType",
    "DatasetDiff",
    "DatasetDiffCategory",
    "DatasetFingerprint",
    "DatasetSnapshotManifest",
    "QualityGateResult",
    "QualityGateStatus",
    "QualityGateType",
]
