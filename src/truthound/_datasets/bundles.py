"""Private machine-readable artifact bundles for dataset repository contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

from truthound._datasets._serialization import (
    DatasetArtifactContractError,
    normalize_mapping_tuple,
    normalize_str_tuple,
    require_field,
    require_mapping,
    require_non_empty_str,
)
from truthound._datasets.contracts import DatasetArtifactEnvelope
from truthound._datasets.primitives import (
    DatasetArtifactType,
    DatasetAssetManifest,
    DatasetDiff,
    DatasetFingerprint,
    DatasetSnapshotManifest,
    QualityGateResult,
)
from truthound._datasets.redaction import assert_dataset_artifact_safe

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class DatasetSnapshotBundle:
    """Versioned snapshot artifact shared by Depot Console and Orchestration."""

    snapshot_manifest: DatasetSnapshotManifest | Mapping[str, Any]
    fingerprint: DatasetFingerprint | Mapping[str, Any]
    asset_manifest: DatasetAssetManifest | Mapping[str, Any] | None = None
    profile_summary: Mapping[str, Any] = field(default_factory=dict)
    validation_refs: tuple[str, ...] = ()
    content_refs: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "snapshot_manifest",
            _coerce_snapshot_manifest(self.snapshot_manifest),
        )
        object.__setattr__(self, "fingerprint", _coerce_fingerprint(self.fingerprint))
        if self.asset_manifest is not None:
            object.__setattr__(
                self,
                "asset_manifest",
                _coerce_asset_manifest(self.asset_manifest),
            )
        object.__setattr__(
            self,
            "profile_summary",
            require_mapping(self.profile_summary, field_name="profile_summary"),
        )
        object.__setattr__(
            self,
            "validation_refs",
            normalize_str_tuple(self.validation_refs, field_name="validation_refs"),
        )
        object.__setattr__(
            self,
            "content_refs",
            require_mapping(self.content_refs, field_name="content_refs"),
        )
        object.__setattr__(self, "metadata", require_mapping(self.metadata, field_name="metadata"))
        self.validate()

    def validate(self) -> None:
        assert_dataset_artifact_safe(
            self.to_dict(redaction_safe=False),
            label="dataset snapshot bundle",
        )

    def to_dict(self, *, redaction_safe: bool = True) -> dict[str, Any]:
        payload = {
            "snapshot_manifest": self.snapshot_manifest.to_dict(),
            "fingerprint": self.fingerprint.to_dict(),
            "asset_manifest": self.asset_manifest.to_dict() if self.asset_manifest else None,
            "profile_summary": dict(self.profile_summary),
            "validation_refs": list(self.validation_refs),
            "content_refs": dict(self.content_refs),
            "metadata": dict(self.metadata),
        }
        if redaction_safe:
            assert_dataset_artifact_safe(payload, label="dataset snapshot bundle")
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        payload = require_mapping(data, field_name="dataset snapshot bundle")
        asset_manifest = payload.get("asset_manifest")
        return cls(
            snapshot_manifest=require_mapping(
                require_field(payload, "snapshot_manifest"),
                field_name="snapshot_manifest",
            ),
            fingerprint=require_mapping(
                require_field(payload, "fingerprint"), field_name="fingerprint"
            ),
            asset_manifest=require_mapping(asset_manifest, field_name="asset_manifest")
            if asset_manifest is not None
            else None,
            profile_summary=require_mapping(
                payload.get("profile_summary", {}),
                field_name="profile_summary",
            ),
            validation_refs=normalize_str_tuple(
                payload.get("validation_refs", ()),
                field_name="validation_refs",
            ),
            content_refs=require_mapping(
                payload.get("content_refs", {}), field_name="content_refs"
            ),
            metadata=require_mapping(payload.get("metadata", {}), field_name="metadata"),
        )

    def to_envelope(self) -> DatasetArtifactEnvelope:
        return DatasetArtifactEnvelope(
            artifact_type=DatasetArtifactType.SNAPSHOT_BUNDLE.value,
            payload=self.to_dict(),
            fingerprint_policy_version=self.fingerprint.fingerprint_policy_version,
            sampling_policy_version=self.fingerprint.sampling_policy_version,
        )

    @classmethod
    def from_envelope(cls, envelope: DatasetArtifactEnvelope) -> Self:
        return cls.from_dict(
            _envelope_payload(envelope, artifact_type=DatasetArtifactType.SNAPSHOT_BUNDLE)
        )

    def to_json(self, *, indent: int | None = None) -> str:
        return _to_envelope_json(self.to_envelope(), indent=indent)

    @classmethod
    def from_json(cls, data: str) -> Self:
        return cls.from_envelope(_envelope_from_json(data))


@dataclass(frozen=True)
class DatasetDiffBundle:
    """Summary-level diff bundle that deliberately excludes row-level diff state."""

    diff: DatasetDiff | Mapping[str, Any]
    source_snapshot_ref: str
    target_snapshot_ref: str
    schema_diff: Mapping[str, Any] = field(default_factory=dict)
    profile_diff: Mapping[str, Any] = field(default_factory=dict)
    digest_diff: Mapping[str, Any] = field(default_factory=dict)
    risk_flags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        diff = _coerce_diff(self.diff)
        object.__setattr__(self, "diff", diff)
        object.__setattr__(
            self,
            "source_snapshot_ref",
            require_non_empty_str(self.source_snapshot_ref, field_name="source_snapshot_ref"),
        )
        object.__setattr__(
            self,
            "target_snapshot_ref",
            require_non_empty_str(self.target_snapshot_ref, field_name="target_snapshot_ref"),
        )
        details = require_mapping(diff.details, field_name="diff.details")
        object.__setattr__(
            self,
            "schema_diff",
            _mapping_or_detail(self.schema_diff, details, "schema", "schema_diff"),
        )
        object.__setattr__(
            self,
            "profile_diff",
            _mapping_or_detail(self.profile_diff, details, "profile", "profile_diff"),
        )
        object.__setattr__(
            self,
            "digest_diff",
            _digest_detail(self.digest_diff, details),
        )
        object.__setattr__(
            self,
            "risk_flags",
            normalize_str_tuple(self.risk_flags, field_name="risk_flags"),
        )
        object.__setattr__(self, "metadata", require_mapping(self.metadata, field_name="metadata"))
        self.validate()

    @classmethod
    def from_diff(
        cls,
        diff: DatasetDiff,
        *,
        risk_flags: tuple[str, ...] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> Self:
        return cls(
            diff=diff,
            source_snapshot_ref=diff.source_snapshot_id,
            target_snapshot_ref=diff.target_snapshot_id,
            risk_flags=risk_flags,
            metadata=metadata or {},
        )

    def validate(self) -> None:
        assert_dataset_artifact_safe(
            self.to_dict(redaction_safe=False),
            label="dataset diff bundle",
        )

    def to_dict(self, *, redaction_safe: bool = True) -> dict[str, Any]:
        payload = {
            "diff": self.diff.to_dict(),
            "source_snapshot_ref": self.source_snapshot_ref,
            "target_snapshot_ref": self.target_snapshot_ref,
            "schema_diff": dict(self.schema_diff),
            "profile_diff": dict(self.profile_diff),
            "digest_diff": dict(self.digest_diff),
            "risk_flags": list(self.risk_flags),
            "row_level_diff_available": False,
            "conflict_resolution_available": False,
            "metadata": dict(self.metadata),
        }
        if redaction_safe:
            assert_dataset_artifact_safe(payload, label="dataset diff bundle")
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        payload = require_mapping(data, field_name="dataset diff bundle")
        return cls(
            diff=require_mapping(require_field(payload, "diff"), field_name="diff"),
            source_snapshot_ref=require_non_empty_str(
                require_field(payload, "source_snapshot_ref"),
                field_name="source_snapshot_ref",
            ),
            target_snapshot_ref=require_non_empty_str(
                require_field(payload, "target_snapshot_ref"),
                field_name="target_snapshot_ref",
            ),
            schema_diff=require_mapping(payload.get("schema_diff", {}), field_name="schema_diff"),
            profile_diff=require_mapping(
                payload.get("profile_diff", {}), field_name="profile_diff"
            ),
            digest_diff=require_mapping(payload.get("digest_diff", {}), field_name="digest_diff"),
            risk_flags=normalize_str_tuple(payload.get("risk_flags", ()), field_name="risk_flags"),
            metadata=require_mapping(payload.get("metadata", {}), field_name="metadata"),
        )

    def to_envelope(self) -> DatasetArtifactEnvelope:
        return DatasetArtifactEnvelope(
            artifact_type=DatasetArtifactType.DIFF_BUNDLE.value,
            payload=self.to_dict(),
        )

    @classmethod
    def from_envelope(cls, envelope: DatasetArtifactEnvelope) -> Self:
        return cls.from_dict(
            _envelope_payload(envelope, artifact_type=DatasetArtifactType.DIFF_BUNDLE)
        )

    def to_json(self, *, indent: int | None = None) -> str:
        return _to_envelope_json(self.to_envelope(), indent=indent)

    @classmethod
    def from_json(cls, data: str) -> Self:
        return cls.from_envelope(_envelope_from_json(data))


@dataclass(frozen=True)
class QualityGateBundle:
    """Quality gate artifact without embedding the original ValidationRunResult."""

    gate_result: QualityGateResult | Mapping[str, Any]
    source_validation_run_refs: tuple[str, ...] = ()
    blocking_failure_summary: Mapping[str, Any] = field(default_factory=dict)
    reviewer_summary: Mapping[str, Any] = field(default_factory=dict)
    machine_status: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        gate_result = _coerce_quality_gate_result(self.gate_result)
        object.__setattr__(self, "gate_result", gate_result)
        refs = self.source_validation_run_refs or (gate_result.run_ref,)
        object.__setattr__(
            self,
            "source_validation_run_refs",
            normalize_str_tuple(refs, field_name="source_validation_run_refs"),
        )
        object.__setattr__(
            self,
            "blocking_failure_summary",
            _quality_blocking_summary(self.blocking_failure_summary, gate_result),
        )
        object.__setattr__(
            self,
            "reviewer_summary",
            _quality_reviewer_summary(self.reviewer_summary, gate_result),
        )
        object.__setattr__(
            self,
            "machine_status",
            require_non_empty_str(
                self.machine_status or gate_result.status.value,
                field_name="machine_status",
            ),
        )
        object.__setattr__(self, "metadata", require_mapping(self.metadata, field_name="metadata"))
        self.validate()

    @classmethod
    def from_gate_result(
        cls,
        gate_result: QualityGateResult,
        *,
        source_validation_run_refs: tuple[str, ...] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> Self:
        return cls(
            gate_result=gate_result,
            source_validation_run_refs=source_validation_run_refs or (gate_result.run_ref,),
            metadata=metadata or {},
        )

    def validate(self) -> None:
        assert_dataset_artifact_safe(
            self.to_dict(redaction_safe=False),
            label="quality gate bundle",
        )

    def to_dict(self, *, redaction_safe: bool = True) -> dict[str, Any]:
        payload = {
            "gate_result": self.gate_result.to_dict(),
            "source_validation_run_refs": list(self.source_validation_run_refs),
            "blocking_failure_summary": dict(self.blocking_failure_summary),
            "reviewer_summary": dict(self.reviewer_summary),
            "machine_status": self.machine_status,
            "metadata": dict(self.metadata),
        }
        if redaction_safe:
            assert_dataset_artifact_safe(payload, label="quality gate bundle")
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        payload = require_mapping(data, field_name="quality gate bundle")
        return cls(
            gate_result=require_mapping(
                require_field(payload, "gate_result"), field_name="gate_result"
            ),
            source_validation_run_refs=normalize_str_tuple(
                payload.get("source_validation_run_refs", ()),
                field_name="source_validation_run_refs",
            ),
            blocking_failure_summary=require_mapping(
                payload.get("blocking_failure_summary", {}),
                field_name="blocking_failure_summary",
            ),
            reviewer_summary=require_mapping(
                payload.get("reviewer_summary", {}),
                field_name="reviewer_summary",
            ),
            machine_status=payload.get("machine_status"),
            metadata=require_mapping(payload.get("metadata", {}), field_name="metadata"),
        )

    def to_envelope(self) -> DatasetArtifactEnvelope:
        return DatasetArtifactEnvelope(
            artifact_type=DatasetArtifactType.QUALITY_GATE_BUNDLE.value,
            payload=self.to_dict(),
        )

    @classmethod
    def from_envelope(cls, envelope: DatasetArtifactEnvelope) -> Self:
        return cls.from_dict(
            _envelope_payload(envelope, artifact_type=DatasetArtifactType.QUALITY_GATE_BUNDLE)
        )

    def to_json(self, *, indent: int | None = None) -> str:
        return _to_envelope_json(self.to_envelope(), indent=indent)

    @classmethod
    def from_json(cls, data: str) -> Self:
        return cls.from_envelope(_envelope_from_json(data))


@dataclass(frozen=True)
class DatasetEvidenceInputPayload:
    """Redacted AI Evidence input payload derived from dataset artifacts."""

    evidence_id: str
    source_artifact_refs: tuple[str, ...]
    artifact_summaries: tuple[Mapping[str, Any], ...]
    risk_flags: tuple[str, ...] = ()
    reviewer_summary: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_id",
            require_non_empty_str(self.evidence_id, field_name="evidence_id"),
        )
        object.__setattr__(
            self,
            "source_artifact_refs",
            normalize_str_tuple(self.source_artifact_refs, field_name="source_artifact_refs"),
        )
        object.__setattr__(
            self,
            "artifact_summaries",
            normalize_mapping_tuple(self.artifact_summaries, field_name="artifact_summaries"),
        )
        object.__setattr__(
            self,
            "risk_flags",
            normalize_str_tuple(self.risk_flags, field_name="risk_flags"),
        )
        object.__setattr__(
            self,
            "reviewer_summary",
            require_mapping(self.reviewer_summary, field_name="reviewer_summary"),
        )
        object.__setattr__(self, "metadata", require_mapping(self.metadata, field_name="metadata"))
        self.validate()

    def validate(self) -> None:
        assert_dataset_artifact_safe(
            self.to_dict(redaction_safe=False),
            label="dataset evidence input payload",
        )

    def to_dict(self, *, redaction_safe: bool = True) -> dict[str, Any]:
        payload = {
            "evidence_id": self.evidence_id,
            "source_artifact_refs": list(self.source_artifact_refs),
            "artifact_summaries": [dict(item) for item in self.artifact_summaries],
            "risk_flags": list(self.risk_flags),
            "reviewer_summary": dict(self.reviewer_summary),
            "metadata": dict(self.metadata),
        }
        if redaction_safe:
            assert_dataset_artifact_safe(payload, label="dataset evidence input payload")
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        payload = require_mapping(data, field_name="dataset evidence input payload")
        return cls(
            evidence_id=require_non_empty_str(
                require_field(payload, "evidence_id"),
                field_name="evidence_id",
            ),
            source_artifact_refs=normalize_str_tuple(
                require_field(payload, "source_artifact_refs"),
                field_name="source_artifact_refs",
            ),
            artifact_summaries=normalize_mapping_tuple(
                require_field(payload, "artifact_summaries"),
                field_name="artifact_summaries",
            ),
            risk_flags=normalize_str_tuple(payload.get("risk_flags", ()), field_name="risk_flags"),
            reviewer_summary=require_mapping(
                payload.get("reviewer_summary", {}),
                field_name="reviewer_summary",
            ),
            metadata=require_mapping(payload.get("metadata", {}), field_name="metadata"),
        )

    def to_envelope(self) -> DatasetArtifactEnvelope:
        return DatasetArtifactEnvelope(
            artifact_type=DatasetArtifactType.EVIDENCE_INPUT_PAYLOAD.value,
            payload=self.to_dict(),
        )

    @classmethod
    def from_envelope(cls, envelope: DatasetArtifactEnvelope) -> Self:
        return cls.from_dict(
            _envelope_payload(envelope, artifact_type=DatasetArtifactType.EVIDENCE_INPUT_PAYLOAD)
        )

    def to_json(self, *, indent: int | None = None) -> str:
        return _to_envelope_json(self.to_envelope(), indent=indent)

    @classmethod
    def from_json(cls, data: str) -> Self:
        return cls.from_envelope(_envelope_from_json(data))


def restore_dataset_artifact(
    envelope_or_mapping: DatasetArtifactEnvelope | Mapping[str, Any],
) -> Any:
    """Restore a supported dataset artifact from a versioned envelope."""

    envelope = (
        envelope_or_mapping
        if isinstance(envelope_or_mapping, DatasetArtifactEnvelope)
        else DatasetArtifactEnvelope.from_dict(
            require_mapping(envelope_or_mapping, field_name="dataset artifact envelope")
        )
    )
    restorers = {
        DatasetArtifactType.ASSET_MANIFEST.value: DatasetAssetManifest.from_envelope,
        DatasetArtifactType.SNAPSHOT_MANIFEST.value: DatasetSnapshotManifest.from_envelope,
        DatasetArtifactType.FINGERPRINT.value: DatasetFingerprint.from_envelope,
        DatasetArtifactType.DIFF.value: DatasetDiff.from_envelope,
        DatasetArtifactType.QUALITY_GATE_RESULT.value: QualityGateResult.from_envelope,
        DatasetArtifactType.SNAPSHOT_BUNDLE.value: DatasetSnapshotBundle.from_envelope,
        DatasetArtifactType.DIFF_BUNDLE.value: DatasetDiffBundle.from_envelope,
        DatasetArtifactType.QUALITY_GATE_BUNDLE.value: QualityGateBundle.from_envelope,
        DatasetArtifactType.EVIDENCE_INPUT_PAYLOAD.value: DatasetEvidenceInputPayload.from_envelope,
    }
    restorer = restorers.get(envelope.artifact_type)
    if restorer is None:
        raise DatasetArtifactContractError(
            f"Unsupported dataset artifact type: {envelope.artifact_type!r}"
        )
    return restorer(envelope)


def _coerce_asset_manifest(value: DatasetAssetManifest | Mapping[str, Any]) -> DatasetAssetManifest:
    if isinstance(value, DatasetAssetManifest):
        return value
    return DatasetAssetManifest.from_dict(require_mapping(value, field_name="asset_manifest"))


def _coerce_snapshot_manifest(
    value: DatasetSnapshotManifest | Mapping[str, Any],
) -> DatasetSnapshotManifest:
    if isinstance(value, DatasetSnapshotManifest):
        return value
    return DatasetSnapshotManifest.from_dict(require_mapping(value, field_name="snapshot_manifest"))


def _coerce_fingerprint(value: DatasetFingerprint | Mapping[str, Any]) -> DatasetFingerprint:
    if isinstance(value, DatasetFingerprint):
        return value
    return DatasetFingerprint.from_dict(require_mapping(value, field_name="fingerprint"))


def _coerce_diff(value: DatasetDiff | Mapping[str, Any]) -> DatasetDiff:
    if isinstance(value, DatasetDiff):
        return value
    return DatasetDiff.from_dict(require_mapping(value, field_name="diff"))


def _coerce_quality_gate_result(
    value: QualityGateResult | Mapping[str, Any],
) -> QualityGateResult:
    if isinstance(value, QualityGateResult):
        return value
    return QualityGateResult.from_dict(require_mapping(value, field_name="gate_result"))


def _mapping_or_detail(
    value: Mapping[str, Any],
    details: Mapping[str, Any],
    detail_key: str,
    field_name: str,
) -> dict[str, Any]:
    explicit = require_mapping(value, field_name=field_name)
    if explicit:
        return explicit
    detail_value = details.get(detail_key, {})
    if detail_value is None:
        return {}
    return require_mapping(detail_value, field_name=f"diff.details.{detail_key}")


def _digest_detail(value: Mapping[str, Any], details: Mapping[str, Any]) -> dict[str, Any]:
    explicit = require_mapping(value, field_name="digest_diff")
    if explicit:
        return explicit
    return {
        "sampled_row_digest": require_mapping(
            details.get("sampled_row_digest", {}),
            field_name="diff.details.sampled_row_digest",
        ),
        "content_checksum": require_mapping(
            details.get("content_checksum", {}),
            field_name="diff.details.content_checksum",
        ),
        "fingerprints": require_mapping(
            details.get("fingerprints", {}),
            field_name="diff.details.fingerprints",
        ),
    }


def _quality_blocking_summary(
    value: Mapping[str, Any],
    gate_result: QualityGateResult,
) -> dict[str, Any]:
    explicit = require_mapping(value, field_name="blocking_failure_summary")
    if explicit:
        return explicit
    reasons: list[str] = []
    for failure in gate_result.blocking_failures:
        reason = failure.get("reason") or failure.get("issue_type") or failure.get("source")
        if reason is not None:
            reasons.append(str(reason))
    return {
        "blocking_count": len(gate_result.blocking_failures),
        "warning_count": len(gate_result.warnings),
        "reasons": reasons,
    }


def _quality_reviewer_summary(
    value: Mapping[str, Any],
    gate_result: QualityGateResult,
) -> dict[str, Any]:
    explicit = require_mapping(value, field_name="reviewer_summary")
    if explicit:
        return explicit
    return {
        "status": gate_result.status.value,
        "gate_type": gate_result.gate_type.value,
        "blocking_count": len(gate_result.blocking_failures),
        "warning_count": len(gate_result.warnings),
        "run_ref": gate_result.run_ref,
    }


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


def _to_envelope_json(envelope: DatasetArtifactEnvelope, *, indent: int | None) -> str:
    return json.dumps(envelope.to_dict(), indent=indent, sort_keys=True, default=str)


def _envelope_from_json(data: str) -> DatasetArtifactEnvelope:
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise DatasetArtifactContractError("Dataset artifact JSON is invalid") from exc
    return DatasetArtifactEnvelope.from_dict(
        require_mapping(payload, field_name="dataset artifact JSON")
    )


__all__ = [
    "DatasetDiffBundle",
    "DatasetEvidenceInputPayload",
    "DatasetSnapshotBundle",
    "QualityGateBundle",
    "restore_dataset_artifact",
]
