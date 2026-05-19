from __future__ import annotations

from datetime import UTC, datetime

import pytest

from truthound._datasets import (
    DatasetArtifactContractError,
    DatasetArtifactEnvelope,
    DatasetArtifactType,
    DatasetAssetManifest,
    DatasetAssetType,
    DatasetDiff,
    DatasetDiffCategory,
    DatasetFingerprint,
    DatasetSnapshotManifest,
    QualityGateResult,
    QualityGateStatus,
    QualityGateType,
)
from truthound._redaction import RedactionViolationError

CREATED_AT = datetime(2026, 5, 19, 12, 0, tzinfo=UTC)


def test_dataset_asset_manifest_round_trips_through_envelope() -> None:
    manifest = DatasetAssetManifest(
        asset_id="asset_customers",
        asset_name="Customers",
        asset_type=DatasetAssetType.TABLE_SNAPSHOT,
        logical_path="crm/customers",
        source_kind="warehouse_table",
        schema_ref="schema://customers",
        content_ref="table://crm.customers",
        created_at=CREATED_AT,
        metadata={"owner": "data-platform"},
    )

    restored = DatasetAssetManifest.from_envelope(manifest.to_envelope())

    assert restored == manifest
    assert manifest.to_envelope().artifact_type == DatasetArtifactType.ASSET_MANIFEST.value
    assert manifest.to_dict()["asset_type"] == "table_snapshot"


def test_dataset_snapshot_manifest_round_trips_and_normalizes_refs() -> None:
    manifest = DatasetSnapshotManifest(
        snapshot_id="snapshot_20260519",
        asset_id="asset_customers",
        parent_snapshot_id=None,
        fingerprint="fp-main",
        schema_fingerprint="schema-fp",
        profile_fingerprint="profile-fp",
        row_count=42,
        column_count=4,
        size_bytes=1024,
        created_at=CREATED_AT,
        created_by="system",
        validation_refs=["run_001", "run_002"],
    )

    restored = DatasetSnapshotManifest.from_envelope(manifest.to_envelope())

    assert restored == manifest
    assert restored.validation_refs == ("run_001", "run_002")


def test_dataset_fingerprint_round_trips_with_policy_versions() -> None:
    fingerprint = DatasetFingerprint(
        schema_hash="schema-hash",
        column_list_hash="columns-hash",
        row_count=42,
        null_profile_hash="nulls-hash",
        sampled_row_hash="sampled-hash",
        content_checksum="content-hash",
        metadata={"method": "fixture"},
    )

    envelope = fingerprint.to_envelope()
    restored = DatasetFingerprint.from_envelope(envelope)

    assert restored == fingerprint
    assert envelope.fingerprint_policy_version == fingerprint.fingerprint_policy_version
    assert envelope.sampling_policy_version == fingerprint.sampling_policy_version


def test_dataset_diff_round_trips_and_normalizes_categories() -> None:
    diff = DatasetDiff(
        source_snapshot_id="snapshot_a",
        target_snapshot_id="snapshot_b",
        categories=["schema_changed", DatasetDiffCategory.ROW_COUNT_CHANGED],
        summary={"changed": True, "risk": "medium"},
        details={"schema": {"added": ["status"]}},
    )

    restored = DatasetDiff.from_envelope(diff.to_envelope())

    assert restored == diff
    assert restored.categories == (
        DatasetDiffCategory.SCHEMA_CHANGED,
        DatasetDiffCategory.ROW_COUNT_CHANGED,
    )
    assert restored.to_dict()["categories"] == ["schema_changed", "row_count_changed"]


def test_quality_gate_result_round_trips_with_failure_and_warning_rows() -> None:
    result = QualityGateResult(
        quality_gate_id="gate_001",
        gate_type=QualityGateType.MERGE,
        status=QualityGateStatus.FAILED,
        asset_id="asset_customers",
        snapshot_id="snapshot_20260519",
        suite_ref="suite://merge",
        run_ref="run_001",
        summary={"issue_count": 1},
        blocking_failures=({"check": "not_null", "count": 1},),
        warnings=({"check": "freshness", "count": 1},),
        created_at=CREATED_AT,
    )

    restored = QualityGateResult.from_envelope(result.to_envelope())

    assert restored == result
    assert restored.blocking_failures == ({"check": "not_null", "count": 1},)
    assert restored.warnings == ({"check": "freshness", "count": 1},)


def test_primitive_from_envelope_rejects_wrong_artifact_type() -> None:
    envelope = DatasetArtifactEnvelope(
        artifact_type=DatasetArtifactType.DIFF.value,
        payload={
            "source_snapshot_id": "snapshot_a",
            "target_snapshot_id": "snapshot_b",
            "categories": ["schema_changed"],
            "summary": {},
            "details": {},
            "metadata": {},
        },
    )

    with pytest.raises(DatasetArtifactContractError, match="type mismatch"):
        DatasetAssetManifest.from_envelope(envelope)


def test_primitives_reject_invalid_enum_and_negative_counts() -> None:
    with pytest.raises(DatasetArtifactContractError, match="asset_type"):
        DatasetAssetManifest(
            asset_id="asset_customers",
            asset_name="Customers",
            asset_type="not-real",
            logical_path="crm/customers",
            source_kind="warehouse_table",
            schema_ref="schema://customers",
            content_ref="table://crm.customers",
        )

    with pytest.raises(DatasetArtifactContractError, match="row_count"):
        DatasetFingerprint(
            schema_hash="schema-hash",
            column_list_hash="columns-hash",
            row_count=-1,
            null_profile_hash="nulls-hash",
            sampled_row_hash="sampled-hash",
        )


def test_primitives_reject_non_mapping_metadata_and_rows() -> None:
    with pytest.raises(DatasetArtifactContractError, match="metadata must be a mapping"):
        DatasetAssetManifest(
            asset_id="asset_customers",
            asset_name="Customers",
            asset_type=DatasetAssetType.TABLE_SNAPSHOT,
            logical_path="crm/customers",
            source_kind="warehouse_table",
            schema_ref="schema://customers",
            content_ref="table://crm.customers",
            metadata=["not", "mapping"],  # type: ignore[arg-type]
        )

    with pytest.raises(DatasetArtifactContractError, match="blocking_failures"):
        QualityGateResult(
            quality_gate_id="gate_001",
            gate_type=QualityGateType.MERGE,
            status=QualityGateStatus.FAILED,
            asset_id="asset_customers",
            snapshot_id="snapshot_20260519",
            suite_ref="suite://merge",
            run_ref="run_001",
            summary={"issue_count": 1},
            blocking_failures=("not-mapping",),  # type: ignore[arg-type]
            warnings=(),
        )


def test_primitives_reject_raw_rows_and_pii_literals() -> None:
    with pytest.raises(RedactionViolationError, match="raw_rows"):
        DatasetDiff(
            source_snapshot_id="snapshot_a",
            target_snapshot_id="snapshot_b",
            categories=(DatasetDiffCategory.UNKNOWN_CHANGED,),
            summary={},
            details={"raw_rows": [{"email": "person@example.com"}]},
        )

    with pytest.raises(RedactionViolationError, match="PII-like"):
        QualityGateResult(
            quality_gate_id="gate_001",
            gate_type=QualityGateType.MERGE,
            status=QualityGateStatus.FAILED,
            asset_id="asset_customers",
            snapshot_id="snapshot_20260519",
            suite_ref="suite://merge",
            run_ref="run_001",
            summary={"reason": "contact person@example.com"},
            blocking_failures=(),
            warnings=(),
        )


def test_typed_primitives_remain_private_to_root_package() -> None:
    import truthound

    exported = dir(truthound)

    assert "DatasetAssetManifest" not in exported
    assert "DatasetSnapshotManifest" not in exported
    assert "DatasetDiff" not in exported
    assert "QualityGateResult" not in exported
