from __future__ import annotations

from datetime import UTC, datetime

import pytest

from truthound._datasets import (
    DatasetArtifactContractError,
    DatasetArtifactEnvelope,
    DatasetArtifactType,
    DatasetArtifactVersionError,
    DatasetAssetManifest,
    DatasetAssetType,
    DatasetDiff,
    DatasetDiffBundle,
    DatasetDiffCategory,
    DatasetEvidenceInputPayload,
    DatasetFingerprint,
    DatasetSnapshotBundle,
    DatasetSnapshotManifest,
    QualityGateBundle,
    QualityGateResult,
    QualityGateStatus,
    QualityGateType,
    restore_dataset_artifact,
)
from truthound._redaction import RedactionViolationError

CREATED_AT = datetime(2026, 5, 19, 12, 0, tzinfo=UTC)


def _asset_manifest() -> DatasetAssetManifest:
    return DatasetAssetManifest(
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


def _fingerprint() -> DatasetFingerprint:
    return DatasetFingerprint(
        schema_hash="sha256:schema",
        column_list_hash="sha256:columns",
        row_count=42,
        null_profile_hash="sha256:nulls",
        sampled_row_hash="sha256:sampled",
        content_checksum="sha256:content",
        metadata={
            "schema": [{"name": "id", "dtype": "Int64", "position": 0}],
            "profile_hash": "sha256:profile",
        },
    )


def _snapshot_manifest() -> DatasetSnapshotManifest:
    return DatasetSnapshotManifest(
        snapshot_id="snapshot_001",
        asset_id="asset_customers",
        parent_snapshot_id=None,
        fingerprint="sha256:content",
        schema_fingerprint="sha256:schema",
        profile_fingerprint="sha256:profile",
        row_count=42,
        column_count=2,
        size_bytes=2048,
        created_at=CREATED_AT,
        created_by="system",
        validation_refs=("run_001",),
        metadata={"retention": "short"},
    )


def _diff() -> DatasetDiff:
    return DatasetDiff(
        source_snapshot_id="snapshot_001",
        target_snapshot_id="snapshot_002",
        categories=(DatasetDiffCategory.SCHEMA_ADDED, DatasetDiffCategory.SAMPLE_CHANGED),
        summary={"changed": True, "row_level_diff_available": False},
        details={
            "schema": {"added": [{"name": "status", "dtype": "String"}]},
            "profile": {"changed": True, "changed_columns": ["status"]},
            "sampled_row_digest": {
                "changed": True,
                "source_hash": "sha256:old",
                "target_hash": "sha256:new",
            },
            "content_checksum": {
                "changed": True,
                "source_hash": "sha256:old-content",
                "target_hash": "sha256:new-content",
            },
            "fingerprints": {
                "source": {"schema_hash": "sha256:old-schema"},
                "target": {"schema_hash": "sha256:new-schema"},
            },
        },
    )


def _gate_result() -> QualityGateResult:
    return QualityGateResult(
        quality_gate_id="gate_001",
        gate_type=QualityGateType.MERGE,
        status=QualityGateStatus.FAILED,
        asset_id="asset_customers",
        snapshot_id="snapshot_002",
        suite_ref="suite://merge",
        run_ref="run_001",
        summary={"blocking_count": 1, "warning_count": 0},
        blocking_failures=(
            {
                "source": "validation_issue",
                "check": "not_null",
                "issue_type": "null",
                "count": 2,
                "disposition": "blocking",
            },
        ),
        warnings=(),
        created_at=CREATED_AT,
        metadata={"projection": "ValidationRunResult->QualityGateResult"},
    )


def test_snapshot_bundle_round_trips_through_dict_json_and_envelope() -> None:
    bundle = DatasetSnapshotBundle(
        snapshot_manifest=_snapshot_manifest(),
        fingerprint=_fingerprint(),
        asset_manifest=_asset_manifest(),
        profile_summary={"profile_hash": "sha256:profile", "column_count": 2},
        validation_refs=("run_001", "run_002"),
        content_refs={"table": "table://crm.customers"},
        metadata={"unknown": {"kept": True}},
    )

    restored_from_dict = DatasetSnapshotBundle.from_dict(bundle.to_dict())
    restored_from_json = DatasetSnapshotBundle.from_json(bundle.to_json())
    restored_from_envelope = DatasetSnapshotBundle.from_envelope(bundle.to_envelope())

    assert restored_from_dict == bundle
    assert restored_from_json == bundle
    assert restored_from_envelope == bundle
    assert bundle.to_envelope().artifact_type == DatasetArtifactType.SNAPSHOT_BUNDLE.value
    assert '"artifact_schema_version"' in bundle.to_json()
    assert restored_from_json.metadata["unknown"] == {"kept": True}


def test_diff_bundle_derives_machine_sections_without_row_level_promise() -> None:
    diff = _diff()
    bundle = DatasetDiffBundle.from_diff(diff, risk_flags=("schema_review_required",))
    payload = bundle.to_dict()

    assert payload["schema_diff"] == {"added": [{"name": "status", "dtype": "String"}]}
    assert payload["profile_diff"]["changed"] is True
    assert payload["digest_diff"]["sampled_row_digest"]["changed"] is True
    assert payload["risk_flags"] == ["schema_review_required"]
    assert payload["row_level_diff_available"] is False
    assert payload["conflict_resolution_available"] is False
    assert DatasetDiffBundle.from_json(bundle.to_json()) == bundle


def test_quality_gate_bundle_preserves_machine_and_reviewer_summary() -> None:
    gate_result = _gate_result()
    bundle = QualityGateBundle.from_gate_result(gate_result)
    payload = bundle.to_dict()

    assert payload["machine_status"] == "failed"
    assert payload["source_validation_run_refs"] == ["run_001"]
    assert payload["blocking_failure_summary"]["reasons"] == ["null"]
    assert payload["reviewer_summary"]["status"] == "failed"
    assert "ValidationRunResult" not in payload
    assert QualityGateBundle.from_envelope(bundle.to_envelope()) == bundle


def test_evidence_input_payload_is_redacted_and_round_trips() -> None:
    payload = DatasetEvidenceInputPayload(
        evidence_id="evidence_001",
        source_artifact_refs=("artifact://snapshot_001", "artifact://gate_001"),
        artifact_summaries=(
            {"artifact_type": "dataset_snapshot_bundle", "row_count": 42},
            {"artifact_type": "quality_gate_bundle", "machine_status": "failed"},
        ),
        risk_flags=("blocking_gate",),
        reviewer_summary={"status": "needs_review"},
        metadata={"consumer": "ai-evidence"},
    )

    restored = DatasetEvidenceInputPayload.from_json(payload.to_json())

    assert restored == payload
    assert restored.to_envelope().artifact_type == DatasetArtifactType.EVIDENCE_INPUT_PAYLOAD.value
    assert restored.artifact_summaries[0]["row_count"] == 42


def test_restore_dataset_artifact_dispatches_primitives_and_bundles() -> None:
    snapshot_bundle = DatasetSnapshotBundle(
        snapshot_manifest=_snapshot_manifest(),
        fingerprint=_fingerprint(),
    )
    gate_result = _gate_result()

    restored_bundle = restore_dataset_artifact(snapshot_bundle.to_envelope().to_dict())
    restored_gate = restore_dataset_artifact(gate_result.to_envelope())

    assert restored_bundle == snapshot_bundle
    assert restored_gate == gate_result


def test_bundle_from_envelope_rejects_type_mismatch_and_bad_versions() -> None:
    bundle = DatasetSnapshotBundle(
        snapshot_manifest=_snapshot_manifest(),
        fingerprint=_fingerprint(),
    )
    wrong_envelope = DatasetArtifactEnvelope(
        artifact_type=DatasetArtifactType.DIFF_BUNDLE.value,
        payload=bundle.to_dict(),
    )
    bad_version = bundle.to_envelope().to_dict()
    bad_version["artifact_schema_version"] = "9.9"

    with pytest.raises(DatasetArtifactContractError, match="type mismatch"):
        DatasetSnapshotBundle.from_envelope(wrong_envelope)

    with pytest.raises(DatasetArtifactVersionError, match="schema version"):
        restore_dataset_artifact(bad_version)


def test_restore_dataset_artifact_rejects_unknown_artifact_type() -> None:
    envelope = DatasetArtifactEnvelope(
        artifact_type="unknown_dataset_artifact",
        payload={"summary": {"safe": True}},
    )

    with pytest.raises(DatasetArtifactContractError, match="Unsupported dataset artifact type"):
        restore_dataset_artifact(envelope)


def test_bundles_reject_raw_rows_samples_and_pii_literals() -> None:
    with pytest.raises(RedactionViolationError, match="raw_rows"):
        DatasetSnapshotBundle(
            snapshot_manifest=_snapshot_manifest(),
            fingerprint=_fingerprint(),
            profile_summary={"raw_rows": [{"email": "person@example.com"}]},
        )

    with pytest.raises(RedactionViolationError, match="sample_values"):
        DatasetEvidenceInputPayload(
            evidence_id="evidence_unsafe",
            source_artifact_refs=("artifact://snapshot_001",),
            artifact_summaries=({"sample_values": ["secret"]},),
        )

    with pytest.raises(RedactionViolationError, match="PII-like"):
        DatasetEvidenceInputPayload(
            evidence_id="evidence_pii",
            source_artifact_refs=("artifact://snapshot_001",),
            artifact_summaries=({"summary": "contact person@example.com"},),
        )


def test_quality_gate_bundle_does_not_include_validation_run_raw_payload() -> None:
    gate_result = QualityGateResult(
        quality_gate_id="gate_001",
        gate_type=QualityGateType.MERGE,
        status=QualityGateStatus.ERROR,
        asset_id="asset_customers",
        snapshot_id="snapshot_002",
        suite_ref="suite://merge",
        run_ref="run_001",
        summary={"error_count": 1},
        blocking_failures=(
            {
                "source": "execution_issue",
                "check": "schema",
                "exception_type": "RuntimeError",
                "failure_category": "runtime",
                "retry_count": 1,
                "disposition": "blocking",
            },
        ),
        warnings=(),
        created_at=CREATED_AT,
    )

    bundle = QualityGateBundle.from_gate_result(gate_result)
    serialized = str(bundle.to_dict())

    assert "sample_values" not in serialized
    assert "database failure" not in serialized
    assert "person@example.com" not in serialized


def test_bundle_runtime_remains_private_to_root_package() -> None:
    import truthound

    exported = dir(truthound)

    assert "DatasetSnapshotBundle" not in exported
    assert "DatasetDiffBundle" not in exported
    assert "QualityGateBundle" not in exported
    assert "DatasetEvidenceInputPayload" not in exported
    assert "restore_dataset_artifact" not in exported
