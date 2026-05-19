from __future__ import annotations

from datetime import UTC, datetime

import pytest

from truthound._ai_redaction import SummaryOnlyRedactor as AISummaryOnlyRedactor
from truthound._datasets import (
    TRUTHOUND_DATASET_ARTIFACT_SCHEMA_VERSION,
    TRUTHOUND_DATASET_FINGERPRINT_POLICY_VERSION,
    TRUTHOUND_DATASET_SAMPLING_POLICY_VERSION,
    DatasetArtifactContractError,
    DatasetArtifactEnvelope,
    DatasetArtifactVersionError,
    assert_dataset_artifact_safe,
)
from truthound._redaction import RedactionViolationError, SummaryOnlyRedactor


def test_dataset_artifact_envelope_round_trips_with_policy_versions() -> None:
    envelope = DatasetArtifactEnvelope(
        artifact_type="dataset_snapshot_bundle",
        payload={
            "asset_id": "asset_customers",
            "snapshot_id": "snapshot_20260519",
            "summary": {"row_count": 42, "column_count": 4},
        },
        created_at=datetime(2026, 5, 19, 12, 0, tzinfo=UTC),
        metadata={"source": "unit-test"},
    )

    serialized = envelope.to_dict()

    assert serialized["artifact_schema_version"] == TRUTHOUND_DATASET_ARTIFACT_SCHEMA_VERSION
    assert serialized["fingerprint_policy_version"] == TRUTHOUND_DATASET_FINGERPRINT_POLICY_VERSION
    assert serialized["sampling_policy_version"] == TRUTHOUND_DATASET_SAMPLING_POLICY_VERSION

    restored = DatasetArtifactEnvelope.from_dict(serialized)

    assert restored == envelope


def test_dataset_artifact_envelope_rejects_unsupported_versions() -> None:
    payload = {
        "artifact_schema_version": "9.9",
        "artifact_type": "dataset_snapshot_bundle",
        "payload": {"asset_id": "asset_customers"},
        "fingerprint_policy_version": TRUTHOUND_DATASET_FINGERPRINT_POLICY_VERSION,
        "sampling_policy_version": TRUTHOUND_DATASET_SAMPLING_POLICY_VERSION,
        "created_at": "2026-05-19T12:00:00+00:00",
        "metadata": {},
    }

    with pytest.raises(DatasetArtifactVersionError, match="schema version"):
        DatasetArtifactEnvelope.from_dict(payload)


def test_dataset_artifact_envelope_rejects_missing_fixed_fields() -> None:
    payload = {
        "artifact_type": "dataset_snapshot_bundle",
        "payload": {"asset_id": "asset_customers"},
        "fingerprint_policy_version": TRUTHOUND_DATASET_FINGERPRINT_POLICY_VERSION,
        "sampling_policy_version": TRUTHOUND_DATASET_SAMPLING_POLICY_VERSION,
        "created_at": "2026-05-19T12:00:00+00:00",
        "metadata": {},
    }

    with pytest.raises(DatasetArtifactContractError, match="artifact_schema_version"):
        DatasetArtifactEnvelope.from_dict(payload)


def test_dataset_artifact_envelope_requires_mapping_payload() -> None:
    with pytest.raises(DatasetArtifactContractError, match="payload must be a mapping"):
        DatasetArtifactEnvelope(
            artifact_type="dataset_snapshot_bundle",
            payload=["not", "a", "mapping"],  # type: ignore[arg-type]
        )


def test_dataset_artifact_safety_rejects_raw_rows() -> None:
    unsafe_payload = {
        "asset_id": "asset_customers",
        "raw_rows": [{"email": "person@example.com"}],
    }

    with pytest.raises(RedactionViolationError, match="raw_rows"):
        assert_dataset_artifact_safe(unsafe_payload)


def test_dataset_artifact_safety_rejects_pii_literals() -> None:
    unsafe_payload = {
        "asset_id": "asset_customers",
        "risk_summary": "contact owner at person@example.com",
    }

    with pytest.raises(RedactionViolationError, match="PII-like"):
        DatasetArtifactEnvelope(
            artifact_type="quality_gate_bundle",
            payload=unsafe_payload,
        )


def test_ai_redaction_path_remains_compatibility_facade() -> None:
    assert AISummaryOnlyRedactor is SummaryOnlyRedactor


def test_dataset_contract_remains_private_to_root_package() -> None:
    import truthound

    assert "datasets" not in dir(truthound)
    assert "depot" not in dir(truthound)
    assert "DatasetArtifactEnvelope" not in dir(truthound)
