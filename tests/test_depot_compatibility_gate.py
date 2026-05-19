from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import polars as pl
import pytest

from truthound._datasets import (
    DatasetArtifactContractError,
    DatasetArtifactEnvelope,
    DatasetArtifactType,
    DatasetAssetManifest,
    DatasetAssetType,
    DatasetDiffBundle,
    DatasetDiffCategory,
    DatasetEvidenceInputPayload,
    DatasetFingerprint,
    DatasetSnapshotBundle,
    DatasetSnapshotManifest,
    QualityGateBundle,
    QualityGateContext,
    QualityGateDisposition,
    QualityGatePolicy,
    QualityGateStatus,
    QualityGateType,
    diff_datasets,
    evaluate_quality_gate,
    fingerprint_dataset,
    restore_dataset_artifact,
)
from truthound.core.results import CheckResult, ExecutionIssue, ValidationRunResult
from truthound.types import Severity
from truthound.validators.base import ValidationIssue

if TYPE_CHECKING:
    from pathlib import Path

CREATED_AT = datetime(2026, 5, 19, 12, 0, tzinfo=UTC)
EXPECTED_CANONICAL_FINGERPRINT = {
    "schema_hash": "sha256:85cecbb23682443cc02c85fc6eb972f4013d85bf24087a108ed3790c41fd4b7d",
    "column_list_hash": "sha256:75d83242203bf67749f463183ffcc69fa05a3bdff4995fcc4ead2627f494f0d0",
    "row_count": 4,
    "null_profile_hash": "sha256:ac8abb57996be1c22dca0a81b4b1c41ced8ac223cc3433da86b1b045d2b6ec1a",
    "sampled_row_hash": "sha256:7e9729a471b8df2af31d012362bb5c52888916419802952532bb90d62360fba3",
    "content_checksum": None,
    "profile_hash": "sha256:028317f5771b8cb587c834dfde7ba92ca62955fd7fb895ffb2d532394b131a56",
}


def _canonical_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["alpha", "beta", "gamma", "delta"],
            "score": [10.5, 20.0, None, 40.25],
            "active": [True, False, True, True],
        }
    )


def _fingerprint_signature(fingerprint: DatasetFingerprint) -> dict[str, Any]:
    return {
        "schema_hash": fingerprint.schema_hash,
        "column_list_hash": fingerprint.column_list_hash,
        "row_count": fingerprint.row_count,
        "null_profile_hash": fingerprint.null_profile_hash,
        "sampled_row_hash": fingerprint.sampled_row_hash,
        "content_checksum": fingerprint.content_checksum,
        "profile_hash": fingerprint.metadata["profile_hash"],
    }


def _write_canonical_input(kind: str, tmp_path: Path) -> pl.DataFrame | Path:
    frame = _canonical_frame()
    if kind == "dataframe":
        return frame
    if kind == "csv":
        path = tmp_path / "canonical.csv"
        frame.write_csv(path)
        return path
    if kind == "parquet":
        path = tmp_path / "canonical.parquet"
        frame.write_parquet(path)
        return path
    raise AssertionError(f"unsupported fixture kind: {kind}")


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
        metadata={"owner": "data-platform", "unknown": {"preserved": True}},
    )


def _snapshot_manifest(fingerprint: DatasetFingerprint | None = None) -> DatasetSnapshotManifest:
    resolved = fingerprint or fingerprint_dataset(_canonical_frame())
    return DatasetSnapshotManifest(
        snapshot_id="snapshot_001",
        asset_id="asset_customers",
        parent_snapshot_id=None,
        fingerprint=resolved.sampled_row_hash,
        schema_fingerprint=resolved.schema_hash,
        profile_fingerprint=resolved.metadata["profile_hash"],
        row_count=resolved.row_count,
        column_count=len(resolved.metadata["schema"]),
        size_bytes=4096,
        created_at=CREATED_AT,
        created_by="system",
        validation_refs=("run_001",),
        metadata={"retention": "short", "unknown": {"kept": True}},
    )


def _issue(*, severity: Severity = Severity.HIGH) -> ValidationIssue:
    return ValidationIssue(
        column="name",
        issue_type="null",
        count=2,
        severity=severity,
        details="debug rows and sample values must not be copied",
        expected="non-null",
        actual="null",
        sample_values=["person@example.com"],
        validator_name="not_null",
    )


def _run(
    *,
    checks: tuple[CheckResult, ...],
    issues: tuple[ValidationIssue, ...] = (),
    execution_issues: tuple[ExecutionIssue, ...] = (),
) -> ValidationRunResult:
    return ValidationRunResult(
        run_id="run_001",
        run_time=CREATED_AT,
        suite_name="suite",
        source="dict",
        row_count=4,
        column_count=4,
        checks=checks,
        issues=issues,
        execution_issues=execution_issues,
    )


def _context(**overrides: Any) -> QualityGateContext:
    values: dict[str, Any] = {
        "quality_gate_id": "gate_001",
        "gate_type": QualityGateType.MERGE,
        "asset_id": "asset_customers",
        "snapshot_id": "snapshot_001",
        "suite_ref": "suite://merge",
    }
    values.update(overrides)
    return QualityGateContext(**values)


def test_snapshot_bundle_json_round_trip_preserves_wire_contract() -> None:
    fingerprint = fingerprint_dataset(_canonical_frame())
    bundle = DatasetSnapshotBundle(
        snapshot_manifest=_snapshot_manifest(fingerprint),
        fingerprint=fingerprint,
        asset_manifest=_asset_manifest(),
        profile_summary={"profile_hash": fingerprint.metadata["profile_hash"], "column_count": 4},
        validation_refs=("run_001",),
        content_refs={"table": "table://crm.customers"},
        metadata={"unknown": {"nested": True}},
    )

    envelope = json.loads(bundle.to_json())
    restored = DatasetSnapshotBundle.from_json(bundle.to_json())

    assert restored == bundle
    assert envelope["artifact_schema_version"] == "0.1"
    assert envelope["artifact_type"] == DatasetArtifactType.SNAPSHOT_BUNDLE.value
    assert envelope["created_at"]
    assert envelope["payload"]["asset_manifest"]["asset_type"] == "table_snapshot"
    assert envelope["payload"]["metadata"]["unknown"] == {"nested": True}


def test_missing_artifact_schema_version_fails_explicitly() -> None:
    envelope = fingerprint_dataset(_canonical_frame()).to_envelope().to_dict()
    envelope.pop("artifact_schema_version")

    with pytest.raises(DatasetArtifactContractError, match="artifact_schema_version"):
        DatasetArtifactEnvelope.from_dict(envelope)


@pytest.mark.parametrize("kind", ["dataframe", "csv", "parquet"])
def test_canonical_fingerprint_hashes_are_stable_across_inputs(
    kind: str,
    tmp_path: Path,
) -> None:
    data = _write_canonical_input(kind, tmp_path)
    fingerprint = fingerprint_dataset(data, sample_size=3, stable_key_columns=("id",))
    serialized = json.dumps(fingerprint.to_dict(), sort_keys=True)

    assert _fingerprint_signature(fingerprint) == EXPECTED_CANONICAL_FINGERPRINT
    assert "row_hashes" not in serialized
    assert "sample_values" not in serialized
    assert fingerprint.content_checksum is None


@pytest.mark.parametrize(
    ("name", "source", "target", "expected_categories"),
    [
        (
            "schema_added",
            pl.DataFrame({"id": [1, 2]}),
            pl.DataFrame({"id": [1, 2], "status": ["ok", "ok"]}),
            (
                DatasetDiffCategory.SCHEMA_ADDED,
                DatasetDiffCategory.NULL_PROFILE_CHANGED,
                DatasetDiffCategory.PROFILE_CHANGED,
                DatasetDiffCategory.SAMPLE_CHANGED,
            ),
        ),
        (
            "schema_removed",
            pl.DataFrame({"id": [1, 2], "status": ["ok", "ok"]}),
            pl.DataFrame({"id": [1, 2]}),
            (
                DatasetDiffCategory.SCHEMA_REMOVED,
                DatasetDiffCategory.NULL_PROFILE_CHANGED,
                DatasetDiffCategory.PROFILE_CHANGED,
                DatasetDiffCategory.SAMPLE_CHANGED,
            ),
        ),
        (
            "schema_changed",
            pl.DataFrame({"id": [1, 2], "score": [1, 2]}),
            pl.DataFrame({"id": [1, 2], "score": ["1", "2"]}),
            (
                DatasetDiffCategory.SCHEMA_CHANGED,
                DatasetDiffCategory.PROFILE_CHANGED,
                DatasetDiffCategory.SAMPLE_CHANGED,
            ),
        ),
        (
            "row_count_changed",
            pl.DataFrame({"id": [1, 2], "score": [1, 2]}),
            pl.DataFrame({"id": [1, 2, 3], "score": [1, 2, 3]}),
            (
                DatasetDiffCategory.ROW_COUNT_CHANGED,
                DatasetDiffCategory.PROFILE_CHANGED,
                DatasetDiffCategory.SAMPLE_CHANGED,
            ),
        ),
        (
            "null_profile_changed",
            pl.DataFrame({"id": [1, 2], "score": [1, 2]}),
            pl.DataFrame({"id": [1, 2], "score": [1, None]}),
            (
                DatasetDiffCategory.NULL_PROFILE_CHANGED,
                DatasetDiffCategory.PROFILE_CHANGED,
                DatasetDiffCategory.SAMPLE_CHANGED,
            ),
        ),
        (
            "sample_changed",
            pl.DataFrame({"id": [1, 2], "score": [1, 2]}),
            pl.DataFrame({"id": [1, 2], "score": [1, 3]}),
            (DatasetDiffCategory.PROFILE_CHANGED, DatasetDiffCategory.SAMPLE_CHANGED),
        ),
        (
            "unchanged",
            pl.DataFrame({"id": [1, 2], "score": [1, 2]}),
            pl.DataFrame({"id": [2, 1], "score": [2, 1]}),
            (),
        ),
    ],
)
def test_diff_categories_are_regression_locked(
    name: str,
    source: pl.DataFrame,
    target: pl.DataFrame,
    expected_categories: tuple[DatasetDiffCategory, ...],
) -> None:
    diff = diff_datasets(
        source,
        target,
        source_snapshot_id=f"{name}_source",
        target_snapshot_id=f"{name}_target",
        stable_key_columns=("id",),
    )
    bundle = DatasetDiffBundle.from_diff(diff)

    assert diff.categories == expected_categories
    assert diff.summary["row_level_diff_available"] is False
    assert bundle.to_dict()["row_level_diff_available"] is False
    assert bundle.to_dict()["conflict_resolution_available"] is False


def test_quality_gate_projection_shape_is_regression_locked() -> None:
    issue = _issue()
    run = _run(
        checks=(CheckResult(name="not_null", success=False, issue_count=1, issues=(issue,)),),
        issues=(issue,),
    )

    result = evaluate_quality_gate(run, _context())
    bundle = QualityGateBundle.from_gate_result(result)
    payload = result.to_dict()

    assert payload["status"] == "failed"
    assert bundle.machine_status == "failed"
    assert bundle.source_validation_run_refs == ("run_001",)
    assert payload["run_ref"] == "run_001"
    assert payload["summary"] == {
        "gate_type": "merge",
        "skipped": False,
        "skip_reason": None,
        "check_count": 1,
        "passed_check_count": 0,
        "failed_check_count": 1,
        "validation_issue_count": 1,
        "execution_issue_count": 0,
        "blocking_count": 1,
        "warning_count": 0,
        "informational_count": 0,
        "error_count": 0,
        "run_success": False,
        "row_count": 4,
        "column_count": 4,
        "rollback_target_exists": None,
        "previous_successful_gate_count": 0,
    }
    assert payload["blocking_failures"] == [
        {
            "source": "validation_issue",
            "check": "not_null",
            "validator": "not_null",
            "issue_type": "null",
            "column": "name",
            "severity": "high",
            "count": 2,
            "disposition": "blocking",
        }
    ]
    assert "sample_values" not in str(payload)
    assert "person@example.com" not in str(payload)


def test_quality_gate_status_regressions() -> None:
    clean = _run(checks=(CheckResult(name="schema", success=True),))
    low_issue = _issue(severity=Severity.LOW)
    warning_run = _run(
        checks=(CheckResult(name="not_null", success=False, issue_count=1, issues=(low_issue,)),),
        issues=(low_issue,),
    )
    execution_error = _run(
        checks=(CheckResult(name="schema", success=True),),
        execution_issues=(
            ExecutionIssue(
                check_name="schema",
                message="database failed for person@example.com",
                exception_type="RuntimeError",
                failure_category="runtime",
                retry_count=1,
            ),
        ),
    )

    assert evaluate_quality_gate(clean, _context()).status == QualityGateStatus.PASSED
    assert (
        evaluate_quality_gate(
            warning_run,
            _context(),
            QualityGatePolicy(severity_dispositions={Severity.LOW: QualityGateDisposition.WARNING}),
        ).status
        == QualityGateStatus.WARNING
    )
    assert evaluate_quality_gate(execution_error, _context()).status == QualityGateStatus.ERROR
    assert evaluate_quality_gate(_run(checks=()), _context()).status == QualityGateStatus.ERROR
    with pytest.raises(DatasetArtifactContractError, match="suite_ref"):
        _context(suite_ref="")
    assert (
        evaluate_quality_gate(clean, _context(gate_type=QualityGateType.ROLLBACK)).status
        == QualityGateStatus.ERROR
    )


def test_artifact_shape_contract_prevents_row_level_payload_growth() -> None:
    large_frame = pl.DataFrame(
        {"id": list(range(1_000)), "value": [f"value-{idx}" for idx in range(1_000)]}
    )
    fingerprint = fingerprint_dataset(large_frame)
    bundle = DatasetSnapshotBundle(
        snapshot_manifest=_snapshot_manifest(fingerprint),
        fingerprint=fingerprint,
        profile_summary={"profile_hash": fingerprint.metadata["profile_hash"]},
    )
    evidence = DatasetEvidenceInputPayload(
        evidence_id="evidence_001",
        source_artifact_refs=("artifact://snapshot_001",),
        artifact_summaries=(
            {
                "artifact_type": DatasetArtifactType.SNAPSHOT_BUNDLE.value,
                "row_count": fingerprint.row_count,
                "schema_hash": fingerprint.schema_hash,
            },
        ),
    )
    serialized = json.dumps(
        {
            "fingerprint": fingerprint.to_dict(),
            "bundle": bundle.to_dict(),
            "evidence": evidence.to_dict(),
        },
        sort_keys=True,
    )

    assert fingerprint.content_checksum is None
    assert "row_hashes" not in serialized
    assert "raw_rows" not in serialized
    assert "sample_values" not in serialized
    assert len(json.dumps(bundle.to_dict(), sort_keys=True)) < 20_000


def test_restore_dispatcher_and_root_surface_remain_private() -> None:
    import truthound

    fingerprint = fingerprint_dataset(_canonical_frame())
    restored = restore_dataset_artifact(fingerprint.to_envelope().to_dict())

    assert restored == fingerprint
    assert "datasets" not in dir(truthound)
    assert "depot" not in dir(truthound)
    assert "restore_dataset_artifact" not in dir(truthound)
    assert "DatasetSnapshotBundle" not in dir(truthound)
