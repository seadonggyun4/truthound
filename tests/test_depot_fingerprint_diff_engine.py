from __future__ import annotations

import polars as pl
import pytest

from truthound._datasets import (
    DatasetDiffCategory,
    DatasetFingerprintOptions,
    diff_datasets,
    diff_fingerprints,
    fingerprint_dataset,
    normalize_dataset_input,
)
from truthound._datasets._serialization import DatasetArtifactContractError
from truthound._redaction import RedactionViolationError


def test_fingerprint_is_deterministic_for_row_order_changes() -> None:
    left = pl.DataFrame({"id": [1, 2, 3], "value": ["aa", "bb", "cc"]})
    right = pl.DataFrame({"id": [3, 1, 2], "value": ["cc", "aa", "bb"]})

    left_fingerprint = fingerprint_dataset(left)
    right_fingerprint = fingerprint_dataset(right)

    assert left_fingerprint.schema_hash == right_fingerprint.schema_hash
    assert left_fingerprint.column_list_hash == right_fingerprint.column_list_hash
    assert left_fingerprint.null_profile_hash == right_fingerprint.null_profile_hash
    assert left_fingerprint.sampled_row_hash == right_fingerprint.sampled_row_hash


def test_fingerprint_supports_stable_key_sampling() -> None:
    left = pl.DataFrame({"id": [1, 2, 3], "value": ["aa", "bb", "cc"]})
    right = pl.DataFrame({"id": [2, 3, 1], "value": ["bb", "cc", "aa"]})

    left_fingerprint = fingerprint_dataset(
        left,
        sample_size=2,
        stable_key_columns=("id",),
    )
    right_fingerprint = fingerprint_dataset(
        right,
        sample_size=2,
        stable_key_columns=("id",),
    )

    assert left_fingerprint.sampled_row_hash == right_fingerprint.sampled_row_hash
    assert left_fingerprint.metadata["stable_key_columns"] == ["id"]


def test_diff_detects_schema_added_without_row_level_promise() -> None:
    left = pl.DataFrame({"id": [1, 2], "value": ["aa", "bb"]})
    right = pl.DataFrame({"id": [1, 2], "value": ["aa", "bb"], "status": ["ok", "ok"]})

    diff = diff_datasets(
        left,
        right,
        source_snapshot_id="snapshot_a",
        target_snapshot_id="snapshot_b",
    )

    assert DatasetDiffCategory.SCHEMA_ADDED in diff.categories
    assert diff.summary["row_level_diff_available"] is False
    assert diff.details["schema"]["added"] == [{"name": "status", "dtype": "String"}]


def test_diff_detects_schema_order_changes() -> None:
    left = pl.DataFrame({"id": [1, 2], "value": ["aa", "bb"]})
    right = pl.DataFrame({"value": ["aa", "bb"], "id": [1, 2]})

    diff = diff_datasets(
        left,
        right,
        source_snapshot_id="snapshot_a",
        target_snapshot_id="snapshot_b",
    )

    assert diff.categories == (DatasetDiffCategory.SCHEMA_CHANGED,)
    assert diff.details["schema"]["column_order_changed"] is True


def test_diff_detects_null_profile_and_profile_changes() -> None:
    left = pl.DataFrame({"id": [1, 2, 3], "value": ["aa", "bb", "cc"]})
    right = pl.DataFrame({"id": [1, 2, 3], "value": ["aa", None, "cc"]})

    diff = diff_datasets(
        left,
        right,
        source_snapshot_id="snapshot_a",
        target_snapshot_id="snapshot_b",
    )

    assert DatasetDiffCategory.NULL_PROFILE_CHANGED in diff.categories
    assert DatasetDiffCategory.PROFILE_CHANGED in diff.categories
    assert diff.details["profile"]["changed_columns"] == ["value"]


def test_diff_detects_sample_changes_without_raw_rows() -> None:
    left = pl.DataFrame({"id": [1, 2], "value": ["aa", "bb"]})
    right = pl.DataFrame({"id": [1, 2], "value": ["aa", "cc"]})

    diff = diff_datasets(
        left,
        right,
        source_snapshot_id="snapshot_a",
        target_snapshot_id="snapshot_b",
    )

    assert DatasetDiffCategory.SAMPLE_CHANGED in diff.categories
    assert "raw_rows" not in str(diff.to_dict())
    assert diff.details["sampled_row_digest"]["changed"] is True


def test_diff_detects_content_checksum_only_changes_as_unknown() -> None:
    left = fingerprint_dataset(pl.DataFrame({"id": [1]}), include_content_checksum=True)
    right = fingerprint_dataset(pl.DataFrame({"id": [1]}), include_content_checksum=True)
    mutated = type(right)(
        schema_hash=right.schema_hash,
        column_list_hash=right.column_list_hash,
        row_count=right.row_count,
        null_profile_hash=right.null_profile_hash,
        sampled_row_hash=right.sampled_row_hash,
        content_checksum="sha256:changed",
        metadata=right.metadata,
    )

    checksum_diff = diff_fingerprints(
        left,
        mutated,
        source_snapshot_id="snapshot_a",
        target_snapshot_id="snapshot_b",
    )

    assert checksum_diff.categories == (DatasetDiffCategory.UNKNOWN_CHANGED,)
    assert checksum_diff.details["content_checksum"]["changed"] is True


def test_normalize_dataset_input_reads_local_csv(tmp_path) -> None:
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text("id,value\n1,aa\n2,bb\n", encoding="utf-8")

    frame = normalize_dataset_input(csv_path)
    fingerprint = fingerprint_dataset(csv_path)

    assert frame.shape == (2, 2)
    assert fingerprint.row_count == 2


def test_fingerprint_options_reject_zero_sample_size() -> None:
    with pytest.raises(DatasetArtifactContractError, match="sample_size"):
        DatasetFingerprintOptions(sample_size=0)


def test_fingerprint_and_diff_reject_unsafe_metadata() -> None:
    frame = pl.DataFrame({"id": [1], "value": ["aa"]})

    with pytest.raises(RedactionViolationError, match="raw_rows"):
        fingerprint_dataset(frame, metadata={"raw_rows": [{"value": "aa"}]})

    with pytest.raises(RedactionViolationError, match="PII-like"):
        diff_datasets(
            frame,
            frame,
            source_snapshot_id="snapshot_a",
            target_snapshot_id="snapshot_b",
            metadata={"note": "contact person@example.com"},
        )


def test_fingerprint_and_diff_engine_remain_private_to_root_package() -> None:
    import truthound

    exported = dir(truthound)

    assert "DatasetFingerprintOptions" not in exported
    assert "fingerprint_dataset" not in exported
    assert "diff_datasets" not in exported
