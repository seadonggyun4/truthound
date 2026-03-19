from __future__ import annotations

from unittest.mock import patch

import polars as pl
import pytest

from truthound.realtime import (
    CheckpointManager,
    ConnectionError,
    IncrementalValidator,
    MockStreamingSource,
    StreamingValidator,
)


pytestmark = pytest.mark.fault


@pytest.mark.contract
def test_incremental_validator_tracks_cross_batch_duplicates():
    validator = IncrementalValidator(
        track_duplicates=True,
        duplicate_columns=["id"],
    )

    first = pl.DataFrame({"id": [1, 2], "value": [10.0, 20.0]})
    second = pl.DataFrame({"id": [2, 3], "value": [20.0, 30.0]})

    first_result = validator.validate_batch(first, batch_id="b1")
    second_result = validator.validate_batch(second, batch_id="b2")
    aggregate = validator.get_aggregate_stats()

    assert first_result.metadata["duplicate_count"] == 0
    assert second_result.metadata["duplicate_count"] == 1
    assert aggregate["unique_records_seen"] == 3


def test_mock_streaming_source_requires_a_connection_before_reads():
    source = MockStreamingSource(records_per_batch=2, num_batches=1, error_rate=0.0)

    with pytest.raises(ConnectionError):
        source.read_batch(max_records=2)


def test_streaming_validator_isolates_check_failures_by_default():
    validator = StreamingValidator(validators=["null"])
    batch = pl.DataFrame({"id": [1], "value": [10.0]})

    with patch("truthound.api.check", side_effect=RuntimeError("validator backend offline")):
        result = validator.validate_batch(batch, batch_id="batch-1")

    assert result.record_count == 1
    assert result.issue_count == 0


def test_incremental_validator_restore_rejects_unknown_checkpoints(tmp_path):
    manager = CheckpointManager(tmp_path)
    validator = IncrementalValidator(checkpoint_manager=manager)

    assert validator.restore("missing-checkpoint") is False
