"""Tests for replication base module."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, Mock
from datetime import datetime

from truthound.stores.replication.base import (
    ConflictResolution,
    ReadPreference,
    ReplicaHealth,
    ReplicaState,
    ReplicaTarget,
    ReplicationConfig,
    ReplicationMetrics,
    ReplicationMode,
)


class TestReplicationConfig:
    """Tests for ReplicationConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ReplicationConfig()
        assert config.mode == ReplicationMode.ASYNC
        assert config.read_preference == ReadPreference.PRIMARY
        assert config.conflict_resolution == ConflictResolution.LAST_WRITE_WINS
        assert config.min_sync_replicas == 1
        assert config.enable_health_checks is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ReplicationConfig(
            mode=ReplicationMode.SYNC,
            read_preference=ReadPreference.NEAREST,
            min_sync_replicas=2,
        )
        assert config.mode == ReplicationMode.SYNC
        assert config.read_preference == ReadPreference.NEAREST
        assert config.min_sync_replicas == 2

    def test_validate_semi_sync(self) -> None:
        """Test validation for semi-sync mode."""
        mock_store = MagicMock()
        targets = [
            ReplicaTarget(name="r1", store=mock_store, region="us-east-1"),
        ]

        # min_sync_replicas > targets should fail
        config = ReplicationConfig(
            mode=ReplicationMode.SEMI_SYNC,
            targets=targets,
            min_sync_replicas=2,
        )
        with pytest.raises(ValueError, match="min_sync_replicas cannot exceed"):
            config.validate()

    def test_validate_health_check_interval(self) -> None:
        """Test validation for health check interval."""
        config = ReplicationConfig(health_check_interval_seconds=0)
        with pytest.raises(ValueError, match="health_check_interval_seconds"):
            config.validate()


class TestReplicaTarget:
    """Tests for ReplicaTarget."""

    def test_creation(self) -> None:
        """Test target creation."""
        mock_store = MagicMock()
        target = ReplicaTarget(
            name="replica-1",
            store=mock_store,
            region="us-east-1",
        )
        assert target.name == "replica-1"
        assert target.region == "us-east-1"
        assert target.health == ReplicaHealth.UNKNOWN
        assert target.state == ReplicaState.ACTIVE

    def test_mark_healthy(self) -> None:
        """Test marking as healthy."""
        mock_store = MagicMock()
        target = ReplicaTarget(name="r1", store=mock_store, region="us-east-1")

        target.mark_healthy()
        assert target.health == ReplicaHealth.HEALTHY
        assert target.state == ReplicaState.ACTIVE
        assert target.last_sync_time is not None

    def test_mark_degraded(self) -> None:
        """Test marking as degraded."""
        mock_store = MagicMock()
        target = ReplicaTarget(name="r1", store=mock_store, region="us-east-1")

        target.mark_degraded(lag_ms=5000.0)
        assert target.health == ReplicaHealth.DEGRADED
        assert target.replication_lag_ms == 5000.0

    def test_mark_unhealthy(self) -> None:
        """Test marking as unhealthy."""
        mock_store = MagicMock()
        target = ReplicaTarget(name="r1", store=mock_store, region="us-east-1")

        target.mark_unhealthy()
        assert target.health == ReplicaHealth.UNHEALTHY
        assert target.state == ReplicaState.FAILED

    def test_pause_resume(self) -> None:
        """Test pause and resume."""
        mock_store = MagicMock()
        target = ReplicaTarget(name="r1", store=mock_store, region="us-east-1")

        target.pause()
        assert target.state == ReplicaState.PAUSED

        target.resume()
        assert target.state == ReplicaState.ACTIVE


class TestReplicationMetrics:
    """Tests for ReplicationMetrics."""

    def test_default_values(self) -> None:
        """Test default metrics values."""
        metrics = ReplicationMetrics()
        assert metrics.writes_to_primary == 0
        assert metrics.writes_replicated == 0
        assert metrics.writes_failed == 0
        assert metrics.conflicts_detected == 0

    def test_record_primary_write(self) -> None:
        """Test recording primary write."""
        metrics = ReplicationMetrics()
        metrics.record_primary_write()
        assert metrics.writes_to_primary == 1

    def test_record_replication_success(self) -> None:
        """Test recording successful replication."""
        metrics = ReplicationMetrics()
        metrics.record_replication_success("replica-1", 50.0)

        assert metrics.writes_replicated == 1
        assert metrics.total_replication_time_ms == 50.0
        assert metrics._replica_success_count["replica-1"] == 1

    def test_record_replication_failure(self) -> None:
        """Test recording replication failure."""
        metrics = ReplicationMetrics()
        metrics.record_replication_failure("replica-1")

        assert metrics.writes_failed == 1
        assert metrics._replica_failure_count["replica-1"] == 1

    def test_record_conflict(self) -> None:
        """Test recording conflict."""
        metrics = ReplicationMetrics()
        metrics.record_conflict(resolved=True)
        assert metrics.conflicts_detected == 1
        assert metrics.conflicts_resolved == 1

        metrics.record_conflict(resolved=False)
        assert metrics.conflicts_detected == 2
        assert metrics.conflicts_resolved == 1

    def test_update_lag(self) -> None:
        """Test updating lag."""
        metrics = ReplicationMetrics()
        metrics.update_lag("replica-1", 100.0)
        assert metrics.current_lag_ms["replica-1"] == 100.0

    def test_get_replica_success_rate(self) -> None:
        """Test success rate calculation."""
        metrics = ReplicationMetrics()

        # No data
        assert metrics.get_replica_success_rate("replica-1") == 100.0

        # Add some results
        metrics.record_replication_success("replica-1", 50.0)
        metrics.record_replication_success("replica-1", 50.0)
        metrics.record_replication_failure("replica-1")

        rate = metrics.get_replica_success_rate("replica-1")
        assert rate == pytest.approx(66.67, rel=0.1)

    def test_get_average_replication_time(self) -> None:
        """Test average replication time."""
        metrics = ReplicationMetrics()
        assert metrics.get_average_replication_time() == 0.0

        metrics.record_replication_success("r1", 100.0)
        metrics.record_replication_success("r2", 200.0)

        assert metrics.get_average_replication_time() == 150.0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        metrics = ReplicationMetrics()
        metrics.record_primary_write()
        metrics.record_replication_success("r1", 100.0)

        d = metrics.to_dict()
        assert d["writes_to_primary"] == 1
        assert d["writes_replicated"] == 1
        assert "replica_success_rates" in d


class TestEnums:
    """Tests for enum values."""

    def test_replication_mode(self) -> None:
        """Test ReplicationMode values."""
        assert ReplicationMode.SYNC.value == "sync"
        assert ReplicationMode.ASYNC.value == "async"
        assert ReplicationMode.SEMI_SYNC.value == "semi_sync"

    def test_read_preference(self) -> None:
        """Test ReadPreference values."""
        assert ReadPreference.PRIMARY.value == "primary"
        assert ReadPreference.SECONDARY.value == "secondary"
        assert ReadPreference.NEAREST.value == "nearest"
        assert ReadPreference.ANY.value == "any"

    def test_conflict_resolution(self) -> None:
        """Test ConflictResolution values."""
        assert ConflictResolution.LAST_WRITE_WINS.value == "last_write_wins"
        assert ConflictResolution.FIRST_WRITE_WINS.value == "first_write_wins"
        assert ConflictResolution.PRIMARY_WINS.value == "primary_wins"

    def test_replica_health(self) -> None:
        """Test ReplicaHealth values."""
        assert ReplicaHealth.HEALTHY.value == "healthy"
        assert ReplicaHealth.DEGRADED.value == "degraded"
        assert ReplicaHealth.UNHEALTHY.value == "unhealthy"

    def test_replica_state(self) -> None:
        """Test ReplicaState values."""
        assert ReplicaState.ACTIVE.value == "active"
        assert ReplicaState.PAUSED.value == "paused"
        assert ReplicaState.SYNCING.value == "syncing"
        assert ReplicaState.FAILED.value == "failed"
