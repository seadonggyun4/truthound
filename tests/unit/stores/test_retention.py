"""Unit tests for the retention module."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from truthound.stores.retention.base import (
    ItemMetadata,
    PolicyMode,
    RetentionAction,
    RetentionConfig,
    RetentionResult,
    RetentionSchedule,
)
from truthound.stores.retention.policies import (
    CompositePolicy,
    CountBasedPolicy,
    SizeBasedPolicy,
    StatusBasedPolicy,
    TagBasedPolicy,
    TimeBasedPolicy,
)


class TestItemMetadata:
    """Tests for ItemMetadata data class."""

    def test_metadata_creation(self) -> None:
        """Test creating item metadata."""
        metadata = ItemMetadata(
            item_id="test-item",
            data_asset="test.csv",
            created_at=datetime.now(),
            size_bytes=1024,
            status="success",
        )

        assert metadata.item_id == "test-item"
        assert metadata.size_bytes == 1024


class TestTimeBasedPolicy:
    """Tests for time-based retention policy."""

    def test_retain_recent_items(self) -> None:
        """Test that recent items are retained."""
        policy = TimeBasedPolicy(max_age_days=7)

        recent = ItemMetadata(
            item_id="recent",
            data_asset="test.csv",
            created_at=datetime.now() - timedelta(days=3),
        )

        assert policy.should_retain(recent)

    def test_delete_old_items(self) -> None:
        """Test that old items are not retained."""
        policy = TimeBasedPolicy(max_age_days=7)

        old = ItemMetadata(
            item_id="old",
            data_asset="test.csv",
            created_at=datetime.now() - timedelta(days=10),
        )

        assert not policy.should_retain(old)

    def test_with_hours(self) -> None:
        """Test policy with hours."""
        policy = TimeBasedPolicy(max_age_hours=12)

        recent = ItemMetadata(
            item_id="recent",
            data_asset="test.csv",
            created_at=datetime.now() - timedelta(hours=6),
        )

        old = ItemMetadata(
            item_id="old",
            data_asset="test.csv",
            created_at=datetime.now() - timedelta(hours=18),
        )

        assert policy.should_retain(recent)
        assert not policy.should_retain(old)

    def test_expiry_time(self) -> None:
        """Test getting expiry time."""
        policy = TimeBasedPolicy(max_age_days=30)

        item = ItemMetadata(
            item_id="test",
            data_asset="test.csv",
            created_at=datetime(2025, 1, 1),
        )

        expiry = policy.get_expiry_time(item)
        assert expiry == datetime(2025, 1, 31)

    def test_invalid_threshold(self) -> None:
        """Test that invalid threshold raises error."""
        with pytest.raises(ValueError):
            TimeBasedPolicy()  # No age specified


class TestCountBasedPolicy:
    """Tests for count-based retention policy."""

    def test_prepare_batch(self) -> None:
        """Test batch preparation."""
        policy = CountBasedPolicy(max_count=2)

        items = [
            ItemMetadata(
                item_id=f"item{i}",
                data_asset="test.csv",
                created_at=datetime.now() - timedelta(days=i),
            )
            for i in range(5)
        ]

        policy.prepare_batch(items)

        # Newest items should be retained
        assert policy.should_retain(items[0])  # Newest
        assert policy.should_retain(items[1])  # Second newest
        assert not policy.should_retain(items[2])  # Third
        assert not policy.should_retain(items[3])
        assert not policy.should_retain(items[4])  # Oldest

    def test_per_asset(self) -> None:
        """Test per-asset counting."""
        policy = CountBasedPolicy(max_count=2, per_asset=True)

        items = [
            ItemMetadata(
                item_id="a1",
                data_asset="asset_a.csv",
                created_at=datetime.now() - timedelta(days=1),
            ),
            ItemMetadata(
                item_id="a2",
                data_asset="asset_a.csv",
                created_at=datetime.now() - timedelta(days=2),
            ),
            ItemMetadata(
                item_id="a3",
                data_asset="asset_a.csv",
                created_at=datetime.now() - timedelta(days=3),
            ),
            ItemMetadata(
                item_id="b1",
                data_asset="asset_b.csv",
                created_at=datetime.now() - timedelta(days=1),
            ),
        ]

        policy.prepare_batch(items)

        # Asset A: keep 2 newest
        assert policy.should_retain(items[0])
        assert policy.should_retain(items[1])
        assert not policy.should_retain(items[2])

        # Asset B: keep all (only 1 item)
        assert policy.should_retain(items[3])

    def test_invalid_max_count(self) -> None:
        """Test that invalid max_count raises error."""
        with pytest.raises(ValueError):
            CountBasedPolicy(max_count=0)


class TestSizeBasedPolicy:
    """Tests for size-based retention policy."""

    def test_size_limit(self) -> None:
        """Test size limit enforcement."""
        policy = SizeBasedPolicy(max_size_mb=1)  # 1 MB

        items = [
            ItemMetadata(
                item_id=f"item{i}",
                data_asset="test.csv",
                created_at=datetime.now() - timedelta(days=i),
                size_bytes=300 * 1024,  # 300 KB each
            )
            for i in range(5)
        ]

        policy.prepare_batch(items)

        # Only ~3 items fit in 1MB
        assert policy.should_retain(items[0])
        assert policy.should_retain(items[1])
        assert policy.should_retain(items[2])
        assert not policy.should_retain(items[3])
        assert not policy.should_retain(items[4])

    def test_invalid_size(self) -> None:
        """Test that invalid size raises error."""
        with pytest.raises(ValueError):
            SizeBasedPolicy()  # No size specified


class TestStatusBasedPolicy:
    """Tests for status-based retention policy."""

    def test_delete_failures(self) -> None:
        """Test deleting failed results."""
        policy = StatusBasedPolicy(status="failure", retain=False)

        success = ItemMetadata(
            item_id="success",
            data_asset="test.csv",
            created_at=datetime.now(),
            status="success",
        )

        failure = ItemMetadata(
            item_id="failure",
            data_asset="test.csv",
            created_at=datetime.now(),
            status="failure",
        )

        assert policy.should_retain(success)
        assert not policy.should_retain(failure)

    def test_keep_failures_for_limited_time(self) -> None:
        """Test keeping failures for limited time."""
        policy = StatusBasedPolicy(
            status="failure",
            max_age_days=7,
            retain=True,
        )

        recent_failure = ItemMetadata(
            item_id="recent",
            data_asset="test.csv",
            created_at=datetime.now() - timedelta(days=3),
            status="failure",
        )

        old_failure = ItemMetadata(
            item_id="old",
            data_asset="test.csv",
            created_at=datetime.now() - timedelta(days=10),
            status="failure",
        )

        assert policy.should_retain(recent_failure)
        assert not policy.should_retain(old_failure)


class TestTagBasedPolicy:
    """Tests for tag-based retention policy."""

    def test_required_tags(self) -> None:
        """Test requiring specific tags."""
        policy = TagBasedPolicy(required_tags={"env": "production"})

        prod = ItemMetadata(
            item_id="prod",
            data_asset="test.csv",
            created_at=datetime.now(),
            tags={"env": "production"},
        )

        dev = ItemMetadata(
            item_id="dev",
            data_asset="test.csv",
            created_at=datetime.now(),
            tags={"env": "development"},
        )

        assert policy.should_retain(prod)
        assert not policy.should_retain(dev)

    def test_delete_tags(self) -> None:
        """Test deleting items with specific tags."""
        policy = TagBasedPolicy(delete_tags={"temp": "true"})

        permanent = ItemMetadata(
            item_id="perm",
            data_asset="test.csv",
            created_at=datetime.now(),
            tags={"temp": "false"},
        )

        temp = ItemMetadata(
            item_id="temp",
            data_asset="test.csv",
            created_at=datetime.now(),
            tags={"temp": "true"},
        )

        assert policy.should_retain(permanent)
        assert not policy.should_retain(temp)


class TestCompositePolicy:
    """Tests for composite policy."""

    def test_all_mode(self) -> None:
        """Test ALL mode (all policies must agree)."""
        policy = CompositePolicy(
            policies=[
                TimeBasedPolicy(max_age_days=7),
                StatusBasedPolicy(status="failure", retain=False),
            ],
            mode=PolicyMode.ALL,
        )

        # Recent failure - time says keep, status says delete
        recent_failure = ItemMetadata(
            item_id="rf",
            data_asset="test.csv",
            created_at=datetime.now(),
            status="failure",
        )

        # Old success - time says delete, status says keep
        old_success = ItemMetadata(
            item_id="os",
            data_asset="test.csv",
            created_at=datetime.now() - timedelta(days=10),
            status="success",
        )

        # In ALL mode, both must agree to keep
        assert not policy.should_retain(recent_failure)
        assert not policy.should_retain(old_success)

    def test_any_mode(self) -> None:
        """Test ANY mode (any policy can keep)."""
        policy = CompositePolicy(
            policies=[
                TimeBasedPolicy(max_age_days=7),
                TagBasedPolicy(required_tags={"keep": "true"}),
            ],
            mode=PolicyMode.ANY,
        )

        # Old but tagged to keep
        old_tagged = ItemMetadata(
            item_id="old",
            data_asset="test.csv",
            created_at=datetime.now() - timedelta(days=10),
            tags={"keep": "true"},
        )

        # In ANY mode, either policy can save the item
        assert policy.should_retain(old_tagged)


class TestRetentionConfig:
    """Tests for retention configuration."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = RetentionConfig()

        assert len(config.policies) == 0
        assert config.mode == PolicyMode.ALL
        assert config.default_action == RetentionAction.DELETE

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = RetentionConfig(
            policies=[TimeBasedPolicy(max_age_days=30)],
            preserve_latest=True,
            dry_run=True,
        )

        assert len(config.policies) == 1
        assert config.preserve_latest is True
        assert config.dry_run is True


class TestRetentionResult:
    """Tests for retention result."""

    def test_result_properties(self) -> None:
        """Test result properties."""
        result = RetentionResult(
            start_time=datetime(2025, 1, 1, 12, 0, 0),
            end_time=datetime(2025, 1, 1, 12, 1, 0),
            items_scanned=100,
            items_deleted=10,
            items_preserved=90,
            bytes_freed=1024,
        )

        assert result.duration_seconds == 60.0
        assert result.total_processed == 100

    def test_result_to_dict(self) -> None:
        """Test serializing result."""
        result = RetentionResult(
            start_time=datetime.now(),
            items_deleted=5,
        )

        data = result.to_dict()
        assert "items_deleted" in data
        assert data["items_deleted"] == 5
