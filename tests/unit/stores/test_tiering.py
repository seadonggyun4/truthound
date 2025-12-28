"""Unit tests for the tiered storage module."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from truthound.stores.tiering.base import (
    InMemoryTierMetadataStore,
    MigrationDirection,
    StorageTier,
    TierAccessError,
    TierInfo,
    TierMigrationError,
    TierNotFoundError,
    TierPolicy,
    TierType,
    TieringConfig,
    TieringResult,
)
from truthound.stores.tiering.policies import (
    AccessBasedTierPolicy,
    AgeBasedTierPolicy,
    CompositeTierPolicy,
    ScheduledTierPolicy,
    SizeBasedTierPolicy,
)


class TestStorageTier:
    """Tests for StorageTier data class."""

    def test_tier_creation(self) -> None:
        """Test creating a storage tier."""
        mock_store = MagicMock()

        tier = StorageTier(
            name="hot",
            store=mock_store,
            tier_type=TierType.HOT,
            priority=1,
        )

        assert tier.name == "hot"
        assert tier.tier_type == TierType.HOT
        assert tier.priority == 1

    def test_tier_with_cost(self) -> None:
        """Test tier with cost information."""
        mock_store = MagicMock()

        tier = StorageTier(
            name="archive",
            store=mock_store,
            tier_type=TierType.ARCHIVE,
            cost_per_gb=0.01,
            retrieval_time_ms=3600000,  # 1 hour
        )

        assert tier.cost_per_gb == 0.01
        assert tier.retrieval_time_ms == 3600000

    def test_tier_invalid_priority(self) -> None:
        """Test that invalid priority raises error."""
        mock_store = MagicMock()

        with pytest.raises(ValueError):
            StorageTier(
                name="invalid",
                store=mock_store,
                priority=0,  # Must be >= 1
            )

    def test_tier_to_dict(self) -> None:
        """Test serializing tier."""
        mock_store = MagicMock()

        tier = StorageTier(
            name="warm",
            store=mock_store,
            tier_type=TierType.WARM,
            metadata={"region": "us-east-1"},
        )

        data = tier.to_dict()

        assert data["name"] == "warm"
        assert data["tier_type"] == "warm"
        assert data["metadata"]["region"] == "us-east-1"
        # Store should not be serialized
        assert "store" not in data


class TestTierInfo:
    """Tests for TierInfo data class."""

    def test_info_creation(self) -> None:
        """Test creating tier info."""
        info = TierInfo(
            item_id="test-item",
            tier_name="hot",
            created_at=datetime.now(),
            size_bytes=1024,
        )

        assert info.item_id == "test-item"
        assert info.tier_name == "hot"
        assert info.size_bytes == 1024

    def test_info_with_access_tracking(self) -> None:
        """Test tier info with access tracking."""
        info = TierInfo(
            item_id="test-item",
            tier_name="hot",
            created_at=datetime.now(),
            access_count=10,
            last_accessed=datetime.now(),
        )

        assert info.access_count == 10
        assert info.last_accessed is not None

    def test_info_to_dict(self) -> None:
        """Test serializing tier info."""
        created = datetime(2025, 1, 1, 12, 0, 0)

        info = TierInfo(
            item_id="test-item",
            tier_name="warm",
            created_at=created,
        )

        data = info.to_dict()

        assert data["item_id"] == "test-item"
        assert data["tier_name"] == "warm"
        assert data["created_at"] == "2025-01-01T12:00:00"

    def test_info_from_dict(self) -> None:
        """Test deserializing tier info."""
        data = {
            "item_id": "test-item",
            "tier_name": "cold",
            "created_at": "2025-01-01T12:00:00",
            "access_count": 5,
        }

        info = TierInfo.from_dict(data)

        assert info.item_id == "test-item"
        assert info.tier_name == "cold"
        assert info.access_count == 5


class TestTieringResult:
    """Tests for TieringResult data class."""

    def test_result_duration(self) -> None:
        """Test result duration calculation."""
        result = TieringResult(
            start_time=datetime(2025, 1, 1, 12, 0, 0),
            end_time=datetime(2025, 1, 1, 12, 5, 0),
        )

        assert result.duration_seconds == 300.0

    def test_result_to_dict(self) -> None:
        """Test serializing result."""
        result = TieringResult(
            start_time=datetime.now(),
            items_scanned=100,
            items_migrated=10,
            bytes_migrated=10240,
        )

        data = result.to_dict()

        assert data["items_scanned"] == 100
        assert data["items_migrated"] == 10
        assert data["bytes_migrated"] == 10240


class TestTieringConfig:
    """Tests for TieringConfig data class."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = TieringConfig()

        assert config.default_tier == "hot"
        assert config.enable_promotion is True
        assert config.promotion_threshold == 10

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = TieringConfig(
            default_tier="warm",
            enable_promotion=False,
            batch_size=50,
        )

        assert config.default_tier == "warm"
        assert config.enable_promotion is False
        assert config.batch_size == 50


class TestAgeBasedTierPolicy:
    """Tests for age-based tier policy."""

    def test_demote_old_items(self) -> None:
        """Test demoting old items."""
        policy = AgeBasedTierPolicy(
            from_tier="hot",
            to_tier="warm",
            after_days=7,
        )

        old_item = TierInfo(
            item_id="old",
            tier_name="hot",
            created_at=datetime.now() - timedelta(days=10),
        )

        recent_item = TierInfo(
            item_id="recent",
            tier_name="hot",
            created_at=datetime.now() - timedelta(days=3),
        )

        assert policy.should_migrate(old_item) is True
        assert policy.should_migrate(recent_item) is False

    def test_policy_properties(self) -> None:
        """Test policy properties."""
        policy = AgeBasedTierPolicy(
            from_tier="warm",
            to_tier="cold",
            after_days=30,
        )

        assert policy.from_tier == "warm"
        assert policy.to_tier == "cold"
        assert "30 days" in policy.description


class TestAccessBasedTierPolicy:
    """Tests for access-based tier policy."""

    def test_promote_frequently_accessed(self) -> None:
        """Test promoting frequently accessed items."""
        policy = AccessBasedTierPolicy(
            from_tier="warm",
            to_tier="hot",
            min_access_count=10,
            access_window_days=7,
            direction=MigrationDirection.PROMOTE,
        )

        frequent = TierInfo(
            item_id="frequent",
            tier_name="warm",
            created_at=datetime.now(),
            access_count=15,
            last_accessed=datetime.now() - timedelta(days=1),  # Recent access
        )

        infrequent = TierInfo(
            item_id="infrequent",
            tier_name="warm",
            created_at=datetime.now(),
            access_count=3,
            last_accessed=datetime.now() - timedelta(days=1),
        )

        assert policy.should_migrate(frequent) is True
        assert policy.should_migrate(infrequent) is False

    def test_demote_rarely_accessed(self) -> None:
        """Test demoting rarely accessed items."""
        policy = AccessBasedTierPolicy(
            from_tier="hot",
            to_tier="warm",
            inactive_days=30,
            direction=MigrationDirection.DEMOTE,
        )

        # No access for 35 days -> should demote
        rare = TierInfo(
            item_id="rare",
            tier_name="hot",
            created_at=datetime.now() - timedelta(days=60),
            access_count=2,
            last_accessed=datetime.now() - timedelta(days=35),
        )

        assert policy.should_migrate(rare) is True


class TestSizeBasedTierPolicy:
    """Tests for size-based tier policy."""

    def test_demote_large_items(self) -> None:
        """Test demoting items over size threshold."""
        policy = SizeBasedTierPolicy(
            from_tier="hot",
            to_tier="cold",
            min_size_mb=10,
        )

        large = TierInfo(
            item_id="large",
            tier_name="hot",
            created_at=datetime.now(),
            size_bytes=15 * 1024 * 1024,  # 15 MB
        )

        small = TierInfo(
            item_id="small",
            tier_name="hot",
            created_at=datetime.now(),
            size_bytes=5 * 1024 * 1024,  # 5 MB
        )

        assert policy.should_migrate(large) is True
        assert policy.should_migrate(small) is False

    def test_priority_based_on_size(self) -> None:
        """Test migration priority based on size."""
        policy = SizeBasedTierPolicy(
            from_tier="hot",
            to_tier="cold",
            min_size_mb=10,
        )

        item = TierInfo(
            item_id="item",
            tier_name="hot",
            created_at=datetime.now(),
            size_bytes=20 * 1024 * 1024,
        )

        # Larger items should have higher priority
        priority = policy.get_priority(item)
        assert priority > 0


class TestScheduledTierPolicy:
    """Tests for scheduled tier policy."""

    def test_scheduled_migration_by_day(self) -> None:
        """Test scheduled migration based on day of week."""
        today = datetime.now().weekday()

        policy = ScheduledTierPolicy(
            from_tier="hot",
            to_tier="archive",
            on_days=[today],  # Today's day of week
            min_age_days=7,
        )

        # Old item on the right day -> should migrate
        old_item = TierInfo(
            item_id="item",
            tier_name="hot",
            created_at=datetime.now() - timedelta(days=30),
        )

        assert policy.should_migrate(old_item) is True

    def test_scheduled_migration_wrong_day(self) -> None:
        """Test scheduled migration on wrong day."""
        today = datetime.now().weekday()
        wrong_day = (today + 1) % 7  # Tomorrow

        policy = ScheduledTierPolicy(
            from_tier="hot",
            to_tier="archive",
            on_days=[wrong_day],
        )

        item = TierInfo(
            item_id="item",
            tier_name="hot",
            created_at=datetime.now(),
        )

        assert policy.should_migrate(item) is False

    def test_scheduled_migration_age_requirement(self) -> None:
        """Test scheduled migration with age requirement."""
        today = datetime.now().weekday()

        policy = ScheduledTierPolicy(
            from_tier="hot",
            to_tier="archive",
            on_days=[today],
            min_age_days=30,
        )

        # Item too young -> should not migrate
        young_item = TierInfo(
            item_id="item",
            tier_name="hot",
            created_at=datetime.now() - timedelta(days=10),
        )

        assert policy.should_migrate(young_item) is False


class TestCompositeTierPolicy:
    """Tests for composite tier policy."""

    def test_all_mode(self) -> None:
        """Test ALL mode (all policies must agree)."""
        policy = CompositeTierPolicy(
            from_tier="hot",
            to_tier="cold",
            policies=[
                AgeBasedTierPolicy("hot", "cold", after_days=7),
                SizeBasedTierPolicy("hot", "cold", min_size_mb=10),
            ],
            require_all=True,
        )

        # Old AND large -> should migrate
        old_large = TierInfo(
            item_id="old_large",
            tier_name="hot",
            created_at=datetime.now() - timedelta(days=10),
            size_bytes=15 * 1024 * 1024,
        )

        # Old but small -> should not migrate (size policy fails)
        old_small = TierInfo(
            item_id="old_small",
            tier_name="hot",
            created_at=datetime.now() - timedelta(days=10),
            size_bytes=5 * 1024 * 1024,
        )

        assert policy.should_migrate(old_large) is True
        assert policy.should_migrate(old_small) is False

    def test_any_mode(self) -> None:
        """Test ANY mode (any policy can trigger)."""
        policy = CompositeTierPolicy(
            from_tier="hot",
            to_tier="cold",
            policies=[
                AgeBasedTierPolicy("hot", "cold", after_days=30),
                SizeBasedTierPolicy("hot", "cold", min_size_mb=10),
            ],
            require_all=False,
        )

        # Young but large -> should migrate (size policy matches)
        young_large = TierInfo(
            item_id="young_large",
            tier_name="hot",
            created_at=datetime.now() - timedelta(days=5),
            size_bytes=15 * 1024 * 1024,
        )

        # Old but small -> should migrate (age policy matches)
        old_small = TierInfo(
            item_id="old_small",
            tier_name="hot",
            created_at=datetime.now() - timedelta(days=60),
            size_bytes=5 * 1024 * 1024,
        )

        # Young and small -> should not migrate
        young_small = TierInfo(
            item_id="young_small",
            tier_name="hot",
            created_at=datetime.now() - timedelta(days=5),
            size_bytes=5 * 1024 * 1024,
        )

        assert policy.should_migrate(young_large) is True
        assert policy.should_migrate(old_small) is True
        assert policy.should_migrate(young_small) is False


class TestInMemoryTierMetadataStore:
    """Tests for in-memory tier metadata store."""

    def test_save_and_get(self) -> None:
        """Test saving and retrieving tier info."""
        store = InMemoryTierMetadataStore()

        info = TierInfo(
            item_id="test-item",
            tier_name="hot",
            created_at=datetime.now(),
        )

        store.save_info(info)
        retrieved = store.get_info("test-item")

        assert retrieved is not None
        assert retrieved.item_id == "test-item"
        assert retrieved.tier_name == "hot"

    def test_get_not_found(self) -> None:
        """Test getting non-existent item."""
        store = InMemoryTierMetadataStore()

        retrieved = store.get_info("nonexistent")

        assert retrieved is None

    def test_delete(self) -> None:
        """Test deleting tier info."""
        store = InMemoryTierMetadataStore()

        info = TierInfo(
            item_id="test-item",
            tier_name="hot",
            created_at=datetime.now(),
        )

        store.save_info(info)
        result = store.delete_info("test-item")

        assert result is True
        assert store.get_info("test-item") is None

    def test_delete_not_found(self) -> None:
        """Test deleting non-existent item."""
        store = InMemoryTierMetadataStore()

        result = store.delete_info("nonexistent")

        assert result is False

    def test_list_by_tier(self) -> None:
        """Test listing items by tier."""
        store = InMemoryTierMetadataStore()

        for i in range(3):
            store.save_info(
                TierInfo(
                    item_id=f"hot-{i}",
                    tier_name="hot",
                    created_at=datetime.now(),
                )
            )

        for i in range(2):
            store.save_info(
                TierInfo(
                    item_id=f"cold-{i}",
                    tier_name="cold",
                    created_at=datetime.now(),
                )
            )

        hot_items = store.list_by_tier("hot")
        cold_items = store.list_by_tier("cold")

        assert len(hot_items) == 3
        assert len(cold_items) == 2

    def test_update_access(self) -> None:
        """Test updating access statistics."""
        store = InMemoryTierMetadataStore()

        info = TierInfo(
            item_id="test-item",
            tier_name="hot",
            created_at=datetime.now(),
            access_count=0,
        )

        store.save_info(info)
        store.update_access("test-item")
        store.update_access("test-item")

        updated = store.get_info("test-item")

        assert updated is not None
        assert updated.access_count == 2
        assert updated.last_accessed is not None


class TestTierExceptions:
    """Tests for tiering exceptions."""

    def test_tier_not_found_error(self) -> None:
        """Test TierNotFoundError."""
        error = TierNotFoundError("archive")

        assert error.tier_name == "archive"
        assert "archive" in str(error)

    def test_tier_migration_error(self) -> None:
        """Test TierMigrationError."""
        error = TierMigrationError(
            "item-123",
            "hot",
            "cold",
            "Connection timeout",
        )

        assert error.item_id == "item-123"
        assert error.from_tier == "hot"
        assert error.to_tier == "cold"
        assert "Connection timeout" in str(error)

    def test_tier_access_error(self) -> None:
        """Test TierAccessError."""
        error = TierAccessError("cold", "Bucket not found")

        assert error.tier_name == "cold"
        assert "Bucket not found" in str(error)


class TestTierType:
    """Tests for TierType enum."""

    def test_tier_types(self) -> None:
        """Test tier type values."""
        assert TierType.HOT.value == "hot"
        assert TierType.WARM.value == "warm"
        assert TierType.COLD.value == "cold"
        assert TierType.ARCHIVE.value == "archive"


class TestMigrationDirection:
    """Tests for MigrationDirection enum."""

    def test_demote_direction(self) -> None:
        """Test demote direction."""
        assert MigrationDirection.DEMOTE.value == "demote"

    def test_promote_direction(self) -> None:
        """Test promote direction."""
        assert MigrationDirection.PROMOTE.value == "promote"
