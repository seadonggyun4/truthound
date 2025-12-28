"""Unit tests for the versioning module."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from truthound.stores.versioning.base import (
    VersionConflictError,
    VersionDiff,
    VersionInfo,
    VersioningConfig,
    VersioningMode,
    VersionNotFoundError,
)
from truthound.stores.versioning.strategies import (
    GitLikeStrategy,
    IncrementalStrategy,
    SemanticStrategy,
    TimestampStrategy,
    get_strategy,
)


class TestVersionInfo:
    """Tests for VersionInfo data class."""

    def test_version_info_creation(self) -> None:
        """Test creating version info."""
        info = VersionInfo(
            version=1,
            item_id="test-item",
            created_at=datetime.now(),
            message="Initial version",
        )

        assert info.version == 1
        assert info.item_id == "test-item"
        assert info.message == "Initial version"

    def test_version_info_to_dict(self) -> None:
        """Test serializing version info."""
        now = datetime.now()
        info = VersionInfo(
            version=1,
            item_id="test-item",
            created_at=now,
            created_by="user1",
        )

        data = info.to_dict()
        assert data["version"] == 1
        assert data["item_id"] == "test-item"
        assert data["created_by"] == "user1"

    def test_version_info_from_dict(self) -> None:
        """Test deserializing version info."""
        data = {
            "version": 2,
            "item_id": "test-item",
            "created_at": "2025-01-01T12:00:00",
            "message": "Test",
        }

        info = VersionInfo.from_dict(data)
        assert info.version == 2
        assert info.item_id == "test-item"


class TestVersioningStrategies:
    """Tests for versioning strategies."""

    def test_incremental_strategy(self) -> None:
        """Test incremental versioning."""
        strategy = IncrementalStrategy()

        assert strategy.get_next_version("item1", None) == 1
        assert strategy.get_next_version("item1", 1) == 2
        assert strategy.get_next_version("item1", 99) == 100

        assert strategy.format_version(1) == "v1"
        assert strategy.parse_version("v5") == 5

    def test_semantic_strategy(self) -> None:
        """Test semantic versioning."""
        strategy = SemanticStrategy()

        # First version
        assert strategy.get_next_version("item1", None) == 10000  # 1.0.0

        # Patch bump
        assert strategy.get_next_version("item1", 10000) == 10001  # 1.0.1

        # Minor bump
        v = strategy.get_next_version("item1", 10000, {"bump": "minor"})
        assert v == 10100  # 1.1.0

        # Major bump
        v = strategy.get_next_version("item1", 10000, {"bump": "major"})
        assert v == 20000  # 2.0.0

        # Format
        assert strategy.format_version(10203) == "1.2.3"
        assert strategy.parse_version("1.2.3") == 10203

    def test_timestamp_strategy(self) -> None:
        """Test timestamp versioning."""
        strategy = TimestampStrategy()

        v1 = strategy.get_next_version("item1", None)
        v2 = strategy.get_next_version("item1", v1)

        assert v2 >= v1

        # Format and parse
        formatted = strategy.format_version(v1)
        parsed = strategy.parse_version(formatted)
        assert abs(parsed - v1) < 1000  # Within 1 second

    def test_git_like_strategy(self) -> None:
        """Test git-like versioning."""
        strategy = GitLikeStrategy(hash_length=7)

        v1 = strategy.get_next_version("item1", None, {"content": "test data"})
        assert v1 == 1

        v2 = strategy.get_next_version("item1", v1, {"content": "more data"})
        assert v2 == 2

        # Should have hashes
        assert len(strategy.format_version(v1)) == 7

    def test_get_strategy(self) -> None:
        """Test strategy factory function."""
        assert isinstance(get_strategy("incremental"), IncrementalStrategy)
        assert isinstance(get_strategy("semantic"), SemanticStrategy)
        assert isinstance(get_strategy("timestamp"), TimestampStrategy)
        assert isinstance(get_strategy("git_like"), GitLikeStrategy)

        with pytest.raises(ValueError):
            get_strategy("unknown")


class TestVersioningConfig:
    """Tests for versioning configuration."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = VersioningConfig()

        assert config.mode == VersioningMode.INCREMENTAL
        assert config.max_versions == 0
        assert config.auto_cleanup is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = VersioningConfig(
            mode=VersioningMode.SEMANTIC,
            max_versions=10,
            require_message=True,
        )

        assert config.mode == VersioningMode.SEMANTIC
        assert config.max_versions == 10
        assert config.require_message is True


class TestVersionDiff:
    """Tests for version diff."""

    def test_diff_creation(self) -> None:
        """Test creating version diff."""
        diff = VersionDiff(
            item_id="test",
            version_a=1,
            version_b=2,
            changes=[
                {"path": "status", "type": "modified", "old": "a", "new": "b"},
            ],
            summary="1 change",
        )

        assert diff.item_id == "test"
        assert len(diff.changes) == 1

    def test_diff_to_dict(self) -> None:
        """Test serializing diff."""
        diff = VersionDiff(
            item_id="test",
            version_a=1,
            version_b=2,
        )

        data = diff.to_dict()
        assert data["version_a"] == 1
        assert data["version_b"] == 2


class TestVersioningExceptions:
    """Tests for versioning exceptions."""

    def test_version_conflict_error(self) -> None:
        """Test version conflict exception."""
        error = VersionConflictError("item1", 5, 7)

        assert error.item_id == "item1"
        assert error.expected_version == 5
        assert error.actual_version == 7
        assert "item1" in str(error)

    def test_version_not_found_error(self) -> None:
        """Test version not found exception."""
        error = VersionNotFoundError("item1", 99)

        assert error.item_id == "item1"
        assert error.version == 99
        assert "99" in str(error)
