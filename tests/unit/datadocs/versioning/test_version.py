"""Tests for versioning version module."""

import pytest
import time
from datetime import datetime
from truthound.datadocs.versioning.version import (
    VersionInfo,
    ReportVersion,
    VersioningStrategy,
    IncrementalStrategy,
    SemanticStrategy,
    TimestampStrategy,
    GitLikeStrategy,
)


class TestVersionInfo:
    """Tests for VersionInfo class."""

    def test_create_version_info(self):
        """Test creating version info."""
        info = VersionInfo(
            version=1,
            report_id="test_report",
            created_at=datetime.now(),
        )
        assert info.version == 1
        assert info.report_id == "test_report"
        assert info.parent_version is None

    def test_version_info_with_all_fields(self):
        """Test version info with all fields."""
        now = datetime.now()
        info = VersionInfo(
            version=2,
            report_id="test_report",
            created_at=now,
            created_by="user@example.com",
            message="Updated report",
            parent_version=1,
            checksum="abc123",
            size_bytes=1024,
            metadata={"format": "html"},
        )
        assert info.version == 2
        assert info.created_by == "user@example.com"
        assert info.message == "Updated report"
        assert info.parent_version == 1
        assert info.checksum == "abc123"
        assert info.size_bytes == 1024

    def test_version_info_to_dict(self):
        """Test converting to dict."""
        info = VersionInfo(
            version=1,
            report_id="test",
            created_at=datetime(2025, 12, 28, 12, 0, 0),
        )
        d = info.to_dict()
        assert d["version"] == 1
        assert d["report_id"] == "test"
        assert "created_at" in d

    def test_version_info_from_dict(self):
        """Test creating from dict."""
        d = {
            "version": 1,
            "report_id": "test",
            "created_at": "2025-12-28T12:00:00",
        }
        info = VersionInfo.from_dict(d)
        assert info.version == 1
        assert info.report_id == "test"


class TestReportVersion:
    """Tests for ReportVersion class."""

    def test_create_report_version(self):
        """Test creating report version."""
        info = VersionInfo(
            version=1,
            report_id="test",
            created_at=datetime.now(),
        )
        version = ReportVersion(
            info=info,
            content="<html>Report</html>",
            format="html",
        )
        assert version.version == 1
        assert version.content == "<html>Report</html>"
        assert version.format == "html"

    def test_report_version_with_bytes(self):
        """Test report version with bytes content."""
        info = VersionInfo(
            version=1,
            report_id="test",
            created_at=datetime.now(),
        )
        version = ReportVersion(
            info=info,
            content=b"PDF content",
            format="pdf",
        )
        assert isinstance(version.content, bytes)


class TestIncrementalStrategy:
    """Tests for IncrementalStrategy."""

    def test_first_version(self):
        """Test generating first version."""
        strategy = IncrementalStrategy()
        version = strategy.next_version(None)
        assert version == 1

    def test_increment_version(self):
        """Test incrementing version."""
        strategy = IncrementalStrategy()
        assert strategy.next_version(1) == 2
        assert strategy.next_version(10) == 11

    def test_format_version(self):
        """Test formatting incremental version."""
        strategy = IncrementalStrategy()
        assert strategy.format_version(1) == "1"
        assert strategy.format_version(42) == "42"

    def test_parse_version(self):
        """Test parsing incremental version string."""
        strategy = IncrementalStrategy()
        assert strategy.parse_version("1") == 1
        assert strategy.parse_version("42") == 42


class TestSemanticStrategy:
    """Tests for SemanticStrategy."""

    def test_first_version(self):
        """Test generating first semantic version."""
        strategy = SemanticStrategy()  # default initial is "1.0.0"
        version = strategy.next_version(None)
        # SemanticStrategy encodes version as: major * 1000000 + minor * 1000 + patch
        # "1.0.0" -> 1 * 1000000 + 0 * 1000 + 0 = 1000000
        assert version == 1000000

    def test_patch_increment(self):
        """Test patch version increment."""
        strategy = SemanticStrategy()
        # 1.0.0 (1000000) -> 1.0.1 (1000001)
        version = strategy.next_version(1000000)
        assert version == 1000001

    def test_format_version(self):
        """Test formatting as semver string."""
        strategy = SemanticStrategy()
        assert strategy.format_version(1000000) == "1.0.0"
        assert strategy.format_version(1000001) == "1.0.1"
        assert strategy.format_version(1001000) == "1.1.0"
        assert strategy.format_version(2000000) == "2.0.0"

    def test_parse_version(self):
        """Test parsing semver string."""
        strategy = SemanticStrategy()
        assert strategy.parse_version("1.0.0") == 1000000
        assert strategy.parse_version("0.1.0") == 1000
        assert strategy.parse_version("0.0.1") == 1
        assert strategy.parse_version("2.3.4") == 2003004

    def test_minor_bump(self):
        """Test minor version bump."""
        strategy = SemanticStrategy()
        # 1.0.0 -> 1.1.0
        version = strategy.next_version(1000000, metadata={"bump": "minor"})
        assert version == 1001000
        assert strategy.format_version(version) == "1.1.0"

    def test_major_bump(self):
        """Test major version bump."""
        strategy = SemanticStrategy()
        # 1.0.0 -> 2.0.0
        version = strategy.next_version(1000000, metadata={"bump": "major"})
        assert version == 2000000
        assert strategy.format_version(version) == "2.0.0"


class TestTimestampStrategy:
    """Tests for TimestampStrategy."""

    def test_generates_timestamp(self):
        """Test that version is timestamp-based."""
        strategy = TimestampStrategy()
        version = strategy.next_version(None)
        # Should be a Unix timestamp in seconds (not milliseconds)
        # After 2023 but reasonable range
        assert version > 1700000000  # After Nov 2023
        assert version < 2000000000  # Before 2033

    def test_versions_increase(self):
        """Test that versions always increase."""
        strategy = TimestampStrategy()
        v1 = strategy.next_version(None)
        time.sleep(0.1)  # Small delay to ensure timestamp changes
        v2 = strategy.next_version(v1)
        assert v2 >= v1  # May be equal if within same second

    def test_format_version(self):
        """Test formatting timestamp as ISO datetime."""
        strategy = TimestampStrategy()
        version = strategy.next_version(None)
        formatted = strategy.format_version(version)
        # Should be ISO format datetime
        assert "T" in formatted or "-" in formatted

    def test_parse_version(self):
        """Test parsing ISO datetime to timestamp."""
        strategy = TimestampStrategy()
        # Parse a known datetime
        timestamp = strategy.parse_version("2025-12-28T12:00:00")
        assert timestamp > 1700000000


class TestGitLikeStrategy:
    """Tests for GitLikeStrategy."""

    def test_first_commit(self):
        """Test first commit hash."""
        strategy = GitLikeStrategy()
        version = strategy.next_version(None, metadata={"content": "test"})
        # Returns an integer representation of hash
        assert isinstance(version, int)

    def test_same_content_same_hash(self):
        """Test that same content produces same hash."""
        strategy = GitLikeStrategy()
        v1 = strategy.next_version(None, metadata={"content": "test"})
        v2 = strategy.next_version(None, metadata={"content": "test"})
        # Same content should produce same hash
        assert v1 == v2

    def test_different_content_different_hash(self):
        """Test that different content produces different hash."""
        strategy = GitLikeStrategy()
        v1 = strategy.next_version(None, metadata={"content": "test1"})
        v2 = strategy.next_version(None, metadata={"content": "test2"})
        assert v1 != v2

    def test_format_version(self):
        """Test formatting as short hash."""
        strategy = GitLikeStrategy()
        version = strategy.next_version(None, metadata={"content": "test"})
        formatted = strategy.format_version(version)
        # Should be 8-character hex string
        assert len(formatted) == 8
        assert all(c in "0123456789abcdef" for c in formatted)

    def test_parse_version(self):
        """Test parsing hex string to integer."""
        strategy = GitLikeStrategy()
        # Parse a known hex value
        assert strategy.parse_version("0000000a") == 10
        assert strategy.parse_version("000000ff") == 255
