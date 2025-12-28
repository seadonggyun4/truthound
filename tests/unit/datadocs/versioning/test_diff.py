"""Tests for versioning diff module."""

import pytest
from datetime import datetime
from truthound.datadocs.versioning.version import VersionInfo, ReportVersion
from truthound.datadocs.versioning.diff import (
    Change,
    ChangeType,
    DiffResult,
    TextDiffStrategy,
    StructuralDiffStrategy,
    SemanticDiffStrategy,
    ReportDiffer,
    diff_versions,
)


def make_version(version: int, content: str, format: str = "html") -> ReportVersion:
    """Helper to create a ReportVersion."""
    info = VersionInfo(
        version=version,
        report_id="test_report",
        created_at=datetime.now(),
    )
    return ReportVersion(info=info, content=content, format=format)


class TestChangeType:
    """Tests for ChangeType enum."""

    def test_change_types(self):
        """Test all change types exist."""
        assert ChangeType.ADDED.value == "added"
        assert ChangeType.REMOVED.value == "removed"
        assert ChangeType.MODIFIED.value == "modified"
        assert ChangeType.UNCHANGED.value == "unchanged"


class TestChange:
    """Tests for Change class."""

    def test_create_change(self):
        """Test creating a change."""
        change = Change(
            change_type=ChangeType.ADDED,
            path="content",
            new_value="new line",
        )
        assert change.change_type == ChangeType.ADDED
        assert change.path == "content"
        assert change.new_value == "new line"

    def test_change_to_dict(self):
        """Test converting to dict."""
        change = Change(
            change_type=ChangeType.MODIFIED,
            path="title",
            old_value="Old",
            new_value="New",
        )
        d = change.to_dict()
        assert d["change_type"] == "modified"
        assert d["path"] == "title"
        assert d["old_value"] == "Old"
        assert d["new_value"] == "New"


class TestDiffResult:
    """Tests for DiffResult class."""

    def test_create_result(self):
        """Test creating a result."""
        result = DiffResult(
            old_version=1,
            new_version=2,
            report_id="test",
        )
        assert result.old_version == 1
        assert result.new_version == 2
        assert len(result.changes) == 0

    def test_has_changes(self):
        """Test has_changes method."""
        result = DiffResult(
            old_version=1,
            new_version=2,
            report_id="test",
        )
        assert result.has_changes() is False

        result.changes.append(Change(ChangeType.ADDED, "path"))
        assert result.has_changes() is True

    def test_count_properties(self):
        """Test count properties."""
        result = DiffResult(
            old_version=1,
            new_version=2,
            report_id="test",
            changes=[
                Change(ChangeType.ADDED, "a"),
                Change(ChangeType.ADDED, "b"),
                Change(ChangeType.REMOVED, "c"),
                Change(ChangeType.MODIFIED, "d"),
            ],
        )
        assert result.added_count == 2
        assert result.removed_count == 1
        assert result.modified_count == 1


class TestTextDiffStrategy:
    """Tests for TextDiffStrategy."""

    def test_identical_versions(self):
        """Test diffing identical versions."""
        strategy = TextDiffStrategy()
        v1 = make_version(1, "Line 1\nLine 2\nLine 3")
        v2 = make_version(2, "Line 1\nLine 2\nLine 3")

        result = strategy.diff(v1, v2)
        assert len(result.changes) == 0

    def test_added_line(self):
        """Test detecting added line."""
        strategy = TextDiffStrategy()
        v1 = make_version(1, "Line 1\nLine 2")
        v2 = make_version(2, "Line 1\nLine 2\nLine 3")

        result = strategy.diff(v1, v2)
        added = [c for c in result.changes if c.change_type == ChangeType.ADDED]
        assert len(added) > 0

    def test_removed_line(self):
        """Test detecting removed line."""
        strategy = TextDiffStrategy()
        v1 = make_version(1, "Line 1\nLine 2\nLine 3")
        v2 = make_version(2, "Line 1\nLine 3")

        result = strategy.diff(v1, v2)
        removed = [c for c in result.changes if c.change_type == ChangeType.REMOVED]
        assert len(removed) > 0

    def test_unified_diff_output(self):
        """Test unified diff output."""
        strategy = TextDiffStrategy()
        v1 = make_version(1, "Line 1\nLine 2")
        v2 = make_version(2, "Line 1\nLine 3")

        result = strategy.diff(v1, v2)
        assert result.unified_diff is not None
        assert "@@" in result.unified_diff

    def test_ignore_whitespace(self):
        """Test ignoring whitespace."""
        strategy = TextDiffStrategy(ignore_whitespace=True)
        v1 = make_version(1, "Line 1  \nLine 2")
        v2 = make_version(2, "Line 1\nLine 2  ")

        result = strategy.diff(v1, v2)
        assert len(result.changes) == 0


class TestStructuralDiffStrategy:
    """Tests for StructuralDiffStrategy."""

    def test_metadata_change(self):
        """Test detecting metadata change."""
        strategy = StructuralDiffStrategy()
        v1 = make_version(1, "{}")
        v2 = make_version(2, "{}")

        # Modify version info metadata
        v1.info = VersionInfo(
            version=1,
            report_id="test",
            created_at=datetime.now(),
            metadata={"key": "value1"},
        )
        v2.info = VersionInfo(
            version=2,
            report_id="test",
            created_at=datetime.now(),
            metadata={"key": "value2"},
        )

        result = strategy.diff(v1, v2)
        # Should detect the metadata change
        assert result.has_changes()

    def test_json_content_diff(self):
        """Test diffing JSON content."""
        strategy = StructuralDiffStrategy(include_content=True)
        v1 = make_version(1, '{"key": "value1"}', format="json")
        v2 = make_version(2, '{"key": "value2"}', format="json")

        result = strategy.diff(v1, v2)
        # Should detect the content change
        modified = [c for c in result.changes if c.change_type == ChangeType.MODIFIED]
        assert len(modified) > 0


class TestReportDiffer:
    """Tests for ReportDiffer class."""

    def test_compare_basic(self):
        """Test basic comparison."""
        differ = ReportDiffer()
        v1 = make_version(1, "Content v1")
        v2 = make_version(2, "Content v2")

        result = differ.compare(v1, v2)
        assert result.has_changes()

    def test_compare_with_strategy(self):
        """Test comparing with named strategy."""
        differ = ReportDiffer()
        v1 = make_version(1, "Line 1\nLine 2")
        v2 = make_version(2, "Line 1\nLine 3")

        result = differ.compare_with_strategy(v1, v2, "text")
        assert result.has_changes()

    def test_format_diff_unified(self):
        """Test formatting as unified diff."""
        differ = ReportDiffer()
        v1 = make_version(1, "Line 1")
        v2 = make_version(2, "Line 2")

        result = differ.compare(v1, v2)
        formatted = differ.format_diff(result, "unified")
        assert "v1" in formatted or "Line" in formatted

    def test_format_diff_summary(self):
        """Test formatting as summary."""
        differ = ReportDiffer()
        v1 = make_version(1, "Content")
        v2 = make_version(2, "Content changed")

        result = differ.compare(v1, v2)
        formatted = differ.format_diff(result, "summary")
        assert "Report:" in formatted
        assert "Versions:" in formatted

    def test_format_diff_json(self):
        """Test formatting as JSON."""
        differ = ReportDiffer()
        v1 = make_version(1, "Content")
        v2 = make_version(2, "New content")

        result = differ.compare(v1, v2)
        formatted = differ.format_diff(result, "json")
        assert "old_version" in formatted
        assert "new_version" in formatted


class TestDiffVersions:
    """Tests for diff_versions function."""

    def test_diff_versions_function(self):
        """Test convenience function."""
        v1 = make_version(1, "Old content")
        v2 = make_version(2, "New content")

        result = diff_versions(v1, v2)
        assert isinstance(result, DiffResult)
        assert result.old_version == 1
        assert result.new_version == 2

    def test_diff_with_strategy_arg(self):
        """Test with strategy argument."""
        v1 = make_version(1, "Content")
        v2 = make_version(2, "Content")

        result = diff_versions(v1, v2, strategy="structural")
        assert isinstance(result, DiffResult)
