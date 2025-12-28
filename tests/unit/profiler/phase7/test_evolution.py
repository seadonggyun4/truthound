"""Tests for schema evolution detection."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import Mock

import pytest
import polars as pl

from truthound.profiler.evolution import (
    # Change types
    ChangeType,
    ChangeSeverity,
    CompatibilityLevel,
    SchemaChange,
    SchemaChangeSummary,
    # Detector
    SchemaEvolutionDetector,
    ColumnAddedChange,
    ColumnRemovedChange,
    TypeChangedChange,
    # Compatibility
    TypeCompatibilityChecker,
    SchemaCompatibilityAnalyzer,
    # Alerts
    SchemaChangeAlert,
    SchemaChangeAlertManager,
    ConsoleAlertHandler,
    LoggingAlertHandler,
)


class TestSchemaChange:
    """Tests for SchemaChange."""

    def test_create_change(self) -> None:
        """Test creating a schema change."""
        change = SchemaChange(
            change_type=ChangeType.COLUMN_ADDED,
            column="new_col",
            new_value=pl.Int64,
        )

        assert change.change_type == ChangeType.COLUMN_ADDED
        assert change.column == "new_col"
        assert not change.breaking

    def test_breaking_change_sets_severity(self) -> None:
        """Test that breaking changes have critical severity."""
        change = SchemaChange(
            change_type=ChangeType.COLUMN_REMOVED,
            column="old_col",
            breaking=True,
        )

        assert change.severity == ChangeSeverity.CRITICAL

    def test_auto_description(self) -> None:
        """Test automatic description generation."""
        change = SchemaChange(
            change_type=ChangeType.COLUMN_ADDED,
            column="new_col",
            new_value=pl.Int64,
        )

        assert "new_col" in change.description
        assert "added" in change.description.lower()

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        change = SchemaChange(
            change_type=ChangeType.COLUMN_ADDED,
            column="new_col",
            new_value=pl.Int64,
        )

        data = change.to_dict()

        assert data["change_type"] == "column_added"
        assert data["column"] == "new_col"

    def test_from_dict(self) -> None:
        """Test deserialization from dict."""
        data = {
            "change_type": "column_added",
            "column": "new_col",
            "breaking": False,
        }

        change = SchemaChange.from_dict(data)

        assert change.change_type == ChangeType.COLUMN_ADDED
        assert change.column == "new_col"


class TestSchemaChangeSummary:
    """Tests for SchemaChangeSummary."""

    def test_from_changes_empty(self) -> None:
        """Test summary with no changes."""
        summary = SchemaChangeSummary.from_changes([])

        assert summary.total_changes == 0
        assert summary.breaking_changes == 0
        assert summary.compatibility_level == CompatibilityLevel.FULL

    def test_from_changes_with_additions(self) -> None:
        """Test summary with column additions."""
        changes = [
            ColumnAddedChange("col1", pl.Int64),
            ColumnAddedChange("col2", pl.Utf8),
        ]

        summary = SchemaChangeSummary.from_changes(changes)

        assert summary.total_changes == 2
        assert summary.columns_added == 2
        assert summary.breaking_changes == 0
        assert summary.compatibility_level == CompatibilityLevel.BACKWARD

    def test_from_changes_with_removals(self) -> None:
        """Test summary with column removals (breaking)."""
        changes = [
            ColumnRemovedChange("col1", pl.Int64),
        ]

        summary = SchemaChangeSummary.from_changes(changes)

        assert summary.total_changes == 1
        assert summary.columns_removed == 1
        assert summary.breaking_changes == 1
        assert summary.compatibility_level == CompatibilityLevel.NONE

    def test_is_breaking(self) -> None:
        """Test is_breaking method."""
        summary = SchemaChangeSummary(breaking_changes=0)
        assert not summary.is_breaking()

        summary = SchemaChangeSummary(breaking_changes=1)
        assert summary.is_breaking()


class TestSchemaEvolutionDetector:
    """Tests for SchemaEvolutionDetector."""

    def test_detect_no_changes(self) -> None:
        """Test detecting no changes."""
        detector = SchemaEvolutionDetector()

        current = {"id": pl.Int64, "name": pl.Utf8}
        baseline = {"id": pl.Int64, "name": pl.Utf8}

        changes = detector.detect_changes(current, baseline)

        assert len(changes) == 0

    def test_detect_column_added(self) -> None:
        """Test detecting column addition."""
        detector = SchemaEvolutionDetector()

        current = {"id": pl.Int64, "name": pl.Utf8, "age": pl.Int32}
        baseline = {"id": pl.Int64, "name": pl.Utf8}

        changes = detector.detect_changes(current, baseline)

        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.COLUMN_ADDED
        assert changes[0].column == "age"
        assert not changes[0].breaking

    def test_detect_column_removed(self) -> None:
        """Test detecting column removal."""
        detector = SchemaEvolutionDetector()

        current = {"id": pl.Int64}
        baseline = {"id": pl.Int64, "name": pl.Utf8}

        changes = detector.detect_changes(current, baseline)

        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.COLUMN_REMOVED
        assert changes[0].column == "name"
        assert changes[0].breaking

    def test_detect_type_change(self) -> None:
        """Test detecting type change."""
        detector = SchemaEvolutionDetector()

        current = {"id": pl.Int64, "value": pl.Float64}
        baseline = {"id": pl.Int64, "value": pl.Int32}

        changes = detector.detect_changes(current, baseline)

        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.TYPE_CHANGED
        assert changes[0].column == "value"

    def test_detect_compatible_type_change(self) -> None:
        """Test detecting compatible type widening."""
        detector = SchemaEvolutionDetector()

        current = {"id": pl.Int64}  # Widened from Int32
        baseline = {"id": pl.Int32}

        changes = detector.detect_changes(current, baseline)

        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.TYPE_CHANGED
        assert not changes[0].breaking  # Int32 -> Int64 is compatible

    def test_detect_column_rename(self) -> None:
        """Test detecting column rename."""
        detector = SchemaEvolutionDetector(detect_renames=True)

        current = {"id": pl.Int64, "full_name": pl.Utf8}
        baseline = {"id": pl.Int64, "name": pl.Utf8}

        changes = detector.detect_changes(current, baseline)

        # Should detect as rename if similarity is high enough
        # Otherwise will be add + remove
        assert len(changes) >= 1

    def test_normalize_polars_schema(self) -> None:
        """Test normalizing Polars schema."""
        detector = SchemaEvolutionDetector()

        df = pl.DataFrame({"id": [1, 2], "name": ["a", "b"]})
        schema = df.schema

        current = {"id": pl.Int64, "name": pl.Utf8}
        changes = detector.detect_changes(schema, current)

        # Should work with Polars schema directly
        assert len(changes) == 0

    def test_get_change_summary(self) -> None:
        """Test getting change summary."""
        detector = SchemaEvolutionDetector()

        changes = [
            ColumnAddedChange("col1", pl.Int64),
            ColumnRemovedChange("col2", pl.Utf8),
        ]

        summary = detector.get_change_summary(changes)

        assert summary.total_changes == 2
        assert summary.breaking_changes == 1


class TestSchemaCompatibilityAnalyzer:
    """Tests for SchemaCompatibilityAnalyzer."""

    def test_fully_compatible(self) -> None:
        """Test fully compatible schemas."""
        analyzer = SchemaCompatibilityAnalyzer()

        old = {"id": pl.Int64}
        new = {"id": pl.Int64}

        result = analyzer.analyze(old, new)

        assert result.is_compatible
        assert result.level == CompatibilityLevel.FULL

    def test_backward_compatible_addition(self) -> None:
        """Test backward compatible addition."""
        analyzer = SchemaCompatibilityAnalyzer()

        old = {"id": pl.Int64}
        new = {"id": pl.Int64, "name": pl.Utf8}

        result = analyzer.analyze(old, new)

        assert result.is_compatible
        assert result.level == CompatibilityLevel.BACKWARD

    def test_not_compatible_removal(self) -> None:
        """Test not compatible due to removal."""
        analyzer = SchemaCompatibilityAnalyzer()

        old = {"id": pl.Int64, "name": pl.Utf8}
        new = {"id": pl.Int64}

        result = analyzer.analyze(old, new)

        assert not result.is_compatible
        assert len(result.issues) > 0

    def test_is_compatible_method(self) -> None:
        """Test is_compatible method."""
        analyzer = SchemaCompatibilityAnalyzer()

        old = {"id": pl.Int64}
        new = {"id": pl.Int64}

        assert analyzer.is_compatible(old, new)

    def test_get_compatibility_level_method(self) -> None:
        """Test get_compatibility_level method."""
        analyzer = SchemaCompatibilityAnalyzer()

        old = {"id": pl.Int64}
        new = {"id": pl.Int64, "name": pl.Utf8}

        level = analyzer.get_compatibility_level(old, new)
        assert level == "backward"


class TestTypeCompatibilityChecker:
    """Tests for TypeCompatibilityChecker."""

    def test_same_types_compatible(self) -> None:
        """Test that same types are compatible."""
        checker = TypeCompatibilityChecker()

        old = {"value": pl.Int64}
        new = {"value": pl.Int64}

        compatible, issues = checker.check(old, new)

        assert compatible
        assert len(issues) == 0

    def test_integer_widening_compatible(self) -> None:
        """Test that integer widening is compatible."""
        checker = TypeCompatibilityChecker()

        old = {"value": pl.Int32}
        new = {"value": pl.Int64}

        compatible, issues = checker.check(old, new)

        assert compatible

    def test_incompatible_type_change(self) -> None:
        """Test detecting incompatible type change."""
        checker = TypeCompatibilityChecker()

        old = {"value": pl.Utf8}
        new = {"value": pl.Int64}

        compatible, issues = checker.check(old, new)

        assert not compatible
        assert len(issues) == 1


class TestSchemaChangeAlertManager:
    """Tests for SchemaChangeAlertManager."""

    def test_alert_sends_to_handlers(self) -> None:
        """Test that alert sends to all handlers."""
        handler = Mock(spec=ConsoleAlertHandler)
        handler.supports_severity.return_value = True

        manager = SchemaChangeAlertManager(handlers=[handler])
        changes = [ColumnAddedChange("col1", pl.Int64)]

        manager.alert(changes)

        handler.notify.assert_called_once()

    def test_alert_if_breaking_only_alerts_breaking(self) -> None:
        """Test that alert_if_breaking only alerts for breaking changes."""
        handler = Mock(spec=ConsoleAlertHandler)
        handler.supports_severity.return_value = True

        manager = SchemaChangeAlertManager(handlers=[handler])

        # Non-breaking change
        non_breaking = [ColumnAddedChange("col1", pl.Int64)]
        manager.alert_if_breaking(non_breaking)
        handler.notify.assert_not_called()

        # Breaking change
        breaking = [ColumnRemovedChange("col1", pl.Int64)]
        manager.alert_if_breaking(breaking)
        handler.notify.assert_called_once()

    def test_alert_returns_alert_object(self) -> None:
        """Test that alert returns an alert object."""
        manager = SchemaChangeAlertManager()
        changes = [ColumnAddedChange("col1", pl.Int64)]

        alert = manager.alert(changes)

        assert alert is not None
        assert isinstance(alert, SchemaChangeAlert)
        assert len(alert.changes) == 1

    def test_alert_history(self) -> None:
        """Test alert history tracking."""
        manager = SchemaChangeAlertManager()
        changes = [ColumnAddedChange("col1", pl.Int64)]

        manager.alert(changes)
        manager.alert(changes)

        history = manager.get_alert_history()
        assert len(history) == 2


class TestConsoleAlertHandler:
    """Tests for ConsoleAlertHandler."""

    def test_supports_severity(self) -> None:
        """Test severity support checking."""
        handler = ConsoleAlertHandler(min_severity=ChangeSeverity.WARNING)

        assert handler.supports_severity("warning")
        assert handler.supports_severity("critical")
        assert not handler.supports_severity("info")

    def test_notify_prints_changes(self, capsys) -> None:
        """Test that notify prints changes."""
        handler = ConsoleAlertHandler(
            min_severity=ChangeSeverity.INFO,
            use_colors=False,
        )

        changes = [ColumnAddedChange("col1", pl.Int64)]
        handler.notify(changes, "info")

        captured = capsys.readouterr()
        assert "SCHEMA CHANGES" in captured.out


class TestLoggingAlertHandler:
    """Tests for LoggingAlertHandler."""

    def test_supports_severity(self) -> None:
        """Test severity support checking."""
        handler = LoggingAlertHandler(min_severity=ChangeSeverity.WARNING)

        assert handler.supports_severity("warning")
        assert handler.supports_severity("critical")
        assert not handler.supports_severity("info")

    def test_notify_logs_changes(self, caplog) -> None:
        """Test that notify logs changes."""
        import logging

        handler = LoggingAlertHandler(min_severity=ChangeSeverity.INFO)

        with caplog.at_level(logging.INFO):
            changes = [ColumnAddedChange("col1", pl.Int64)]
            handler.notify(changes, "info")

        assert len(caplog.records) > 0
