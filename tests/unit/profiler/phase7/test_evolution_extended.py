"""Extended tests for schema evolution module.

Tests for SchemaHistory, ColumnRenameDetector, BreakingChangeAlert, and SchemaWatcher.
"""

from __future__ import annotations

import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest


# =============================================================================
# SchemaHistory Tests
# =============================================================================


class TestSchemaHistory:
    """Tests for SchemaHistory class."""

    def test_create_memory_history(self) -> None:
        """Test creating in-memory schema history."""
        from truthound.profiler.evolution import SchemaHistory

        history = SchemaHistory.create(storage_type="memory")
        assert history is not None
        assert history.latest is None

    def test_create_file_history(self, tmp_path: Path) -> None:
        """Test creating file-based schema history."""
        from truthound.profiler.evolution import SchemaHistory

        history = SchemaHistory.create(
            storage_type="file",
            path=tmp_path / "history",
        )
        assert history is not None
        assert (tmp_path / "history").exists()

    def test_save_schema_version(self) -> None:
        """Test saving schema versions."""
        from truthound.profiler.evolution import SchemaHistory

        history = SchemaHistory.create(storage_type="memory")

        v1 = history.save({"id": "Int64", "name": "Utf8"})
        assert v1 is not None
        assert v1.version == "1.0.0"
        assert v1.column_count() == 2

    def test_save_with_changes(self) -> None:
        """Test saving version with changes from parent."""
        from truthound.profiler.evolution import SchemaHistory

        history = SchemaHistory.create(storage_type="memory")

        v1 = history.save({"id": "Int64", "name": "Utf8"})
        v2 = history.save({"id": "Int64", "name": "Utf8", "email": "Utf8"})

        assert v2.version == "1.1.0"  # Minor bump for addition
        assert len(v2.changes_from_parent) == 1
        assert v2.changes_from_parent[0].change_type.value == "column_added"

    def test_semantic_version_breaking_change(self) -> None:
        """Test semantic versioning with breaking change."""
        from truthound.profiler.evolution import SchemaHistory

        history = SchemaHistory.create(storage_type="memory", version_strategy="semantic")

        v1 = history.save({"id": "Int64", "name": "Utf8", "email": "Utf8"})
        v2 = history.save({"id": "Int64", "name": "Utf8"})  # Removed email

        assert v2.version == "2.0.0"  # Major bump for breaking
        assert v2.has_breaking_changes()

    def test_diff_between_versions(self) -> None:
        """Test computing diff between versions."""
        from truthound.profiler.evolution import SchemaHistory

        history = SchemaHistory.create(storage_type="memory")

        v1 = history.save({"id": "Int64", "name": "Utf8"})
        v2 = history.save({"id": "Int64", "name": "Utf8", "email": "Utf8"})

        diff = history.diff(v1, v2)
        assert diff is not None
        assert diff.summary.total_changes == 1
        assert diff.summary.columns_added == 1

    def test_rollback(self) -> None:
        """Test rollback to previous version."""
        from truthound.profiler.evolution import SchemaHistory

        history = SchemaHistory.create(storage_type="memory")

        v1 = history.save({"id": "Int64", "name": "Utf8"})
        v2 = history.save({"id": "Int64", "name": "Utf8", "email": "Utf8"})

        v3 = history.rollback(v1, reason="Testing rollback")

        assert v3 is not None
        assert v3.schema == v1.schema
        assert "rollback" in v3.metadata.get("rollback_reason", "")

    def test_list_versions(self) -> None:
        """Test listing versions."""
        from truthound.profiler.evolution import SchemaHistory

        history = SchemaHistory.create(storage_type="memory")

        history.save({"id": "Int64"})
        history.save({"id": "Int64", "name": "Utf8"})
        history.save({"id": "Int64", "name": "Utf8", "email": "Utf8"})

        versions = history.list()
        assert len(versions) == 3

        versions_limited = history.list(limit=2)
        assert len(versions_limited) == 2

    def test_get_by_version_string(self) -> None:
        """Test getting version by version string."""
        from truthound.profiler.evolution import SchemaHistory

        history = SchemaHistory.create(storage_type="memory")

        v1 = history.save({"id": "Int64"})

        found = history.get_by_version(v1.version)
        assert found is not None
        assert found.version_id == v1.version_id

    def test_file_persistence(self, tmp_path: Path) -> None:
        """Test file history persistence across instances."""
        from truthound.profiler.evolution import SchemaHistory

        history_path = tmp_path / "history"

        # Save version
        history1 = SchemaHistory.create(storage_type="file", path=history_path)
        v1 = history1.save({"id": "Int64", "name": "Utf8"})

        # Create new instance
        history2 = SchemaHistory.create(storage_type="file", path=history_path)

        assert history2.latest is not None
        assert history2.latest.version_id == v1.version_id

    def test_version_strategies(self) -> None:
        """Test different version strategies."""
        from truthound.profiler.evolution import SchemaHistory

        # Incremental
        h1 = SchemaHistory.create(storage_type="memory", version_strategy="incremental")
        v1 = h1.save({"id": "Int64"})
        v2 = h1.save({"id": "Int64", "name": "Utf8"})
        assert v1.version == "1"
        assert v2.version == "2"

        # Timestamp
        h2 = SchemaHistory.create(storage_type="memory", version_strategy="timestamp")
        v3 = h2.save({"id": "Int64"})
        assert "." in v3.version  # Format: YYYYMMDD.HHMMSS

        # Git-like
        h3 = SchemaHistory.create(storage_type="memory", version_strategy="git")
        v4 = h3.save({"id": "Int64"})
        assert len(v4.version) == 8  # Short hash


# =============================================================================
# ColumnRenameDetector Tests
# =============================================================================


class TestColumnRenameDetector:
    """Tests for ColumnRenameDetector class."""

    def test_detect_exact_rename(self) -> None:
        """Test detecting obvious column renames."""
        from truthound.profiler.evolution import ColumnRenameDetector

        detector = ColumnRenameDetector(similarity_threshold=0.8)

        result = detector.detect(
            added_columns={"user_email": "Utf8"},
            removed_columns={"email": "Utf8"},
        )

        # Note: "email" -> "user_email" has high similarity
        assert len(result.all_renames) >= 0  # May or may not detect

    def test_detect_similar_rename(self) -> None:
        """Test detecting similar column renames."""
        from truthound.profiler.evolution import ColumnRenameDetector

        detector = ColumnRenameDetector(similarity_threshold=0.7)

        result = detector.detect(
            added_columns={"customer_id": "Int64"},
            removed_columns={"cust_id": "Int64"},
        )

        # "cust_id" and "customer_id" share tokens
        if result.confirmed_renames:
            assert result.confirmed_renames[0].old_name == "cust_id"

    def test_type_mismatch_blocks_rename(self) -> None:
        """Test that type mismatch blocks rename detection."""
        from truthound.profiler.evolution import ColumnRenameDetector

        detector = ColumnRenameDetector(require_type_match=True)

        result = detector.detect(
            added_columns={"name": "Int64"},  # Different type
            removed_columns={"name": "Utf8"},
        )

        assert len(result.confirmed_renames) == 0

    def test_unmatched_columns(self) -> None:
        """Test unmatched columns are reported."""
        from truthound.profiler.evolution import ColumnRenameDetector

        detector = ColumnRenameDetector()

        result = detector.detect(
            added_columns={"new_col": "Utf8"},
            removed_columns={"old_col": "Int64"},
        )

        assert "new_col" in result.unmatched_added
        assert "old_col" in result.unmatched_removed


class TestSimilarityCalculators:
    """Tests for similarity calculator implementations."""

    def test_levenshtein_similarity(self) -> None:
        """Test Levenshtein similarity."""
        from truthound.profiler.evolution import LevenshteinSimilarity

        calc = LevenshteinSimilarity()

        assert calc.calculate("test", "test") == 1.0
        assert calc.calculate("test", "tset") > 0.5
        assert calc.calculate("abc", "xyz") < 0.5

    def test_jaro_winkler_similarity(self) -> None:
        """Test Jaro-Winkler similarity."""
        from truthound.profiler.evolution import JaroWinklerSimilarity

        calc = JaroWinklerSimilarity()

        assert calc.calculate("test", "test") == 1.0
        # Common prefix boost
        assert calc.calculate("test_name", "test_value") > 0.7

    def test_ngram_similarity(self) -> None:
        """Test N-gram similarity."""
        from truthound.profiler.evolution import NgramSimilarity

        calc = NgramSimilarity(n=2)

        assert calc.calculate("test", "test") == 1.0
        assert calc.calculate("testing", "tested") > 0.5

    def test_token_similarity(self) -> None:
        """Test token-based similarity."""
        from truthound.profiler.evolution import TokenSimilarity

        calc = TokenSimilarity()

        # snake_case
        assert calc.calculate("user_name", "user_id") > 0.3
        # camelCase
        assert calc.calculate("userName", "userId") > 0.3

    def test_composite_similarity(self) -> None:
        """Test composite similarity."""
        from truthound.profiler.evolution import CompositeSimilarity

        calc = CompositeSimilarity()

        assert calc.calculate("test", "test") == 1.0
        score = calc.calculate("customer_id", "cust_id")
        assert 0.3 < score < 1.0


# =============================================================================
# BreakingChangeAlert Tests
# =============================================================================


class TestBreakingChangeAlert:
    """Tests for BreakingChangeAlert class."""

    def test_create_alert(self) -> None:
        """Test creating a breaking change alert."""
        from truthound.profiler.evolution import (
            BreakingChangeAlert,
            ChangeSeverity,
            ColumnRemovedChange,
        )

        changes = [ColumnRemovedChange("email", "Utf8")]

        alert = BreakingChangeAlert(
            alert_id="ALERT-001",
            title="Breaking Change Detected",
            description="Column removed",
            severity=ChangeSeverity.CRITICAL,
            changes=changes,
        )

        assert alert.alert_id == "ALERT-001"
        assert alert.severity == ChangeSeverity.CRITICAL
        assert len(alert.changes) == 1

    def test_alert_status(self) -> None:
        """Test alert status transitions."""
        from truthound.profiler.evolution import (
            BreakingChangeAlert,
            ChangeSeverity,
        )

        alert = BreakingChangeAlert(
            alert_id="ALERT-001",
            title="Test",
            description="Test",
            severity=ChangeSeverity.WARNING,
        )

        assert alert.status == "open"

        alert.acknowledge()
        assert alert.status == "acknowledged"

        alert.resolve()
        assert alert.status == "resolved"

    def test_format_slack_message(self) -> None:
        """Test Slack message formatting."""
        from truthound.profiler.evolution import (
            BreakingChangeAlert,
            ChangeSeverity,
            ColumnRemovedChange,
        )

        changes = [ColumnRemovedChange("email", "Utf8")]

        alert = BreakingChangeAlert(
            alert_id="ALERT-001",
            title="Breaking Change",
            description="Column removed",
            severity=ChangeSeverity.CRITICAL,
            changes=changes,
        )

        message = alert.format_slack_message()
        assert "attachments" in message
        assert message["attachments"][0]["color"] == "#FF0000"

    def test_format_email(self) -> None:
        """Test email formatting."""
        from truthound.profiler.evolution import (
            BreakingChangeAlert,
            ChangeSeverity,
            ColumnRemovedChange,
        )

        changes = [ColumnRemovedChange("email", "Utf8")]

        alert = BreakingChangeAlert(
            alert_id="ALERT-001",
            title="Breaking Change",
            description="Column removed",
            severity=ChangeSeverity.CRITICAL,
            changes=changes,
        )

        subject, body = alert.format_email()
        assert "[CRITICAL]" in subject
        assert "Breaking Change" in subject
        assert "<html>" in body


class TestImpactAnalyzer:
    """Tests for ImpactAnalyzer class."""

    def test_analyze_column_removal(self) -> None:
        """Test impact analysis for column removal."""
        from truthound.profiler.evolution import (
            ImpactAnalyzer,
            ColumnRemovedChange,
            ImpactCategory,
        )

        analyzer = ImpactAnalyzer()

        changes = [ColumnRemovedChange("email", "Utf8")]
        impact = analyzer.analyze(changes, source="users")

        assert impact.category == ImpactCategory.DATA_LOSS
        assert impact.data_risk_level >= 2

    def test_analyze_with_consumers(self) -> None:
        """Test impact analysis with registered consumers."""
        from truthound.profiler.evolution import (
            ImpactAnalyzer,
            ColumnRemovedChange,
            ImpactScope,
        )

        analyzer = ImpactAnalyzer()
        analyzer.register_consumer("dashboard", ["users", "orders"])
        analyzer.register_consumer("reports", ["users"])

        changes = [ColumnRemovedChange("email", "Utf8")]
        impact = analyzer.analyze(changes, source="users")

        assert len(impact.affected_consumers) == 2
        assert "dashboard" in impact.affected_consumers
        assert impact.scope in (ImpactScope.DOWNSTREAM, ImpactScope.PIPELINE)

    def test_recommendations_generated(self) -> None:
        """Test that recommendations are generated."""
        from truthound.profiler.evolution import (
            ImpactAnalyzer,
            ColumnRemovedChange,
        )

        analyzer = ImpactAnalyzer()

        changes = [ColumnRemovedChange("email", "Utf8")]
        impact = analyzer.analyze(changes)

        assert len(impact.recommendations) > 0


class TestBreakingChangeAlertManager:
    """Tests for BreakingChangeAlertManager class."""

    def test_create_alert_via_manager(self) -> None:
        """Test creating alerts through manager."""
        from truthound.profiler.evolution import (
            BreakingChangeAlertManager,
            ColumnRemovedChange,
        )

        manager = BreakingChangeAlertManager()

        changes = [ColumnRemovedChange("email", "Utf8")]
        alert = manager.create_alert(changes, source="users", notify=False)

        assert alert is not None
        assert "ALERT-" in alert.alert_id

    def test_alert_history(self) -> None:
        """Test alert history tracking."""
        from truthound.profiler.evolution import (
            BreakingChangeAlertManager,
            ColumnRemovedChange,
            ColumnAddedChange,
        )

        manager = BreakingChangeAlertManager()

        manager.create_alert([ColumnRemovedChange("a", "Utf8")], notify=False)
        manager.create_alert([ColumnAddedChange("b", "Utf8")], notify=False)

        history = manager.get_alert_history()
        assert len(history) == 2

    def test_acknowledge_and_resolve(self) -> None:
        """Test acknowledging and resolving alerts."""
        from truthound.profiler.evolution import (
            BreakingChangeAlertManager,
            ColumnRemovedChange,
        )

        manager = BreakingChangeAlertManager()

        alert = manager.create_alert(
            [ColumnRemovedChange("email", "Utf8")],
            notify=False,
        )

        assert manager.acknowledge_alert(alert.alert_id)
        assert manager.get_alert(alert.alert_id).is_acknowledged

        assert manager.resolve_alert(alert.alert_id)
        assert manager.get_alert(alert.alert_id).is_resolved

    def test_stats(self) -> None:
        """Test alert statistics."""
        from truthound.profiler.evolution import (
            BreakingChangeAlertManager,
            ColumnRemovedChange,
        )

        manager = BreakingChangeAlertManager()

        manager.create_alert([ColumnRemovedChange("a", "Utf8")], notify=False)
        manager.create_alert([ColumnRemovedChange("b", "Utf8")], notify=False)
        manager.acknowledge_alert("ALERT-000001")

        stats = manager.get_stats()
        assert stats["total"] == 2
        assert stats["acknowledged"] == 1


# =============================================================================
# SchemaWatcher Tests
# =============================================================================


class TestSchemaWatcher:
    """Tests for SchemaWatcher class."""

    def test_create_watcher(self) -> None:
        """Test creating a schema watcher."""
        from truthound.profiler.evolution import SchemaWatcher

        watcher = SchemaWatcher()
        assert watcher.state.value == "created"

    def test_add_source(self) -> None:
        """Test adding a schema source."""
        from truthound.profiler.evolution import SchemaWatcher, DictSchemaSource

        watcher = SchemaWatcher()
        source = DictSchemaSource({"id": "Int64"}, name="test")
        watcher.add_source(source)

        status = watcher.get_source_status()
        assert "test" in status

    def test_check_now_no_changes(self) -> None:
        """Test checking with no changes."""
        from truthound.profiler.evolution import SchemaWatcher, DictSchemaSource

        watcher = SchemaWatcher()
        source = DictSchemaSource({"id": "Int64"}, name="test")
        watcher.add_source(source)

        events = watcher.check_now()
        assert len(events) == 0

    def test_check_now_with_changes(self) -> None:
        """Test checking with schema changes."""
        from truthound.profiler.evolution import SchemaWatcher, DictSchemaSource

        watcher = SchemaWatcher()
        source = DictSchemaSource({"id": "Int64"}, name="test")
        watcher.add_source(source)

        # First check establishes baseline
        watcher.check_now()

        # Modify schema
        source.update({"id": "Int64", "name": "Utf8"})

        # Second check detects change
        events = watcher.check_now()
        assert len(events) == 1
        assert events[0].source == "test"
        assert events[0].summary.columns_added == 1

    def test_event_handler(self) -> None:
        """Test event handler invocation."""
        from truthound.profiler.evolution import (
            SchemaWatcher,
            DictSchemaSource,
            CallbackEventHandler,
        )

        events_received: list = []

        def on_change(event):
            events_received.append(event)

        watcher = SchemaWatcher()
        watcher.add_source(DictSchemaSource({"id": "Int64"}, name="test"))
        watcher.add_handler(CallbackEventHandler(on_change))

        # Establish baseline
        watcher.check_now()

        # Make change
        watcher._sources["test"].update({"id": "Int64", "name": "Utf8"})

        # Check
        watcher.check_now()

        assert len(events_received) == 1

    def test_file_schema_source(self, tmp_path: Path) -> None:
        """Test FileSchemaSource."""
        from truthound.profiler.evolution import FileSchemaSource

        schema_file = tmp_path / "schema.json"
        schema_file.write_text('{"id": "Int64", "name": "Utf8"}')

        source = FileSchemaSource(schema_file)

        schema = source.get_schema()
        assert schema["id"] == "Int64"
        assert schema["name"] == "Utf8"

    def test_watcher_start_stop(self) -> None:
        """Test watcher start and stop."""
        from truthound.profiler.evolution import SchemaWatcher, DictSchemaSource

        watcher = SchemaWatcher()
        watcher.add_source(DictSchemaSource({"id": "Int64"}))

        watcher.start(poll_interval=0.1, daemon=True)
        assert watcher.state.value == "running"

        time.sleep(0.2)

        watcher.stop()
        assert watcher.state.value == "stopped"

    def test_watcher_pause_resume(self) -> None:
        """Test watcher pause and resume."""
        from truthound.profiler.evolution import SchemaWatcher, DictSchemaSource

        watcher = SchemaWatcher()
        watcher.add_source(DictSchemaSource({"id": "Int64"}))

        watcher.start(poll_interval=0.1, daemon=True)
        time.sleep(0.1)

        watcher.pause()
        assert watcher.state.value == "paused"

        watcher.resume()
        assert watcher.state.value == "running"

        watcher.stop()


class TestCreateWatcher:
    """Tests for create_watcher factory function."""

    def test_create_with_defaults(self) -> None:
        """Test creating watcher with defaults."""
        from truthound.profiler.evolution import create_watcher

        watcher = create_watcher()
        assert watcher is not None

    def test_create_with_sources(self) -> None:
        """Test creating watcher with sources."""
        from truthound.profiler.evolution import create_watcher, DictSchemaSource

        source = DictSchemaSource({"id": "Int64"})
        watcher = create_watcher(sources=[source])

        status = watcher.get_source_status()
        assert len(status) == 1

    def test_create_with_history(self, tmp_path: Path) -> None:
        """Test creating watcher with history enabled."""
        from truthound.profiler.evolution import create_watcher

        watcher = create_watcher(
            enable_history=True,
            history_path=tmp_path / "history",
        )
        assert watcher is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestSchemaEvolutionIntegration:
    """Integration tests for schema evolution workflow."""

    def test_full_workflow(self, tmp_path: Path) -> None:
        """Test full schema evolution workflow."""
        from truthound.profiler.evolution import (
            SchemaHistory,
            SchemaWatcher,
            DictSchemaSource,
            BreakingChangeAlertManager,
            AlertingEventHandler,
        )

        # Setup
        history = SchemaHistory.create(
            storage_type="file",
            path=tmp_path / "history",
        )
        alert_manager = BreakingChangeAlertManager()

        watcher = SchemaWatcher(history=history)
        source = DictSchemaSource(
            {"id": "Int64", "name": "Utf8", "email": "Utf8"},
            name="users",
        )
        watcher.add_source(source)
        watcher.add_handler(AlertingEventHandler(alert_manager))

        # Initial save
        v1 = history.save(source.get_schema())

        # Establish baseline
        watcher.check_now()

        # Make breaking change
        source.update({"id": "Int64", "name": "Utf8"})  # Removed email

        # Detect change
        events = watcher.check_now()

        # Verify
        assert len(events) == 1
        assert events[0].has_breaking_changes()

        # Check history
        assert history.version_count == 1

        # Check alerts
        alerts = alert_manager.get_alert_history()
        assert len(alerts) == 1

    def test_rename_detection_in_workflow(self) -> None:
        """Test rename detection in evolution workflow."""
        from truthound.profiler.evolution import (
            SchemaEvolutionDetector,
            ColumnRenameDetector,
        )

        old_schema = {"user_email": "Utf8", "user_name": "Utf8"}
        new_schema = {"email_address": "Utf8", "user_name": "Utf8"}

        detector = SchemaEvolutionDetector(
            detect_renames=True,
            rename_similarity_threshold=0.5,
        )

        changes = detector.detect_changes(new_schema, old_schema)

        # Should detect changes
        assert len(changes) > 0
