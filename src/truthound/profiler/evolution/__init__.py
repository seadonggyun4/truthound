"""Schema evolution detection module.

This module provides comprehensive schema change detection, compatibility analysis,
history management, and continuous monitoring capabilities.

Key Features:
- Detect schema changes (columns added/removed, types changed, renames)
- Classify changes as breaking or non-breaking
- Compatibility analysis for downstream consumers
- Schema versioning with diff and rollback support
- Continuous schema monitoring with file watching
- Integration with notification systems

Example - Basic Detection:
    from truthound.profiler.evolution import (
        SchemaEvolutionDetector,
        SchemaChangeAlertManager,
    )

    detector = SchemaEvolutionDetector(storage=profile_storage)
    changes = detector.detect_changes(current_schema)

    for change in changes:
        if change.breaking:
            print(f"Breaking change: {change}")

    # Send alerts for breaking changes
    alert_manager = SchemaChangeAlertManager(notifiers=[slack_notifier])
    alert_manager.alert_if_breaking(changes)

Example - Schema History:
    from truthound.profiler.evolution import SchemaHistory

    # Create history with semantic versioning
    history = SchemaHistory.create(
        storage_type="file",
        path="./schema_history",
        version_strategy="semantic",
    )

    # Save schema versions
    v1 = history.save({"id": "Int64", "name": "Utf8"})
    v2 = history.save({"id": "Int64", "name": "Utf8", "email": "Utf8"})

    # Get diff between versions
    diff = history.diff(v1, v2)
    print(diff.format_text())

    # Rollback to previous version
    v3 = history.rollback(v1, reason="Incompatible change")

Example - Schema Watcher:
    from truthound.profiler.evolution import SchemaWatcher, FileSchemaSource

    # Create watcher
    watcher = SchemaWatcher()
    watcher.add_source(FileSchemaSource("schema.json"))
    watcher.add_handler(LoggingEventHandler())

    # Start watching
    watcher.start(poll_interval=60)

    # ... later ...
    watcher.stop()

Example - Advanced Rename Detection:
    from truthound.profiler.evolution import ColumnRenameDetector

    detector = ColumnRenameDetector(
        similarity_threshold=0.8,
        require_type_match=True,
    )

    result = detector.detect(
        added_columns={"user_email": "Utf8"},
        removed_columns={"email": "Utf8"},
    )

    for rename in result.confirmed_renames:
        print(f"Rename detected: {rename.old_name} -> {rename.new_name}")
"""

from truthound.profiler.evolution.protocols import (
    SchemaChangeDetector,
    CompatibilityChecker,
    ChangeNotifier,
)
from truthound.profiler.evolution.changes import (
    ChangeType,
    ChangeSeverity,
    CompatibilityLevel,
    SchemaChange,
    SchemaChangeSummary,
)
from truthound.profiler.evolution.detector import (
    SchemaEvolutionDetector,
    ColumnAddedChange,
    ColumnRemovedChange,
    ColumnRenamedChange,
    TypeChangedChange,
    NullabilityChangedChange,
)
from truthound.profiler.evolution.compatibility import (
    CompatibilityRule,
    TypeCompatibilityChecker,
    NullabilityCompatibilityChecker,
    SchemaCompatibilityAnalyzer,
)
from truthound.profiler.evolution.alerts import (
    SchemaChangeAlert,
    SchemaChangeAlertManager,
    ConsoleAlertHandler,
    LoggingAlertHandler,
)

# New modules
from truthound.profiler.evolution.history import (
    SchemaVersion,
    SchemaDiff,
    SchemaHistory,
    SchemaHistoryStorage,
    InMemorySchemaHistory,
    FileSchemaHistory,
    VersionStrategy,
    IncrementalVersionStrategy,
    SemanticVersionStrategy,
    TimestampVersionStrategy,
    GitLikeVersionStrategy,
)
from truthound.profiler.evolution.rename_detector import (
    ColumnRenameDetector,
    RenameCandidate,
    RenameDetectionResult,
    RenameConfidence,
    SimilarityCalculator,
    LevenshteinSimilarity,
    JaroWinklerSimilarity,
    NgramSimilarity,
    TokenSimilarity,
    CompositeSimilarity,
    create_rename_detector,
)
from truthound.profiler.evolution.breaking_alerts import (
    BreakingChangeAlert,
    BreakingChangeAlertManager,
    ImpactAssessment,
    ImpactScope,
    ImpactCategory,
    ImpactAnalyzer,
    AlertNotifier,
    CallbackNotifier,
    FileNotifier,
)
from truthound.profiler.evolution.watcher import (
    SchemaWatcher,
    AsyncSchemaWatcher,
    WatchEvent,
    WatcherState,
    SchemaSource,
    WatchEventHandler,
    FileSchemaSource,
    PolarsSchemaSource,
    DictSchemaSource,
    LoggingEventHandler,
    CallbackEventHandler,
    AlertingEventHandler,
    HistoryEventHandler,
    create_watcher,
)

__all__ = [
    # Protocols
    "SchemaChangeDetector",
    "CompatibilityChecker",
    "ChangeNotifier",
    "SchemaHistoryStorage",
    "VersionStrategy",
    "SimilarityCalculator",
    "SchemaSource",
    "WatchEventHandler",
    "AlertNotifier",
    # Change types
    "ChangeType",
    "ChangeSeverity",
    "CompatibilityLevel",
    "SchemaChange",
    "SchemaChangeSummary",
    # Detector
    "SchemaEvolutionDetector",
    "ColumnAddedChange",
    "ColumnRemovedChange",
    "ColumnRenamedChange",
    "TypeChangedChange",
    "NullabilityChangedChange",
    # Compatibility
    "CompatibilityRule",
    "TypeCompatibilityChecker",
    "NullabilityCompatibilityChecker",
    "SchemaCompatibilityAnalyzer",
    # Basic Alerts
    "SchemaChangeAlert",
    "SchemaChangeAlertManager",
    "ConsoleAlertHandler",
    "LoggingAlertHandler",
    # History
    "SchemaVersion",
    "SchemaDiff",
    "SchemaHistory",
    "InMemorySchemaHistory",
    "FileSchemaHistory",
    "IncrementalVersionStrategy",
    "SemanticVersionStrategy",
    "TimestampVersionStrategy",
    "GitLikeVersionStrategy",
    # Rename Detection
    "ColumnRenameDetector",
    "RenameCandidate",
    "RenameDetectionResult",
    "RenameConfidence",
    "LevenshteinSimilarity",
    "JaroWinklerSimilarity",
    "NgramSimilarity",
    "TokenSimilarity",
    "CompositeSimilarity",
    "create_rename_detector",
    # Breaking Alerts
    "BreakingChangeAlert",
    "BreakingChangeAlertManager",
    "ImpactAssessment",
    "ImpactScope",
    "ImpactCategory",
    "ImpactAnalyzer",
    "CallbackNotifier",
    "FileNotifier",
    # Watcher
    "SchemaWatcher",
    "AsyncSchemaWatcher",
    "WatchEvent",
    "WatcherState",
    "FileSchemaSource",
    "PolarsSchemaSource",
    "DictSchemaSource",
    "LoggingEventHandler",
    "CallbackEventHandler",
    "AlertingEventHandler",
    "HistoryEventHandler",
    "create_watcher",
]
