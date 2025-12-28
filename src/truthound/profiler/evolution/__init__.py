"""Schema evolution detection module.

This module provides schema change detection and compatibility analysis,
enabling proactive alerting when data structures evolve.

Key Features:
- Detect schema changes (columns added/removed, types changed)
- Classify changes as breaking or non-breaking
- Compatibility analysis for downstream consumers
- Integration with notification systems

Example:
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

__all__ = [
    # Protocols
    "SchemaChangeDetector",
    "CompatibilityChecker",
    "ChangeNotifier",
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
    # Alerts
    "SchemaChangeAlert",
    "SchemaChangeAlertManager",
    "ConsoleAlertHandler",
    "LoggingAlertHandler",
]
