"""Profile comparison and drift detection.

This module provides tools for comparing profiles across time
to detect data drift, schema changes, and quality degradation.

Key features:
- Column-level comparison
- Statistical drift detection
- Schema change detection
- Configurable alerting thresholds
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Iterator, Sequence

from truthound.profiler.base import (
    ColumnProfile,
    DataType,
    DistributionStats,
    TableProfile,
)


# =============================================================================
# Drift Types and Severity
# =============================================================================


class DriftType(str, Enum):
    """Types of data drift."""

    SCHEMA = "schema"           # Column added/removed/type changed
    DISTRIBUTION = "distribution"  # Statistical distribution changed
    COMPLETENESS = "completeness"  # Null ratio changed significantly
    UNIQUENESS = "uniqueness"   # Unique ratio changed
    PATTERN = "pattern"         # Pattern match ratio changed
    RANGE = "range"            # Value range changed
    CARDINALITY = "cardinality"  # Distinct count changed


class DriftSeverity(str, Enum):
    """Severity levels for detected drift."""

    INFO = "info"           # Minor change, informational
    WARNING = "warning"     # Notable change, may need attention
    CRITICAL = "critical"   # Significant change, likely requires action


class ChangeDirection(str, Enum):
    """Direction of change."""

    INCREASED = "increased"
    DECREASED = "decreased"
    UNCHANGED = "unchanged"


# =============================================================================
# Drift Detection Results
# =============================================================================


@dataclass(frozen=True)
class DriftResult:
    """Result of a single drift detection.

    Attributes:
        drift_type: Type of drift detected
        severity: Severity level
        column: Column name (None for table-level)
        metric: Specific metric that changed
        old_value: Previous value
        new_value: Current value
        change_ratio: Relative change (new - old) / old
        direction: Direction of change
        message: Human-readable description
        threshold: Threshold that was exceeded
    """

    drift_type: DriftType
    severity: DriftSeverity
    column: str | None
    metric: str
    old_value: Any
    new_value: Any
    change_ratio: float | None = None
    direction: ChangeDirection = ChangeDirection.UNCHANGED
    message: str = ""
    threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "drift_type": self.drift_type.value,
            "severity": self.severity.value,
            "column": self.column,
            "metric": self.metric,
            "old_value": self._serialize_value(self.old_value),
            "new_value": self._serialize_value(self.new_value),
            "change_ratio": self.change_ratio,
            "direction": self.direction.value,
            "message": self.message,
            "threshold": self.threshold,
        }

    def _serialize_value(self, value: Any) -> Any:
        """Serialize value for JSON."""
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Enum):
            return value.value
        return value


@dataclass(frozen=True)
class ColumnComparison:
    """Comparison result for a single column."""

    column_name: str
    exists_in_old: bool
    exists_in_new: bool
    drifts: tuple[DriftResult, ...] = field(default_factory=tuple)

    @property
    def has_drift(self) -> bool:
        """Check if any drift was detected."""
        return len(self.drifts) > 0

    @property
    def max_severity(self) -> DriftSeverity | None:
        """Get maximum severity among drifts."""
        if not self.drifts:
            return None
        severity_order = {
            DriftSeverity.INFO: 0,
            DriftSeverity.WARNING: 1,
            DriftSeverity.CRITICAL: 2,
        }
        return max(self.drifts, key=lambda d: severity_order[d.severity]).severity

    @property
    def is_new(self) -> bool:
        """Check if column is newly added."""
        return not self.exists_in_old and self.exists_in_new

    @property
    def is_removed(self) -> bool:
        """Check if column was removed."""
        return self.exists_in_old and not self.exists_in_new

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "column_name": self.column_name,
            "exists_in_old": self.exists_in_old,
            "exists_in_new": self.exists_in_new,
            "drifts": [d.to_dict() for d in self.drifts],
            "has_drift": self.has_drift,
            "max_severity": self.max_severity.value if self.max_severity else None,
        }


@dataclass(frozen=True)
class ProfileComparison:
    """Complete comparison between two profiles.

    This is the main result type for profile comparison, containing
    all detected drifts organized by column and type.
    """

    old_profile_name: str
    new_profile_name: str
    old_profiled_at: datetime
    new_profiled_at: datetime
    columns: tuple[ColumnComparison, ...] = field(default_factory=tuple)
    table_drifts: tuple[DriftResult, ...] = field(default_factory=tuple)
    comparison_timestamp: datetime = field(default_factory=datetime.now)

    def __iter__(self) -> Iterator[ColumnComparison]:
        """Iterate over column comparisons."""
        return iter(self.columns)

    def __len__(self) -> int:
        """Get number of columns compared."""
        return len(self.columns)

    @property
    def has_drift(self) -> bool:
        """Check if any drift was detected."""
        if self.table_drifts:
            return True
        return any(c.has_drift for c in self.columns)

    @property
    def has_schema_changes(self) -> bool:
        """Check if schema changed (columns added/removed)."""
        return any(c.is_new or c.is_removed for c in self.columns)

    @property
    def drift_count(self) -> int:
        """Get total number of drifts detected."""
        return len(self.table_drifts) + sum(len(c.drifts) for c in self.columns)

    @property
    def all_drifts(self) -> list[DriftResult]:
        """Get all drifts as flat list."""
        result = list(self.table_drifts)
        for col in self.columns:
            result.extend(col.drifts)
        return result

    def get_by_severity(self, severity: DriftSeverity) -> list[DriftResult]:
        """Get all drifts of a specific severity."""
        return [d for d in self.all_drifts if d.severity == severity]

    def get_by_type(self, drift_type: DriftType) -> list[DriftResult]:
        """Get all drifts of a specific type."""
        return [d for d in self.all_drifts if d.drift_type == drift_type]

    def get_column(self, name: str) -> ColumnComparison | None:
        """Get comparison for a specific column."""
        for col in self.columns:
            if col.column_name == name:
                return col
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "old_profile_name": self.old_profile_name,
            "new_profile_name": self.new_profile_name,
            "old_profiled_at": self.old_profiled_at.isoformat(),
            "new_profiled_at": self.new_profiled_at.isoformat(),
            "comparison_timestamp": self.comparison_timestamp.isoformat(),
            "has_drift": self.has_drift,
            "has_schema_changes": self.has_schema_changes,
            "drift_count": self.drift_count,
            "summary": self._build_summary(),
            "columns": [c.to_dict() for c in self.columns],
            "table_drifts": [d.to_dict() for d in self.table_drifts],
        }

    def _build_summary(self) -> dict[str, Any]:
        """Build summary statistics."""
        by_severity = {
            s.value: len(self.get_by_severity(s))
            for s in DriftSeverity
        }
        by_type = {
            t.value: len(self.get_by_type(t))
            for t in DriftType
        }
        return {
            "by_severity": by_severity,
            "by_type": by_type,
            "columns_added": sum(1 for c in self.columns if c.is_new),
            "columns_removed": sum(1 for c in self.columns if c.is_removed),
            "columns_with_drift": sum(1 for c in self.columns if c.has_drift),
        }

    def to_report(self) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 60,
            "PROFILE COMPARISON REPORT",
            "=" * 60,
            f"Old Profile: {self.old_profile_name} ({self.old_profiled_at})",
            f"New Profile: {self.new_profile_name} ({self.new_profiled_at})",
            f"Compared At: {self.comparison_timestamp}",
            "",
            "SUMMARY",
            "-" * 40,
            f"Total Drifts: {self.drift_count}",
            f"  Critical: {len(self.get_by_severity(DriftSeverity.CRITICAL))}",
            f"  Warning:  {len(self.get_by_severity(DriftSeverity.WARNING))}",
            f"  Info:     {len(self.get_by_severity(DriftSeverity.INFO))}",
            "",
        ]

        if self.has_schema_changes:
            lines.extend([
                "SCHEMA CHANGES",
                "-" * 40,
            ])
            for col in self.columns:
                if col.is_new:
                    lines.append(f"  + Added: {col.column_name}")
                elif col.is_removed:
                    lines.append(f"  - Removed: {col.column_name}")
            lines.append("")

        # Group drifts by severity
        for severity in [DriftSeverity.CRITICAL, DriftSeverity.WARNING, DriftSeverity.INFO]:
            drifts = self.get_by_severity(severity)
            if drifts:
                lines.extend([
                    f"{severity.value.upper()} DRIFTS",
                    "-" * 40,
                ])
                for drift in drifts:
                    col_str = f"[{drift.column}] " if drift.column else ""
                    lines.append(f"  {col_str}{drift.message}")
                lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# Drift Thresholds Configuration
# =============================================================================


@dataclass
class DriftThresholds:
    """Configuration for drift detection thresholds.

    All thresholds are relative changes (0.0 to 1.0) unless noted.

    Attributes:
        null_ratio_warning: Null ratio change for warning
        null_ratio_critical: Null ratio change for critical
        unique_ratio_warning: Unique ratio change for warning
        unique_ratio_critical: Unique ratio change for critical
        mean_warning: Mean change (relative) for warning
        mean_critical: Mean change (relative) for critical
        std_warning: Standard deviation change for warning
        std_critical: Standard deviation change for critical
        min_warning: Min value change for warning
        max_warning: Max value change for warning
        cardinality_warning: Distinct count change for warning
        cardinality_critical: Distinct count change for critical
        pattern_match_warning: Pattern match ratio change for warning
    """

    # Completeness
    null_ratio_warning: float = 0.05      # 5% change
    null_ratio_critical: float = 0.20     # 20% change

    # Uniqueness
    unique_ratio_warning: float = 0.10    # 10% change
    unique_ratio_critical: float = 0.30   # 30% change

    # Distribution (numeric)
    mean_warning: float = 0.10            # 10% relative change
    mean_critical: float = 0.30           # 30% relative change
    std_warning: float = 0.20             # 20% change
    std_critical: float = 0.50            # 50% change

    # Range
    min_warning: float = 0.10             # 10% change
    max_warning: float = 0.10             # 10% change

    # Cardinality
    cardinality_warning: float = 0.20     # 20% change
    cardinality_critical: float = 0.50    # 50% change

    # Patterns
    pattern_match_warning: float = 0.10   # 10% change

    @classmethod
    def strict(cls) -> "DriftThresholds":
        """Create strict thresholds (lower tolerance)."""
        return cls(
            null_ratio_warning=0.02,
            null_ratio_critical=0.10,
            unique_ratio_warning=0.05,
            unique_ratio_critical=0.15,
            mean_warning=0.05,
            mean_critical=0.15,
            std_warning=0.10,
            std_critical=0.30,
        )

    @classmethod
    def loose(cls) -> "DriftThresholds":
        """Create loose thresholds (higher tolerance)."""
        return cls(
            null_ratio_warning=0.10,
            null_ratio_critical=0.40,
            unique_ratio_warning=0.20,
            unique_ratio_critical=0.50,
            mean_warning=0.20,
            mean_critical=0.50,
            std_warning=0.30,
            std_critical=0.70,
        )


# =============================================================================
# Drift Detectors
# =============================================================================


class DriftDetector(ABC):
    """Abstract base for drift detection strategies."""

    name: str = "base"
    drift_type: DriftType = DriftType.DISTRIBUTION

    @abstractmethod
    def detect(
        self,
        old: ColumnProfile,
        new: ColumnProfile,
        thresholds: DriftThresholds,
    ) -> list[DriftResult]:
        """Detect drift between two column profiles.

        Args:
            old: Previous column profile
            new: Current column profile
            thresholds: Detection thresholds

        Returns:
            List of detected drifts
        """
        pass


class CompletenessDriftDetector(DriftDetector):
    """Detects changes in null ratio (completeness)."""

    name = "completeness"
    drift_type = DriftType.COMPLETENESS

    def detect(
        self,
        old: ColumnProfile,
        new: ColumnProfile,
        thresholds: DriftThresholds,
    ) -> list[DriftResult]:
        results = []

        old_null = old.null_ratio
        new_null = new.null_ratio
        change = abs(new_null - old_null)

        if change >= thresholds.null_ratio_critical:
            severity = DriftSeverity.CRITICAL
        elif change >= thresholds.null_ratio_warning:
            severity = DriftSeverity.WARNING
        else:
            return results

        direction = (
            ChangeDirection.INCREASED if new_null > old_null
            else ChangeDirection.DECREASED
        )

        results.append(DriftResult(
            drift_type=DriftType.COMPLETENESS,
            severity=severity,
            column=new.name,
            metric="null_ratio",
            old_value=old_null,
            new_value=new_null,
            change_ratio=change,
            direction=direction,
            message=f"Null ratio {direction.value} from {old_null:.2%} to {new_null:.2%}",
            threshold=thresholds.null_ratio_warning if severity == DriftSeverity.WARNING else thresholds.null_ratio_critical,
        ))

        return results


class UniquenessDriftDetector(DriftDetector):
    """Detects changes in unique ratio."""

    name = "uniqueness"
    drift_type = DriftType.UNIQUENESS

    def detect(
        self,
        old: ColumnProfile,
        new: ColumnProfile,
        thresholds: DriftThresholds,
    ) -> list[DriftResult]:
        results = []

        old_ratio = old.unique_ratio
        new_ratio = new.unique_ratio
        change = abs(new_ratio - old_ratio)

        if change >= thresholds.unique_ratio_critical:
            severity = DriftSeverity.CRITICAL
        elif change >= thresholds.unique_ratio_warning:
            severity = DriftSeverity.WARNING
        else:
            return results

        direction = (
            ChangeDirection.INCREASED if new_ratio > old_ratio
            else ChangeDirection.DECREASED
        )

        results.append(DriftResult(
            drift_type=DriftType.UNIQUENESS,
            severity=severity,
            column=new.name,
            metric="unique_ratio",
            old_value=old_ratio,
            new_value=new_ratio,
            change_ratio=change,
            direction=direction,
            message=f"Unique ratio {direction.value} from {old_ratio:.2%} to {new_ratio:.2%}",
            threshold=thresholds.unique_ratio_warning if severity == DriftSeverity.WARNING else thresholds.unique_ratio_critical,
        ))

        return results


class DistributionDriftDetector(DriftDetector):
    """Detects changes in statistical distribution."""

    name = "distribution"
    drift_type = DriftType.DISTRIBUTION

    def detect(
        self,
        old: ColumnProfile,
        new: ColumnProfile,
        thresholds: DriftThresholds,
    ) -> list[DriftResult]:
        results = []

        old_dist = old.distribution
        new_dist = new.distribution

        if old_dist is None or new_dist is None:
            return results

        # Check mean
        if old_dist.mean is not None and new_dist.mean is not None:
            mean_drift = self._check_relative_change(
                old_dist.mean,
                new_dist.mean,
                "mean",
                new.name,
                thresholds.mean_warning,
                thresholds.mean_critical,
            )
            if mean_drift:
                results.append(mean_drift)

        # Check standard deviation
        if old_dist.std is not None and new_dist.std is not None:
            std_drift = self._check_relative_change(
                old_dist.std,
                new_dist.std,
                "std",
                new.name,
                thresholds.std_warning,
                thresholds.std_critical,
            )
            if std_drift:
                results.append(std_drift)

        return results

    def _check_relative_change(
        self,
        old_val: float,
        new_val: float,
        metric: str,
        column: str,
        warning_threshold: float,
        critical_threshold: float,
    ) -> DriftResult | None:
        """Check for relative change in a metric."""
        if old_val == 0:
            if new_val != 0:
                return DriftResult(
                    drift_type=DriftType.DISTRIBUTION,
                    severity=DriftSeverity.WARNING,
                    column=column,
                    metric=metric,
                    old_value=old_val,
                    new_value=new_val,
                    change_ratio=None,
                    direction=ChangeDirection.INCREASED,
                    message=f"{metric} changed from 0 to {new_val:.4f}",
                )
            return None

        change_ratio = abs(new_val - old_val) / abs(old_val)

        if change_ratio >= critical_threshold:
            severity = DriftSeverity.CRITICAL
        elif change_ratio >= warning_threshold:
            severity = DriftSeverity.WARNING
        else:
            return None

        direction = (
            ChangeDirection.INCREASED if new_val > old_val
            else ChangeDirection.DECREASED
        )

        return DriftResult(
            drift_type=DriftType.DISTRIBUTION,
            severity=severity,
            column=column,
            metric=metric,
            old_value=old_val,
            new_value=new_val,
            change_ratio=change_ratio,
            direction=direction,
            message=f"{metric} {direction.value} by {change_ratio:.1%} ({old_val:.4f} -> {new_val:.4f})",
            threshold=warning_threshold if severity == DriftSeverity.WARNING else critical_threshold,
        )


class RangeDriftDetector(DriftDetector):
    """Detects changes in value range (min/max)."""

    name = "range"
    drift_type = DriftType.RANGE

    def detect(
        self,
        old: ColumnProfile,
        new: ColumnProfile,
        thresholds: DriftThresholds,
    ) -> list[DriftResult]:
        results = []

        old_dist = old.distribution
        new_dist = new.distribution

        if old_dist is None or new_dist is None:
            return results

        # Check min
        if old_dist.min is not None and new_dist.min is not None:
            min_drift = self._check_range_change(
                old_dist.min, new_dist.min,
                "min", new.name, thresholds.min_warning,
            )
            if min_drift:
                results.append(min_drift)

        # Check max
        if old_dist.max is not None and new_dist.max is not None:
            max_drift = self._check_range_change(
                old_dist.max, new_dist.max,
                "max", new.name, thresholds.max_warning,
            )
            if max_drift:
                results.append(max_drift)

        return results

    def _check_range_change(
        self,
        old_val: float,
        new_val: float,
        metric: str,
        column: str,
        threshold: float,
    ) -> DriftResult | None:
        """Check for range boundary changes."""
        if old_val == 0:
            return None

        change_ratio = abs(new_val - old_val) / abs(old_val)
        if change_ratio < threshold:
            return None

        direction = (
            ChangeDirection.INCREASED if new_val > old_val
            else ChangeDirection.DECREASED
        )

        return DriftResult(
            drift_type=DriftType.RANGE,
            severity=DriftSeverity.WARNING,
            column=column,
            metric=metric,
            old_value=old_val,
            new_value=new_val,
            change_ratio=change_ratio,
            direction=direction,
            message=f"{metric} {direction.value} from {old_val} to {new_val}",
            threshold=threshold,
        )


class CardinalityDriftDetector(DriftDetector):
    """Detects changes in cardinality (distinct count)."""

    name = "cardinality"
    drift_type = DriftType.CARDINALITY

    def detect(
        self,
        old: ColumnProfile,
        new: ColumnProfile,
        thresholds: DriftThresholds,
    ) -> list[DriftResult]:
        results = []

        old_count = old.distinct_count
        new_count = new.distinct_count

        if old_count == 0:
            return results

        change_ratio = abs(new_count - old_count) / old_count

        if change_ratio >= thresholds.cardinality_critical:
            severity = DriftSeverity.CRITICAL
        elif change_ratio >= thresholds.cardinality_warning:
            severity = DriftSeverity.WARNING
        else:
            return results

        direction = (
            ChangeDirection.INCREASED if new_count > old_count
            else ChangeDirection.DECREASED
        )

        results.append(DriftResult(
            drift_type=DriftType.CARDINALITY,
            severity=severity,
            column=new.name,
            metric="distinct_count",
            old_value=old_count,
            new_value=new_count,
            change_ratio=change_ratio,
            direction=direction,
            message=f"Distinct count {direction.value} by {change_ratio:.1%} ({old_count} -> {new_count})",
            threshold=thresholds.cardinality_warning if severity == DriftSeverity.WARNING else thresholds.cardinality_critical,
        ))

        return results


# Default detectors
DEFAULT_DETECTORS: tuple[DriftDetector, ...] = (
    CompletenessDriftDetector(),
    UniquenessDriftDetector(),
    DistributionDriftDetector(),
    RangeDriftDetector(),
    CardinalityDriftDetector(),
)


# =============================================================================
# Profile Comparator
# =============================================================================


class ProfileComparator:
    """Compares two profiles to detect drift.

    This is the main entry point for profile comparison. It orchestrates
    multiple drift detectors and builds a comprehensive comparison result.

    Example:
        comparator = ProfileComparator()
        comparison = comparator.compare(old_profile, new_profile)

        if comparison.has_drift:
            print(comparison.to_report())

        # With custom thresholds
        comparator = ProfileComparator(thresholds=DriftThresholds.strict())
    """

    def __init__(
        self,
        detectors: Sequence[DriftDetector] | None = None,
        thresholds: DriftThresholds | None = None,
    ):
        """Initialize comparator.

        Args:
            detectors: Custom drift detectors (uses defaults if None)
            thresholds: Detection thresholds
        """
        self.detectors = list(detectors) if detectors else list(DEFAULT_DETECTORS)
        self.thresholds = thresholds or DriftThresholds()

    def add_detector(self, detector: DriftDetector) -> None:
        """Add a custom drift detector."""
        self.detectors.append(detector)

    def compare(
        self,
        old_profile: TableProfile,
        new_profile: TableProfile,
    ) -> ProfileComparison:
        """Compare two profiles.

        Args:
            old_profile: Previous/baseline profile
            new_profile: Current profile to compare

        Returns:
            Complete comparison result
        """
        column_comparisons = []
        table_drifts = []

        # Build column name sets
        old_columns = {col.name: col for col in old_profile.columns}
        new_columns = {col.name: col for col in new_profile.columns}
        all_column_names = set(old_columns.keys()) | set(new_columns.keys())

        # Compare each column
        for col_name in sorted(all_column_names):
            old_col = old_columns.get(col_name)
            new_col = new_columns.get(col_name)

            comparison = self._compare_column(col_name, old_col, new_col)
            column_comparisons.append(comparison)

        # Check table-level changes
        table_drifts.extend(self._check_table_drift(old_profile, new_profile))

        return ProfileComparison(
            old_profile_name=old_profile.name,
            new_profile_name=new_profile.name,
            old_profiled_at=old_profile.profiled_at,
            new_profiled_at=new_profile.profiled_at,
            columns=tuple(column_comparisons),
            table_drifts=tuple(table_drifts),
        )

    def _compare_column(
        self,
        name: str,
        old_col: ColumnProfile | None,
        new_col: ColumnProfile | None,
    ) -> ColumnComparison:
        """Compare a single column."""
        drifts: list[DriftResult] = []

        # Check for schema changes
        if old_col is None:
            # New column added
            drifts.append(DriftResult(
                drift_type=DriftType.SCHEMA,
                severity=DriftSeverity.WARNING,
                column=name,
                metric="column",
                old_value=None,
                new_value=new_col.physical_type if new_col else None,
                message=f"New column added: {name}",
            ))
        elif new_col is None:
            # Column removed
            drifts.append(DriftResult(
                drift_type=DriftType.SCHEMA,
                severity=DriftSeverity.CRITICAL,
                column=name,
                metric="column",
                old_value=old_col.physical_type,
                new_value=None,
                message=f"Column removed: {name}",
            ))
        else:
            # Both exist - check for type change
            if old_col.physical_type != new_col.physical_type:
                drifts.append(DriftResult(
                    drift_type=DriftType.SCHEMA,
                    severity=DriftSeverity.CRITICAL,
                    column=name,
                    metric="physical_type",
                    old_value=old_col.physical_type,
                    new_value=new_col.physical_type,
                    message=f"Type changed from {old_col.physical_type} to {new_col.physical_type}",
                ))

            # Run all detectors
            for detector in self.detectors:
                try:
                    detector_drifts = detector.detect(old_col, new_col, self.thresholds)
                    drifts.extend(detector_drifts)
                except Exception:
                    pass  # Skip failed detectors

        return ColumnComparison(
            column_name=name,
            exists_in_old=old_col is not None,
            exists_in_new=new_col is not None,
            drifts=tuple(drifts),
        )

    def _check_table_drift(
        self,
        old_profile: TableProfile,
        new_profile: TableProfile,
    ) -> list[DriftResult]:
        """Check for table-level drift."""
        drifts = []

        # Row count change
        if old_profile.row_count > 0:
            row_change = abs(new_profile.row_count - old_profile.row_count) / old_profile.row_count
            if row_change >= 0.5:  # 50% change
                direction = (
                    ChangeDirection.INCREASED if new_profile.row_count > old_profile.row_count
                    else ChangeDirection.DECREASED
                )
                drifts.append(DriftResult(
                    drift_type=DriftType.CARDINALITY,
                    severity=DriftSeverity.WARNING,
                    column=None,
                    metric="row_count",
                    old_value=old_profile.row_count,
                    new_value=new_profile.row_count,
                    change_ratio=row_change,
                    direction=direction,
                    message=f"Row count {direction.value} by {row_change:.1%}",
                ))

        # Column count change
        if old_profile.column_count != new_profile.column_count:
            direction = (
                ChangeDirection.INCREASED if new_profile.column_count > old_profile.column_count
                else ChangeDirection.DECREASED
            )
            drifts.append(DriftResult(
                drift_type=DriftType.SCHEMA,
                severity=DriftSeverity.INFO,
                column=None,
                metric="column_count",
                old_value=old_profile.column_count,
                new_value=new_profile.column_count,
                direction=direction,
                message=f"Column count {direction.value} from {old_profile.column_count} to {new_profile.column_count}",
            ))

        return drifts


# =============================================================================
# Convenience Functions
# =============================================================================


def compare_profiles(
    old: TableProfile,
    new: TableProfile,
    *,
    thresholds: DriftThresholds | None = None,
) -> ProfileComparison:
    """Compare two profiles for drift detection.

    Args:
        old: Previous/baseline profile
        new: Current profile
        thresholds: Detection thresholds (uses defaults if None)

    Returns:
        Comparison result

    Example:
        from truthound.profiler import profile_file, compare_profiles

        old_profile = profile_file("data_v1.parquet")
        new_profile = profile_file("data_v2.parquet")

        comparison = compare_profiles(old_profile, new_profile)

        if comparison.has_drift:
            print(comparison.to_report())
    """
    comparator = ProfileComparator(thresholds=thresholds)
    return comparator.compare(old, new)


def detect_drift(
    old: TableProfile,
    new: TableProfile,
    *,
    min_severity: DriftSeverity = DriftSeverity.WARNING,
) -> list[DriftResult]:
    """Detect drifts above a minimum severity.

    Args:
        old: Previous profile
        new: Current profile
        min_severity: Minimum severity to return

    Returns:
        List of detected drifts
    """
    comparison = compare_profiles(old, new)

    severity_order = {
        DriftSeverity.INFO: 0,
        DriftSeverity.WARNING: 1,
        DriftSeverity.CRITICAL: 2,
    }
    min_level = severity_order[min_severity]

    return [
        d for d in comparison.all_drifts
        if severity_order[d.severity] >= min_level
    ]
