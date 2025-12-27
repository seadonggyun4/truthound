"""Distributed validator adapters for running validators on distributed data.

This module provides adapters that allow standard Truthound validators
to run efficiently on distributed computing frameworks (Spark, Dask, Ray).

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                  DistributedValidatorAdapter                     │
    │                                                                  │
    │   ┌──────────────────────────────────────────────────────────┐  │
    │   │                    Standard Validator                     │  │
    │   │              (Polars-based implementation)                │  │
    │   └──────────────────────────────────────────────────────────┘  │
    │                              │                                   │
    │                              ▼                                   │
    │   ┌──────────────────────────────────────────────────────────┐  │
    │   │               Distributed Execution Path                  │  │
    │   │                                                           │  │
    │   │  ┌─────────────────┐   ┌─────────────────┐               │  │
    │   │  │ Native Path     │   │ Fallback Path   │               │  │
    │   │  │ (count, null,   │   │ (complex ML,    │               │  │
    │   │  │  stats, etc.)   │   │  pattern-based) │               │  │
    │   │  └─────────────────┘   └─────────────────┘               │  │
    │   │          │                      │                        │  │
    │   │          ▼                      ▼                        │  │
    │   │   ┌──────────────┐      ┌──────────────┐                 │  │
    │   │   │ Spark-native │      │ Arrow Bridge │                 │  │
    │   │   │  aggregation │      │  to Polars   │                 │  │
    │   │   └──────────────┘      └──────────────┘                 │  │
    │   │                                                           │  │
    │   └──────────────────────────────────────────────────────────┘  │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘

Execution Strategies:
1. NATIVE: Use distributed engine's native operations (fastest)
2. SAMPLE_AND_LOCAL: Sample data, run validator locally (medium)
3. FULL_CONVERSION: Convert all data to Polars (slowest, most accurate)

Example:
    >>> from truthound.execution.distributed import (
    ...     SparkExecutionEngine,
    ...     DistributedValidatorAdapter,
    ... )
    >>> from truthound.validators import NullValidator
    >>>
    >>> engine = SparkExecutionEngine.from_dataframe(spark_df)
    >>> adapter = DistributedValidatorAdapter(engine)
    >>>
    >>> # Run validator on distributed data
    >>> issues = adapter.validate(NullValidator())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable

from truthound.execution.distributed.protocols import (
    AggregationSpec,
    ComputeBackend,
)
from truthound.types import Severity

if TYPE_CHECKING:
    from truthound.execution.distributed.base import BaseDistributedEngine
    from truthound.validators.base import ValidationIssue, Validator

logger = logging.getLogger(__name__)


# =============================================================================
# Execution Strategy
# =============================================================================


class ExecutionStrategy(str, Enum):
    """Strategies for executing validators on distributed data."""

    NATIVE = "native"  # Use distributed engine's native operations
    SAMPLE_AND_LOCAL = "sample_and_local"  # Sample data, run locally
    FULL_CONVERSION = "full_conversion"  # Convert all to Polars
    AUTO = "auto"  # Auto-detect best strategy


@dataclass
class AdapterConfig:
    """Configuration for validator adapter.

    Attributes:
        strategy: Execution strategy to use.
        sample_size: Sample size for SAMPLE_AND_LOCAL strategy.
        sample_seed: Random seed for sampling.
        native_threshold: Row count threshold for native execution.
        parallel_validators: Run multiple validators in parallel.
        max_parallel: Maximum parallel validators.
    """

    strategy: ExecutionStrategy = ExecutionStrategy.AUTO
    sample_size: int = 100_000
    sample_seed: int | None = 42
    native_threshold: int = 1_000_000
    parallel_validators: bool = True
    max_parallel: int = 4


# =============================================================================
# Validator Classification
# =============================================================================


class ValidatorCapability(str, Enum):
    """Capabilities that affect distributed execution."""

    NULL_CHECK = "null_check"  # Can use native null counting
    DUPLICATE_CHECK = "duplicate_check"  # Can use native distinct
    RANGE_CHECK = "range_check"  # Can use native min/max
    STATS_CHECK = "stats_check"  # Can use native stats
    PATTERN_CHECK = "pattern_check"  # Requires string operations
    ML_BASED = "ml_based"  # Requires ML libraries
    CROSS_COLUMN = "cross_column"  # Requires multiple columns
    CUSTOM = "custom"  # Custom logic


def classify_validator(validator: "Validator") -> set[ValidatorCapability]:
    """Classify a validator's capabilities.

    Args:
        validator: Validator to classify.

    Returns:
        Set of capabilities.
    """
    capabilities = set()
    name = validator.name.lower()
    category = getattr(validator, "category", "").lower()

    # Null-related validators
    if "null" in name or "completeness" in name or "missing" in name:
        capabilities.add(ValidatorCapability.NULL_CHECK)

    # Duplicate-related validators
    if "duplicate" in name or "unique" in name:
        capabilities.add(ValidatorCapability.DUPLICATE_CHECK)

    # Range-related validators
    if "range" in name or "bound" in name or "outlier" in name:
        capabilities.add(ValidatorCapability.RANGE_CHECK)
        capabilities.add(ValidatorCapability.STATS_CHECK)

    # Pattern-related validators
    if "pattern" in name or "regex" in name or "format" in name:
        capabilities.add(ValidatorCapability.PATTERN_CHECK)

    # Statistics-related validators
    if "stat" in name or "distribution" in name or "anomaly" in name:
        capabilities.add(ValidatorCapability.STATS_CHECK)

    # ML-related validators
    if category == "ml" or "learning" in name or "model" in name:
        capabilities.add(ValidatorCapability.ML_BASED)

    # Cross-column validators
    if "cross" in name or "correlation" in name or "dependency" in name:
        capabilities.add(ValidatorCapability.CROSS_COLUMN)

    # If no capabilities detected, mark as custom
    if not capabilities:
        capabilities.add(ValidatorCapability.CUSTOM)

    return capabilities


def can_run_natively(
    validator: "Validator",
    engine: "BaseDistributedEngine",
) -> bool:
    """Check if a validator can run natively on the distributed engine.

    Args:
        validator: Validator to check.
        engine: Distributed engine.

    Returns:
        True if validator can run natively.
    """
    capabilities = classify_validator(validator)

    # These capabilities can run natively on Spark
    native_capabilities = {
        ValidatorCapability.NULL_CHECK,
        ValidatorCapability.DUPLICATE_CHECK,
        ValidatorCapability.RANGE_CHECK,
        ValidatorCapability.STATS_CHECK,
    }

    # Check if all capabilities are native
    return capabilities.issubset(native_capabilities)


# =============================================================================
# Native Validator Runners
# =============================================================================


class NativeValidatorRunner:
    """Runs validators using native distributed operations.

    This class provides implementations for common validation checks
    using the distributed engine's native operations, avoiding the
    overhead of converting to Polars.
    """

    def __init__(self, engine: "BaseDistributedEngine") -> None:
        """Initialize runner.

        Args:
            engine: Distributed execution engine.
        """
        self._engine = engine

    def run_null_check(
        self,
        validator: "Validator",
        columns: list[str] | None = None,
    ) -> list["ValidationIssue"]:
        """Run null check using native operations.

        Args:
            validator: Null validator.
            columns: Columns to check.

        Returns:
            List of validation issues.
        """
        from truthound.validators.base import ValidationIssue

        columns = columns or self._engine.get_columns()
        issues = []

        # Get all null counts in one distributed operation
        null_counts = self._engine.count_nulls_all()
        total_rows = self._engine.count_rows()

        # Get threshold from validator if available
        threshold = getattr(validator, "threshold", 0.0)
        max_null_ratio = getattr(validator, "max_null_ratio", 1.0)

        for column in columns:
            null_count = null_counts.get(column, 0)
            if null_count > 0:
                null_ratio = null_count / total_rows if total_rows > 0 else 0

                # Determine severity
                if null_ratio > max_null_ratio:
                    severity = Severity.HIGH
                elif null_ratio > threshold:
                    severity = Severity.MEDIUM
                else:
                    severity = Severity.LOW

                if null_ratio > threshold:
                    issues.append(
                        ValidationIssue(
                            column=column,
                            issue_type="null_values",
                            severity=severity,
                            count=null_count,
                            details=f"Found {null_count:,} null values ({null_ratio:.2%})",
                        )
                    )

        return issues

    def run_duplicate_check(
        self,
        validator: "Validator",
        columns: list[str] | None = None,
    ) -> list["ValidationIssue"]:
        """Run duplicate check using native operations.

        Args:
            validator: Duplicate validator.
            columns: Columns to check for duplicates.

        Returns:
            List of validation issues.
        """
        from truthound.validators.base import ValidationIssue

        columns = columns or getattr(validator, "columns", self._engine.get_columns())
        issues = []

        duplicate_count = self._engine.count_duplicates(list(columns))

        if duplicate_count > 0:
            issues.append(
                ValidationIssue(
                    column=",".join(columns),
                    issue_type="duplicates",
                    severity=Severity.MEDIUM,
                    count=duplicate_count,
                    details=f"Found {duplicate_count:,} duplicate rows",
                )
            )

        return issues

    def run_range_check(
        self,
        validator: "Validator",
        column: str,
    ) -> list["ValidationIssue"]:
        """Run range check using native operations.

        Args:
            validator: Range validator.
            column: Column to check.

        Returns:
            List of validation issues.
        """
        from truthound.validators.base import ValidationIssue

        issues = []

        # Get bounds from validator
        min_value = getattr(validator, "min_value", None)
        max_value = getattr(validator, "max_value", None)

        if min_value is None and max_value is None:
            return issues

        # Count out of range
        out_of_range = self._engine.count_outside_range(
            column, min_value, max_value
        )

        if out_of_range > 0:
            issues.append(
                ValidationIssue(
                    column=column,
                    issue_type="out_of_range",
                    severity=Severity.MEDIUM,
                    count=out_of_range,
                    details=f"Found {out_of_range:,} values outside range [{min_value}, {max_value}]",
                )
            )

        return issues

    def run_stats_check(
        self,
        validator: "Validator",
        column: str,
    ) -> list["ValidationIssue"]:
        """Run statistical check using native operations.

        Args:
            validator: Statistics validator.
            column: Column to check.

        Returns:
            List of validation issues.
        """
        from truthound.validators.base import ValidationIssue

        issues = []

        stats = self._engine.get_stats(column)

        # Check for expected mean
        expected_mean = getattr(validator, "expected_mean", None)
        mean_tolerance = getattr(validator, "mean_tolerance", 0.1)

        if expected_mean is not None:
            actual_mean = stats.get("mean", 0)
            if actual_mean is not None:
                diff = abs(actual_mean - expected_mean)
                if diff > mean_tolerance * abs(expected_mean):
                    issues.append(
                        ValidationIssue(
                            column=column,
                            issue_type="mean_deviation",
                            severity=Severity.MEDIUM,
                            count=1,
                            details=f"Mean ({actual_mean:.2f}) deviates from expected ({expected_mean:.2f})",
                        )
                    )

        # Check for high std
        max_std = getattr(validator, "max_std", None)
        if max_std is not None:
            actual_std = stats.get("std", 0)
            if actual_std is not None and actual_std > max_std:
                issues.append(
                    ValidationIssue(
                        column=column,
                        issue_type="high_variance",
                        severity=Severity.LOW,
                        count=1,
                        details=f"Standard deviation ({actual_std:.2f}) exceeds threshold ({max_std:.2f})",
                    )
                )

        return issues


# =============================================================================
# Distributed Validator Adapter
# =============================================================================


class DistributedValidatorAdapter:
    """Adapter for running validators on distributed execution engines.

    This adapter determines the best execution strategy for each
    validator and runs it efficiently on the distributed engine.

    Example:
        >>> engine = SparkExecutionEngine.from_dataframe(spark_df)
        >>> adapter = DistributedValidatorAdapter(engine)
        >>>
        >>> # Run single validator
        >>> issues = adapter.validate(NullValidator())
        >>>
        >>> # Run multiple validators
        >>> issues = adapter.validate_all([
        ...     NullValidator(),
        ...     DuplicateValidator(columns=["id"]),
        ... ])
    """

    def __init__(
        self,
        engine: "BaseDistributedEngine",
        config: AdapterConfig | None = None,
    ) -> None:
        """Initialize adapter.

        Args:
            engine: Distributed execution engine.
            config: Optional configuration.
        """
        self._engine = engine
        self._config = config or AdapterConfig()
        self._native_runner = NativeValidatorRunner(engine)

    @property
    def engine(self) -> "BaseDistributedEngine":
        """Get the execution engine."""
        return self._engine

    @property
    def config(self) -> AdapterConfig:
        """Get adapter configuration."""
        return self._config

    def validate(
        self,
        validator: "Validator",
        columns: list[str] | None = None,
    ) -> list["ValidationIssue"]:
        """Run a validator on the distributed data.

        Args:
            validator: Validator to run.
            columns: Columns to validate.

        Returns:
            List of validation issues.
        """
        strategy = self._determine_strategy(validator)

        if strategy == ExecutionStrategy.NATIVE:
            return self._run_native(validator, columns)
        elif strategy == ExecutionStrategy.SAMPLE_AND_LOCAL:
            return self._run_sampled(validator, columns)
        else:
            return self._run_full_conversion(validator, columns)

    def validate_all(
        self,
        validators: list["Validator"],
        columns: list[str] | None = None,
    ) -> list["ValidationIssue"]:
        """Run multiple validators on the distributed data.

        Args:
            validators: Validators to run.
            columns: Columns to validate.

        Returns:
            Combined list of validation issues.
        """
        all_issues = []

        if self._config.parallel_validators:
            # TODO: Implement parallel validator execution
            # For now, run sequentially
            for validator in validators:
                issues = self.validate(validator, columns)
                all_issues.extend(issues)
        else:
            for validator in validators:
                issues = self.validate(validator, columns)
                all_issues.extend(issues)

        return all_issues

    def _determine_strategy(self, validator: "Validator") -> ExecutionStrategy:
        """Determine the best execution strategy for a validator.

        Args:
            validator: Validator to analyze.

        Returns:
            Recommended execution strategy.
        """
        if self._config.strategy != ExecutionStrategy.AUTO:
            return self._config.strategy

        # Check if validator can run natively
        if can_run_natively(validator, self._engine):
            return ExecutionStrategy.NATIVE

        # Check row count
        row_count = self._engine.count_rows()

        if row_count <= self._config.native_threshold:
            # Small enough for full conversion
            return ExecutionStrategy.FULL_CONVERSION
        else:
            # Large dataset, use sampling
            return ExecutionStrategy.SAMPLE_AND_LOCAL

    def _run_native(
        self,
        validator: "Validator",
        columns: list[str] | None = None,
    ) -> list["ValidationIssue"]:
        """Run validator using native distributed operations.

        Args:
            validator: Validator to run.
            columns: Columns to validate.

        Returns:
            List of validation issues.
        """
        capabilities = classify_validator(validator)

        issues = []

        if ValidatorCapability.NULL_CHECK in capabilities:
            issues.extend(self._native_runner.run_null_check(validator, columns))

        if ValidatorCapability.DUPLICATE_CHECK in capabilities:
            issues.extend(self._native_runner.run_duplicate_check(validator, columns))

        if ValidatorCapability.RANGE_CHECK in capabilities:
            for column in (columns or self._engine.get_columns()):
                issues.extend(self._native_runner.run_range_check(validator, column))

        if ValidatorCapability.STATS_CHECK in capabilities:
            for column in (columns or self._engine.get_columns()):
                issues.extend(self._native_runner.run_stats_check(validator, column))

        return issues

    def _run_sampled(
        self,
        validator: "Validator",
        columns: list[str] | None = None,
    ) -> list["ValidationIssue"]:
        """Run validator on sampled data.

        Args:
            validator: Validator to run.
            columns: Columns to validate.

        Returns:
            List of validation issues.
        """
        # Sample the distributed data
        sampled_engine = self._engine.sample(
            n=self._config.sample_size,
            seed=self._config.sample_seed,
        )

        # Convert to Polars LazyFrame
        lf = sampled_engine.to_polars_lazyframe()

        # Run validator
        issues = validator.validate(lf)

        # Adjust counts based on sampling ratio
        total_rows = self._engine.count_rows()
        sample_rows = sampled_engine.count_rows()
        ratio = total_rows / sample_rows if sample_rows > 0 else 1

        for issue in issues:
            if hasattr(issue, "count") and issue.count:
                # Estimate actual count
                issue.count = int(issue.count * ratio)
                issue.details = f"(Estimated from sample) {issue.details}"

        return issues

    def _run_full_conversion(
        self,
        validator: "Validator",
        columns: list[str] | None = None,
    ) -> list["ValidationIssue"]:
        """Run validator with full data conversion to Polars.

        Args:
            validator: Validator to run.
            columns: Columns to validate.

        Returns:
            List of validation issues.
        """
        # Convert distributed data to Polars
        lf = self._engine.to_polars_lazyframe()

        # Select columns if specified
        if columns:
            lf = lf.select(columns)

        # Run validator
        return validator.validate(lf)


# =============================================================================
# Convenience Functions
# =============================================================================


def validate_distributed(
    engine: "BaseDistributedEngine",
    validators: list["Validator"],
    columns: list[str] | None = None,
    config: AdapterConfig | None = None,
) -> list["ValidationIssue"]:
    """Run validators on distributed data.

    Args:
        engine: Distributed execution engine.
        validators: Validators to run.
        columns: Columns to validate.
        config: Adapter configuration.

    Returns:
        List of validation issues.
    """
    adapter = DistributedValidatorAdapter(engine, config)
    return adapter.validate_all(validators, columns)


def create_distributed_adapter(
    engine: "BaseDistributedEngine",
    strategy: ExecutionStrategy = ExecutionStrategy.AUTO,
    sample_size: int = 100_000,
) -> DistributedValidatorAdapter:
    """Create a distributed validator adapter.

    Args:
        engine: Distributed execution engine.
        strategy: Execution strategy.
        sample_size: Sample size for sampling strategy.

    Returns:
        Configured adapter.
    """
    config = AdapterConfig(
        strategy=strategy,
        sample_size=sample_size,
    )
    return DistributedValidatorAdapter(engine, config)
