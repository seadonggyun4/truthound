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
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING

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

    def __post_init__(self) -> None:
        self.strategy = ExecutionStrategy(self.strategy)
        for name in ("sample_size", "native_threshold", "max_parallel"):
            value = getattr(self, name)
            minimum = 0 if name == "native_threshold" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.sample_seed is not None and (
            type(self.sample_seed) is not int or self.sample_seed < 0
        ):
            raise ValueError("sample_seed must be a non-negative integer")
        if type(self.parallel_validators) is not bool:
            raise ValueError("parallel_validators must be a boolean")


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


def classify_validator(validator: Validator) -> set[ValidatorCapability]:
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
    validator: Validator,
    engine: BaseDistributedEngine,
) -> bool:
    """Check if a validator can run natively on the distributed engine.

    Args:
        validator: Validator to check.
        engine: Distributed engine.

    Returns:
        True if validator can run natively.
    """
    from truthound.validators.completeness.null import NotNullValidator, NullValidator

    # A name/category is descriptive, never permission to replace validate().
    return type(validator) in {NullValidator, NotNullValidator} or callable(
        getattr(validator, "validate_distributed", None)
    )


class NativeValidatorRunner:
    """Dispatch explicit native ports and reuse Core's null issue construction.

    Null aggregates are native. Detailed result enrichment uses the Arrow bridge
    only after a failing aggregate requires samples, counts or complete rows.
    Other validators can implement validate_distributed(engine, columns=...).
    """

    def __init__(self, engine: BaseDistributedEngine) -> None:
        self._engine = engine

    def run(self, validator: Validator, columns: list[str] | None = None) -> list[ValidationIssue]:
        from truthound.validators.base import ValidationIssue
        from truthound.validators.completeness.null import NotNullValidator, NullValidator
        from truthound.validators.pushdown_support import PushdownResult

        if type(validator) not in {NullValidator, NotNullValidator}:
            port = getattr(validator, "validate_distributed", None)
            if not callable(port):
                raise ValueError("Validator has no native distributed execution port")
            issues = port(self._engine, columns=columns)
            if not isinstance(issues, list) or any(
                not isinstance(issue, ValidationIssue) for issue in issues
            ):
                raise TypeError("Native validator must return a list of ValidationIssue")
            return issues
        available = self._engine.get_columns()
        if columns is not None:
            if any(column not in available for column in columns):
                raise ValueError("Unknown validation column")
            available = columns
        targets = validator._resolve_pushdown_columns(available)
        total = self._engine.count_rows()
        if not targets or total == 0:
            return []
        results = [
            PushdownResult(
                column=column,
                check_type="null_count",
                value=self._engine.count_nulls(column),
                total_rows=total,
            )
            for column in targets
        ]
        issues = validator.process_pushdown_results(results)
        fmt = validator.config.get_result_format_config()
        if issues and (
            fmt.includes_unexpected_samples()
            or fmt.includes_unexpected_counts()
            or fmt.includes_full_results()
        ):
            lf = self._engine.to_polars_lazyframe()
            if columns is not None:
                lf = lf.select(columns)
            specs = validator.get_validation_exprs(lf, targets)
            if fmt.includes_unexpected_samples():
                validator._enrich_with_samples(lf, issues, specs, fmt)
            if fmt.includes_unexpected_counts():
                validator._enrich_with_value_counts(lf, issues, specs, fmt)
            if fmt.includes_full_results():
                validator._enrich_with_full_results(lf, issues, specs, fmt)
        return issues

    def run_null_check(
        self, validator: Validator, columns: list[str] | None = None
    ) -> list[ValidationIssue]:
        return self.run(validator, columns)

    def run_duplicate_check(
        self, validator: Validator, columns: list[str] | None = None
    ) -> list[ValidationIssue]:
        return self.run(validator, columns)

    def run_range_check(self, validator: Validator, column: str) -> list[ValidationIssue]:
        return self.run(validator, [column])

    def run_stats_check(self, validator: Validator, column: str) -> list[ValidationIssue]:
        return self.run(validator, [column])


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
        engine: BaseDistributedEngine,
        config: AdapterConfig | None = None,
    ) -> None:
        """Initialize adapter.

        Args:
            engine: Distributed execution engine.
            config: Optional configuration.
        """
        self._engine = engine
        self._config = replace(config) if config is not None else AdapterConfig()
        self._native_runner = NativeValidatorRunner(engine)
        self._execution_guard = None

    def set_execution_guard(self, guard) -> None:
        """Install the owner's authority check around each validator operation."""
        self._execution_guard = guard

    def _check_guard(self) -> None:
        if self._execution_guard is not None:
            self._execution_guard()

    @property
    def engine(self) -> BaseDistributedEngine:
        """Get the execution engine."""
        return self._engine

    @property
    def config(self) -> AdapterConfig:
        """Get adapter configuration."""
        return replace(self._config)

    def validate(
        self,
        validator: Validator,
        columns: list[str] | None = None,
    ) -> list[ValidationIssue]:
        """Run a validator on the distributed data.

        Args:
            validator: Validator to run.
            columns: Columns to validate.

        Returns:
            List of validation issues.
        """
        self._check_guard()
        strategy = self._determine_strategy(validator)
        if strategy == ExecutionStrategy.NATIVE:
            issues = self._run_native(validator, columns)
        elif strategy == ExecutionStrategy.SAMPLE_AND_LOCAL:
            issues = self._run_sampled(validator, columns)
        else:
            issues = self._run_full_conversion(validator, columns)
        self._check_guard()
        return issues

    def validate_all(
        self,
        validators: list[Validator],
        columns: list[str] | None = None,
    ) -> list[ValidationIssue]:
        """Run multiple validators on the distributed data.

        Args:
            validators: Validators to run.
            columns: Columns to validate.

        Returns:
            Combined list of validation issues.
        """
        if self._config.parallel_validators and len(validators) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=self._config.max_parallel) as executor:
                # map keeps input order and the context drains every submitted
                # operation before returning or propagating a validator failure.
                results = list(
                    executor.map(lambda validator: self.validate(validator, columns), validators)
                )
        else:
            results = [self.validate(validator, columns) for validator in validators]
        return [issue for issues in results for issue in issues]

    def _determine_strategy(self, validator: Validator) -> ExecutionStrategy:
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
        validator: Validator,
        columns: list[str] | None = None,
    ) -> list[ValidationIssue]:
        """Run validator using native distributed operations.

        Args:
            validator: Validator to run.
            columns: Columns to validate.

        Returns:
            List of validation issues.
        """
        return self._native_runner.run(validator, columns)

    def _run_sampled(
        self,
        validator: Validator,
        columns: list[str] | None = None,
    ) -> list[ValidationIssue]:
        """Run validator on sampled data.

        Args:
            validator: Validator to run.
            columns: Columns to validate.

        Returns:
            List of validation issues.
        """
        from contextlib import nullcontext

        from truthound.validators.base import ValidationSamplingInfo

        total_rows = self._engine.count_rows()
        sampled_engine = self._engine.sample(
            n=self._config.sample_size, seed=self._config.sample_seed
        )
        # sample() may return the root when the request covers the entire input.
        lifetime = nullcontext(sampled_engine) if sampled_engine is self._engine else sampled_engine
        with lifetime:
            lf = sampled_engine.to_polars_lazyframe()
            if columns is not None:
                lf = lf.select(columns)
            issues = validator.validate(lf)
            sample_rows = sampled_engine.count_rows()
            if sample_rows < total_rows:
                evidence = ValidationSamplingInfo(
                    total_rows=total_rows, sample_rows=sample_rows, seed=self._config.sample_seed
                )
                for issue in issues:
                    issue.sampling = evidence
            # Counts and percentages describe observed rows. Extrapolating only
            # count would contradict ValidationDetail and arbitrary custom logic.
            return issues

    def _run_full_conversion(
        self,
        validator: Validator,
        columns: list[str] | None = None,
    ) -> list[ValidationIssue]:
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
        if columns is not None:
            lf = lf.select(columns)

        # Run validator
        return validator.validate(lf)


# =============================================================================
# Convenience Functions
# =============================================================================


def validate_distributed(
    engine: BaseDistributedEngine,
    validators: list[Validator],
    columns: list[str] | None = None,
    config: AdapterConfig | None = None,
) -> list[ValidationIssue]:
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
    engine: BaseDistributedEngine,
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
