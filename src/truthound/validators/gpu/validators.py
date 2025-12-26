"""GPU-accelerated validators implementation.

This module provides concrete GPU-accelerated validators for common
data validation tasks using NVIDIA RAPIDS cuDF.

Validators:
    GPUNullValidator: GPU-accelerated null value detection
    GPURangeValidator: GPU-accelerated range validation
    GPUPatternValidator: GPU-accelerated regex pattern matching
    GPUUniqueValidator: GPU-accelerated uniqueness validation
    GPUStatisticsValidator: GPU-accelerated statistical analysis
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING
import re

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.registry import register_validator
from truthound.validators.gpu.base import (
    GPUValidator,
    GPUConfig,
    CUDF_AVAILABLE,
    is_gpu_available,
)

if TYPE_CHECKING or CUDF_AVAILABLE:
    try:
        import cudf
    except ImportError:
        cudf = None  # type: ignore


@register_validator
class GPUNullValidator(GPUValidator):
    """GPU-accelerated null value detection.

    Significantly faster than CPU for large datasets (>1M rows).

    Example:
        validator = GPUNullValidator(
            columns=["customer_id", "order_id"],
            max_null_ratio=0.05,
        )
        issues = validator.validate(df.lazy())
    """

    name = "gpu_null"
    category = "gpu_completeness"

    def __init__(
        self,
        columns: list[str] | None = None,
        max_null_ratio: float = 0.0,
        gpu_config: GPUConfig | None = None,
        **kwargs: Any,
    ):
        """Initialize GPU null validator.

        Args:
            columns: Columns to check (None = all columns)
            max_null_ratio: Maximum acceptable null ratio
            gpu_config: GPU configuration
            **kwargs: Additional config
        """
        super().__init__(gpu_config=gpu_config, columns=columns, **kwargs)
        self.max_null_ratio = max_null_ratio

    def _validate_gpu(self, gdf: "cudf.DataFrame") -> list[ValidationIssue]:
        """GPU-accelerated null validation using cuDF."""
        issues: list[ValidationIssue] = []
        total_rows = len(gdf)

        if total_rows == 0:
            return issues

        for column in gdf.columns:
            null_count = gdf[column].isna().sum()

            if null_count > 0:
                null_ratio = null_count / total_rows

                if null_ratio > self.max_null_ratio:
                    # Get sample null indices
                    null_mask = gdf[column].isna()
                    sample_indices = gdf[null_mask].head(5).index.tolist()

                    issues.append(
                        ValidationIssue(
                            column=column,
                            issue_type="gpu_null_values_detected",
                            count=int(null_count),
                            severity=self._calculate_severity(null_ratio),
                            details=(
                                f"Found {null_count:,} null values ({null_ratio:.2%}) "
                                f"in column '{column}' using GPU acceleration. "
                                f"Max allowed: {self.max_null_ratio:.2%}"
                            ),
                            expected=f"<= {self.max_null_ratio:.2%} nulls",
                            actual=f"{null_ratio:.2%} nulls",
                            sample_values=sample_indices[:5],
                        )
                    )

        return issues

    def _validate_cpu(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """CPU fallback using Polars."""
        from truthound.validators.completeness import NullValidator

        # Delegate to standard NullValidator for each column
        issues: list[ValidationIssue] = []
        columns = self._get_target_columns(lf)

        for column in columns:
            validator = NullValidator(column=column, max_null_ratio=self.max_null_ratio)
            issues.extend(validator.validate(lf))

        return issues


@register_validator
class GPURangeValidator(GPUValidator):
    """GPU-accelerated range validation.

    Validates that numeric values fall within specified bounds.

    Example:
        validator = GPURangeValidator(
            column="price",
            min_value=0,
            max_value=10000,
        )
        issues = validator.validate(df.lazy())
    """

    name = "gpu_range"
    category = "gpu_distribution"

    def __init__(
        self,
        column: str,
        min_value: float | None = None,
        max_value: float | None = None,
        inclusive: bool = True,
        gpu_config: GPUConfig | None = None,
        **kwargs: Any,
    ):
        """Initialize GPU range validator.

        Args:
            column: Column to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            inclusive: Whether bounds are inclusive
            gpu_config: GPU configuration
            **kwargs: Additional config
        """
        super().__init__(gpu_config=gpu_config, columns=[column], **kwargs)
        self.column = column
        self.min_value = min_value
        self.max_value = max_value
        self.inclusive = inclusive

    def _validate_gpu(self, gdf: "cudf.DataFrame") -> list[ValidationIssue]:
        """GPU-accelerated range validation using cuDF."""
        issues: list[ValidationIssue] = []

        if self.column not in gdf.columns:
            return issues

        col_data = gdf[self.column].dropna()
        total_rows = len(col_data)

        if total_rows == 0:
            return issues

        # Check minimum
        if self.min_value is not None:
            if self.inclusive:
                below_min = (col_data < self.min_value).sum()
            else:
                below_min = (col_data <= self.min_value).sum()

            if below_min > 0:
                violation_ratio = below_min / total_rows
                sample_below = col_data[col_data < self.min_value].head(5).to_pandas().tolist()

                issues.append(
                    ValidationIssue(
                        column=self.column,
                        issue_type="gpu_below_minimum",
                        count=int(below_min),
                        severity=self._calculate_severity(violation_ratio),
                        details=(
                            f"Found {below_min:,} values ({violation_ratio:.2%}) "
                            f"below minimum {self.min_value} using GPU acceleration"
                        ),
                        expected=f">{'=' if self.inclusive else ''} {self.min_value}",
                        actual=f"min: {col_data.min()}",
                        sample_values=sample_below,
                    )
                )

        # Check maximum
        if self.max_value is not None:
            if self.inclusive:
                above_max = (col_data > self.max_value).sum()
            else:
                above_max = (col_data >= self.max_value).sum()

            if above_max > 0:
                violation_ratio = above_max / total_rows
                sample_above = col_data[col_data > self.max_value].head(5).to_pandas().tolist()

                issues.append(
                    ValidationIssue(
                        column=self.column,
                        issue_type="gpu_above_maximum",
                        count=int(above_max),
                        severity=self._calculate_severity(violation_ratio),
                        details=(
                            f"Found {above_max:,} values ({violation_ratio:.2%}) "
                            f"above maximum {self.max_value} using GPU acceleration"
                        ),
                        expected=f"<{'=' if self.inclusive else ''} {self.max_value}",
                        actual=f"max: {col_data.max()}",
                        sample_values=sample_above,
                    )
                )

        return issues

    def _validate_cpu(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """CPU fallback using Polars."""
        from truthound.validators.distribution import RangeValidator

        validator = RangeValidator(
            column=self.column,
            min_value=self.min_value,
            max_value=self.max_value,
            inclusive=self.inclusive,
        )
        return validator.validate(lf)


@register_validator
class GPUPatternValidator(GPUValidator):
    """GPU-accelerated regex pattern validation.

    Uses cuDF's GPU-accelerated string operations for pattern matching.

    Example:
        validator = GPUPatternValidator(
            column="email",
            pattern=r"^[\\w.-]+@[\\w.-]+\\.\\w+$",
        )
        issues = validator.validate(df.lazy())
    """

    name = "gpu_pattern"
    category = "gpu_string"

    def __init__(
        self,
        column: str,
        pattern: str,
        invert: bool = False,
        gpu_config: GPUConfig | None = None,
        **kwargs: Any,
    ):
        """Initialize GPU pattern validator.

        Args:
            column: Column to validate
            pattern: Regex pattern to match
            invert: If True, values should NOT match the pattern
            gpu_config: GPU configuration
            **kwargs: Additional config
        """
        super().__init__(gpu_config=gpu_config, columns=[column], **kwargs)
        self.column = column
        self.pattern = pattern
        self.invert = invert
        # Pre-compile for CPU fallback
        self._compiled_pattern = re.compile(pattern)

    def _validate_gpu(self, gdf: "cudf.DataFrame") -> list[ValidationIssue]:
        """GPU-accelerated pattern validation using cuDF."""
        issues: list[ValidationIssue] = []

        if self.column not in gdf.columns:
            return issues

        col_data = gdf[self.column].dropna()
        total_rows = len(col_data)

        if total_rows == 0:
            return issues

        # cuDF string operations
        matches = col_data.str.match(self.pattern)

        if self.invert:
            # Values should NOT match
            violations = matches.sum()
            violation_mask = matches
        else:
            # Values should match
            violations = (~matches).sum()
            violation_mask = ~matches

        if violations > 0:
            violation_ratio = violations / total_rows

            # Get sample violations
            sample_values = col_data[violation_mask].head(5).to_pandas().tolist()

            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="gpu_pattern_mismatch",
                    count=int(violations),
                    severity=self._calculate_severity(violation_ratio),
                    details=(
                        f"Found {violations:,} values ({violation_ratio:.2%}) "
                        f"{'matching' if self.invert else 'not matching'} pattern "
                        f"'{self.pattern}' using GPU acceleration"
                    ),
                    expected=f"{'not ' if self.invert else ''}match pattern",
                    sample_values=sample_values,
                )
            )

        return issues

    def _validate_cpu(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """CPU fallback using Polars."""
        issues: list[ValidationIssue] = []

        df = lf.select(self.column).collect()
        col_data = df[self.column].drop_nulls()
        total_rows = len(col_data)

        if total_rows == 0:
            return issues

        # Polars regex matching
        matches = col_data.str.contains(self.pattern)

        if self.invert:
            violations = matches.sum()
        else:
            violations = (~matches).sum()

        if violations > 0:
            violation_ratio = violations / total_rows

            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="pattern_mismatch",
                    count=int(violations),
                    severity=self._calculate_severity(violation_ratio),
                    details=(
                        f"Found {violations:,} values ({violation_ratio:.2%}) "
                        f"{'matching' if self.invert else 'not matching'} pattern"
                    ),
                    expected=f"{'not ' if self.invert else ''}match pattern",
                )
            )

        return issues


@register_validator
class GPUUniqueValidator(GPUValidator):
    """GPU-accelerated uniqueness validation.

    Efficiently detects duplicate values using GPU acceleration.

    Example:
        validator = GPUUniqueValidator(
            columns=["customer_id"],
            allow_duplicates=False,
        )
        issues = validator.validate(df.lazy())
    """

    name = "gpu_unique"
    category = "gpu_uniqueness"

    def __init__(
        self,
        columns: list[str] | None = None,
        allow_duplicates: bool = False,
        max_duplicate_ratio: float = 0.0,
        gpu_config: GPUConfig | None = None,
        **kwargs: Any,
    ):
        """Initialize GPU unique validator.

        Args:
            columns: Columns to check for uniqueness
            allow_duplicates: Whether to allow any duplicates
            max_duplicate_ratio: Maximum acceptable duplicate ratio
            gpu_config: GPU configuration
            **kwargs: Additional config
        """
        super().__init__(gpu_config=gpu_config, columns=columns, **kwargs)
        self.allow_duplicates = allow_duplicates
        self.max_duplicate_ratio = max_duplicate_ratio

    def _validate_gpu(self, gdf: "cudf.DataFrame") -> list[ValidationIssue]:
        """GPU-accelerated uniqueness validation using cuDF."""
        issues: list[ValidationIssue] = []
        total_rows = len(gdf)

        if total_rows == 0:
            return issues

        for column in gdf.columns:
            col_data = gdf[column].dropna()
            unique_count = col_data.nunique()
            total_values = len(col_data)

            if total_values == 0:
                continue

            duplicate_count = total_values - unique_count
            duplicate_ratio = duplicate_count / total_values

            if not self.allow_duplicates and duplicate_count > 0:
                if duplicate_ratio > self.max_duplicate_ratio:
                    # Find duplicate values
                    value_counts = col_data.value_counts()
                    duplicates = value_counts[value_counts > 1]
                    sample_dups = duplicates.head(5).to_pandas().to_dict()

                    issues.append(
                        ValidationIssue(
                            column=column,
                            issue_type="gpu_duplicate_values",
                            count=int(duplicate_count),
                            severity=self._calculate_severity(duplicate_ratio),
                            details=(
                                f"Found {duplicate_count:,} duplicate values "
                                f"({duplicate_ratio:.2%}) in column '{column}' "
                                f"using GPU acceleration"
                            ),
                            expected=f"<= {self.max_duplicate_ratio:.2%} duplicates",
                            actual=f"{duplicate_ratio:.2%} duplicates",
                            sample_values=list(sample_dups.keys())[:5],
                        )
                    )

        return issues

    def _validate_cpu(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """CPU fallback using Polars."""
        from truthound.validators.uniqueness import UniqueValidator

        issues: list[ValidationIssue] = []
        columns = self._get_target_columns(lf)

        for column in columns:
            validator = UniqueValidator(column=column)
            issues.extend(validator.validate(lf))

        return issues


@register_validator
class GPUStatisticsValidator(GPUValidator):
    """GPU-accelerated statistical validation.

    Computes and validates statistical properties using GPU acceleration.

    Example:
        validator = GPUStatisticsValidator(
            column="revenue",
            expected_mean_range=(1000, 5000),
            expected_std_range=(100, 1000),
        )
        issues = validator.validate(df.lazy())
    """

    name = "gpu_statistics"
    category = "gpu_aggregate"

    def __init__(
        self,
        column: str,
        expected_mean: float | None = None,
        expected_mean_range: tuple[float, float] | None = None,
        expected_std: float | None = None,
        expected_std_range: tuple[float, float] | None = None,
        expected_min: float | None = None,
        expected_max: float | None = None,
        mean_tolerance: float = 0.1,
        std_tolerance: float = 0.1,
        gpu_config: GPUConfig | None = None,
        **kwargs: Any,
    ):
        """Initialize GPU statistics validator.

        Args:
            column: Column to analyze
            expected_mean: Expected mean value
            expected_mean_range: (min, max) range for mean
            expected_std: Expected standard deviation
            expected_std_range: (min, max) range for std
            expected_min: Expected minimum value
            expected_max: Expected maximum value
            mean_tolerance: Tolerance for mean comparison (fraction)
            std_tolerance: Tolerance for std comparison (fraction)
            gpu_config: GPU configuration
            **kwargs: Additional config
        """
        super().__init__(gpu_config=gpu_config, columns=[column], **kwargs)
        self.column = column
        self.expected_mean = expected_mean
        self.expected_mean_range = expected_mean_range
        self.expected_std = expected_std
        self.expected_std_range = expected_std_range
        self.expected_min = expected_min
        self.expected_max = expected_max
        self.mean_tolerance = mean_tolerance
        self.std_tolerance = std_tolerance

    def _validate_gpu(self, gdf: "cudf.DataFrame") -> list[ValidationIssue]:
        """GPU-accelerated statistics validation using cuDF."""
        issues: list[ValidationIssue] = []

        if self.column not in gdf.columns:
            return issues

        col_data = gdf[self.column].dropna()

        if len(col_data) == 0:
            return issues

        # Compute statistics on GPU
        actual_mean = float(col_data.mean())
        actual_std = float(col_data.std())
        actual_min = float(col_data.min())
        actual_max = float(col_data.max())

        # Check mean
        if self.expected_mean is not None:
            tolerance = abs(self.expected_mean * self.mean_tolerance)
            if abs(actual_mean - self.expected_mean) > tolerance:
                issues.append(
                    ValidationIssue(
                        column=self.column,
                        issue_type="gpu_mean_out_of_tolerance",
                        count=1,
                        severity=Severity.MEDIUM,
                        details=(
                            f"Mean {actual_mean:.4f} differs from expected "
                            f"{self.expected_mean:.4f} by more than {self.mean_tolerance:.1%} "
                            f"(GPU accelerated)"
                        ),
                        expected=f"~{self.expected_mean:.4f}",
                        actual=f"{actual_mean:.4f}",
                    )
                )

        # Check mean range
        if self.expected_mean_range is not None:
            min_mean, max_mean = self.expected_mean_range
            if actual_mean < min_mean or actual_mean > max_mean:
                issues.append(
                    ValidationIssue(
                        column=self.column,
                        issue_type="gpu_mean_out_of_range",
                        count=1,
                        severity=Severity.MEDIUM,
                        details=(
                            f"Mean {actual_mean:.4f} outside expected range "
                            f"[{min_mean:.4f}, {max_mean:.4f}] (GPU accelerated)"
                        ),
                        expected=f"[{min_mean:.4f}, {max_mean:.4f}]",
                        actual=f"{actual_mean:.4f}",
                    )
                )

        # Check std
        if self.expected_std is not None:
            tolerance = abs(self.expected_std * self.std_tolerance)
            if abs(actual_std - self.expected_std) > tolerance:
                issues.append(
                    ValidationIssue(
                        column=self.column,
                        issue_type="gpu_std_out_of_tolerance",
                        count=1,
                        severity=Severity.MEDIUM,
                        details=(
                            f"Std {actual_std:.4f} differs from expected "
                            f"{self.expected_std:.4f} by more than {self.std_tolerance:.1%} "
                            f"(GPU accelerated)"
                        ),
                        expected=f"~{self.expected_std:.4f}",
                        actual=f"{actual_std:.4f}",
                    )
                )

        # Check std range
        if self.expected_std_range is not None:
            min_std, max_std = self.expected_std_range
            if actual_std < min_std or actual_std > max_std:
                issues.append(
                    ValidationIssue(
                        column=self.column,
                        issue_type="gpu_std_out_of_range",
                        count=1,
                        severity=Severity.MEDIUM,
                        details=(
                            f"Std {actual_std:.4f} outside expected range "
                            f"[{min_std:.4f}, {max_std:.4f}] (GPU accelerated)"
                        ),
                        expected=f"[{min_std:.4f}, {max_std:.4f}]",
                        actual=f"{actual_std:.4f}",
                    )
                )

        # Check min/max bounds
        if self.expected_min is not None and actual_min < self.expected_min:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="gpu_below_expected_minimum",
                    count=1,
                    severity=Severity.HIGH,
                    details=(
                        f"Minimum value {actual_min:.4f} is below expected "
                        f"{self.expected_min:.4f} (GPU accelerated)"
                    ),
                    expected=f">= {self.expected_min:.4f}",
                    actual=f"{actual_min:.4f}",
                )
            )

        if self.expected_max is not None and actual_max > self.expected_max:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="gpu_above_expected_maximum",
                    count=1,
                    severity=Severity.HIGH,
                    details=(
                        f"Maximum value {actual_max:.4f} exceeds expected "
                        f"{self.expected_max:.4f} (GPU accelerated)"
                    ),
                    expected=f"<= {self.expected_max:.4f}",
                    actual=f"{actual_max:.4f}",
                )
            )

        return issues

    def _validate_cpu(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """CPU fallback using Polars."""
        issues: list[ValidationIssue] = []

        df = lf.select(self.column).collect()
        col_data = df[self.column].drop_nulls()

        if len(col_data) == 0:
            return issues

        actual_mean = col_data.mean()
        actual_std = col_data.std()
        actual_min = col_data.min()
        actual_max = col_data.max()

        # Similar checks as GPU version but with "cpu_" prefix
        if self.expected_mean is not None:
            tolerance = abs(self.expected_mean * self.mean_tolerance)
            if abs(actual_mean - self.expected_mean) > tolerance:
                issues.append(
                    ValidationIssue(
                        column=self.column,
                        issue_type="mean_out_of_tolerance",
                        count=1,
                        severity=Severity.MEDIUM,
                        details=(
                            f"Mean {actual_mean:.4f} differs from expected "
                            f"{self.expected_mean:.4f}"
                        ),
                        expected=f"~{self.expected_mean:.4f}",
                        actual=f"{actual_mean:.4f}",
                    )
                )

        if self.expected_mean_range is not None:
            min_mean, max_mean = self.expected_mean_range
            if actual_mean < min_mean or actual_mean > max_mean:
                issues.append(
                    ValidationIssue(
                        column=self.column,
                        issue_type="mean_out_of_range",
                        count=1,
                        severity=Severity.MEDIUM,
                        details=(
                            f"Mean {actual_mean:.4f} outside expected range "
                            f"[{min_mean:.4f}, {max_mean:.4f}]"
                        ),
                        expected=f"[{min_mean:.4f}, {max_mean:.4f}]",
                        actual=f"{actual_mean:.4f}",
                    )
                )

        return issues


def create_gpu_validator(
    validator_type: str,
    **kwargs: Any,
) -> Validator:
    """Factory function to create GPU validators with automatic fallback.

    Args:
        validator_type: Type of validator ("null", "range", "pattern", "unique", "statistics")
        **kwargs: Validator-specific arguments

    Returns:
        GPU validator if available, otherwise CPU equivalent

    Example:
        validator = create_gpu_validator(
            "range",
            column="price",
            min_value=0,
            max_value=10000,
        )
    """
    validators = {
        "null": (GPUNullValidator, "truthound.validators.completeness.NullValidator"),
        "range": (GPURangeValidator, "truthound.validators.distribution.RangeValidator"),
        "pattern": (GPUPatternValidator, "truthound.validators.string.PatternValidator"),
        "unique": (GPUUniqueValidator, "truthound.validators.uniqueness.UniqueValidator"),
        "statistics": (GPUStatisticsValidator, None),
    }

    if validator_type not in validators:
        raise ValueError(
            f"Unknown validator type: {validator_type}. "
            f"Available: {list(validators.keys())}"
        )

    gpu_cls, cpu_path = validators[validator_type]

    if is_gpu_available():
        return gpu_cls(**kwargs)

    if cpu_path is None:
        raise RuntimeError(
            f"GPU not available and no CPU fallback for {validator_type}"
        )

    # Dynamic import of CPU validator
    module_path, cls_name = cpu_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    cpu_cls = getattr(module, cls_name)

    return cpu_cls(**kwargs)
