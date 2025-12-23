"""Value frequency profiling validators.

This module provides validators that analyze value frequency
distributions for quality assessment.
"""

from typing import Any

import polars as pl
import numpy as np

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.registry import register_validator
from truthound.validators.profiling.base import (
    ProfileMetrics,
    ProfilingValidator,
)


@register_validator
class ValueFrequencyValidator(ProfilingValidator):
    """Validates value frequency distribution patterns.

    Checks:
    - Most common value frequency bounds
    - Least common value frequency bounds
    - Distribution shape (uniform, skewed, etc.)
    - Dominance of top values

    Example:
        validator = ValueFrequencyValidator(
            column="category",
            max_top_frequency=0.5,  # No single value > 50%
            min_bottom_frequency=0.01,  # Each value at least 1%
        )
    """

    name = "value_frequency"

    def __init__(
        self,
        column: str,
        min_top_frequency: float | None = None,
        max_top_frequency: float | None = None,
        min_bottom_frequency: float | None = None,
        max_bottom_frequency: float | None = None,
        top_n_max_ratio: float | None = None,
        top_n: int = 5,
        expected_values: list[Any] | None = None,
        expected_frequencies: dict[Any, float] | None = None,
        frequency_tolerance: float = 0.1,
        **kwargs: Any,
    ):
        """Initialize value frequency validator.

        Args:
            column: Column to validate
            min_top_frequency: Minimum frequency of most common value
            max_top_frequency: Maximum frequency of most common value
            min_bottom_frequency: Minimum frequency of least common value
            max_bottom_frequency: Maximum frequency of least common value
            top_n_max_ratio: Maximum ratio for top N values combined
            top_n: Number of top values for ratio calculation
            expected_values: List of expected values
            expected_frequencies: Expected frequency per value
            frequency_tolerance: Tolerance for expected frequencies
            **kwargs: Additional config
        """
        super().__init__(column=column, **kwargs)
        self.min_top_frequency = min_top_frequency
        self.max_top_frequency = max_top_frequency
        self.min_bottom_frequency = min_bottom_frequency
        self.max_bottom_frequency = max_bottom_frequency
        self.top_n_max_ratio = top_n_max_ratio
        self.top_n = top_n
        self.expected_values = expected_values
        self.expected_frequencies = expected_frequencies
        self.frequency_tolerance = frequency_tolerance

    def _get_frequency_distribution(
        self, df: pl.DataFrame
    ) -> list[tuple[Any, int, float]]:
        """Get value frequency distribution.

        Args:
            df: Input DataFrame

        Returns:
            List of (value, count, frequency) tuples, sorted by count desc
        """
        value_counts = self._compute_value_counts(df)

        if len(value_counts) == 0:
            return []

        total = value_counts["count"].sum()
        if total == 0:
            return []

        result = []
        for row in value_counts.iter_rows():
            value, count = row
            freq = count / total
            result.append((value, count, freq))

        return result

    def validate_profile(
        self, df: pl.DataFrame, metrics: ProfileMetrics
    ) -> list[ValidationIssue]:
        """Validate value frequency distribution.

        Args:
            df: Input DataFrame
            metrics: Profile metrics

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        if metrics.total_count == 0:
            return issues

        distribution = self._get_frequency_distribution(df)
        if not distribution:
            return issues

        # Top frequency checks
        top_value, top_count, top_freq = distribution[0]

        if self.min_top_frequency is not None:
            if top_freq < self.min_top_frequency:
                issues.append(
                    ValidationIssue(
                        column=self.column,
                        issue_type="top_frequency_too_low",
                        count=1,
                        severity=Severity.LOW,
                        details=(
                            f"Top value '{top_value}' frequency ({top_freq:.4f}) "
                            f"below minimum ({self.min_top_frequency:.4f}). "
                            f"Distribution may be more uniform than expected."
                        ),
                        expected=f"Top frequency >= {self.min_top_frequency}",
                    )
                )

        if self.max_top_frequency is not None:
            if top_freq > self.max_top_frequency:
                issues.append(
                    ValidationIssue(
                        column=self.column,
                        issue_type="top_frequency_too_high",
                        count=top_count,
                        severity=Severity.MEDIUM,
                        details=(
                            f"Top value '{top_value}' frequency ({top_freq:.4f}) "
                            f"exceeds maximum ({self.max_top_frequency:.4f}). "
                            f"Single value dominates the column."
                        ),
                        expected=f"Top frequency <= {self.max_top_frequency}",
                    )
                )

        # Bottom frequency checks
        bottom_value, bottom_count, bottom_freq = distribution[-1]

        if self.min_bottom_frequency is not None:
            if bottom_freq < self.min_bottom_frequency:
                # Count how many values are below threshold
                rare_count = sum(
                    1 for _, _, f in distribution
                    if f < self.min_bottom_frequency
                )
                issues.append(
                    ValidationIssue(
                        column=self.column,
                        issue_type="rare_values_detected",
                        count=rare_count,
                        severity=Severity.LOW,
                        details=(
                            f"Found {rare_count} values with frequency below "
                            f"{self.min_bottom_frequency:.4f}. "
                            f"Least common: '{bottom_value}' ({bottom_freq:.4f})"
                        ),
                        expected=f"All values frequency >= {self.min_bottom_frequency}",
                    )
                )

        # Top N ratio check
        if self.top_n_max_ratio is not None:
            top_n_freq = sum(f for _, _, f in distribution[: self.top_n])
            if top_n_freq > self.top_n_max_ratio:
                top_values = [str(v) for v, _, _ in distribution[: self.top_n]]
                issues.append(
                    ValidationIssue(
                        column=self.column,
                        issue_type="top_n_ratio_too_high",
                        count=1,
                        severity=Severity.MEDIUM,
                        details=(
                            f"Top {self.top_n} values account for {top_n_freq:.4f} "
                            f"of data, exceeds {self.top_n_max_ratio:.4f}. "
                            f"Values: {top_values}"
                        ),
                        expected=f"Top {self.top_n} ratio <= {self.top_n_max_ratio}",
                    )
                )

        # Expected values check
        if self.expected_values is not None:
            actual_values = {v for v, _, _ in distribution}
            missing = set(self.expected_values) - actual_values
            if missing:
                issues.append(
                    ValidationIssue(
                        column=self.column,
                        issue_type="expected_values_missing",
                        count=len(missing),
                        severity=Severity.MEDIUM,
                        details=(
                            f"Missing {len(missing)} expected values: "
                            f"{list(missing)[:5]}"
                        ),
                        expected=f"All of {self.expected_values}",
                    )
                )

        # Expected frequencies check
        if self.expected_frequencies is not None:
            freq_dict = {v: f for v, _, f in distribution}

            deviations = []
            for value, expected_freq in self.expected_frequencies.items():
                actual_freq = freq_dict.get(value, 0.0)
                deviation = abs(actual_freq - expected_freq)
                if deviation > self.frequency_tolerance:
                    deviations.append((value, expected_freq, actual_freq))

            if deviations:
                dev_strs = [
                    f"{v}: expected {ef:.4f}, got {af:.4f}"
                    for v, ef, af in deviations[:3]
                ]
                issues.append(
                    ValidationIssue(
                        column=self.column,
                        issue_type="frequency_deviation",
                        count=len(deviations),
                        severity=Severity.MEDIUM,
                        details=(
                            f"Found {len(deviations)} values with frequency "
                            f"deviation > {self.frequency_tolerance}. "
                            f"Examples: {'; '.join(dev_strs)}"
                        ),
                        expected="Frequencies within tolerance",
                    )
                )

        return issues

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate the LazyFrame.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        df = lf.select(pl.col(self.column)).collect()
        metrics = self._compute_metrics(df)
        return self.validate_profile(df, metrics)


@register_validator
class DistributionShapeValidator(ProfilingValidator):
    """Validates the shape of value distribution.

    Checks for:
    - Uniform distribution
    - Power law distribution
    - Normal-like distribution (for numeric)
    - Custom distribution matching

    Example:
        validator = DistributionShapeValidator(
            column="user_activity",
            expected_shape="power_law",
            shape_tolerance=0.1,
        )
    """

    name = "distribution_shape"

    def __init__(
        self,
        column: str,
        expected_shape: str = "any",
        shape_tolerance: float = 0.2,
        min_gini: float | None = None,
        max_gini: float | None = None,
        **kwargs: Any,
    ):
        """Initialize distribution shape validator.

        Args:
            column: Column to validate
            expected_shape: Expected distribution shape
                ("uniform", "power_law", "any")
            shape_tolerance: Tolerance for shape matching
            min_gini: Minimum Gini coefficient (0=perfect equality)
            max_gini: Maximum Gini coefficient (1=perfect inequality)
            **kwargs: Additional config
        """
        super().__init__(column=column, **kwargs)
        self.expected_shape = expected_shape
        self.shape_tolerance = shape_tolerance
        self.min_gini = min_gini
        self.max_gini = max_gini

    def _compute_gini(self, frequencies: np.ndarray) -> float:
        """Compute Gini coefficient of frequency distribution.

        Args:
            frequencies: Array of frequencies (must sum to 1)

        Returns:
            Gini coefficient (0-1)
        """
        if len(frequencies) == 0:
            return 0.0

        # Sort frequencies
        sorted_freqs = np.sort(frequencies)
        n = len(sorted_freqs)

        # Compute Gini
        cumsum = np.cumsum(sorted_freqs)
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_freqs))) / (n * np.sum(sorted_freqs)) - (n + 1) / n

        return float(max(0.0, gini))

    def _check_uniform(self, frequencies: np.ndarray) -> float:
        """Check how close to uniform distribution.

        Args:
            frequencies: Array of frequencies

        Returns:
            Distance from uniform (0 = perfect uniform)
        """
        if len(frequencies) == 0:
            return 0.0

        expected = 1.0 / len(frequencies)
        deviations = np.abs(frequencies - expected)
        return float(np.mean(deviations))

    def validate_profile(
        self, df: pl.DataFrame, metrics: ProfileMetrics
    ) -> list[ValidationIssue]:
        """Validate distribution shape.

        Args:
            df: Input DataFrame
            metrics: Profile metrics

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        if metrics.total_count == 0:
            return issues

        # Get frequencies
        value_counts = self._compute_value_counts(df)
        if len(value_counts) == 0:
            return issues

        counts = value_counts["count"].to_numpy()
        frequencies = counts / counts.sum()

        # Compute Gini coefficient
        gini = self._compute_gini(frequencies)

        # Check Gini bounds
        if self.min_gini is not None and gini < self.min_gini:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="gini_too_low",
                    count=1,
                    severity=Severity.LOW,
                    details=(
                        f"Gini coefficient ({gini:.4f}) below minimum "
                        f"({self.min_gini:.4f}). "
                        f"Distribution is more equal than expected."
                    ),
                    expected=f"Gini >= {self.min_gini}",
                )
            )

        if self.max_gini is not None and gini > self.max_gini:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="gini_too_high",
                    count=1,
                    severity=Severity.MEDIUM,
                    details=(
                        f"Gini coefficient ({gini:.4f}) above maximum "
                        f"({self.max_gini:.4f}). "
                        f"Distribution is more unequal than expected."
                    ),
                    expected=f"Gini <= {self.max_gini}",
                )
            )

        # Check expected shape
        if self.expected_shape == "uniform":
            deviation = self._check_uniform(frequencies)
            if deviation > self.shape_tolerance:
                issues.append(
                    ValidationIssue(
                        column=self.column,
                        issue_type="non_uniform_distribution",
                        count=1,
                        severity=Severity.MEDIUM,
                        details=(
                            f"Distribution deviates from uniform by {deviation:.4f} "
                            f"(tolerance: {self.shape_tolerance:.4f}). "
                            f"Gini: {gini:.4f}"
                        ),
                        expected="Uniform distribution",
                    )
                )

        return issues

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate the LazyFrame.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        df = lf.select(pl.col(self.column)).collect()
        metrics = self._compute_metrics(df)
        return self.validate_profile(df, metrics)
