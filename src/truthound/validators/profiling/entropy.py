"""Entropy-based profiling validators.

This module provides validators that analyze information entropy
and related statistical properties of column values.
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
class EntropyValidator(ProfilingValidator):
    """Validates column entropy is within expected range.

    Shannon entropy measures the average information content
    (in bits) of a column's values.

    Use cases:
    - Detect constant columns (entropy ≈ 0)
    - Detect highly random data (high entropy)
    - Quality control for categorical columns

    Entropy interpretation:
    - 0 bits: All values are the same
    - log2(n) bits: All n values are unique and equally distributed
    - Higher entropy = more randomness/diversity

    Example:
        validator = EntropyValidator(
            column="status",
            min_entropy=1.0,  # At least 2 distinct values with some distribution
            max_entropy=3.0,  # Not too many distinct values
        )
    """

    name = "entropy"

    def __init__(
        self,
        column: str,
        min_entropy: float = 0.0,
        max_entropy: float | None = None,
        min_normalized_entropy: float | None = None,
        max_normalized_entropy: float | None = None,
        **kwargs: Any,
    ):
        """Initialize entropy validator.

        Args:
            column: Column to validate
            min_entropy: Minimum Shannon entropy (bits)
            max_entropy: Maximum Shannon entropy (bits)
            min_normalized_entropy: Minimum normalized entropy (0-1)
            max_normalized_entropy: Maximum normalized entropy (0-1)
            **kwargs: Additional config
        """
        super().__init__(column=column, **kwargs)
        self.min_entropy = min_entropy
        self.max_entropy = max_entropy
        self.min_normalized_entropy = min_normalized_entropy
        self.max_normalized_entropy = max_normalized_entropy

    def _compute_normalized_entropy(
        self, entropy: float, unique_count: int
    ) -> float:
        """Compute normalized entropy (0-1).

        Normalized entropy = entropy / max_possible_entropy

        Args:
            entropy: Shannon entropy
            unique_count: Number of unique values

        Returns:
            Normalized entropy (0-1)
        """
        if unique_count <= 1:
            return 0.0

        max_entropy = np.log2(unique_count)
        if max_entropy == 0:
            return 0.0

        return entropy / max_entropy

    def validate_profile(
        self, df: pl.DataFrame, metrics: ProfileMetrics
    ) -> list[ValidationIssue]:
        """Validate entropy constraints.

        Args:
            df: Input DataFrame
            metrics: Profile metrics

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        if metrics.total_count == 0:
            return issues

        entropy = metrics.entropy
        normalized = self._compute_normalized_entropy(
            entropy, metrics.unique_count
        )

        # Check raw entropy
        if entropy < self.min_entropy:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="entropy_too_low",
                    count=1,
                    severity=Severity.MEDIUM,
                    details=(
                        f"Entropy ({entropy:.4f} bits) below minimum "
                        f"({self.min_entropy:.4f} bits). "
                        f"Normalized: {normalized:.4f}. "
                        f"The column may have too little diversity."
                    ),
                    expected=f"Entropy >= {self.min_entropy} bits",
                )
            )

        if self.max_entropy is not None and entropy > self.max_entropy:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="entropy_too_high",
                    count=1,
                    severity=Severity.LOW,
                    details=(
                        f"Entropy ({entropy:.4f} bits) above maximum "
                        f"({self.max_entropy:.4f} bits). "
                        f"Normalized: {normalized:.4f}. "
                        f"The column may be too random or diverse."
                    ),
                    expected=f"Entropy <= {self.max_entropy} bits",
                )
            )

        # Check normalized entropy
        if self.min_normalized_entropy is not None:
            if normalized < self.min_normalized_entropy:
                issues.append(
                    ValidationIssue(
                        column=self.column,
                        issue_type="normalized_entropy_too_low",
                        count=1,
                        severity=Severity.MEDIUM,
                        details=(
                            f"Normalized entropy ({normalized:.4f}) below minimum "
                            f"({self.min_normalized_entropy:.4f}). "
                            f"Distribution may be too skewed."
                        ),
                        expected=f"Normalized entropy >= {self.min_normalized_entropy}",
                    )
                )

        if self.max_normalized_entropy is not None:
            if normalized > self.max_normalized_entropy:
                issues.append(
                    ValidationIssue(
                        column=self.column,
                        issue_type="normalized_entropy_too_high",
                        count=1,
                        severity=Severity.LOW,
                        details=(
                            f"Normalized entropy ({normalized:.4f}) above maximum "
                            f"({self.max_normalized_entropy:.4f}). "
                            f"Distribution may be too uniform."
                        ),
                        expected=f"Normalized entropy <= {self.max_normalized_entropy}",
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
class InformationGainValidator(ProfilingValidator):
    """Validates information gain between two columns.

    Information gain measures how much knowing one column
    reduces uncertainty about another column.

    Use cases:
    - Feature selection validation
    - Detecting redundant columns
    - Validating expected dependencies

    Example:
        validator = InformationGainValidator(
            column="category",
            target_column="subcategory",
            min_information_gain=0.5,
        )
    """

    name = "information_gain"

    def __init__(
        self,
        column: str,
        target_column: str,
        min_information_gain: float = 0.0,
        max_information_gain: float | None = None,
        **kwargs: Any,
    ):
        """Initialize information gain validator.

        Args:
            column: Feature column
            target_column: Target column to measure against
            min_information_gain: Minimum information gain
            max_information_gain: Maximum information gain
            **kwargs: Additional config
        """
        super().__init__(column=column, **kwargs)
        self.target_column = target_column
        self.min_information_gain = min_information_gain
        self.max_information_gain = max_information_gain

    def _compute_conditional_entropy(
        self, df: pl.DataFrame
    ) -> float:
        """Compute conditional entropy H(target|feature).

        Args:
            df: DataFrame with both columns

        Returns:
            Conditional entropy in bits
        """
        total = len(df)
        if total == 0:
            return 0.0

        # Group by feature column
        conditional_entropy = 0.0

        feature_groups = df.group_by(self.column).agg(
            pl.col(self.target_column).alias("targets")
        )

        for row in feature_groups.iter_rows():
            feature_val, targets = row
            group_size = len(targets)
            group_prob = group_size / total

            # Compute entropy within this group
            target_counts: dict[Any, int] = {}
            for t in targets:
                target_counts[t] = target_counts.get(t, 0) + 1

            group_entropy = 0.0
            for count in target_counts.values():
                if count > 0:
                    p = count / group_size
                    group_entropy -= p * np.log2(p)

            conditional_entropy += group_prob * group_entropy

        return conditional_entropy

    def _compute_target_entropy(self, df: pl.DataFrame) -> float:
        """Compute entropy of target column.

        Args:
            df: DataFrame

        Returns:
            Target entropy in bits
        """
        value_counts = (
            df.group_by(self.target_column)
            .agg(pl.len().alias("count"))
        )

        if len(value_counts) == 0:
            return 0.0

        counts = value_counts["count"].to_numpy()
        total = counts.sum()
        if total == 0:
            return 0.0

        probs = counts / total
        probs = probs[probs > 0]

        return float(-np.sum(probs * np.log2(probs)))

    def validate_profile(
        self, df: pl.DataFrame, metrics: ProfileMetrics
    ) -> list[ValidationIssue]:
        """Validate information gain.

        Args:
            df: Input DataFrame (must contain target_column too)
            metrics: Profile metrics (for feature column)

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        if len(df) == 0:
            return issues

        # Compute information gain
        target_entropy = self._compute_target_entropy(df)
        conditional_entropy = self._compute_conditional_entropy(df)
        info_gain = target_entropy - conditional_entropy

        if info_gain < self.min_information_gain:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="information_gain_too_low",
                    count=1,
                    severity=Severity.MEDIUM,
                    details=(
                        f"Information gain ({info_gain:.4f} bits) below minimum "
                        f"({self.min_information_gain:.4f} bits). "
                        f"'{self.column}' provides less predictive power for "
                        f"'{self.target_column}' than expected."
                    ),
                    expected=f"Information gain >= {self.min_information_gain}",
                )
            )

        if self.max_information_gain is not None:
            if info_gain > self.max_information_gain:
                issues.append(
                    ValidationIssue(
                        column=self.column,
                        issue_type="information_gain_too_high",
                        count=1,
                        severity=Severity.LOW,
                        details=(
                            f"Information gain ({info_gain:.4f} bits) above maximum "
                            f"({self.max_information_gain:.4f} bits). "
                            f"Columns may be too closely related."
                        ),
                        expected=f"Information gain <= {self.max_information_gain}",
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
        df = lf.select([
            pl.col(self.column),
            pl.col(self.target_column),
        ]).collect()
        metrics = self._compute_metrics(
            df.select(pl.col(self.column))
        )
        return self.validate_profile(df, metrics)
