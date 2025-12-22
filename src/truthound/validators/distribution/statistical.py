"""Statistical distribution validators."""

import math
from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator, NumericValidatorMixin
from truthound.validators.registry import register_validator


@register_validator
class KLDivergenceValidator(Validator, NumericValidatorMixin):
    """Validates that column distribution is similar to a reference distribution using KL divergence.

    KL divergence measures how one probability distribution differs from another.
    Lower values indicate more similar distributions.

    Example:
        # Compare current data distribution to historical baseline
        validator = KLDivergenceValidator(
            column="age",
            reference_distribution={"18-25": 0.2, "26-35": 0.3, "36-50": 0.35, "51+": 0.15},
            max_divergence=0.1,
        )
    """

    name = "kl_divergence"
    category = "distribution"

    def __init__(
        self,
        column: str,
        reference_distribution: dict[Any, float] | None = None,
        reference_data: pl.DataFrame | pl.LazyFrame | None = None,
        max_divergence: float = 0.1,
        num_bins: int = 10,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.column = column
        self.max_divergence = max_divergence
        self.num_bins = num_bins

        # Build reference distribution
        if reference_distribution:
            self.reference_dist = reference_distribution
        elif reference_data is not None:
            self.reference_dist = self._build_distribution(reference_data)
        else:
            raise ValueError("Either reference_distribution or reference_data must be provided")

    def _build_distribution(self, data: pl.DataFrame | pl.LazyFrame) -> dict[Any, float]:
        """Build probability distribution from data."""
        if isinstance(data, pl.LazyFrame):
            df = data.collect()
        else:
            df = data

        counts = df.group_by(self.column).len().sort(self.column)
        total = counts["len"].sum()

        dist = {}
        for row in counts.iter_rows():
            dist[row[0]] = row[1] / total

        return dist

    def _kl_divergence(self, p: dict[Any, float], q: dict[Any, float]) -> float:
        """Calculate KL divergence D(P || Q)."""
        divergence = 0.0
        epsilon = 1e-10  # Small value to avoid log(0)

        all_keys = set(p.keys()) | set(q.keys())

        for key in all_keys:
            p_val = p.get(key, epsilon)
            q_val = q.get(key, epsilon)

            if p_val > 0:
                divergence += p_val * math.log(p_val / q_val)

        return divergence

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # Build current distribution
        current_dist = self._build_distribution(lf)

        # Calculate KL divergence
        divergence = self._kl_divergence(current_dist, self.reference_dist)

        if divergence > self.max_divergence:
            severity = Severity.CRITICAL if divergence > self.max_divergence * 3 else \
                       Severity.HIGH if divergence > self.max_divergence * 2 else Severity.MEDIUM

            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="kl_divergence_exceeded",
                    count=1,
                    severity=severity,
                    details=f"KL divergence {divergence:.4f} exceeds threshold {self.max_divergence}",
                    expected=f"<= {self.max_divergence}",
                    actual=f"{divergence:.4f}",
                )
            )

        return issues


@register_validator
class ChiSquareValidator(Validator):
    """Validates categorical distribution using Chi-square test.

    Compares observed frequencies to expected frequencies.

    Example:
        validator = ChiSquareValidator(
            column="category",
            expected_frequencies={"A": 0.5, "B": 0.3, "C": 0.2},
            significance_level=0.05,
        )
    """

    name = "chi_square"
    category = "distribution"

    def __init__(
        self,
        column: str,
        expected_frequencies: dict[Any, float] | None = None,
        reference_data: pl.DataFrame | pl.LazyFrame | None = None,
        significance_level: float = 0.05,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.column = column
        self.significance_level = significance_level

        if expected_frequencies:
            self.expected_freq = expected_frequencies
        elif reference_data is not None:
            self.expected_freq = self._build_frequencies(reference_data)
        else:
            # Assume uniform distribution
            self.expected_freq = None

    def _build_frequencies(self, data: pl.DataFrame | pl.LazyFrame) -> dict[Any, float]:
        """Build expected frequencies from reference data."""
        if isinstance(data, pl.LazyFrame):
            df = data.collect()
        else:
            df = data

        counts = df.group_by(self.column).len()
        total = counts["len"].sum()

        freq = {}
        for row in counts.iter_rows():
            freq[row[0]] = row[1] / total

        return freq

    def _chi_square_statistic(
        self,
        observed: dict[Any, int],
        expected_freq: dict[Any, float],
        total: int,
    ) -> tuple[float, int]:
        """Calculate chi-square statistic and degrees of freedom."""
        chi_sq = 0.0
        df = 0

        all_keys = set(observed.keys()) | set(expected_freq.keys())

        for key in all_keys:
            obs = observed.get(key, 0)
            exp_ratio = expected_freq.get(key, 0)
            expected = exp_ratio * total

            if expected > 0:
                chi_sq += ((obs - expected) ** 2) / expected
                df += 1

        return chi_sq, max(df - 1, 1)

    def _chi_square_critical_value(self, df: int, alpha: float) -> float:
        """Approximate chi-square critical value."""
        # Simple approximation for common significance levels
        # For production, use scipy.stats.chi2.ppf
        critical_values = {
            0.05: {1: 3.84, 2: 5.99, 3: 7.81, 4: 9.49, 5: 11.07, 10: 18.31, 20: 31.41},
            0.01: {1: 6.63, 2: 9.21, 3: 11.34, 4: 13.28, 5: 15.09, 10: 23.21, 20: 37.57},
        }

        if alpha in critical_values:
            cv_table = critical_values[alpha]
            if df in cv_table:
                return cv_table[df]
            # Linear interpolation for other df values
            return df * 2.0 + 3.0  # Rough approximation

        return df * 2.0 + 3.0

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        df = lf.collect()
        counts = df.group_by(self.column).len()
        total = len(df)

        observed = {}
        for row in counts.iter_rows():
            observed[row[0]] = row[1]

        # Determine expected frequencies
        if self.expected_freq:
            expected_freq = self.expected_freq
        else:
            # Uniform distribution
            n_categories = len(observed)
            expected_freq = {k: 1 / n_categories for k in observed}

        chi_sq, degrees_freedom = self._chi_square_statistic(observed, expected_freq, total)
        critical_value = self._chi_square_critical_value(degrees_freedom, self.significance_level)

        if chi_sq > critical_value:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="chi_square_distribution_mismatch",
                    count=1,
                    severity=Severity.HIGH if chi_sq > critical_value * 2 else Severity.MEDIUM,
                    details=f"Chi-square {chi_sq:.2f} > critical {critical_value:.2f} (α={self.significance_level})",
                    expected=f"χ² <= {critical_value:.2f}",
                    actual=f"χ² = {chi_sq:.2f}",
                )
            )

        return issues


@register_validator
class MostCommonValueValidator(Validator):
    """Validates that the most common value(s) are within an expected set.

    Example:
        validator = MostCommonValueValidator(
            column="country",
            expected_values=["US", "UK", "CA"],
            top_n=3,
        )
    """

    name = "most_common_value"
    category = "distribution"

    def __init__(
        self,
        column: str,
        expected_values: set[Any] | list[Any],
        top_n: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.column = column
        self.expected_values = set(expected_values)
        self.top_n = top_n

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        df = lf.collect()
        counts = (
            df.group_by(self.column)
            .len()
            .sort("len", descending=True)
            .head(self.top_n)
        )

        top_values = counts[self.column].to_list()
        unexpected = [v for v in top_values if v not in self.expected_values]

        if unexpected:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="unexpected_most_common_value",
                    count=len(unexpected),
                    severity=Severity.MEDIUM,
                    details=f"Top {self.top_n} values include unexpected: {unexpected}",
                    expected=list(self.expected_values),
                    sample_values=unexpected,
                )
            )

        return issues
