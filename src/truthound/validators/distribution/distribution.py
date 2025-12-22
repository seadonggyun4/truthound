"""Distribution comparison validators."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import (
    ValidationIssue,
    Validator,
    NumericValidatorMixin,
)
from truthound.validators.registry import register_validator


@register_validator
class DistributionValidator(Validator, NumericValidatorMixin):
    """Validates distribution similarity using Jensen-Shannon divergence."""

    name = "distribution"
    category = "distribution"

    def __init__(
        self,
        reference_data: pl.DataFrame | pl.LazyFrame,
        threshold: float = 0.1,
        n_bins: int = 10,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if isinstance(reference_data, pl.LazyFrame):
            self.reference_data = reference_data.collect()
        else:
            self.reference_data = reference_data
        self.threshold = threshold
        self.n_bins = n_bins

    def _compute_histogram(
        self,
        data: pl.Series,
        min_val: float,
        max_val: float,
    ) -> list[float]:
        """Compute normalized histogram."""
        if len(data) == 0:
            return [1.0 / self.n_bins] * self.n_bins

        bin_width = (max_val - min_val) / self.n_bins
        if bin_width == 0:
            return [1.0 / self.n_bins] * self.n_bins

        counts = []
        for i in range(self.n_bins):
            lower = min_val + i * bin_width
            upper = min_val + (i + 1) * bin_width
            if i == self.n_bins - 1:
                count = ((data >= lower) & (data <= upper)).sum()
            else:
                count = ((data >= lower) & (data < upper)).sum()
            counts.append(count)

        total = sum(counts)
        if total == 0:
            return [1.0 / self.n_bins] * self.n_bins

        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        return [(c / total) + epsilon for c in counts]

    def _js_divergence(self, p: list[float], q: list[float]) -> float:
        """Compute Jensen-Shannon divergence."""
        import math

        m = [(pi + qi) / 2 for pi, qi in zip(p, q)]

        def kl_div(a: list[float], b: list[float]) -> float:
            return sum(ai * math.log(ai / bi) for ai, bi in zip(a, b) if ai > 0)

        return (kl_div(p, m) + kl_div(q, m)) / 2

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_numeric_columns(lf)

        if not columns:
            return issues

        df = lf.collect()

        for col in columns:
            if col not in self.reference_data.columns:
                continue

            current = df.get_column(col).drop_nulls()
            reference = self.reference_data.get_column(col).drop_nulls()

            if len(current) == 0 or len(reference) == 0:
                continue

            # Compute common range
            all_values = pl.concat([current.to_frame(), reference.to_frame()])
            min_val = all_values.min().item()
            max_val = all_values.max().item()

            if min_val == max_val:
                continue

            # Compute histograms
            p = self._compute_histogram(current, min_val, max_val)
            q = self._compute_histogram(reference, min_val, max_val)

            # Compute JS divergence
            js = self._js_divergence(p, q)

            if js > self.threshold:
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="distribution_drift",
                        count=1,
                        severity=Severity.HIGH if js > 0.2 else Severity.MEDIUM,
                        details=f"JS divergence {js:.4f} > threshold {self.threshold}",
                        expected=f"<= {self.threshold}",
                        actual=js,
                    )
                )

        return issues
