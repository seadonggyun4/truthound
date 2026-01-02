"""Statistical drift detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from math import log, sqrt
from typing import Any

import polars as pl


class DriftLevel(str, Enum):
    """Drift severity levels."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class DriftResult:
    """Result of a drift detection test."""

    statistic: float
    p_value: float | None
    threshold: float
    drifted: bool
    level: DriftLevel
    method: str
    details: str | None = None

    def to_dict(self) -> dict:
        return {
            "statistic": round(self.statistic, 6),
            "p_value": round(self.p_value, 6) if self.p_value is not None else None,
            "threshold": self.threshold,
            "drifted": self.drifted,
            "level": self.level.value,
            "method": self.method,
            "details": self.details,
        }


class DriftDetector(ABC):
    """Abstract base class for drift detectors."""

    name: str = "base"

    @abstractmethod
    def detect(
        self,
        baseline: pl.Series,
        current: pl.Series,
    ) -> DriftResult:
        """Detect drift between baseline and current data.

        Args:
            baseline: Reference data series.
            current: Current data series to compare.

        Returns:
            DriftResult with test statistics and drift determination.
        """
        pass

    @staticmethod
    def _get_drift_level(statistic: float, thresholds: tuple[float, float, float]) -> DriftLevel:
        """Determine drift level based on thresholds."""
        low, medium, high = thresholds
        if statistic >= high:
            return DriftLevel.HIGH
        elif statistic >= medium:
            return DriftLevel.MEDIUM
        elif statistic >= low:
            return DriftLevel.LOW
        return DriftLevel.NONE


class KSTestDetector(DriftDetector):
    """Kolmogorov-Smirnov test for numeric distributions.

    Best for: Continuous numeric data.
    Detects: Any difference in distribution shape.
    """

    name = "ks_test"

    def __init__(self, threshold: float = 0.05):
        """Initialize KS test detector.

        Args:
            threshold: P-value threshold for significance (default 0.05).
        """
        self.threshold = threshold

    def detect(self, baseline: pl.Series, current: pl.Series) -> DriftResult:
        """Perform two-sample Kolmogorov-Smirnov test."""
        # Clean data
        b = baseline.drop_nulls().cast(pl.Float64).sort().to_list()
        c = current.drop_nulls().cast(pl.Float64).sort().to_list()

        if len(b) == 0 or len(c) == 0:
            return DriftResult(
                statistic=0.0,
                p_value=1.0,
                threshold=self.threshold,
                drifted=False,
                level=DriftLevel.NONE,
                method=self.name,
                details="Insufficient data",
            )

        # Calculate KS statistic (maximum difference between ECDFs)
        n1, n2 = len(b), len(c)
        all_values = sorted(set(b + c))

        max_diff = 0.0
        for val in all_values:
            ecdf1 = sum(1 for x in b if x <= val) / n1
            ecdf2 = sum(1 for x in c if x <= val) / n2
            diff = abs(ecdf1 - ecdf2)
            max_diff = max(max_diff, diff)

        # Approximate p-value using asymptotic formula
        en = sqrt(n1 * n2 / (n1 + n2))
        p_value = self._ks_p_value(max_diff * en)

        drifted = p_value < self.threshold
        level = self._get_drift_level(max_diff, (0.1, 0.2, 0.3))

        return DriftResult(
            statistic=max_diff,
            p_value=p_value,
            threshold=self.threshold,
            drifted=drifted,
            level=level if drifted else DriftLevel.NONE,
            method=self.name,
        )

    @staticmethod
    def _ks_p_value(d: float) -> float:
        """Approximate KS p-value using asymptotic distribution."""
        if d <= 0:
            return 1.0
        # Kolmogorov distribution approximation
        lam = d * d
        p = 2 * sum(
            ((-1) ** (k - 1)) * (2.718281828 ** (-2 * k * k * lam))
            for k in range(1, 101)
        )
        return max(0.0, min(1.0, p))


class PSIDetector(DriftDetector):
    """Population Stability Index for distribution shift.

    Best for: Monitoring model features over time.
    Detects: Changes in population distribution.
    Industry standard thresholds: <0.1 no drift, 0.1-0.25 moderate, >0.25 significant.
    """

    name = "psi"

    def __init__(self, n_bins: int = 10, threshold: float = 0.1):
        """Initialize PSI detector.

        Args:
            n_bins: Number of bins for histogram.
            threshold: PSI threshold for drift detection.
        """
        self.n_bins = n_bins
        self.threshold = threshold

    def detect(self, baseline: pl.Series, current: pl.Series) -> DriftResult:
        """Calculate Population Stability Index."""
        b = baseline.drop_nulls().cast(pl.Float64).to_list()
        c = current.drop_nulls().cast(pl.Float64).to_list()

        if len(b) == 0 or len(c) == 0:
            return DriftResult(
                statistic=0.0,
                p_value=None,
                threshold=self.threshold,
                drifted=False,
                level=DriftLevel.NONE,
                method=self.name,
                details="Insufficient data",
            )

        # Create bins from baseline
        min_val = min(min(b), min(c))
        max_val = max(max(b), max(c))

        if min_val == max_val:
            return DriftResult(
                statistic=0.0,
                p_value=None,
                threshold=self.threshold,
                drifted=False,
                level=DriftLevel.NONE,
                method=self.name,
                details="No variation in data",
            )

        bin_edges = [min_val + i * (max_val - min_val) / self.n_bins for i in range(self.n_bins + 1)]
        bin_edges[-1] = max_val + 1  # Ensure max value is included

        # Calculate bin proportions
        def get_proportions(data: list, edges: list) -> list:
            counts = [0] * self.n_bins
            for val in data:
                for i in range(self.n_bins):
                    if edges[i] <= val < edges[i + 1]:
                        counts[i] += 1
                        break
            total = sum(counts)
            # Add small epsilon to avoid division by zero
            return [(c + 0.0001) / (total + 0.0001 * self.n_bins) for c in counts]

        baseline_props = get_proportions(b, bin_edges)
        current_props = get_proportions(c, bin_edges)

        # Calculate PSI
        psi = sum(
            (cp - bp) * log(cp / bp) if bp > 0 and cp > 0 else 0
            for bp, cp in zip(baseline_props, current_props)
        )

        drifted = psi >= self.threshold
        level = self._get_drift_level(psi, (0.1, 0.2, 0.25))

        return DriftResult(
            statistic=psi,
            p_value=None,
            threshold=self.threshold,
            drifted=drifted,
            level=level if drifted else DriftLevel.NONE,
            method=self.name,
        )


class ChiSquareDetector(DriftDetector):
    """Chi-square test for categorical distributions.

    Best for: Categorical data with finite categories.
    Detects: Changes in category frequencies.
    """

    name = "chi_square"

    def __init__(self, threshold: float = 0.05):
        """Initialize Chi-square detector.

        Args:
            threshold: P-value threshold for significance.
        """
        self.threshold = threshold

    def detect(self, baseline: pl.Series, current: pl.Series) -> DriftResult:
        """Perform Chi-square test on categorical distributions."""
        b = baseline.drop_nulls().to_list()
        c = current.drop_nulls().to_list()

        if len(b) == 0 or len(c) == 0:
            return DriftResult(
                statistic=0.0,
                p_value=1.0,
                threshold=self.threshold,
                drifted=False,
                level=DriftLevel.NONE,
                method=self.name,
                details="Insufficient data",
            )

        # Get all categories
        all_categories = set(b) | set(c)

        if len(all_categories) == 1:
            return DriftResult(
                statistic=0.0,
                p_value=1.0,
                threshold=self.threshold,
                drifted=False,
                level=DriftLevel.NONE,
                method=self.name,
                details="Only one category",
            )

        # Count frequencies
        b_counts = {cat: b.count(cat) for cat in all_categories}
        c_counts = {cat: c.count(cat) for cat in all_categories}

        n_b, n_c = len(b), len(c)
        n_total = n_b + n_c

        # Calculate chi-square statistic
        chi2 = 0.0
        for cat in all_categories:
            observed_b = b_counts.get(cat, 0)
            observed_c = c_counts.get(cat, 0)
            total_cat = observed_b + observed_c

            expected_b = total_cat * n_b / n_total
            expected_c = total_cat * n_c / n_total

            if expected_b > 0:
                chi2 += (observed_b - expected_b) ** 2 / expected_b
            if expected_c > 0:
                chi2 += (observed_c - expected_c) ** 2 / expected_c

        # Degrees of freedom
        df = len(all_categories) - 1

        # Approximate p-value using chi-square distribution
        p_value = self._chi2_p_value(chi2, df)

        drifted = p_value < self.threshold
        # Normalize statistic for level calculation
        normalized = chi2 / max(df, 1)
        level = self._get_drift_level(normalized, (3.84, 6.64, 10.83))  # Chi2 critical values

        return DriftResult(
            statistic=chi2,
            p_value=p_value,
            threshold=self.threshold,
            drifted=drifted,
            level=level if drifted else DriftLevel.NONE,
            method=self.name,
            details=f"df={df}",
        )

    @staticmethod
    def _chi2_p_value(x: float, df: int) -> float:
        """Approximate chi-square p-value."""
        if x <= 0 or df <= 0:
            return 1.0
        # Wilson-Hilferty approximation
        z = ((x / df) ** (1 / 3) - (1 - 2 / (9 * df))) / sqrt(2 / (9 * df))
        # Standard normal CDF approximation
        # Use z * z (always positive) to avoid complex numbers from negative sqrt argument
        p = 0.5 * (1 + (1 - 2.718281828 ** (-0.5 * z * z)) ** 0.5 * (1 if z >= 0 else -1))
        return max(0.0, min(1.0, 1 - p))


class JensenShannonDetector(DriftDetector):
    """Jensen-Shannon divergence for distribution comparison.

    Best for: Any distribution comparison.
    Advantage: Symmetric and bounded [0, 1].
    """

    name = "jensen_shannon"

    def __init__(self, n_bins: int = 10, threshold: float = 0.1):
        """Initialize JS divergence detector.

        Args:
            n_bins: Number of bins for numeric data.
            threshold: JS divergence threshold for drift.
        """
        self.n_bins = n_bins
        self.threshold = threshold

    def detect(self, baseline: pl.Series, current: pl.Series) -> DriftResult:
        """Calculate Jensen-Shannon divergence."""
        b = baseline.drop_nulls()
        c = current.drop_nulls()

        if len(b) == 0 or len(c) == 0:
            return DriftResult(
                statistic=0.0,
                p_value=None,
                threshold=self.threshold,
                drifted=False,
                level=DriftLevel.NONE,
                method=self.name,
                details="Insufficient data",
            )

        # Handle numeric vs categorical
        if b.dtype in (pl.String, pl.Utf8, pl.Categorical):
            p, q = self._categorical_distributions(b, c)
        else:
            p, q = self._numeric_distributions(b.cast(pl.Float64), c.cast(pl.Float64))

        if len(p) == 0:
            return DriftResult(
                statistic=0.0,
                p_value=None,
                threshold=self.threshold,
                drifted=False,
                level=DriftLevel.NONE,
                method=self.name,
            )

        # Calculate JS divergence
        m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
        js = 0.5 * self._kl_divergence(p, m) + 0.5 * self._kl_divergence(q, m)

        # JS divergence is bounded [0, log(2)] ≈ [0, 0.693]
        # Normalize to [0, 1]
        js_normalized = js / 0.693 if js > 0 else 0

        drifted = js_normalized >= self.threshold
        level = self._get_drift_level(js_normalized, (0.05, 0.1, 0.2))

        return DriftResult(
            statistic=js_normalized,
            p_value=None,
            threshold=self.threshold,
            drifted=drifted,
            level=level if drifted else DriftLevel.NONE,
            method=self.name,
        )

    def _numeric_distributions(
        self, b: pl.Series, c: pl.Series
    ) -> tuple[list[float], list[float]]:
        """Get binned distributions for numeric data."""
        b_list = b.to_list()
        c_list = c.to_list()

        min_val = min(min(b_list), min(c_list))
        max_val = max(max(b_list), max(c_list))

        if min_val == max_val:
            return [1.0], [1.0]

        bin_width = (max_val - min_val) / self.n_bins

        def bin_counts(data: list) -> list[float]:
            counts = [0] * self.n_bins
            for val in data:
                idx = min(int((val - min_val) / bin_width), self.n_bins - 1)
                counts[idx] += 1
            total = sum(counts) + 0.0001 * self.n_bins
            return [(c + 0.0001) / total for c in counts]

        return bin_counts(b_list), bin_counts(c_list)

    def _categorical_distributions(
        self, b: pl.Series, c: pl.Series
    ) -> tuple[list[float], list[float]]:
        """Get distributions for categorical data."""
        b_list = b.to_list()
        c_list = c.to_list()

        all_cats = sorted(set(b_list) | set(c_list))
        n_cats = len(all_cats)

        def cat_probs(data: list) -> list[float]:
            counts = {cat: data.count(cat) for cat in all_cats}
            total = len(data) + 0.0001 * n_cats
            return [(counts.get(cat, 0) + 0.0001) / total for cat in all_cats]

        return cat_probs(b_list), cat_probs(c_list)

    @staticmethod
    def _kl_divergence(p: list[float], q: list[float]) -> float:
        """Calculate KL divergence."""
        return sum(pi * log(pi / qi) for pi, qi in zip(p, q) if pi > 0 and qi > 0)
