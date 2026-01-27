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
        # Check if data is numeric
        numeric_types = {
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
        }

        if baseline.dtype not in numeric_types:
            raise TypeError(
                f"KS test requires numeric columns, but '{baseline.name}' is {baseline.dtype}\n\n"
                f"For non-numeric columns, use one of these methods:\n"
                f"  • chi2  - Chi-square test for categorical data\n"
                f"  • js    - Jensen-Shannon divergence (works with any type)\n"
                f"  • auto  - Automatically selects appropriate method\n\n"
                f"Example: truthound compare baseline.csv current.csv --method chi2\n"
                f"     or: truthound compare baseline.csv current.csv --columns numeric_col1,numeric_col2"
            )

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
        # Check if data is numeric
        numeric_types = {
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
        }

        if baseline.dtype not in numeric_types:
            raise TypeError(
                f"PSI requires numeric columns, but '{baseline.name}' is {baseline.dtype}\n\n"
                f"For non-numeric columns, use one of these methods:\n"
                f"  • chi2  - Chi-square test for categorical data\n"
                f"  • js    - Jensen-Shannon divergence (works with any type)\n"
                f"  • auto  - Automatically selects appropriate method\n\n"
                f"Example: truthound compare baseline.csv current.csv --method chi2\n"
                f"     or: truthound compare baseline.csv current.csv --columns numeric_col1,numeric_col2"
            )

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


class KLDivergenceDetector(DriftDetector):
    """Kullback-Leibler divergence detector.

    Best for: Measuring information loss when approximating one distribution with another.
    Note: KL divergence is asymmetric (KL(P||Q) != KL(Q||P)).
    Range: [0, +∞), where 0 means identical distributions.
    """

    name = "kl_divergence"

    def __init__(self, n_bins: int = 10, threshold: float = 0.1):
        """Initialize KL divergence detector.

        Args:
            n_bins: Number of bins for histogram (numeric data).
            threshold: KL divergence threshold for drift detection.
        """
        self.n_bins = n_bins
        self.threshold = threshold

    def detect(self, baseline: pl.Series, current: pl.Series) -> DriftResult:
        """Calculate Kullback-Leibler divergence."""
        # Check if data is numeric
        numeric_types = {
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
        }

        if baseline.dtype not in numeric_types:
            raise TypeError(
                f"KL divergence requires numeric columns, but '{baseline.name}' is {baseline.dtype}\n\n"
                f"For non-numeric columns, use one of these methods:\n"
                f"  • chi2  - Chi-square test for categorical data\n"
                f"  • js    - Jensen-Shannon divergence (works with any type)\n"
                f"  • auto  - Automatically selects appropriate method"
            )

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

        # Create bins from combined data
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

        # Calculate bin proportions with smoothing
        def get_proportions(data: list, edges: list) -> list:
            counts = [0] * self.n_bins
            for val in data:
                for i in range(self.n_bins):
                    if edges[i] <= val < edges[i + 1]:
                        counts[i] += 1
                        break
            total = sum(counts)
            # Add small epsilon to avoid division by zero
            eps = 1e-10
            return [(c + eps) / (total + eps * self.n_bins) for c in counts]

        p = get_proportions(b, bin_edges)  # baseline distribution
        q = get_proportions(c, bin_edges)  # current distribution

        # Calculate KL divergence: KL(P||Q) = sum(p * log(p/q))
        kl_div = sum(pi * log(pi / qi) for pi, qi in zip(p, q) if pi > 0 and qi > 0)

        drifted = kl_div >= self.threshold
        level = self._get_drift_level(kl_div, (0.05, 0.1, 0.2))

        return DriftResult(
            statistic=kl_div,
            p_value=None,
            threshold=self.threshold,
            drifted=drifted,
            level=level if drifted else DriftLevel.NONE,
            method=self.name,
        )


class WassersteinDetector(DriftDetector):
    """Wasserstein (Earth Mover's) distance detector.

    Best for: Measuring the "work" required to transform one distribution into another.
    Advantage: Intuitive interpretation as physical distance.
    Range: [0, +∞), where 0 means identical distributions.
    """

    name = "wasserstein"

    def __init__(self, threshold: float = 0.1):
        """Initialize Wasserstein distance detector.

        Args:
            threshold: Normalized Wasserstein threshold for drift detection.
                      The statistic is normalized by baseline std for comparability.
        """
        self.threshold = threshold

    def detect(self, baseline: pl.Series, current: pl.Series) -> DriftResult:
        """Calculate Wasserstein (Earth Mover's) distance."""
        # Check if data is numeric
        numeric_types = {
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
        }

        if baseline.dtype not in numeric_types:
            raise TypeError(
                f"Wasserstein distance requires numeric columns, but '{baseline.name}' is {baseline.dtype}\n\n"
                f"For non-numeric columns, use one of these methods:\n"
                f"  • chi2  - Chi-square test for categorical data\n"
                f"  • js    - Jensen-Shannon divergence (works with any type)\n"
                f"  • auto  - Automatically selects appropriate method"
            )

        b = baseline.drop_nulls().cast(pl.Float64).sort().to_list()
        c = current.drop_nulls().cast(pl.Float64).sort().to_list()

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

        # Calculate Wasserstein distance (1-Wasserstein / Earth Mover's Distance)
        # For 1D data, this is the integral of |F(x) - G(x)| dx
        # where F and G are the cumulative distribution functions

        # Combine and sort all unique values
        all_values = sorted(set(b + c))

        # Calculate CDFs at each point
        n_b, n_c = len(b), len(c)
        wasserstein = 0.0

        prev_val = all_values[0]
        cdf_b = 0.0
        cdf_c = 0.0

        b_idx, c_idx = 0, 0

        for val in all_values:
            # Update CDFs
            while b_idx < n_b and b[b_idx] <= val:
                b_idx += 1
            while c_idx < n_c and c[c_idx] <= val:
                c_idx += 1

            new_cdf_b = b_idx / n_b
            new_cdf_c = c_idx / n_c

            # Add area between CDFs
            if val > prev_val:
                wasserstein += abs(cdf_b - cdf_c) * (val - prev_val)

            cdf_b = new_cdf_b
            cdf_c = new_cdf_c
            prev_val = val

        # Normalize by baseline std for comparability
        b_std = (sum((x - sum(b) / n_b) ** 2 for x in b) / n_b) ** 0.5 if n_b > 0 else 1.0
        normalized_stat = wasserstein / b_std if b_std > 0 else wasserstein

        drifted = normalized_stat >= self.threshold
        level = self._get_drift_level(normalized_stat, (0.05, 0.1, 0.2))

        return DriftResult(
            statistic=normalized_stat,
            p_value=None,
            threshold=self.threshold,
            drifted=drifted,
            level=level if drifted else DriftLevel.NONE,
            method=self.name,
            details=f"raw_distance={wasserstein:.4f}",
        )


class CramervonMisesDetector(DriftDetector):
    """Cramér-von Mises test for distribution comparison.

    Best for: Detecting differences in entire distribution shape.
    Advantage: More sensitive than KS test to differences in distribution tails.
    """

    name = "cramer_von_mises"

    def __init__(self, threshold: float = 0.05):
        """Initialize Cramér-von Mises detector.

        Args:
            threshold: P-value threshold for significance (default 0.05).
        """
        self.threshold = threshold

    def detect(self, baseline: pl.Series, current: pl.Series) -> DriftResult:
        """Perform two-sample Cramér-von Mises test."""
        # Check if data is numeric
        numeric_types = {
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
        }

        if baseline.dtype not in numeric_types:
            raise TypeError(
                f"Cramér-von Mises test requires numeric columns, but '{baseline.name}' is {baseline.dtype}\n\n"
                f"For non-numeric columns, use one of these methods:\n"
                f"  • chi2  - Chi-square test for categorical data\n"
                f"  • js    - Jensen-Shannon divergence (works with any type)\n"
                f"  • auto  - Automatically selects appropriate method"
            )

        b = baseline.drop_nulls().cast(pl.Float64).to_list()
        c = current.drop_nulls().cast(pl.Float64).to_list()

        if len(b) < 2 or len(c) < 2:
            return DriftResult(
                statistic=0.0,
                p_value=1.0,
                threshold=self.threshold,
                drifted=False,
                level=DriftLevel.NONE,
                method=self.name,
                details="Insufficient data (need at least 2 samples each)",
            )

        n1, n2 = len(b), len(c)
        n = n1 + n2

        # Pool and rank all observations
        pooled = [(val, 0) for val in b] + [(val, 1) for val in c]
        pooled.sort(key=lambda x: x[0])

        # Assign ranks (handle ties by averaging)
        ranks: list[float] = []
        i = 0
        while i < n:
            j = i
            while j < n and pooled[j][0] == pooled[i][0]:
                j += 1
            avg_rank = (i + j + 1) / 2  # Average rank for ties
            for k in range(i, j):
                ranks.append(avg_rank)
            i = j

        # Calculate U statistic (sum of ranks for first sample)
        u = sum(ranks[i] for i in range(n) if pooled[i][1] == 0)

        # Calculate Cramér-von Mises statistic
        # T = (U - n1*(n+1)/2)^2 / (n1*n2*(n+1)/3)
        expected_u = n1 * (n + 1) / 2
        var_u = n1 * n2 * (n + 1) / 12

        # Two-sample CvM statistic approximation
        t_stat = (u - expected_u) ** 2 / (var_u * 12) if var_u > 0 else 0

        # Approximate p-value using asymptotic distribution
        # CvM has a complex distribution; use approximation
        p_value = self._cvm_p_value(t_stat, n1, n2)

        drifted = p_value < self.threshold
        level = self._get_drift_level(t_stat, (0.1, 0.3, 0.5))

        return DriftResult(
            statistic=t_stat,
            p_value=p_value,
            threshold=self.threshold,
            drifted=drifted,
            level=level if drifted else DriftLevel.NONE,
            method=self.name,
        )

    @staticmethod
    def _cvm_p_value(t: float, n1: int, n2: int) -> float:
        """Approximate Cramér-von Mises p-value.

        Uses asymptotic approximation based on chi-square distribution.
        """
        if t <= 0:
            return 1.0

        # Approximate using exponential decay
        # This is a simplified approximation
        p = 2.718281828 ** (-t * 2)
        return max(0.0, min(1.0, p))


class AndersonDarlingDetector(DriftDetector):
    """Anderson-Darling test for distribution comparison.

    Best for: Detecting differences with emphasis on distribution tails.
    Advantage: More sensitive than KS test to tail differences.
    """

    name = "anderson_darling"

    def __init__(self, threshold: float = 0.05):
        """Initialize Anderson-Darling detector.

        Args:
            threshold: P-value threshold for significance (default 0.05).
        """
        self.threshold = threshold

    def detect(self, baseline: pl.Series, current: pl.Series) -> DriftResult:
        """Perform two-sample Anderson-Darling test."""
        # Check if data is numeric
        numeric_types = {
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
        }

        if baseline.dtype not in numeric_types:
            raise TypeError(
                f"Anderson-Darling test requires numeric columns, but '{baseline.name}' is {baseline.dtype}\n\n"
                f"For non-numeric columns, use one of these methods:\n"
                f"  • chi2  - Chi-square test for categorical data\n"
                f"  • js    - Jensen-Shannon divergence (works with any type)\n"
                f"  • auto  - Automatically selects appropriate method"
            )

        b = baseline.drop_nulls().cast(pl.Float64).to_list()
        c = current.drop_nulls().cast(pl.Float64).to_list()

        if len(b) < 2 or len(c) < 2:
            return DriftResult(
                statistic=0.0,
                p_value=1.0,
                threshold=self.threshold,
                drifted=False,
                level=DriftLevel.NONE,
                method=self.name,
                details="Insufficient data (need at least 2 samples each)",
            )

        n1, n2 = len(b), len(c)
        n = n1 + n2

        # Pool and sort all observations
        pooled = sorted(b + c)

        # Calculate empirical CDFs at each pooled value
        def ecdf(data: list, val: float) -> float:
            return sum(1 for x in data if x <= val) / len(data)

        # Calculate A-D statistic
        # A² = -n - (1/n) * sum((2i-1) * [ln(F(zi)) + ln(1-F(z(n+1-i)))])
        # For two-sample case, use combined sample

        ad_stat = 0.0
        for i, val in enumerate(pooled, 1):
            f_b = ecdf(b, val)
            f_c = ecdf(c, val)
            f_combined = (i - 0.5) / n  # Smoothed combined ECDF

            # Weight for Anderson-Darling (emphasizes tails)
            weight = 1.0 / (f_combined * (1 - f_combined)) if 0 < f_combined < 1 else 1.0

            # Squared difference weighted by tail sensitivity
            diff = (f_b - f_c) ** 2
            ad_stat += diff * weight / n

        # Normalize
        ad_stat = ad_stat * n1 * n2 / n

        # Approximate p-value
        p_value = self._ad_p_value(ad_stat, n1, n2)

        drifted = p_value < self.threshold
        level = self._get_drift_level(ad_stat, (1.0, 2.0, 3.5))

        return DriftResult(
            statistic=ad_stat,
            p_value=p_value,
            threshold=self.threshold,
            drifted=drifted,
            level=level if drifted else DriftLevel.NONE,
            method=self.name,
        )

    @staticmethod
    def _ad_p_value(a2: float, n1: int, n2: int) -> float:
        """Approximate Anderson-Darling p-value for two samples.

        Uses critical value approximations for k-sample A-D test.
        """
        if a2 <= 0:
            return 1.0

        # Approximate p-value using asymptotic distribution
        # Based on Scholz & Stephens (1987) approximation
        # Critical values at common significance levels:
        # α=0.25: 1.248, α=0.10: 1.933, α=0.05: 2.492, α=0.025: 3.070, α=0.01: 3.857

        if a2 < 1.248:
            return 0.25
        elif a2 < 1.933:
            return 0.10 + (0.25 - 0.10) * (1.933 - a2) / (1.933 - 1.248)
        elif a2 < 2.492:
            return 0.05 + (0.10 - 0.05) * (2.492 - a2) / (2.492 - 1.933)
        elif a2 < 3.070:
            return 0.025 + (0.05 - 0.025) * (3.070 - a2) / (3.070 - 2.492)
        elif a2 < 3.857:
            return 0.01 + (0.025 - 0.01) * (3.857 - a2) / (3.857 - 3.070)
        else:
            # Very small p-value
            return 0.001


class HellingerDetector(DriftDetector):
    """Hellinger distance detector for probability distributions.

    Best for: Comparing probability distributions with well-defined support.
    Advantage: Bounded [0, 1], symmetric, satisfies triangle inequality.
    Range: [0, 1], where 0 means identical distributions and 1 means no overlap.

    Mathematical definition:
        H(P, Q) = (1/√2) * √(Σ(√p_i - √q_i)²)

    The Hellinger distance is related to the Bhattacharyya coefficient:
        H(P, Q) = √(1 - BC(P, Q))
    """

    name = "hellinger"

    def __init__(self, n_bins: int = 10, threshold: float = 0.1):
        """Initialize Hellinger distance detector.

        Args:
            n_bins: Number of bins for numeric data histogram.
            threshold: Hellinger distance threshold for drift detection.
                      Values: 0.0 = identical, 1.0 = completely different.
        """
        self.n_bins = n_bins
        self.threshold = threshold

    def detect(self, baseline: pl.Series, current: pl.Series) -> DriftResult:
        """Calculate Hellinger distance between two distributions."""
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

        # Calculate Hellinger distance: H(P,Q) = (1/√2) * √(Σ(√p_i - √q_i)²)
        sum_sq = sum((sqrt(pi) - sqrt(qi)) ** 2 for pi, qi in zip(p, q))
        hellinger = sqrt(sum_sq) / sqrt(2)

        # Ensure bounded [0, 1]
        hellinger = max(0.0, min(1.0, hellinger))

        drifted = hellinger >= self.threshold
        level = self._get_drift_level(hellinger, (0.05, 0.1, 0.2))

        return DriftResult(
            statistic=hellinger,
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


class BhattacharyyaDetector(DriftDetector):
    """Bhattacharyya distance detector for probability distributions.

    Best for: Measuring overlap between probability distributions.
    Advantage: Related to classification error bounds.
    Range: [0, +∞), where 0 means identical distributions.

    Mathematical definition:
        D_B(P, Q) = -ln(BC(P, Q))
        where BC(P, Q) = Σ√(p_i * q_i) is the Bhattacharyya coefficient.

    The Bhattacharyya coefficient (BC) is bounded [0, 1]:
        BC = 1 means identical distributions
        BC = 0 means no overlap
    """

    name = "bhattacharyya"

    def __init__(self, n_bins: int = 10, threshold: float = 0.1):
        """Initialize Bhattacharyya distance detector.

        Args:
            n_bins: Number of bins for numeric data histogram.
            threshold: Bhattacharyya distance threshold for drift detection.
        """
        self.n_bins = n_bins
        self.threshold = threshold

    def detect(self, baseline: pl.Series, current: pl.Series) -> DriftResult:
        """Calculate Bhattacharyya distance between two distributions."""
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

        # Calculate Bhattacharyya coefficient: BC = Σ√(p_i * q_i)
        bc = sum(sqrt(pi * qi) for pi, qi in zip(p, q))
        bc = max(0.0001, min(1.0, bc))  # Clamp to avoid log(0)

        # Bhattacharyya distance: D_B = -ln(BC)
        bhattacharyya = -log(bc)

        drifted = bhattacharyya >= self.threshold
        level = self._get_drift_level(bhattacharyya, (0.05, 0.1, 0.2))

        return DriftResult(
            statistic=bhattacharyya,
            p_value=None,
            threshold=self.threshold,
            drifted=drifted,
            level=level if drifted else DriftLevel.NONE,
            method=self.name,
            details=f"bc_coeff={bc:.4f}",
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


class TotalVariationDetector(DriftDetector):
    """Total Variation distance detector.

    Best for: Measuring maximum probability difference between distributions.
    Advantage: Simple interpretation - "largest difference in probability".
    Range: [0, 1], where 0 means identical, 1 means completely different.

    Mathematical definition:
        TV(P, Q) = (1/2) * Σ|p_i - q_i| = max_A |P(A) - Q(A)|

    Properties:
        - Symmetric: TV(P, Q) = TV(Q, P)
        - Bounded: 0 ≤ TV(P, Q) ≤ 1
        - Triangle inequality holds
        - Related to Hellinger: H²(P,Q) ≤ TV(P,Q) ≤ √2 * H(P,Q)
    """

    name = "total_variation"

    def __init__(self, n_bins: int = 10, threshold: float = 0.1):
        """Initialize Total Variation distance detector.

        Args:
            n_bins: Number of bins for numeric data histogram.
            threshold: TV distance threshold for drift detection.
                      Values: 0.0 = identical, 1.0 = completely different.
        """
        self.n_bins = n_bins
        self.threshold = threshold

    def detect(self, baseline: pl.Series, current: pl.Series) -> DriftResult:
        """Calculate Total Variation distance between two distributions."""
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

        # Calculate Total Variation distance: TV(P,Q) = (1/2) * Σ|p_i - q_i|
        tv = 0.5 * sum(abs(pi - qi) for pi, qi in zip(p, q))

        # Ensure bounded [0, 1]
        tv = max(0.0, min(1.0, tv))

        drifted = tv >= self.threshold
        level = self._get_drift_level(tv, (0.05, 0.1, 0.2))

        return DriftResult(
            statistic=tv,
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


class EnergyDetector(DriftDetector):
    """Energy distance detector for distribution comparison.

    Best for: Detecting differences in location and scale of distributions.
    Advantage: Metric property (triangle inequality), characterizes distributions.
    Range: [0, +∞), where 0 means identical distributions.

    Mathematical definition:
        E(P, Q) = 2*E[|X-Y|] - E[|X-X'|] - E[|Y-Y'|]
        where X, X' ~ P and Y, Y' ~ Q are independent.

    Properties:
        - E(P, Q) = 0 if and only if P = Q
        - Symmetric: E(P, Q) = E(Q, P)
        - Triangle inequality holds
        - Consistent statistical test
    """

    name = "energy"

    def __init__(self, threshold: float = 0.1, max_samples: int = 1000):
        """Initialize Energy distance detector.

        Args:
            threshold: Normalized energy distance threshold for drift detection.
            max_samples: Maximum number of samples to use (for computational efficiency).
        """
        self.threshold = threshold
        self.max_samples = max_samples

    def detect(self, baseline: pl.Series, current: pl.Series) -> DriftResult:
        """Calculate Energy distance between two distributions."""
        # Check if data is numeric
        numeric_types = {
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
        }

        if baseline.dtype not in numeric_types:
            raise TypeError(
                f"Energy distance requires numeric columns, but '{baseline.name}' is {baseline.dtype}\n\n"
                f"For non-numeric columns, use one of these methods:\n"
                f"  • chi2  - Chi-square test for categorical data\n"
                f"  • js    - Jensen-Shannon divergence (works with any type)\n"
                f"  • hellinger - Hellinger distance (works with any type)\n"
                f"  • auto  - Automatically selects appropriate method"
            )

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

        # Subsample for computational efficiency
        if len(b) > self.max_samples:
            step = len(b) // self.max_samples
            b = b[::step][:self.max_samples]
        if len(c) > self.max_samples:
            step = len(c) // self.max_samples
            c = c[::step][:self.max_samples]

        n, m = len(b), len(c)

        # Calculate Energy distance using unbiased estimator
        # E_nm = (2/nm) * Σ|X_i - Y_j| - (1/n²) * Σ|X_i - X_j| - (1/m²) * Σ|Y_i - Y_j|

        # Cross-sample term: E[|X-Y|]
        cross_sum = sum(abs(b[i] - c[j]) for i in range(n) for j in range(m))
        cross_term = cross_sum / (n * m) if n * m > 0 else 0

        # Within-baseline term: E[|X-X'|]
        within_b = sum(abs(b[i] - b[j]) for i in range(n) for j in range(i + 1, n))
        within_b_term = 2 * within_b / (n * (n - 1)) if n > 1 else 0

        # Within-current term: E[|Y-Y'|]
        within_c = sum(abs(c[i] - c[j]) for i in range(m) for j in range(i + 1, m))
        within_c_term = 2 * within_c / (m * (m - 1)) if m > 1 else 0

        # Energy distance
        energy = 2 * cross_term - within_b_term - within_c_term
        energy = max(0.0, energy)  # Ensure non-negative

        # Normalize by pooled standard deviation
        all_data = b + c
        mean_all = sum(all_data) / len(all_data)
        std_all = sqrt(sum((x - mean_all) ** 2 for x in all_data) / len(all_data))
        normalized = energy / std_all if std_all > 0 else energy

        drifted = normalized >= self.threshold
        level = self._get_drift_level(normalized, (0.05, 0.1, 0.2))

        return DriftResult(
            statistic=normalized,
            p_value=None,
            threshold=self.threshold,
            drifted=drifted,
            level=level if drifted else DriftLevel.NONE,
            method=self.name,
            details=f"raw_energy={energy:.4f}",
        )


class MMDDetector(DriftDetector):
    """Maximum Mean Discrepancy (MMD) detector.

    Best for: High-dimensional data and kernel-based distribution comparison.
    Advantage: Non-parametric, works without density estimation.
    Range: [0, +∞), where 0 means identical distributions (in RKHS).

    Mathematical definition:
        MMD²(P, Q) = E[k(X,X')] + E[k(Y,Y')] - 2*E[k(X,Y)]
        where k is a kernel function (default: Gaussian RBF).

    Properties:
        - MMD(P, Q) = 0 iff P = Q (for characteristic kernels)
        - Non-parametric: no density estimation required
        - Computationally efficient: O(n²) for exact, O(n) for approximations
        - Works in high dimensions where density estimation fails
    """

    name = "mmd"

    def __init__(
        self,
        threshold: float = 0.1,
        kernel: str = "rbf",
        bandwidth: float | None = None,
        max_samples: int = 1000,
    ):
        """Initialize MMD detector.

        Args:
            threshold: MMD threshold for drift detection.
            kernel: Kernel type - "rbf" (Gaussian), "linear", or "polynomial".
            bandwidth: Kernel bandwidth (gamma) for RBF. If None, uses median heuristic.
            max_samples: Maximum number of samples to use (for computational efficiency).
        """
        self.threshold = threshold
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.max_samples = max_samples

    def detect(self, baseline: pl.Series, current: pl.Series) -> DriftResult:
        """Calculate Maximum Mean Discrepancy between two distributions."""
        # Check if data is numeric
        numeric_types = {
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
        }

        if baseline.dtype not in numeric_types:
            raise TypeError(
                f"MMD requires numeric columns, but '{baseline.name}' is {baseline.dtype}\n\n"
                f"For non-numeric columns, use one of these methods:\n"
                f"  • chi2  - Chi-square test for categorical data\n"
                f"  • js    - Jensen-Shannon divergence (works with any type)\n"
                f"  • hellinger - Hellinger distance (works with any type)\n"
                f"  • auto  - Automatically selects appropriate method"
            )

        b = baseline.drop_nulls().cast(pl.Float64).to_list()
        c = current.drop_nulls().cast(pl.Float64).to_list()

        if len(b) < 2 or len(c) < 2:
            return DriftResult(
                statistic=0.0,
                p_value=None,
                threshold=self.threshold,
                drifted=False,
                level=DriftLevel.NONE,
                method=self.name,
                details="Insufficient data (need at least 2 samples each)",
            )

        # Subsample for computational efficiency
        if len(b) > self.max_samples:
            step = len(b) // self.max_samples
            b = b[::step][:self.max_samples]
        if len(c) > self.max_samples:
            step = len(c) // self.max_samples
            c = c[::step][:self.max_samples]

        n, m = len(b), len(c)

        # Determine bandwidth using median heuristic if not specified
        if self.bandwidth is None:
            all_data = b + c
            # Median of pairwise distances (subsample for efficiency)
            dists = []
            step = max(1, len(all_data) // 100)
            for i in range(0, len(all_data), step):
                for j in range(i + 1, len(all_data), step):
                    dists.append(abs(all_data[i] - all_data[j]))
            if dists:
                dists.sort()
                median_dist = dists[len(dists) // 2]
                gamma = 1.0 / (2 * median_dist ** 2) if median_dist > 0 else 1.0
            else:
                gamma = 1.0
        else:
            gamma = self.bandwidth

        # Kernel function
        def kernel_fn(x: float, y: float) -> float:
            if self.kernel == "rbf":
                return 2.718281828 ** (-gamma * (x - y) ** 2)
            elif self.kernel == "linear":
                return x * y
            elif self.kernel == "polynomial":
                return (1 + x * y) ** 2
            else:
                return 2.718281828 ** (-gamma * (x - y) ** 2)

        # Calculate MMD² using unbiased estimator
        # MMD²_u = (1/n(n-1)) Σ k(x_i, x_j) + (1/m(m-1)) Σ k(y_i, y_j)
        #        - (2/nm) Σ k(x_i, y_j)

        # Within-baseline term
        k_xx = sum(kernel_fn(b[i], b[j]) for i in range(n) for j in range(i + 1, n))
        k_xx_mean = 2 * k_xx / (n * (n - 1)) if n > 1 else 0

        # Within-current term
        k_yy = sum(kernel_fn(c[i], c[j]) for i in range(m) for j in range(i + 1, m))
        k_yy_mean = 2 * k_yy / (m * (m - 1)) if m > 1 else 0

        # Cross term
        k_xy = sum(kernel_fn(b[i], c[j]) for i in range(n) for j in range(m))
        k_xy_mean = k_xy / (n * m) if n * m > 0 else 0

        # MMD² (can be slightly negative due to unbiased estimator)
        mmd_sq = k_xx_mean + k_yy_mean - 2 * k_xy_mean
        mmd = sqrt(max(0.0, mmd_sq))

        drifted = mmd >= self.threshold
        level = self._get_drift_level(mmd, (0.05, 0.1, 0.2))

        return DriftResult(
            statistic=mmd,
            p_value=None,
            threshold=self.threshold,
            drifted=drifted,
            level=level if drifted else DriftLevel.NONE,
            method=self.name,
            details=f"kernel={self.kernel}, gamma={gamma:.4f}",
        )
