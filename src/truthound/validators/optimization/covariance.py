"""Batch and incremental covariance computation.

This module provides memory-efficient covariance matrix estimation
for Mahalanobis distance and other multivariate methods.

Key Optimizations:
    - Incremental (streaming) covariance: O(n) memory instead of O(n×d)
    - Woodbury matrix identity for rank-k updates
    - Robust covariance estimation (MCD) with subsampling
    - Batch processing for large datasets

Usage:
    class OptimizedMahalanobisValidator(MahalanobisValidator, BatchCovarianceMixin):
        def _compute_mahalanobis(self, data):
            # Uses incremental covariance for large data
            cov, mean = self.compute_covariance_incremental(data)
            return self.compute_mahalanobis_distances(data, mean, cov)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CovarianceResult:
    """Result of covariance computation.

    Attributes:
        mean: Sample mean vector
        covariance: Covariance matrix
        precision: Inverse covariance (if computed)
        n_samples: Number of samples used
        is_robust: Whether robust estimation was used
    """

    mean: np.ndarray
    covariance: np.ndarray
    precision: np.ndarray | None = None
    n_samples: int = 0
    is_robust: bool = False

    def get_precision(self, regularization: float = 1e-6) -> np.ndarray:
        """Get precision matrix, computing if needed."""
        if self.precision is not None:
            return self.precision

        # Add regularization for numerical stability
        cov_reg = self.covariance + regularization * np.eye(len(self.covariance))
        self.precision = np.linalg.inv(cov_reg)
        return self.precision


class IncrementalCovariance:
    """Welford's online algorithm for covariance.

    Computes mean and covariance incrementally with O(d²) memory
    regardless of sample size.

    Based on Welford's algorithm extended to covariance:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

    Example:
        cov = IncrementalCovariance(n_features=10)
        for batch in data_stream:
            cov.update(batch)
        result = cov.get_result()
    """

    def __init__(self, n_features: int):
        """Initialize incremental covariance.

        Args:
            n_features: Number of features (dimensions)
        """
        self.n_features = n_features
        self.n_samples = 0
        self._mean = np.zeros(n_features)
        self._C = np.zeros((n_features, n_features))  # Co-moment matrix

    def update(self, x: np.ndarray) -> None:
        """Update with a single sample or batch.

        Args:
            x: Single sample (n_features,) or batch (n_samples, n_features)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        for sample in x:
            self.n_samples += 1
            delta = sample - self._mean
            self._mean += delta / self.n_samples
            delta2 = sample - self._mean
            self._C += np.outer(delta, delta2)

    def update_batch(self, batch: np.ndarray) -> None:
        """Efficient batch update using parallel algorithm.

        Chan et al. (1979) parallel algorithm for combining statistics.

        Args:
            batch: Data batch (n_samples, n_features)
        """
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)

        n_b = len(batch)
        if n_b == 0:
            return

        mean_b = batch.mean(axis=0)
        C_b = np.cov(batch, rowvar=False, ddof=0) * n_b

        if self.n_samples == 0:
            self._mean = mean_b
            self._C = C_b
            self.n_samples = n_b
        else:
            n_total = self.n_samples + n_b
            delta = mean_b - self._mean

            # Update mean
            self._mean = (self.n_samples * self._mean + n_b * mean_b) / n_total

            # Update co-moment matrix
            self._C += C_b + np.outer(delta, delta) * (self.n_samples * n_b / n_total)

            self.n_samples = n_total

    @property
    def mean(self) -> np.ndarray:
        """Get current mean estimate."""
        return self._mean.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Get current covariance estimate."""
        if self.n_samples < 2:
            return np.zeros((self.n_features, self.n_features))
        return self._C / (self.n_samples - 1)

    def get_result(self, regularization: float = 0.0) -> CovarianceResult:
        """Get complete result.

        Args:
            regularization: Ridge regularization for covariance

        Returns:
            CovarianceResult with mean and covariance
        """
        cov = self.covariance
        if regularization > 0:
            cov = cov + regularization * np.eye(self.n_features)

        return CovarianceResult(
            mean=self.mean,
            covariance=cov,
            n_samples=self.n_samples,
            is_robust=False,
        )


class WoodburyCovariance:
    """Covariance matrix with efficient rank-k updates.

    Uses the Woodbury matrix identity for efficient updates to
    the precision matrix when adding/removing samples.

    Woodbury identity:
    (A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA^{-1}

    Useful for:
    - Online anomaly detection with sliding windows
    - Leave-one-out cross-validation
    - Robust estimation with sample removal

    Example:
        cov = WoodburyCovariance.from_data(training_data)

        # Efficiently update with new sample
        cov.add_sample(new_sample)

        # Efficiently remove old sample
        cov.remove_sample(old_sample)
    """

    def __init__(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        precision: np.ndarray,
        n_samples: int,
    ):
        """Initialize from pre-computed statistics."""
        self.mean = mean.copy()
        self.covariance = covariance.copy()
        self.precision = precision.copy()
        self.n_samples = n_samples
        self.n_features = len(mean)

    @classmethod
    def from_data(cls, data: np.ndarray, regularization: float = 1e-6) -> "WoodburyCovariance":
        """Create from data array.

        Args:
            data: Data array (n_samples, n_features)
            regularization: Ridge regularization

        Returns:
            WoodburyCovariance instance
        """
        mean = data.mean(axis=0)
        cov = np.cov(data, rowvar=False)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)

        cov_reg = cov + regularization * np.eye(len(cov))
        precision = np.linalg.inv(cov_reg)

        return cls(mean, cov, precision, len(data))

    def add_sample(self, x: np.ndarray) -> None:
        """Add a sample and update statistics efficiently.

        Uses rank-1 update via Sherman-Morrison formula.

        Args:
            x: New sample
        """
        n = self.n_samples
        n_new = n + 1

        # Update mean
        delta = x - self.mean
        new_mean = self.mean + delta / n_new

        # Update covariance using rank-1 update
        # Cov_new = (n-1)/n * Cov + (n/(n+1)^2) * delta * delta^T
        delta_new = x - new_mean
        cov_update = np.outer(delta, delta_new) * (n / n_new)

        self.covariance = (n - 1) / n * self.covariance + cov_update

        # Update precision using Sherman-Morrison
        # (A + uv^T)^{-1} = A^{-1} - (A^{-1} u v^T A^{-1}) / (1 + v^T A^{-1} u)
        u = delta.reshape(-1, 1)
        v = delta_new.reshape(-1, 1)
        factor = n / n_new

        Au = self.precision @ u
        vA = v.T @ self.precision
        denominator = 1 + factor * (v.T @ Au)

        self.precision -= factor * (Au @ vA) / denominator

        self.mean = new_mean
        self.n_samples = n_new

    def remove_sample(self, x: np.ndarray) -> None:
        """Remove a sample and update statistics.

        Uses rank-1 downdate. Note: assumes x was part of the original data.

        Args:
            x: Sample to remove
        """
        if self.n_samples <= 2:
            raise ValueError("Cannot remove sample: need at least 2 samples")

        n = self.n_samples
        n_new = n - 1

        # Update mean
        delta_old = x - self.mean
        new_mean = (n * self.mean - x) / n_new

        # Downdate covariance
        delta_new = x - new_mean
        cov_update = np.outer(delta_old, delta_new) * (n / n_new)

        self.covariance = (n - 1) / (n_new - 1) * (self.covariance - cov_update / (n - 1))

        # Update precision using Sherman-Morrison (downdate)
        u = delta_old.reshape(-1, 1)
        v = delta_new.reshape(-1, 1)
        factor = -n / n_new  # Negative for downdate

        Au = self.precision @ u
        vA = v.T @ self.precision
        denominator = 1 + factor * (v.T @ Au)

        self.precision -= factor * (Au @ vA) / denominator

        self.mean = new_mean
        self.n_samples = n_new

    def mahalanobis(self, x: np.ndarray) -> float:
        """Compute Mahalanobis distance for a point.

        Args:
            x: Query point

        Returns:
            Squared Mahalanobis distance
        """
        delta = x - self.mean
        return float(delta @ self.precision @ delta)

    def mahalanobis_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distances for multiple points.

        Args:
            X: Query points (n_samples, n_features)

        Returns:
            Array of squared Mahalanobis distances
        """
        delta = X - self.mean
        return np.sum(delta @ self.precision * delta, axis=1)


class RobustCovarianceEstimator:
    """Robust covariance estimation using subsampled MCD.

    Minimum Covariance Determinant (MCD) is robust to outliers but
    expensive O(n³). This implementation uses subsampling for scalability.

    Strategy:
    1. Take multiple random subsamples
    2. Compute MCD on each subsample
    3. Combine estimates using median

    Example:
        estimator = RobustCovarianceEstimator(contamination=0.1)
        result = estimator.fit(large_data)
    """

    def __init__(
        self,
        contamination: float = 0.1,
        n_subsamples: int = 10,
        subsample_size: int = 500,
        random_state: int = 42,
    ):
        """Initialize robust estimator.

        Args:
            contamination: Expected proportion of outliers
            n_subsamples: Number of subsamples to use
            subsample_size: Size of each subsample
            random_state: Random seed
        """
        self.contamination = contamination
        self.n_subsamples = n_subsamples
        self.subsample_size = subsample_size
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> CovarianceResult:
        """Fit robust covariance estimator.

        Args:
            X: Data array (n_samples, n_features)

        Returns:
            CovarianceResult with robust estimates
        """
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        # For small datasets, use standard MCD
        if n_samples <= self.subsample_size:
            return self._fit_mcd(X)

        # Subsample and combine
        means: list[np.ndarray] = []
        covariances: list[np.ndarray] = []

        for _ in range(self.n_subsamples):
            indices = rng.choice(n_samples, size=self.subsample_size, replace=False)
            subsample = X[indices]

            result = self._fit_mcd(subsample)
            means.append(result.mean)
            covariances.append(result.covariance)

        # Combine using geometric median (approximated by coordinate-wise median)
        combined_mean = np.median(means, axis=0)
        combined_cov = np.median(covariances, axis=0)

        return CovarianceResult(
            mean=combined_mean,
            covariance=combined_cov,
            n_samples=n_samples,
            is_robust=True,
        )

    def _fit_mcd(self, X: np.ndarray) -> CovarianceResult:
        """Fit MCD on a single sample.

        Falls back to standard covariance if sklearn not available.
        """
        try:
            from sklearn.covariance import MinCovDet

            mcd = MinCovDet(
                support_fraction=1 - self.contamination,
                random_state=self.random_state,
            )
            mcd.fit(X)

            return CovarianceResult(
                mean=mcd.location_,
                covariance=mcd.covariance_,
                precision=mcd.precision_,
                n_samples=len(X),
                is_robust=True,
            )
        except ImportError:
            # Fallback to standard estimation
            mean = np.mean(X, axis=0)
            cov = np.cov(X, rowvar=False)

            return CovarianceResult(
                mean=mean,
                covariance=cov,
                n_samples=len(X),
                is_robust=False,
            )


class BatchCovarianceMixin:
    """Mixin providing batch covariance computation.

    Use in validators that need covariance matrices for large datasets.

    Features:
        - Automatic selection of computation strategy based on data size
        - Incremental covariance for streaming data
        - Robust estimation with subsampling
        - Efficient precision matrix updates

    Example:
        class OptimizedMahalanobis(MahalanobisValidator, BatchCovarianceMixin):
            def _compute_mahalanobis(self, data, use_robust):
                result = self.compute_covariance_auto(data, use_robust)
                return self.compute_mahalanobis_distances(data, result)
    """

    # Configuration
    _batch_size: int = 10000
    _robust_threshold: int = 5000  # Use subsampled robust estimation above this
    _regularization: float = 1e-6

    def compute_covariance_auto(
        self,
        data: np.ndarray,
        use_robust: bool = False,
    ) -> CovarianceResult:
        """Automatically select best covariance computation method.

        Args:
            data: Data array (n_samples, n_features)
            use_robust: Whether to use robust estimation

        Returns:
            CovarianceResult
        """
        n_samples, n_features = data.shape

        if use_robust:
            if n_samples > self._robust_threshold:
                # Use subsampled robust estimation
                estimator = RobustCovarianceEstimator(
                    subsample_size=min(500, n_samples // 10)
                )
                return estimator.fit(data)
            else:
                # Use full robust estimation
                estimator = RobustCovarianceEstimator()
                return estimator.fit(data)
        else:
            if n_samples > self._batch_size:
                # Use incremental covariance
                return self.compute_covariance_incremental(data)
            else:
                # Standard computation
                mean = np.mean(data, axis=0)
                cov = np.cov(data, rowvar=False)
                return CovarianceResult(
                    mean=mean,
                    covariance=cov,
                    n_samples=n_samples,
                )

    def compute_covariance_incremental(
        self,
        data: np.ndarray,
        batch_size: int | None = None,
    ) -> CovarianceResult:
        """Compute covariance using incremental algorithm.

        Args:
            data: Data array
            batch_size: Batch size for updates

        Returns:
            CovarianceResult
        """
        batch_size = batch_size or self._batch_size
        n_samples, n_features = data.shape

        inc_cov = IncrementalCovariance(n_features)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            inc_cov.update_batch(data[start:end])

        return inc_cov.get_result(regularization=self._regularization)

    def compute_mahalanobis_distances(
        self,
        data: np.ndarray,
        cov_result: CovarianceResult,
    ) -> np.ndarray:
        """Compute Mahalanobis distances efficiently.

        Args:
            data: Data points
            cov_result: Covariance result

        Returns:
            Array of squared Mahalanobis distances
        """
        precision = cov_result.get_precision(self._regularization)
        delta = data - cov_result.mean

        # Vectorized computation
        return np.sum(delta @ precision * delta, axis=1)

    def create_woodbury_covariance(
        self,
        data: np.ndarray,
    ) -> WoodburyCovariance:
        """Create WoodburyCovariance for efficient updates.

        Args:
            data: Initial data

        Returns:
            WoodburyCovariance instance
        """
        return WoodburyCovariance.from_data(data, self._regularization)
