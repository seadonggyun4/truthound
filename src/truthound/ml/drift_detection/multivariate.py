"""Multivariate drift detection.

Detects drift considering correlations and joint distributions
across multiple features simultaneously.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import sqrt
from typing import Any

import polars as pl

from truthound.ml.base import (
    DriftConfig,
    DriftResult,
    MLDriftDetector,
    ModelInfo,
    ModelNotTrainedError,
    ModelState,
    ModelTrainingError,
    ModelType,
    register_model,
)


@dataclass
class MultivariateDriftConfig(DriftConfig):
    """Configuration for multivariate drift detection.

    Attributes:
        method: Detection method ('pca', 'correlation', 'mahalanobis')
        n_components: Number of PCA components (for PCA method)
        correlation_threshold: Threshold for correlation change
    """

    method: str = "correlation"
    n_components: int | None = None
    correlation_threshold: float = 0.3
    columns: list[str] | None = None


@register_model("multivariate_drift")
class MultivariateDriftDetector(MLDriftDetector):
    """Multivariate drift detection.

    Detects drift by analyzing changes in:
    - Feature correlations (correlation matrix changes)
    - Joint distributions (PCA space changes)
    - Multivariate distances (Mahalanobis distance)

    This catches drifts that univariate methods miss, such as:
    - Changes in feature correlations
    - Rotation of the data manifold
    - Shifts in conditional distributions

    Example:
        >>> detector = MultivariateDriftDetector(method="correlation")
        >>> detector.fit(reference_data)
        >>> result = detector.detect(reference_data, current_data)
    """

    def __init__(self, config: MultivariateDriftConfig | None = None, **kwargs: Any):
        super().__init__(config, **kwargs)
        self._reference_corr_matrix: list[list[float]] = []
        self._reference_means: list[float] = []
        self._reference_stds: list[float] = []
        self._columns: list[str] = []
        self._pca_components: list[list[float]] | None = None
        self._pca_explained_var: list[float] | None = None

    @property
    def config(self) -> MultivariateDriftConfig:
        return self._config  # type: ignore

    def _default_config(self) -> MultivariateDriftConfig:
        return MultivariateDriftConfig()

    def _get_model_name(self) -> str:
        return "multivariate_drift"

    def _get_description(self) -> str:
        return "Multivariate drift detection considering feature correlations"

    def fit(self, data: pl.LazyFrame) -> None:
        """Compute reference multivariate statistics.

        Args:
            data: Reference data
        """
        import time

        start = time.perf_counter()
        self._state = ModelState.TRAINING

        try:
            row_count = self._validate_data(data)
            df = self._maybe_sample(data).collect()

            # Get numeric columns
            schema = df.schema
            numeric_types = {
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64,
            }

            if self.config.columns:
                self._columns = [
                    c for c in self.config.columns
                    if c in schema and type(schema[c]) in numeric_types
                ]
            else:
                self._columns = [
                    c for c in schema.keys()
                    if type(schema[c]) in numeric_types
                ]

            if len(self._columns) < 2:
                raise ValueError(
                    "Multivariate drift requires at least 2 numeric columns"
                )

            # Compute correlation matrix
            self._reference_corr_matrix = self._compute_correlation_matrix(df)

            # Compute means and stds
            self._reference_means = []
            self._reference_stds = []
            for col in self._columns:
                stats = df.select([
                    pl.col(col).mean().alias("mean"),
                    pl.col(col).std().alias("std"),
                ]).row(0)
                self._reference_means.append(stats[0] or 0.0)
                self._reference_stds.append(stats[1] or 1.0)

            # Compute PCA if using PCA method
            if self.config.method == "pca":
                self._compute_pca(df)

            self._reference_data = data
            self._training_samples = row_count
            self._trained_at = datetime.now()
            self._state = ModelState.TRAINED

        except Exception as e:
            self._state = ModelState.ERROR
            self._error = e
            raise ModelTrainingError(
                f"Failed to compute multivariate reference: {e}",
                model_name=self.info.name,
            ) from e

    def _compute_correlation_matrix(self, df: pl.DataFrame) -> list[list[float]]:
        """Compute correlation matrix for numeric columns."""
        n = len(self._columns)
        matrix = [[0.0] * n for _ in range(n)]

        for i, col_i in enumerate(self._columns):
            matrix[i][i] = 1.0  # Diagonal is 1
            for j, col_j in enumerate(self._columns[i+1:], i+1):
                corr = self._compute_correlation(df, col_i, col_j)
                matrix[i][j] = corr
                matrix[j][i] = corr  # Symmetric

        return matrix

    def _compute_correlation(
        self, df: pl.DataFrame, col1: str, col2: str
    ) -> float:
        """Compute Pearson correlation between two columns."""
        valid = df.select([col1, col2]).drop_nulls()
        if len(valid) < 2:
            return 0.0

        x = valid[col1].cast(pl.Float64).to_list()
        y = valid[col2].cast(pl.Float64).to_list()

        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n

        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        std_x = sqrt(sum((xi - mean_x) ** 2 for xi in x))
        std_y = sqrt(sum((yi - mean_y) ** 2 for yi in y))

        if std_x == 0 or std_y == 0:
            return 0.0

        return cov / (std_x * std_y)

    def _compute_pca(self, df: pl.DataFrame) -> None:
        """Compute PCA components using power iteration method.

        Simple PCA implementation without external dependencies.
        """
        # Get data matrix
        data = []
        for col in self._columns:
            col_data = df[col].drop_nulls().cast(pl.Float64).to_list()
            data.append(col_data)

        # Ensure all columns have same length
        min_len = min(len(d) for d in data)
        data = [d[:min_len] for d in data]

        n_samples = min_len
        n_features = len(self._columns)

        # Standardize
        standardized = []
        for i, col_data in enumerate(data):
            mean = self._reference_means[i]
            std = self._reference_stds[i] or 1.0
            standardized.append([(x - mean) / std for x in col_data])

        # Compute covariance matrix
        cov_matrix = [[0.0] * n_features for _ in range(n_features)]
        for i in range(n_features):
            for j in range(n_features):
                cov = sum(
                    standardized[i][k] * standardized[j][k]
                    for k in range(n_samples)
                ) / (n_samples - 1)
                cov_matrix[i][j] = cov

        # Power iteration to find principal components
        n_components = self.config.n_components or min(5, n_features)
        self._pca_components = []
        self._pca_explained_var = []

        remaining_cov = [row[:] for row in cov_matrix]

        for _ in range(n_components):
            # Random initial vector
            import random
            rng = random.Random(self.config.random_seed)
            v = [rng.gauss(0, 1) for _ in range(n_features)]

            # Power iteration
            for _ in range(100):
                # Multiply by covariance matrix
                new_v = [0.0] * n_features
                for i in range(n_features):
                    for j in range(n_features):
                        new_v[i] += remaining_cov[i][j] * v[j]

                # Normalize
                norm = sqrt(sum(x * x for x in new_v))
                if norm > 0:
                    v = [x / norm for x in new_v]

            # Eigenvalue (variance explained)
            eigenvalue = sum(
                v[i] * sum(remaining_cov[i][j] * v[j] for j in range(n_features))
                for i in range(n_features)
            )

            self._pca_components.append(v)
            self._pca_explained_var.append(max(0, eigenvalue))

            # Deflate covariance matrix
            for i in range(n_features):
                for j in range(n_features):
                    remaining_cov[i][j] -= eigenvalue * v[i] * v[j]

    def detect(
        self,
        reference: pl.LazyFrame,
        current: pl.LazyFrame,
        columns: list[str] | None = None,
    ) -> DriftResult:
        """Detect multivariate drift.

        Args:
            reference: Reference data
            current: Current data
            columns: Specific columns to analyze

        Returns:
            DriftResult with multivariate drift analysis
        """
        import time
        start = time.perf_counter()

        curr_df = current.collect()

        # Use stored columns
        check_columns = [c for c in self._columns if c in curr_df.columns]

        if len(check_columns) < 2:
            return DriftResult(
                is_drifted=False,
                drift_score=0.0,
                drift_type="error",
                details="Not enough numeric columns",
            )

        method = self.config.method.lower()

        if method == "correlation":
            drift_score, column_scores, details = self._detect_correlation_drift(curr_df)
        elif method == "pca":
            drift_score, column_scores, details = self._detect_pca_drift(curr_df)
        elif method == "mahalanobis":
            drift_score, column_scores, details = self._detect_mahalanobis_drift(curr_df)
        else:
            drift_score, column_scores, details = self._detect_correlation_drift(curr_df)

        is_drifted = drift_score >= self.config.threshold

        # Determine drift type
        if not is_drifted:
            drift_type = "none"
        elif method == "correlation":
            drift_type = "correlation_shift"
        elif method == "pca":
            drift_type = "manifold_shift"
        else:
            drift_type = "multivariate_shift"

        elapsed = (time.perf_counter() - start) * 1000

        return DriftResult(
            is_drifted=is_drifted,
            drift_score=drift_score,
            column_scores=tuple(column_scores),
            drift_type=drift_type,
            details=details,
        )

    def _detect_correlation_drift(
        self, df: pl.DataFrame
    ) -> tuple[float, list[tuple[str, float]], str]:
        """Detect drift in correlation structure."""
        current_corr = self._compute_correlation_matrix(df)

        n = len(self._columns)
        max_diff = 0.0
        total_diff = 0.0
        n_pairs = 0
        changed_pairs = []

        for i in range(n):
            for j in range(i + 1, n):
                ref_corr = self._reference_corr_matrix[i][j]
                curr_corr = current_corr[i][j]
                diff = abs(curr_corr - ref_corr)

                max_diff = max(max_diff, diff)
                total_diff += diff
                n_pairs += 1

                if diff > self.config.correlation_threshold:
                    changed_pairs.append(
                        f"{self._columns[i]}-{self._columns[j]}: {ref_corr:.2f}->{curr_corr:.2f}"
                    )

        avg_diff = total_diff / n_pairs if n_pairs > 0 else 0.0

        # Column scores based on average correlation change per column
        column_scores = []
        for i, col in enumerate(self._columns):
            col_diffs = []
            for j in range(n):
                if i != j:
                    diff = abs(
                        self._reference_corr_matrix[i][j] - current_corr[i][j]
                    )
                    col_diffs.append(diff)
            avg_col_diff = sum(col_diffs) / len(col_diffs) if col_diffs else 0.0
            column_scores.append((col, avg_col_diff))

        # Normalize score
        drift_score = min(1.0, max_diff / 0.5)

        details = f"Max correlation change: {max_diff:.3f}"
        if changed_pairs:
            details += f", Changed pairs: {', '.join(changed_pairs[:3])}"

        return drift_score, column_scores, details

    def _detect_pca_drift(
        self, df: pl.DataFrame
    ) -> tuple[float, list[tuple[str, float]], str]:
        """Detect drift in PCA space."""
        if self._pca_components is None:
            # Fall back to correlation method
            return self._detect_correlation_drift(df)

        # Standardize current data
        data = []
        for i, col in enumerate(self._columns):
            col_data = df[col].drop_nulls().cast(pl.Float64).to_list()
            mean = self._reference_means[i]
            std = self._reference_stds[i] or 1.0
            data.append([(x - mean) / std for x in col_data])

        min_len = min(len(d) for d in data)
        data = [d[:min_len] for d in data]

        # Project onto reference PCA components
        n_samples = min_len
        projections = []

        for component in self._pca_components:
            proj = [0.0] * n_samples
            for k in range(n_samples):
                for i, weight in enumerate(component):
                    proj[k] += weight * data[i][k]
            projections.append(proj)

        # Compute variance of projections
        current_var = []
        for proj in projections:
            mean_proj = sum(proj) / len(proj)
            var = sum((p - mean_proj) ** 2 for p in proj) / (len(proj) - 1)
            current_var.append(var)

        # Compare to reference variance
        max_var_diff = 0.0
        for i, (ref_var, curr_var) in enumerate(zip(
            self._pca_explained_var or [], current_var
        )):
            if ref_var > 0:
                rel_diff = abs(curr_var - ref_var) / ref_var
                max_var_diff = max(max_var_diff, rel_diff)

        drift_score = min(1.0, max_var_diff)

        # Column scores based on PCA loadings
        column_scores = []
        for i, col in enumerate(self._columns):
            # Sum of absolute loadings across components
            loading_sum = sum(
                abs(comp[i]) for comp in self._pca_components
            )
            column_scores.append((col, loading_sum * drift_score))

        details = f"Max variance change ratio: {max_var_diff:.3f}"

        return drift_score, column_scores, details

    def _detect_mahalanobis_drift(
        self, df: pl.DataFrame
    ) -> tuple[float, list[tuple[str, float]], str]:
        """Detect drift using Mahalanobis distance of mean."""
        # Compute current means
        current_means = []
        for col in self._columns:
            mean = df[col].mean()
            current_means.append(mean if mean is not None else 0.0)

        # Compute inverse covariance (use correlation as approximation)
        n = len(self._columns)

        # Mean difference vector
        diff = [
            current_means[i] - self._reference_means[i]
            for i in range(n)
        ]

        # Standardize by reference std
        standardized_diff = [
            diff[i] / (self._reference_stds[i] or 1.0)
            for i in range(n)
        ]

        # Simple Mahalanobis approximation (assuming diagonal covariance)
        mahal_sq = sum(d * d for d in standardized_diff)
        mahal = sqrt(mahal_sq / n) if n > 0 else 0.0

        # Column scores based on standardized difference
        column_scores = [
            (self._columns[i], abs(standardized_diff[i]) / 3.0)  # Normalize
            for i in range(n)
        ]

        drift_score = min(1.0, mahal / 3.0)  # 3 std = score 1.0

        details = f"Mahalanobis distance: {mahal:.3f}"

        return drift_score, column_scores, details

    def get_correlation_matrix(self) -> dict[str, dict[str, float]]:
        """Get reference correlation matrix as dict."""
        result = {}
        for i, col_i in enumerate(self._columns):
            result[col_i] = {}
            for j, col_j in enumerate(self._columns):
                result[col_i][col_j] = self._reference_corr_matrix[i][j]
        return result
