"""Multivariate anomaly detection validators.

Statistical methods for detecting anomalies across multiple dimensions.
"""

from typing import Any

import polars as pl
import numpy as np

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.registry import register_validator
from truthound.validators.anomaly.base import (
    AnomalyValidator,
    StatisticalAnomalyMixin,
)
from truthound.validators.optimization import BatchCovarianceMixin


@register_validator
class MahalanobisValidator(AnomalyValidator, StatisticalAnomalyMixin):
    """Mahalanobis distance based multivariate anomaly detection.

    Mahalanobis distance measures how far a point is from the center
    of a distribution, accounting for correlations between variables.

    For normal data, squared Mahalanobis distances follow a chi-squared
    distribution with degrees of freedom equal to the number of dimensions.

    Example:
        validator = MahalanobisValidator(
            columns=["feature1", "feature2", "feature3"],
            threshold_percentile=97.5,  # Chi-squared critical value
        )
    """

    name = "mahalanobis"

    def __init__(
        self,
        columns: list[str] | None = None,
        threshold_percentile: float = 97.5,
        use_robust_covariance: bool = True,
        max_anomaly_ratio: float = 0.1,
        **kwargs: Any,
    ):
        """Initialize Mahalanobis validator.

        Args:
            columns: Columns to use. If None, uses all numeric columns.
            threshold_percentile: Chi-squared percentile for threshold
            use_robust_covariance: Use robust covariance estimation
            max_anomaly_ratio: Maximum acceptable ratio of anomalies
            **kwargs: Additional config
        """
        super().__init__(columns=columns, max_anomaly_ratio=max_anomaly_ratio, **kwargs)
        self.threshold_percentile = threshold_percentile
        self.use_robust_covariance = use_robust_covariance

    def _compute_mahalanobis(
        self, data: np.ndarray, use_robust: bool = True
    ) -> tuple[np.ndarray, float]:
        """Compute Mahalanobis distances.

        Args:
            data: 2D array (n_samples, n_features)
            use_robust: Use robust covariance estimation

        Returns:
            Tuple of (squared_distances, threshold)
        """
        from scipy import stats

        n_samples, n_features = data.shape

        if use_robust and n_samples > n_features * 5:
            # Use Minimum Covariance Determinant for robust estimation
            try:
                from sklearn.covariance import MinCovDet
                mcd = MinCovDet(random_state=42)
                mcd.fit(data)
                center = mcd.location_
                cov_inv = np.linalg.pinv(mcd.covariance_)
            except (ImportError, Exception):
                # Fall back to standard estimation
                center = np.mean(data, axis=0)
                cov = np.cov(data, rowvar=False)
                cov_inv = np.linalg.pinv(cov)
        else:
            center = np.mean(data, axis=0)
            cov = np.cov(data, rowvar=False)
            cov_inv = np.linalg.pinv(cov)

        # Compute squared Mahalanobis distances
        diff = data - center
        sq_distances = np.sum(diff @ cov_inv * diff, axis=1)

        # Chi-squared threshold
        threshold = stats.chi2.ppf(self.threshold_percentile / 100, df=n_features)

        return sq_distances, threshold

    def detect_anomalies(
        self, data: np.ndarray, column_names: list[str]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Detect anomalies using Mahalanobis distance."""
        sq_distances, threshold = self._compute_mahalanobis(
            data, self.use_robust_covariance
        )

        anomaly_mask = sq_distances > threshold

        return anomaly_mask, {
            "threshold": threshold,
            "n_features": data.shape[1],
            "max_distance": float(np.max(sq_distances)),
            "mean_distance": float(np.mean(sq_distances)),
            "percentile": self.threshold_percentile,
        }

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        columns = self._get_anomaly_columns(lf)
        if len(columns) < 2:
            return issues

        df = lf.select([pl.col(c) for c in columns]).collect()
        df_clean = df.drop_nulls()

        if len(df_clean) < len(columns) * 3:
            return issues

        data = df_clean.to_numpy()

        try:
            anomaly_mask, info = self.detect_anomalies(data, columns)
        except np.linalg.LinAlgError:
            # Singular matrix - skip validation
            return issues

        anomaly_count = int(anomaly_mask.sum())
        anomaly_ratio = anomaly_count / len(data)

        if anomaly_ratio > self.max_anomaly_ratio:
            severity = self._calculate_severity(anomaly_ratio)

            issues.append(
                ValidationIssue(
                    column=", ".join(columns),
                    issue_type="mahalanobis_anomaly",
                    count=anomaly_count,
                    severity=severity,
                    details=(
                        f"Mahalanobis distance (P{info['percentile']}) detected "
                        f"{anomaly_count} anomalies ({anomaly_ratio:.2%}). "
                        f"Threshold: {info['threshold']:.2f}, "
                        f"Max distance: {info['max_distance']:.2f}"
                    ),
                    expected=f"Anomaly ratio <= {self.max_anomaly_ratio:.2%}",
                )
            )

        return issues


@register_validator
class EllipticEnvelopeValidator(AnomalyValidator):
    """Elliptic Envelope for Gaussian-distributed data anomaly detection.

    Fits a robust covariance estimate to the data and identifies
    points that are far from the fitted ellipse as outliers.

    Best for data that is approximately Gaussian.

    Example:
        validator = EllipticEnvelopeValidator(
            columns=["feature1", "feature2"],
            contamination=0.05,
        )
    """

    name = "elliptic_envelope"

    def __init__(
        self,
        columns: list[str] | None = None,
        contamination: float = 0.05,
        random_state: int | None = 42,
        max_anomaly_ratio: float = 0.1,
        **kwargs: Any,
    ):
        """Initialize Elliptic Envelope validator.

        Args:
            columns: Columns to use for detection
            contamination: Expected proportion of outliers
            random_state: Random seed for reproducibility
            max_anomaly_ratio: Maximum acceptable ratio of anomalies
            **kwargs: Additional config
        """
        super().__init__(columns=columns, max_anomaly_ratio=max_anomaly_ratio, **kwargs)
        self.contamination = contamination
        self.random_state = random_state

    def detect_anomalies(
        self, data: np.ndarray, column_names: list[str]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Detect anomalies using Elliptic Envelope."""
        try:
            from sklearn.covariance import EllipticEnvelope
        except ImportError:
            raise ImportError(
                "scikit-learn is required for EllipticEnvelope. "
                "Install with: pip install truthound[anomaly]"
            )

        model = EllipticEnvelope(
            contamination=self.contamination,
            random_state=self.random_state,
        )

        predictions = model.fit_predict(data)
        anomaly_mask = predictions == -1

        scores = model.decision_function(data)

        return anomaly_mask, {
            "contamination": self.contamination,
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "threshold": float(model.offset_),
        }

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        columns = self._get_anomaly_columns(lf)
        if len(columns) < 2:
            return issues

        df = lf.select([pl.col(c) for c in columns]).collect()
        df_clean = df.drop_nulls()

        if len(df_clean) < len(columns) * 3:
            return issues

        data = df_clean.to_numpy()

        try:
            anomaly_mask, info = self.detect_anomalies(data, columns)
        except Exception:
            return issues

        anomaly_count = int(anomaly_mask.sum())
        anomaly_ratio = anomaly_count / len(data)

        if anomaly_ratio > self.max_anomaly_ratio:
            severity = self._calculate_severity(anomaly_ratio)

            issues.append(
                ValidationIssue(
                    column=", ".join(columns),
                    issue_type="elliptic_envelope_anomaly",
                    count=anomaly_count,
                    severity=severity,
                    details=(
                        f"Elliptic Envelope detected {anomaly_count} anomalies "
                        f"({anomaly_ratio:.2%}). "
                        f"Score range: [{info['min_score']:.4f}, {info['max_score']:.4f}]"
                    ),
                    expected=f"Anomaly ratio <= {self.max_anomaly_ratio:.2%}",
                )
            )

        return issues


@register_validator
class PCAAnomalyValidator(AnomalyValidator):
    """PCA-based anomaly detection using reconstruction error.

    Projects data onto principal components and uses reconstruction
    error to identify anomalies. Points with high reconstruction error
    are likely anomalies.

    Useful for high-dimensional data where anomalies deviate from
    the main variance directions.

    Example:
        validator = PCAAnomalyValidator(
            columns=["f1", "f2", "f3", "f4"],
            n_components=2,  # Keep 2 principal components
            error_percentile=95,
        )
    """

    name = "pca_anomaly"

    def __init__(
        self,
        columns: list[str] | None = None,
        n_components: int | float | None = None,
        error_percentile: float = 95.0,
        max_anomaly_ratio: float = 0.1,
        **kwargs: Any,
    ):
        """Initialize PCA anomaly validator.

        Args:
            columns: Columns to use for detection
            n_components: Number of components (int) or variance ratio (float)
            error_percentile: Percentile of reconstruction error for threshold
            max_anomaly_ratio: Maximum acceptable ratio of anomalies
            **kwargs: Additional config
        """
        super().__init__(columns=columns, max_anomaly_ratio=max_anomaly_ratio, **kwargs)
        self.n_components = n_components
        self.error_percentile = error_percentile

    def detect_anomalies(
        self, data: np.ndarray, column_names: list[str]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Detect anomalies using PCA reconstruction error."""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError(
                "scikit-learn is required for PCA anomaly detection. "
                "Install with: pip install truthound[anomaly]"
            )

        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        # Determine number of components
        n_components = self.n_components
        if n_components is None:
            n_components = min(data.shape[1] // 2, max(1, data.shape[1] - 1))

        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(data_scaled)
        reconstructed = pca.inverse_transform(transformed)

        # Calculate reconstruction error (squared Euclidean distance)
        reconstruction_errors = np.sum((data_scaled - reconstructed) ** 2, axis=1)

        # Threshold based on percentile
        threshold = np.percentile(reconstruction_errors, self.error_percentile)
        anomaly_mask = reconstruction_errors > threshold

        return anomaly_mask, {
            "n_components": pca.n_components_,
            "explained_variance_ratio": float(np.sum(pca.explained_variance_ratio_)),
            "threshold": float(threshold),
            "max_error": float(np.max(reconstruction_errors)),
            "mean_error": float(np.mean(reconstruction_errors)),
        }

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        columns = self._get_anomaly_columns(lf)
        if len(columns) < 2:
            return issues

        df = lf.select([pl.col(c) for c in columns]).collect()
        df_clean = df.drop_nulls()

        if len(df_clean) < len(columns) * 2:
            return issues

        data = df_clean.to_numpy()

        try:
            anomaly_mask, info = self.detect_anomalies(data, columns)
        except Exception:
            return issues

        anomaly_count = int(anomaly_mask.sum())
        anomaly_ratio = anomaly_count / len(data)

        if anomaly_ratio > self.max_anomaly_ratio:
            severity = self._calculate_severity(anomaly_ratio)

            issues.append(
                ValidationIssue(
                    column=", ".join(columns),
                    issue_type="pca_anomaly",
                    count=anomaly_count,
                    severity=severity,
                    details=(
                        f"PCA ({info['n_components']} components, "
                        f"{info['explained_variance_ratio']:.1%} variance) detected "
                        f"{anomaly_count} anomalies ({anomaly_ratio:.2%}). "
                        f"Error threshold: {info['threshold']:.4f}"
                    ),
                    expected=f"Anomaly ratio <= {self.max_anomaly_ratio:.2%}",
                )
            )

        return issues


@register_validator
class ZScoreMultivariateValidator(AnomalyValidator, StatisticalAnomalyMixin):
    """Multivariate Z-score based anomaly detection.

    Computes Z-scores for each column and flags rows where any column
    exceeds the threshold, or where the combined score exceeds a threshold.

    Simpler and faster than Mahalanobis but doesn't account for
    correlations between variables.

    Example:
        validator = ZScoreMultivariateValidator(
            columns=["col1", "col2"],
            threshold=3.0,
            method="any",  # Flag if ANY column exceeds threshold
        )
    """

    name = "zscore_multivariate"

    def __init__(
        self,
        columns: list[str] | None = None,
        threshold: float = 3.0,
        method: str = "any",  # "any", "all", or "mean"
        max_anomaly_ratio: float = 0.1,
        **kwargs: Any,
    ):
        """Initialize multivariate Z-score validator.

        Args:
            columns: Columns to use for detection
            threshold: Z-score threshold
            method: How to combine column Z-scores:
                - "any": Anomaly if any column exceeds threshold
                - "all": Anomaly if all columns exceed threshold
                - "mean": Anomaly if mean Z-score exceeds threshold
            max_anomaly_ratio: Maximum acceptable ratio of anomalies
            **kwargs: Additional config
        """
        super().__init__(columns=columns, max_anomaly_ratio=max_anomaly_ratio, **kwargs)
        self.threshold = threshold
        self.method = method

        if method not in ("any", "all", "mean"):
            raise ValueError(f"Invalid method: {method}. Use 'any', 'all', or 'mean'")

    def detect_anomalies(
        self, data: np.ndarray, column_names: list[str]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Detect anomalies using multivariate Z-scores."""
        # Compute Z-scores for each column
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0)
        stds = np.where(stds == 0, 1, stds)  # Avoid division by zero

        z_scores = np.abs((data - means) / stds)

        if self.method == "any":
            anomaly_mask = np.any(z_scores > self.threshold, axis=1)
        elif self.method == "all":
            anomaly_mask = np.all(z_scores > self.threshold, axis=1)
        else:  # mean
            mean_z = np.mean(z_scores, axis=1)
            anomaly_mask = mean_z > self.threshold

        # Per-column anomaly counts
        column_anomalies = {
            col: int(np.sum(z_scores[:, i] > self.threshold))
            for i, col in enumerate(column_names)
        }

        return anomaly_mask, {
            "threshold": self.threshold,
            "method": self.method,
            "column_anomalies": column_anomalies,
            "max_zscore": float(np.max(z_scores)),
        }

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        columns = self._get_anomaly_columns(lf)
        if not columns:
            return issues

        df = lf.select([pl.col(c) for c in columns]).collect()
        df_clean = df.drop_nulls()

        if len(df_clean) < 3:
            return issues

        data = df_clean.to_numpy()

        anomaly_mask, info = self.detect_anomalies(data, columns)
        anomaly_count = int(anomaly_mask.sum())
        anomaly_ratio = anomaly_count / len(data)

        if anomaly_ratio > self.max_anomaly_ratio:
            severity = self._calculate_severity(anomaly_ratio)

            # Format per-column details
            col_details = ", ".join(
                f"{col}:{count}" for col, count in info["column_anomalies"].items()
            )

            issues.append(
                ValidationIssue(
                    column=", ".join(columns),
                    issue_type="zscore_multivariate_anomaly",
                    count=anomaly_count,
                    severity=severity,
                    details=(
                        f"Z-score ({info['method']}, threshold={info['threshold']}) detected "
                        f"{anomaly_count} anomalies ({anomaly_ratio:.2%}). "
                        f"Per-column: {col_details}"
                    ),
                    expected=f"Anomaly ratio <= {self.max_anomaly_ratio:.2%}",
                )
            )

        return issues


@register_validator
class OptimizedMahalanobisValidator(AnomalyValidator, StatisticalAnomalyMixin, BatchCovarianceMixin):
    """Optimized Mahalanobis distance validator with incremental covariance.

    Uses BatchCovarianceMixin for memory-efficient covariance computation:
    - Incremental covariance for large datasets (O(n) memory vs O(n×d))
    - Woodbury matrix updates for streaming scenarios
    - Robust estimation with subsampling for datasets > 5000 rows

    Example:
        validator = OptimizedMahalanobisValidator(
            columns=["feature1", "feature2", "feature3"],
            threshold_percentile=97.5,
            batch_size=10000,  # Process in batches
        )
    """

    name = "optimized_mahalanobis"

    def __init__(
        self,
        columns: list[str] | None = None,
        threshold_percentile: float = 97.5,
        use_robust_covariance: bool = True,
        max_anomaly_ratio: float = 0.1,
        batch_size: int = 10000,
        **kwargs: Any,
    ):
        """Initialize optimized Mahalanobis validator.

        Args:
            columns: Columns to use. If None, uses all numeric columns.
            threshold_percentile: Chi-squared percentile for threshold
            use_robust_covariance: Use robust covariance estimation
            max_anomaly_ratio: Maximum acceptable ratio of anomalies
            batch_size: Batch size for incremental covariance computation
            **kwargs: Additional config
        """
        super().__init__(columns=columns, max_anomaly_ratio=max_anomaly_ratio, **kwargs)
        self.threshold_percentile = threshold_percentile
        self.use_robust_covariance = use_robust_covariance
        self._batch_size = batch_size

    def detect_anomalies(
        self, data: np.ndarray, column_names: list[str]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Detect anomalies using optimized Mahalanobis distance."""
        from scipy import stats

        n_samples, n_features = data.shape

        # Use BatchCovarianceMixin for efficient covariance computation
        cov_result = self.compute_covariance_auto(data, use_robust=self.use_robust_covariance)

        # Compute Mahalanobis distances using the mixin
        sq_distances = self.compute_mahalanobis_distances(data, cov_result)

        # Chi-squared threshold
        threshold = stats.chi2.ppf(self.threshold_percentile / 100, df=n_features)

        anomaly_mask = sq_distances > threshold

        return anomaly_mask, {
            "threshold": threshold,
            "n_features": n_features,
            "max_distance": float(np.max(sq_distances)),
            "mean_distance": float(np.mean(sq_distances)),
            "percentile": self.threshold_percentile,
            "is_robust": cov_result.is_robust,
            "n_samples_used": cov_result.n_samples,
        }

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        columns = self._get_anomaly_columns(lf)
        if len(columns) < 2:
            return issues

        df = lf.select([pl.col(c) for c in columns]).collect()
        df_clean = df.drop_nulls()

        if len(df_clean) < len(columns) * 3:
            return issues

        data = df_clean.to_numpy()

        try:
            anomaly_mask, info = self.detect_anomalies(data, columns)
        except np.linalg.LinAlgError:
            return issues

        anomaly_count = int(anomaly_mask.sum())
        anomaly_ratio = anomaly_count / len(data)

        if anomaly_ratio > self.max_anomaly_ratio:
            severity = self._calculate_severity(anomaly_ratio)

            robust_str = " (robust)" if info["is_robust"] else ""
            issues.append(
                ValidationIssue(
                    column=", ".join(columns),
                    issue_type="optimized_mahalanobis_anomaly",
                    count=anomaly_count,
                    severity=severity,
                    details=(
                        f"Mahalanobis distance{robust_str} (P{info['percentile']}) detected "
                        f"{anomaly_count} anomalies ({anomaly_ratio:.2%}). "
                        f"Threshold: {info['threshold']:.2f}, "
                        f"Max distance: {info['max_distance']:.2f}"
                    ),
                    expected=f"Anomaly ratio <= {self.max_anomaly_ratio:.2%}",
                )
            )

        return issues
