"""Machine learning based anomaly detection validators.

These validators use scikit-learn for advanced anomaly detection.
Requires: pip install truthound[anomaly] (includes scikit-learn)
"""

from typing import Any

import polars as pl
import numpy as np

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.registry import register_validator
from truthound.validators.anomaly.base import (
    AnomalyValidator,
    MLAnomalyMixin,
)


def _check_sklearn_available() -> None:
    """Check if scikit-learn is available."""
    try:
        import sklearn  # noqa: F401
    except ImportError:
        raise ImportError(
            "scikit-learn is required for ML-based anomaly detection. "
            "Install with: pip install truthound[anomaly]"
        )


@register_validator
class IsolationForestValidator(AnomalyValidator, MLAnomalyMixin):
    """Isolation Forest anomaly detection.

    Isolation Forest isolates anomalies by randomly selecting a feature
    and then randomly selecting a split value. Anomalies are easier to
    isolate, so they have shorter path lengths in the tree.

    This is efficient for high-dimensional data and doesn't assume
    any particular distribution.

    Example:
        # Detect anomalies in multiple columns
        validator = IsolationForestValidator(
            columns=["feature1", "feature2", "feature3"],
            contamination=0.05,  # Expected 5% anomalies
        )

        # Auto-detect contamination
        validator = IsolationForestValidator(
            columns=["col1", "col2"],
            contamination="auto",
        )
    """

    name = "isolation_forest"

    def __init__(
        self,
        columns: list[str] | None = None,
        contamination: float | str = "auto",
        n_estimators: int = 100,
        max_samples: int | float | str = "auto",
        random_state: int | None = 42,
        max_anomaly_ratio: float = 0.1,
        **kwargs: Any,
    ):
        """Initialize Isolation Forest validator.

        Args:
            columns: Columns to use for detection. If None, uses all numeric.
            contamination: Expected proportion of outliers ("auto" or 0.0-0.5)
            n_estimators: Number of trees in the forest
            max_samples: Number of samples for each tree
            random_state: Random seed for reproducibility
            max_anomaly_ratio: Maximum acceptable ratio of anomalies
            **kwargs: Additional config
        """
        super().__init__(columns=columns, max_anomaly_ratio=max_anomaly_ratio, **kwargs)
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

    def detect_anomalies(
        self, data: np.ndarray, column_names: list[str]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Detect anomalies using Isolation Forest."""
        _check_sklearn_available()
        from sklearn.ensemble import IsolationForest

        # Normalize data for better performance
        normalized_data, medians, iqrs = self.normalize_data(data)

        # Create and fit model
        model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,  # Use all cores
        )

        # Predict: -1 for anomalies, 1 for normal
        predictions = model.fit_predict(normalized_data)
        anomaly_mask = predictions == -1

        # Get anomaly scores (lower = more anomalous)
        scores = model.decision_function(normalized_data)

        return anomaly_mask, {
            "n_features": data.shape[1],
            "n_samples": data.shape[0],
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "threshold": float(model.offset_),
        }

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        columns = self._get_anomaly_columns(lf)
        if not columns:
            return issues

        # Collect data for selected columns
        df = lf.select([pl.col(c) for c in columns]).collect()

        # Drop rows with any nulls
        df_clean = df.drop_nulls()
        if len(df_clean) < 10:
            return issues

        data = df_clean.to_numpy()

        anomaly_mask, info = self.detect_anomalies(data, columns)
        anomaly_count = int(anomaly_mask.sum())
        anomaly_ratio = anomaly_count / len(data)

        if anomaly_ratio > self.max_anomaly_ratio:
            severity = self._calculate_severity(anomaly_ratio)

            issues.append(
                ValidationIssue(
                    column=", ".join(columns),
                    issue_type="isolation_forest_anomaly",
                    count=anomaly_count,
                    severity=severity,
                    details=(
                        f"Isolation Forest detected {anomaly_count} anomalies "
                        f"({anomaly_ratio:.2%}) across {info['n_features']} features. "
                        f"Score range: [{info['min_score']:.4f}, {info['max_score']:.4f}]"
                    ),
                    expected=f"Anomaly ratio <= {self.max_anomaly_ratio:.2%}",
                )
            )

        return issues


@register_validator
class LOFValidator(AnomalyValidator, MLAnomalyMixin):
    """Local Outlier Factor (LOF) anomaly detection.

    LOF measures the local density deviation of a point with respect to
    its neighbors. Points with substantially lower density than their
    neighbors are considered outliers.

    Best for detecting local anomalies in clustered data.

    Example:
        validator = LOFValidator(
            columns=["x", "y"],
            n_neighbors=20,
            contamination=0.05,
        )
    """

    name = "lof"

    def __init__(
        self,
        columns: list[str] | None = None,
        n_neighbors: int = 20,
        contamination: float | str = "auto",
        metric: str = "minkowski",
        max_anomaly_ratio: float = 0.1,
        **kwargs: Any,
    ):
        """Initialize LOF validator.

        Args:
            columns: Columns to use for detection. If None, uses all numeric.
            n_neighbors: Number of neighbors for LOF calculation
            contamination: Expected proportion of outliers
            metric: Distance metric to use
            max_anomaly_ratio: Maximum acceptable ratio of anomalies
            **kwargs: Additional config
        """
        super().__init__(columns=columns, max_anomaly_ratio=max_anomaly_ratio, **kwargs)
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric

    def detect_anomalies(
        self, data: np.ndarray, column_names: list[str]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Detect anomalies using LOF."""
        _check_sklearn_available()
        from sklearn.neighbors import LocalOutlierFactor

        # Normalize data
        normalized_data, _, _ = self.normalize_data(data)

        # Adjust n_neighbors if needed
        n_neighbors = min(self.n_neighbors, len(data) - 1)

        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.contamination,
            metric=self.metric,
            n_jobs=-1,
        )

        # Predict: -1 for anomalies, 1 for normal
        predictions = model.fit_predict(normalized_data)
        anomaly_mask = predictions == -1

        # Get LOF scores (higher = more anomalous)
        lof_scores = -model.negative_outlier_factor_

        return anomaly_mask, {
            "n_neighbors": n_neighbors,
            "min_lof": float(np.min(lof_scores)),
            "max_lof": float(np.max(lof_scores)),
            "mean_lof": float(np.mean(lof_scores)),
        }

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        columns = self._get_anomaly_columns(lf)
        if not columns:
            return issues

        df = lf.select([pl.col(c) for c in columns]).collect()
        df_clean = df.drop_nulls()

        if len(df_clean) < self.n_neighbors + 1:
            return issues

        data = df_clean.to_numpy()

        anomaly_mask, info = self.detect_anomalies(data, columns)
        anomaly_count = int(anomaly_mask.sum())
        anomaly_ratio = anomaly_count / len(data)

        if anomaly_ratio > self.max_anomaly_ratio:
            severity = self._calculate_severity(anomaly_ratio)

            issues.append(
                ValidationIssue(
                    column=", ".join(columns),
                    issue_type="lof_anomaly",
                    count=anomaly_count,
                    severity=severity,
                    details=(
                        f"LOF (k={info['n_neighbors']}) detected {anomaly_count} anomalies "
                        f"({anomaly_ratio:.2%}). LOF scores: mean={info['mean_lof']:.2f}, "
                        f"max={info['max_lof']:.2f}"
                    ),
                    expected=f"Anomaly ratio <= {self.max_anomaly_ratio:.2%}",
                )
            )

        return issues


@register_validator
class OneClassSVMValidator(AnomalyValidator, MLAnomalyMixin):
    """One-Class SVM for anomaly detection.

    One-Class SVM learns a decision boundary around normal data.
    Points outside this boundary are classified as anomalies.

    Works well for high-dimensional data but can be slower than
    tree-based methods.

    Example:
        validator = OneClassSVMValidator(
            columns=["feature1", "feature2"],
            nu=0.05,  # Upper bound on fraction of anomalies
            kernel="rbf",
        )
    """

    name = "one_class_svm"

    def __init__(
        self,
        columns: list[str] | None = None,
        kernel: str = "rbf",
        nu: float = 0.05,
        gamma: str | float = "scale",
        max_anomaly_ratio: float = 0.1,
        **kwargs: Any,
    ):
        """Initialize One-Class SVM validator.

        Args:
            columns: Columns to use for detection
            kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
            nu: Upper bound on fraction of training errors and support vectors
            gamma: Kernel coefficient
            max_anomaly_ratio: Maximum acceptable ratio of anomalies
            **kwargs: Additional config
        """
        super().__init__(columns=columns, max_anomaly_ratio=max_anomaly_ratio, **kwargs)
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma

    def detect_anomalies(
        self, data: np.ndarray, column_names: list[str]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Detect anomalies using One-Class SVM."""
        _check_sklearn_available()
        from sklearn.svm import OneClassSVM

        # Normalize data
        normalized_data, _, _ = self.normalize_data(data)

        model = OneClassSVM(
            kernel=self.kernel,
            nu=self.nu,
            gamma=self.gamma,
        )

        predictions = model.fit_predict(normalized_data)
        anomaly_mask = predictions == -1

        # Get decision function scores
        scores = model.decision_function(normalized_data)

        return anomaly_mask, {
            "kernel": self.kernel,
            "nu": self.nu,
            "n_support": len(model.support_),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
        }

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        columns = self._get_anomaly_columns(lf)
        if not columns:
            return issues

        df = lf.select([pl.col(c) for c in columns]).collect()
        df_clean = df.drop_nulls()

        if len(df_clean) < 10:
            return issues

        data = df_clean.to_numpy()

        anomaly_mask, info = self.detect_anomalies(data, columns)
        anomaly_count = int(anomaly_mask.sum())
        anomaly_ratio = anomaly_count / len(data)

        if anomaly_ratio > self.max_anomaly_ratio:
            severity = self._calculate_severity(anomaly_ratio)

            issues.append(
                ValidationIssue(
                    column=", ".join(columns),
                    issue_type="svm_anomaly",
                    count=anomaly_count,
                    severity=severity,
                    details=(
                        f"One-Class SVM ({info['kernel']}, nu={info['nu']}) detected "
                        f"{anomaly_count} anomalies ({anomaly_ratio:.2%}). "
                        f"Support vectors: {info['n_support']}"
                    ),
                    expected=f"Anomaly ratio <= {self.max_anomaly_ratio:.2%}",
                )
            )

        return issues


@register_validator
class DBSCANAnomalyValidator(AnomalyValidator, MLAnomalyMixin):
    """DBSCAN-based anomaly detection.

    DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    identifies outliers as noise points that don't belong to any cluster.

    Best for discovering clusters of arbitrary shape while identifying
    noise points as anomalies.

    Example:
        validator = DBSCANAnomalyValidator(
            columns=["x", "y"],
            eps=0.5,  # Maximum distance between points
            min_samples=5,  # Minimum cluster size
        )
    """

    name = "dbscan_anomaly"

    def __init__(
        self,
        columns: list[str] | None = None,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
        max_anomaly_ratio: float = 0.1,
        **kwargs: Any,
    ):
        """Initialize DBSCAN anomaly validator.

        Args:
            columns: Columns to use for detection
            eps: Maximum distance between points in a cluster
            min_samples: Minimum number of points for a core point
            metric: Distance metric
            max_anomaly_ratio: Maximum acceptable ratio of anomalies
            **kwargs: Additional config
        """
        super().__init__(columns=columns, max_anomaly_ratio=max_anomaly_ratio, **kwargs)
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def detect_anomalies(
        self, data: np.ndarray, column_names: list[str]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Detect anomalies using DBSCAN."""
        _check_sklearn_available()
        from sklearn.cluster import DBSCAN

        # Normalize data
        normalized_data, _, _ = self.normalize_data(data)

        model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            n_jobs=-1,
        )

        labels = model.fit_predict(normalized_data)

        # -1 label indicates noise (anomaly)
        anomaly_mask = labels == -1

        # Count clusters
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        return anomaly_mask, {
            "n_clusters": n_clusters,
            "eps": self.eps,
            "min_samples": self.min_samples,
        }

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        columns = self._get_anomaly_columns(lf)
        if not columns:
            return issues

        df = lf.select([pl.col(c) for c in columns]).collect()
        df_clean = df.drop_nulls()

        if len(df_clean) < self.min_samples:
            return issues

        data = df_clean.to_numpy()

        anomaly_mask, info = self.detect_anomalies(data, columns)
        anomaly_count = int(anomaly_mask.sum())
        anomaly_ratio = anomaly_count / len(data)

        if anomaly_ratio > self.max_anomaly_ratio:
            severity = self._calculate_severity(anomaly_ratio)

            issues.append(
                ValidationIssue(
                    column=", ".join(columns),
                    issue_type="dbscan_anomaly",
                    count=anomaly_count,
                    severity=severity,
                    details=(
                        f"DBSCAN (eps={info['eps']}, min_samples={info['min_samples']}) "
                        f"found {info['n_clusters']} clusters and {anomaly_count} noise points "
                        f"({anomaly_ratio:.2%})"
                    ),
                    expected=f"Anomaly ratio <= {self.max_anomaly_ratio:.2%}",
                )
            )

        return issues
