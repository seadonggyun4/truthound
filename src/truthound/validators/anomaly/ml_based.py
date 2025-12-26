"""Machine learning based anomaly detection validators.

These validators use scikit-learn for advanced anomaly detection.
Requires: pip install truthound[anomaly] (includes scikit-learn)

Memory Optimization:
    These validators now support automatic sampling for large datasets:

    # Memory-efficient usage for large datasets:
    validator = IsolationForestValidator(
        columns=["col1", "col2"],
        sample_size=100000,  # Sample if data exceeds this
        batch_size=50000,    # Process in batches for scoring
    )

    # Or use auto-sampling based on available memory:
    validator = IsolationForestValidator(
        columns=["col1", "col2"],
        auto_sample=True,  # Auto-detect optimal sample size
    )
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


# Default thresholds for memory-efficient processing
DEFAULT_SAMPLE_SIZE = 100000  # Default max samples for training
DEFAULT_BATCH_SIZE = 50000    # Default batch size for scoring
MEMORY_THRESHOLD_MB = 500     # Auto-sample when data exceeds this


def _check_sklearn_available() -> None:
    """Check if scikit-learn is available."""
    try:
        import sklearn  # noqa: F401
    except ImportError:
        raise ImportError(
            "scikit-learn is required for ML-based anomaly detection. "
            "Install with: pip install truthound[anomaly]"
        )


def _estimate_data_memory_mb(n_rows: int, n_cols: int) -> float:
    """Estimate memory usage for numpy array in MB."""
    # Assuming float64 (8 bytes per element)
    bytes_needed = n_rows * n_cols * 8
    return bytes_needed / (1024 * 1024)


def _compute_optimal_sample_size(
    n_rows: int,
    n_cols: int,
    max_memory_mb: float = MEMORY_THRESHOLD_MB,
) -> int:
    """Compute optimal sample size based on memory constraints.

    Args:
        n_rows: Total number of rows
        n_cols: Number of columns
        max_memory_mb: Maximum memory to use

    Returns:
        Optimal sample size
    """
    # Calculate max rows that fit in memory
    bytes_per_row = n_cols * 8  # float64
    max_rows = int((max_memory_mb * 1024 * 1024) / bytes_per_row)

    # Apply a safety margin and cap
    safe_rows = int(max_rows * 0.8)
    return min(n_rows, max(safe_rows, 1000))  # At least 1000 samples


class LargeDatasetMixin:
    """Mixin providing large dataset handling utilities for ML validators.

    Provides:
    - Automatic sampling for training
    - Mini-batch scoring for prediction
    - Memory-aware data loading
    """

    def _smart_sample_lazyframe(
        self,
        lf: pl.LazyFrame,
        columns: list[str],
        sample_size: int | None = None,
        random_state: int = 42,
    ) -> tuple[np.ndarray, int, bool]:
        """Efficiently sample data from LazyFrame.

        Uses Polars lazy evaluation to avoid loading full dataset.

        Args:
            lf: Input LazyFrame
            columns: Columns to select
            sample_size: Max samples (None = load all)
            random_state: Random seed

        Returns:
            Tuple of (data_array, original_count, was_sampled)
        """
        # First, get count efficiently
        count_result = lf.select(pl.len()).collect()
        total_count = count_result.item()

        if total_count == 0:
            return np.array([]).reshape(0, len(columns)), 0, False

        # Determine if sampling is needed
        effective_sample_size = sample_size
        should_sample = sample_size is not None and total_count > sample_size

        if should_sample:
            # Collect data first, then sample (more reliable approach)
            # For very large data, we use slice-based sampling
            df = lf.select([pl.col(c) for c in columns]).drop_nulls().collect()

            if len(df) > effective_sample_size:
                # Random sampling from collected dataframe
                df = df.sample(n=effective_sample_size, seed=random_state)
        else:
            df = lf.select([pl.col(c) for c in columns]).drop_nulls().collect()

        if len(df) == 0:
            return np.array([]).reshape(0, len(columns)), total_count, should_sample

        data = df.to_numpy()
        return data, total_count, should_sample

    def _batch_predict(
        self,
        model: Any,
        data: np.ndarray,
        batch_size: int = DEFAULT_BATCH_SIZE,
        predict_method: str = "predict",
    ) -> np.ndarray:
        """Predict in batches to reduce memory usage.

        Args:
            model: Fitted sklearn model
            data: Input data array
            batch_size: Size of each batch
            predict_method: Method to call on model ('predict' or 'decision_function')

        Returns:
            Concatenated predictions
        """
        n_samples = len(data)
        if n_samples <= batch_size:
            method = getattr(model, predict_method)
            return method(data)

        predictions = []
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = data[start_idx:end_idx]
            method = getattr(model, predict_method)
            batch_pred = method(batch)
            predictions.append(batch_pred)

        return np.concatenate(predictions)

    def _streaming_score(
        self,
        lf: pl.LazyFrame,
        columns: list[str],
        model: Any,
        medians: np.ndarray,
        iqrs: np.ndarray,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stream data and score in batches for very large datasets.

        This method processes data in chunks without loading the entire
        dataset into memory at once.

        Args:
            lf: LazyFrame with data
            columns: Columns to process
            model: Fitted sklearn model (must have predict() and optionally decision_function())
            medians: Normalization medians
            iqrs: Normalization IQRs
            batch_size: Size of each batch

        Returns:
            Tuple of (predictions_array, scores_array or empty array)
        """
        # Get total count
        total_count = lf.select(pl.len()).collect().item()

        if total_count == 0:
            return np.array([]), np.array([])

        all_predictions = []
        all_scores = []
        has_decision_function = hasattr(model, 'decision_function')

        # Process in streaming batches
        for offset in range(0, total_count, batch_size):
            # Fetch batch using slice
            batch_lf = (
                lf.select([pl.col(c) for c in columns])
                .slice(offset, batch_size)
                .drop_nulls()
            )
            batch_df = batch_lf.collect()

            if len(batch_df) == 0:
                continue

            batch_data = batch_df.to_numpy()

            # Normalize using training stats
            normalized_batch = (batch_data - medians) / np.where(iqrs == 0, 1, iqrs)

            # Predict
            batch_preds = model.predict(normalized_batch)
            all_predictions.append(batch_preds)

            if has_decision_function:
                batch_scores = model.decision_function(normalized_batch)
                all_scores.append(batch_scores)

        if not all_predictions:
            return np.array([]), np.array([])

        predictions = np.concatenate(all_predictions)
        scores = np.concatenate(all_scores) if all_scores else np.array([])

        return predictions, scores


@register_validator
class IsolationForestValidator(AnomalyValidator, MLAnomalyMixin, LargeDatasetMixin):
    """Isolation Forest anomaly detection.

    Isolation Forest isolates anomalies by randomly selecting a feature
    and then randomly selecting a split value. Anomalies are easier to
    isolate, so they have shorter path lengths in the tree.

    This is efficient for high-dimensional data and doesn't assume
    any particular distribution.

    Memory Optimization:
        For large datasets, use sample_size and batch_size parameters:

        # Memory-efficient for 10M+ rows:
        validator = IsolationForestValidator(
            columns=["col1", "col2"],
            sample_size=100000,   # Train on 100k samples
            batch_size=50000,     # Score in 50k batches
            auto_sample=True,     # Or let it auto-detect
        )

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
        sample_size: int | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        auto_sample: bool = False,
        max_memory_mb: float = MEMORY_THRESHOLD_MB,
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
            sample_size: Max samples for training (None = use all data)
            batch_size: Batch size for scoring large datasets
            auto_sample: If True, automatically determine sample_size
            max_memory_mb: Max memory (MB) for auto_sample mode
            **kwargs: Additional config
        """
        super().__init__(columns=columns, max_anomaly_ratio=max_anomaly_ratio, **kwargs)
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self._sample_size = sample_size
        self._batch_size = batch_size
        self._auto_sample = auto_sample
        self._max_memory_mb = max_memory_mb

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

        # Determine sample size
        sample_size = self._sample_size
        if self._auto_sample and sample_size is None:
            # Get row count first
            total_count = lf.select(pl.len()).collect().item()
            sample_size = _compute_optimal_sample_size(
                total_count, len(columns), self._max_memory_mb
            )
            self.logger.debug(f"Auto-sample: using {sample_size} samples from {total_count}")

        # Smart sampling from LazyFrame
        data, original_count, was_sampled = self._smart_sample_lazyframe(
            lf, columns, sample_size, self.random_state or 42
        )

        if len(data) < 10:
            return issues

        # Detect anomalies on (possibly sampled) data
        anomaly_mask, info = self.detect_anomalies(data, columns)

        # If we sampled, we need to report based on sample
        # For large datasets, we train on sample but can optionally score all data
        if was_sampled and len(data) < original_count:
            # For very large datasets, we estimate anomaly ratio from sample
            sample_anomaly_count = int(anomaly_mask.sum())
            sample_anomaly_ratio = sample_anomaly_count / len(data)
            # Extrapolate to full dataset
            estimated_total_anomalies = int(sample_anomaly_ratio * original_count)
            anomaly_count = estimated_total_anomalies
            anomaly_ratio = sample_anomaly_ratio
            info["sampled"] = True
            info["sample_size"] = len(data)
            info["original_count"] = original_count
        else:
            anomaly_count = int(anomaly_mask.sum())
            anomaly_ratio = anomaly_count / len(data)
            info["sampled"] = False

        if anomaly_ratio > self.max_anomaly_ratio:
            severity = self._calculate_severity(anomaly_ratio)

            sample_note = ""
            if info.get("sampled"):
                sample_note = f" (estimated from {info['sample_size']:,} samples)"

            issues.append(
                ValidationIssue(
                    column=", ".join(columns),
                    issue_type="isolation_forest_anomaly",
                    count=anomaly_count,
                    severity=severity,
                    details=(
                        f"Isolation Forest detected {anomaly_count:,} anomalies "
                        f"({anomaly_ratio:.2%}) across {info['n_features']} features{sample_note}. "
                        f"Score range: [{info['min_score']:.4f}, {info['max_score']:.4f}]"
                    ),
                    expected=f"Anomaly ratio <= {self.max_anomaly_ratio:.2%}",
                )
            )

        return issues


@register_validator
class LOFValidator(AnomalyValidator, MLAnomalyMixin, LargeDatasetMixin):
    """Local Outlier Factor (LOF) anomaly detection.

    LOF measures the local density deviation of a point with respect to
    its neighbors. Points with substantially lower density than their
    neighbors are considered outliers.

    Best for detecting local anomalies in clustered data.

    Memory Optimization:
        LOF is memory-intensive due to distance computations.
        For large datasets, use sampling:

        validator = LOFValidator(
            columns=["x", "y"],
            n_neighbors=20,
            sample_size=50000,  # Sample for large datasets
        )

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
        sample_size: int | None = None,
        auto_sample: bool = False,
        max_memory_mb: float = MEMORY_THRESHOLD_MB,
        **kwargs: Any,
    ):
        """Initialize LOF validator.

        Args:
            columns: Columns to use for detection. If None, uses all numeric.
            n_neighbors: Number of neighbors for LOF calculation
            contamination: Expected proportion of outliers
            metric: Distance metric to use
            max_anomaly_ratio: Maximum acceptable ratio of anomalies
            sample_size: Max samples for training (None = use all data)
            auto_sample: If True, automatically determine sample_size
            max_memory_mb: Max memory (MB) for auto_sample mode
            **kwargs: Additional config
        """
        super().__init__(columns=columns, max_anomaly_ratio=max_anomaly_ratio, **kwargs)
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self._sample_size = sample_size
        self._auto_sample = auto_sample
        self._max_memory_mb = max_memory_mb

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

        # Determine sample size (LOF is O(n^2) memory, so sampling is critical)
        sample_size = self._sample_size
        if self._auto_sample and sample_size is None:
            total_count = lf.select(pl.len()).collect().item()
            # LOF needs distance matrix, so use more aggressive sampling
            # Memory: O(n^2) for distance matrix
            sample_size = _compute_optimal_sample_size(
                total_count, len(columns), self._max_memory_mb / 2  # More conservative
            )
            # Cap at reasonable limit for LOF
            sample_size = min(sample_size, 50000)
            self.logger.debug(f"Auto-sample (LOF): using {sample_size} samples from {total_count}")

        # Smart sampling from LazyFrame
        data, original_count, was_sampled = self._smart_sample_lazyframe(
            lf, columns, sample_size, 42
        )

        if len(data) < self.n_neighbors + 1:
            return issues

        anomaly_mask, info = self.detect_anomalies(data, columns)

        # Handle sampled results
        if was_sampled and len(data) < original_count:
            sample_anomaly_count = int(anomaly_mask.sum())
            sample_anomaly_ratio = sample_anomaly_count / len(data)
            estimated_total_anomalies = int(sample_anomaly_ratio * original_count)
            anomaly_count = estimated_total_anomalies
            anomaly_ratio = sample_anomaly_ratio
            info["sampled"] = True
            info["sample_size"] = len(data)
            info["original_count"] = original_count
        else:
            anomaly_count = int(anomaly_mask.sum())
            anomaly_ratio = anomaly_count / len(data)
            info["sampled"] = False

        if anomaly_ratio > self.max_anomaly_ratio:
            severity = self._calculate_severity(anomaly_ratio)

            sample_note = ""
            if info.get("sampled"):
                sample_note = f" (estimated from {info['sample_size']:,} samples)"

            issues.append(
                ValidationIssue(
                    column=", ".join(columns),
                    issue_type="lof_anomaly",
                    count=anomaly_count,
                    severity=severity,
                    details=(
                        f"LOF (k={info['n_neighbors']}) detected {anomaly_count:,} anomalies "
                        f"({anomaly_ratio:.2%}){sample_note}. LOF scores: mean={info['mean_lof']:.2f}, "
                        f"max={info['max_lof']:.2f}"
                    ),
                    expected=f"Anomaly ratio <= {self.max_anomaly_ratio:.2%}",
                )
            )

        return issues


@register_validator
class OneClassSVMValidator(AnomalyValidator, MLAnomalyMixin, LargeDatasetMixin):
    """One-Class SVM for anomaly detection.

    One-Class SVM learns a decision boundary around normal data.
    Points outside this boundary are classified as anomalies.

    Works well for high-dimensional data but can be slower than
    tree-based methods.

    Memory Optimization:
        SVM training is O(n^2) to O(n^3), so sampling is essential:

        validator = OneClassSVMValidator(
            columns=["feature1", "feature2"],
            nu=0.05,
            sample_size=10000,  # Train on smaller sample
            batch_size=50000,   # Score in batches
        )

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
        sample_size: int | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        auto_sample: bool = False,
        max_memory_mb: float = MEMORY_THRESHOLD_MB,
        **kwargs: Any,
    ):
        """Initialize One-Class SVM validator.

        Args:
            columns: Columns to use for detection
            kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
            nu: Upper bound on fraction of training errors and support vectors
            gamma: Kernel coefficient
            max_anomaly_ratio: Maximum acceptable ratio of anomalies
            sample_size: Max samples for training (None = use all data)
            batch_size: Batch size for scoring large datasets
            auto_sample: If True, automatically determine sample_size
            max_memory_mb: Max memory (MB) for auto_sample mode
            **kwargs: Additional config
        """
        super().__init__(columns=columns, max_anomaly_ratio=max_anomaly_ratio, **kwargs)
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self._sample_size = sample_size
        self._batch_size = batch_size
        self._auto_sample = auto_sample
        self._max_memory_mb = max_memory_mb

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

        # Determine sample size (SVM is O(n^2)-O(n^3), very memory intensive)
        sample_size = self._sample_size
        if self._auto_sample and sample_size is None:
            total_count = lf.select(pl.len()).collect().item()
            # SVM is very expensive, use aggressive sampling
            sample_size = _compute_optimal_sample_size(
                total_count, len(columns), self._max_memory_mb / 4  # Very conservative
            )
            # Cap at reasonable limit for SVM
            sample_size = min(sample_size, 20000)
            self.logger.debug(f"Auto-sample (SVM): using {sample_size} samples from {total_count}")

        # Smart sampling from LazyFrame
        data, original_count, was_sampled = self._smart_sample_lazyframe(
            lf, columns, sample_size, 42
        )

        if len(data) < 10:
            return issues

        anomaly_mask, info = self.detect_anomalies(data, columns)

        # Handle sampled results
        if was_sampled and len(data) < original_count:
            sample_anomaly_count = int(anomaly_mask.sum())
            sample_anomaly_ratio = sample_anomaly_count / len(data)
            estimated_total_anomalies = int(sample_anomaly_ratio * original_count)
            anomaly_count = estimated_total_anomalies
            anomaly_ratio = sample_anomaly_ratio
            info["sampled"] = True
            info["sample_size"] = len(data)
            info["original_count"] = original_count
        else:
            anomaly_count = int(anomaly_mask.sum())
            anomaly_ratio = anomaly_count / len(data)
            info["sampled"] = False

        if anomaly_ratio > self.max_anomaly_ratio:
            severity = self._calculate_severity(anomaly_ratio)

            sample_note = ""
            if info.get("sampled"):
                sample_note = f" (estimated from {info['sample_size']:,} samples)"

            issues.append(
                ValidationIssue(
                    column=", ".join(columns),
                    issue_type="svm_anomaly",
                    count=anomaly_count,
                    severity=severity,
                    details=(
                        f"One-Class SVM ({info['kernel']}, nu={info['nu']}) detected "
                        f"{anomaly_count:,} anomalies ({anomaly_ratio:.2%}){sample_note}. "
                        f"Support vectors: {info['n_support']}"
                    ),
                    expected=f"Anomaly ratio <= {self.max_anomaly_ratio:.2%}",
                )
            )

        return issues


@register_validator
class MemoryEfficientLOFValidator(AnomalyValidator, MLAnomalyMixin):
    """Memory-efficient LOF using approximate k-NN.

    This validator uses approximate nearest neighbor algorithms (BallTree, Annoy, HNSW)
    to compute LOF scores without building a full O(n²) distance matrix.

    Memory Complexity:
        - Standard LOF: O(n²) for distance matrix
        - This implementation: O(n) with O(log n) query time

    Use this for datasets > 50,000 rows where standard LOF would run out of memory.

    Example:
        # For large datasets (100k+ rows)
        validator = MemoryEfficientLOFValidator(
            columns=["feature1", "feature2"],
            n_neighbors=20,
            knn_backend="balltree",  # or "annoy", "hnsw" if installed
        )
    """

    name = "memory_efficient_lof"

    def __init__(
        self,
        columns: list[str] | None = None,
        n_neighbors: int = 20,
        contamination: float = 0.1,
        max_anomaly_ratio: float = 0.1,
        knn_backend: str = "auto",
        sample_size: int | None = None,
        **kwargs: Any,
    ):
        """Initialize memory-efficient LOF validator.

        Args:
            columns: Columns to use for detection
            n_neighbors: Number of neighbors for LOF
            contamination: Expected proportion of outliers
            max_anomaly_ratio: Maximum acceptable ratio of anomalies
            knn_backend: k-NN backend ('auto', 'balltree', 'kdtree', 'annoy', 'hnsw')
            sample_size: Optional sample size for very large datasets
            **kwargs: Additional config
        """
        super().__init__(columns=columns, max_anomaly_ratio=max_anomaly_ratio, **kwargs)
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.knn_backend = knn_backend
        self._sample_size = sample_size

        # Import the mixin at runtime to avoid circular imports
        from truthound.validators.memory import ApproximateKNNMixin
        self._knn_mixin = ApproximateKNNMixin()

    def detect_anomalies(
        self, data: np.ndarray, column_names: list[str]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Detect anomalies using approximate LOF."""
        # Normalize data
        normalized_data, _, _ = self.normalize_data(data)

        n_neighbors = min(self.n_neighbors, len(data) - 1)

        # Build approximate index
        backend = self.knn_backend if self.knn_backend != "auto" else None
        self._knn_mixin.build_approximate_index(
            normalized_data,
            backend=backend,
            metric="euclidean",
        )

        # Compute LOF scores using approximate k-NN
        lof_scores = self._knn_mixin.compute_local_outlier_factor(
            normalized_data, k=n_neighbors
        )

        # Determine threshold based on contamination
        if isinstance(self.contamination, float) and 0 < self.contamination < 0.5:
            threshold = np.percentile(lof_scores, 100 * (1 - self.contamination))
        else:
            # Auto: use 1.5 as threshold (common LOF threshold)
            threshold = 1.5

        anomaly_mask = lof_scores > threshold

        # Clear index to free memory
        self._knn_mixin.clear_index()

        return anomaly_mask, {
            "n_neighbors": n_neighbors,
            "min_lof": float(np.min(lof_scores)),
            "max_lof": float(np.max(lof_scores)),
            "mean_lof": float(np.mean(lof_scores)),
            "threshold": float(threshold),
            "backend": str(self._knn_mixin._knn_backend),
        }

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        columns = self._get_anomaly_columns(lf)
        if not columns:
            return issues

        # Sample if needed
        if self._sample_size:
            sample_lf = lf.head(self._sample_size)
            df = sample_lf.select([pl.col(c) for c in columns]).drop_nulls().collect()
        else:
            df = lf.select([pl.col(c) for c in columns]).drop_nulls().collect()

        if len(df) < self.n_neighbors + 1:
            return issues

        data = df.to_numpy()
        anomaly_mask, info = self.detect_anomalies(data, columns)

        anomaly_count = int(anomaly_mask.sum())
        anomaly_ratio = anomaly_count / len(data)

        if anomaly_ratio > self.max_anomaly_ratio:
            severity = self._calculate_severity(anomaly_ratio)

            issues.append(
                ValidationIssue(
                    column=", ".join(columns),
                    issue_type="memory_efficient_lof_anomaly",
                    count=anomaly_count,
                    severity=severity,
                    details=(
                        f"Approximate LOF (k={info['n_neighbors']}, backend={info['backend']}) "
                        f"detected {anomaly_count:,} anomalies ({anomaly_ratio:.2%}). "
                        f"LOF scores: mean={info['mean_lof']:.2f}, max={info['max_lof']:.2f}"
                    ),
                    expected=f"Anomaly ratio <= {self.max_anomaly_ratio:.2%}",
                )
            )

        return issues


@register_validator
class OnlineSVMValidator(AnomalyValidator, MLAnomalyMixin):
    """Online One-Class SVM using SGD for memory-efficient training.

    This validator uses kernel approximation (Nystroem) and SGD optimization
    to train One-Class SVM incrementally, avoiding the O(n²) kernel matrix.

    Memory Complexity:
        - Standard SVM: O(n²) for kernel matrix
        - This implementation: O(n_components × n_features) constant

    Use this for datasets > 20,000 rows where standard SVM would run out of memory.

    Example:
        # For large datasets
        validator = OnlineSVMValidator(
            columns=["feature1", "feature2"],
            nu=0.05,
            n_components=100,  # Kernel approximation components
        )
    """

    name = "online_svm"

    def __init__(
        self,
        columns: list[str] | None = None,
        nu: float = 0.05,
        n_components: int = 100,
        kernel_approx: str = "nystroem",
        max_anomaly_ratio: float = 0.1,
        n_iterations: int = 3,
        batch_size: int = 1000,
        **kwargs: Any,
    ):
        """Initialize online SVM validator.

        Args:
            columns: Columns to use for detection
            nu: Upper bound on fraction of outliers
            n_components: Number of kernel approximation components
            kernel_approx: Kernel approximation method ('nystroem' or 'rbf_sampler')
            max_anomaly_ratio: Maximum acceptable ratio of anomalies
            n_iterations: Number of passes through data
            batch_size: Mini-batch size for training
            **kwargs: Additional config
        """
        super().__init__(columns=columns, max_anomaly_ratio=max_anomaly_ratio, **kwargs)
        self.nu = nu
        self.n_components = n_components
        self.kernel_approx = kernel_approx
        self.n_iterations = n_iterations
        self.batch_size = batch_size

    def detect_anomalies(
        self, data: np.ndarray, column_names: list[str]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Detect anomalies using online SVM."""
        from truthound.validators.memory import SGDOneClassSVM

        # Normalize data
        normalized_data, _, _ = self.normalize_data(data)

        # Create online SVM
        model = SGDOneClassSVM(
            nu=self.nu,
            n_components=min(self.n_components, len(data)),
            kernel_approx=self.kernel_approx,
        )

        # Train incrementally
        n_samples = len(normalized_data)
        for _ in range(self.n_iterations):
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                model.partial_fit(normalized_data[start:end])

        # Predict
        predictions = model.predict(normalized_data)
        anomaly_mask = predictions == -1

        # Get decision scores
        scores = model.decision_function(normalized_data)

        return anomaly_mask, {
            "nu": self.nu,
            "n_components": self.n_components,
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
        }

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        columns = self._get_anomaly_columns(lf)
        if not columns:
            return issues

        df = lf.select([pl.col(c) for c in columns]).drop_nulls().collect()

        if len(df) < 10:
            return issues

        data = df.to_numpy()
        anomaly_mask, info = self.detect_anomalies(data, columns)

        anomaly_count = int(anomaly_mask.sum())
        anomaly_ratio = anomaly_count / len(data)

        if anomaly_ratio > self.max_anomaly_ratio:
            severity = self._calculate_severity(anomaly_ratio)

            issues.append(
                ValidationIssue(
                    column=", ".join(columns),
                    issue_type="online_svm_anomaly",
                    count=anomaly_count,
                    severity=severity,
                    details=(
                        f"Online SVM (nu={info['nu']}, components={info['n_components']}) "
                        f"detected {anomaly_count:,} anomalies ({anomaly_ratio:.2%}). "
                        f"Score range: [{info['min_score']:.4f}, {info['max_score']:.4f}]"
                    ),
                    expected=f"Anomaly ratio <= {self.max_anomaly_ratio:.2%}",
                )
            )

        return issues


@register_validator
class DBSCANAnomalyValidator(AnomalyValidator, MLAnomalyMixin, LargeDatasetMixin):
    """DBSCAN-based anomaly detection.

    DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    identifies outliers as noise points that don't belong to any cluster.

    Best for discovering clusters of arbitrary shape while identifying
    noise points as anomalies.

    Memory Optimization:
        DBSCAN requires pairwise distance computation. For large datasets:

        validator = DBSCANAnomalyValidator(
            columns=["x", "y"],
            eps=0.5,
            sample_size=50000,  # Sample for large datasets
        )

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
        sample_size: int | None = None,
        auto_sample: bool = False,
        max_memory_mb: float = MEMORY_THRESHOLD_MB,
        **kwargs: Any,
    ):
        """Initialize DBSCAN anomaly validator.

        Args:
            columns: Columns to use for detection
            eps: Maximum distance between points in a cluster
            min_samples: Minimum number of points for a core point
            metric: Distance metric
            max_anomaly_ratio: Maximum acceptable ratio of anomalies
            sample_size: Max samples for processing (None = use all data)
            auto_sample: If True, automatically determine sample_size
            max_memory_mb: Max memory (MB) for auto_sample mode
            **kwargs: Additional config
        """
        super().__init__(columns=columns, max_anomaly_ratio=max_anomaly_ratio, **kwargs)
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self._sample_size = sample_size
        self._auto_sample = auto_sample
        self._max_memory_mb = max_memory_mb

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

        # Determine sample size (DBSCAN needs O(n^2) distance computations)
        sample_size = self._sample_size
        if self._auto_sample and sample_size is None:
            total_count = lf.select(pl.len()).collect().item()
            # DBSCAN is memory intensive, use conservative sampling
            sample_size = _compute_optimal_sample_size(
                total_count, len(columns), self._max_memory_mb / 2
            )
            # Cap at reasonable limit
            sample_size = min(sample_size, 50000)
            self.logger.debug(f"Auto-sample (DBSCAN): using {sample_size} samples from {total_count}")

        # Smart sampling from LazyFrame
        data, original_count, was_sampled = self._smart_sample_lazyframe(
            lf, columns, sample_size, 42
        )

        if len(data) < self.min_samples:
            return issues

        anomaly_mask, info = self.detect_anomalies(data, columns)

        # Handle sampled results
        if was_sampled and len(data) < original_count:
            sample_anomaly_count = int(anomaly_mask.sum())
            sample_anomaly_ratio = sample_anomaly_count / len(data)
            estimated_total_anomalies = int(sample_anomaly_ratio * original_count)
            anomaly_count = estimated_total_anomalies
            anomaly_ratio = sample_anomaly_ratio
            info["sampled"] = True
            info["sample_size"] = len(data)
            info["original_count"] = original_count
        else:
            anomaly_count = int(anomaly_mask.sum())
            anomaly_ratio = anomaly_count / len(data)
            info["sampled"] = False

        if anomaly_ratio > self.max_anomaly_ratio:
            severity = self._calculate_severity(anomaly_ratio)

            sample_note = ""
            if info.get("sampled"):
                sample_note = f" (estimated from {info['sample_size']:,} samples)"

            issues.append(
                ValidationIssue(
                    column=", ".join(columns),
                    issue_type="dbscan_anomaly",
                    count=anomaly_count,
                    severity=severity,
                    details=(
                        f"DBSCAN (eps={info['eps']}, min_samples={info['min_samples']}) "
                        f"found {info['n_clusters']} clusters and {anomaly_count:,} noise points "
                        f"({anomaly_ratio:.2%}){sample_note}"
                    ),
                    expected=f"Anomaly ratio <= {self.max_anomaly_ratio:.2%}",
                )
            )

        return issues
