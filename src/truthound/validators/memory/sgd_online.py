"""SGD-based online learning for memory-efficient model training.

This module provides incremental/online learning implementations for
algorithms that traditionally require O(n²) or O(n³) memory for training.

Key Algorithms:
    - SGDOneClassSVM: Online One-Class SVM using SGD
    - IncrementalPCA: Streaming PCA for dimensionality reduction
    - OnlineIsolationForest: Incremental tree building

Memory Complexity:
    - Traditional SVM: O(n²) for kernel matrix
    - SGD SVM: O(1) per sample, O(d) for model weights

Usage:
    class MemoryEfficientSVM(AnomalyValidator, SGDOnlineMixin):
        def validate(self, lf):
            # Stream data through online learner
            model = self.create_online_svm()
            for chunk in self.iterate_chunks(lf):
                model.partial_fit(chunk)

            # Predict on new data
            predictions = model.predict(current_data)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Iterator, Protocol, TYPE_CHECKING
import warnings

import numpy as np

if TYPE_CHECKING:
    import polars as pl


class OnlineLearnerType(Enum):
    """Type of online learning algorithm."""

    SGD_SVM = auto()  # SGD-based One-Class SVM
    INCREMENTAL_PCA = auto()  # Streaming PCA
    MINI_BATCH_KMEANS = auto()  # Mini-batch K-Means
    ONLINE_COVARIANCE = auto()  # Streaming covariance estimation


@dataclass
class OnlineLearnerConfig:
    """Configuration for online learning algorithms.

    Attributes:
        learning_rate: Initial learning rate
        learning_rate_schedule: Schedule type ('constant', 'optimal', 'invscaling')
        n_iterations: Number of passes through data
        batch_size: Mini-batch size for partial_fit
        regularization: L2 regularization strength
        random_state: Random seed
        warm_start: Whether to continue from previous fit
        tol: Tolerance for convergence
    """

    learning_rate: float = 0.001
    learning_rate_schedule: str = "optimal"
    n_iterations: int = 5
    batch_size: int = 1000
    regularization: float = 0.0001
    random_state: int = 42
    warm_start: bool = True
    tol: float = 1e-4

    # SVM-specific
    nu: float = 0.1  # Upper bound on outlier fraction
    kernel_approx: str = "nystroem"  # Kernel approximation method
    n_components: int = 100  # Number of kernel components


class IncrementalModel(Protocol):
    """Protocol for incremental learning models."""

    def partial_fit(self, X: np.ndarray) -> "IncrementalModel":
        """Fit on a batch of data."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on data."""
        ...


class OnlineStatistics:
    """Welford's online algorithm for computing running statistics.

    Computes mean, variance, and standard deviation in a single pass
    with O(1) memory per feature.

    Example:
        stats = OnlineStatistics(n_features=10)
        for batch in data_stream:
            stats.update(batch)
        mean, std = stats.mean, stats.std
    """

    def __init__(self, n_features: int):
        """Initialize online statistics tracker.

        Args:
            n_features: Number of features
        """
        self.n_features = n_features
        self.n_samples = 0
        self._mean = np.zeros(n_features)
        self._M2 = np.zeros(n_features)  # Sum of squared differences
        self._min = np.full(n_features, np.inf)
        self._max = np.full(n_features, -np.inf)

    def update(self, X: np.ndarray) -> None:
        """Update statistics with new batch.

        Args:
            X: Data batch (n_samples, n_features)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        for x in X:
            self.n_samples += 1
            delta = x - self._mean
            self._mean += delta / self.n_samples
            delta2 = x - self._mean
            self._M2 += delta * delta2
            self._min = np.minimum(self._min, x)
            self._max = np.maximum(self._max, x)

    def update_batch(self, X: np.ndarray) -> None:
        """Batch update using parallel algorithm.

        More efficient than individual updates for large batches.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_batch = len(X)
        batch_mean = X.mean(axis=0)
        batch_var = X.var(axis=0, ddof=0)

        # Combine with existing statistics
        if self.n_samples == 0:
            self._mean = batch_mean
            self._M2 = batch_var * n_batch
        else:
            n_total = self.n_samples + n_batch
            delta = batch_mean - self._mean

            self._mean = (self.n_samples * self._mean + n_batch * batch_mean) / n_total
            self._M2 += batch_var * n_batch + delta**2 * self.n_samples * n_batch / n_total

        self.n_samples += n_batch
        self._min = np.minimum(self._min, X.min(axis=0))
        self._max = np.maximum(self._max, X.max(axis=0))

    @property
    def mean(self) -> np.ndarray:
        """Get current mean."""
        return self._mean.copy()

    @property
    def variance(self) -> np.ndarray:
        """Get current variance."""
        if self.n_samples < 2:
            return np.zeros(self.n_features)
        return self._M2 / (self.n_samples - 1)

    @property
    def std(self) -> np.ndarray:
        """Get current standard deviation."""
        return np.sqrt(self.variance)

    @property
    def min(self) -> np.ndarray:
        """Get minimum values."""
        return self._min.copy()

    @property
    def max(self) -> np.ndarray:
        """Get maximum values."""
        return self._max.copy()


class OnlineScaler:
    """Online standardization scaler.

    Computes scaling parameters incrementally and can transform data
    using the running mean and standard deviation.

    Example:
        scaler = OnlineScaler()
        for batch in training_data:
            scaler.partial_fit(batch)
        scaled = scaler.transform(new_data)
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        """Initialize scaler.

        Args:
            with_mean: Whether to center data
            with_std: Whether to scale by std
        """
        self.with_mean = with_mean
        self.with_std = with_std
        self._stats: OnlineStatistics | None = None

    def partial_fit(self, X: np.ndarray) -> "OnlineScaler":
        """Update scaler with new data.

        Args:
            X: Data batch (n_samples, n_features)

        Returns:
            self
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self._stats is None:
            self._stats = OnlineStatistics(X.shape[1])

        self._stats.update_batch(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using learned parameters.

        Args:
            X: Data to transform

        Returns:
            Transformed data
        """
        if self._stats is None:
            raise RuntimeError("Scaler not fitted. Call partial_fit first.")

        result = X.copy()
        if self.with_mean:
            result = result - self._stats.mean
        if self.with_std:
            std = self._stats.std
            std = np.where(std == 0, 1, std)
            result = result / std
        return result

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.partial_fit(X)
        return self.transform(X)

    @property
    def mean_(self) -> np.ndarray:
        """Get learned mean."""
        if self._stats is None:
            raise RuntimeError("Scaler not fitted")
        return self._stats.mean

    @property
    def scale_(self) -> np.ndarray:
        """Get learned scale (std)."""
        if self._stats is None:
            raise RuntimeError("Scaler not fitted")
        return self._stats.std


class SGDOneClassSVM:
    """SGD-based One-Class SVM for online anomaly detection.

    This implementation uses:
    1. Kernel approximation (Nystroem or RBF Sampler) for scalability
    2. SGD optimization for online learning
    3. Linear SVM in the approximated feature space

    Memory: O(n_components × n_features) instead of O(n_samples²)

    Example:
        model = SGDOneClassSVM(nu=0.05, n_components=100)
        for batch in data_stream:
            model.partial_fit(batch)
        predictions = model.predict(test_data)  # -1 for outliers
    """

    def __init__(
        self,
        nu: float = 0.1,
        kernel_approx: str = "nystroem",
        n_components: int = 100,
        gamma: float | str = "scale",
        learning_rate: str = "optimal",
        eta0: float = 0.01,
        random_state: int = 42,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ):
        """Initialize SGD One-Class SVM.

        Args:
            nu: Upper bound on fraction of outliers (0 < nu <= 0.5)
            kernel_approx: Kernel approximation ('nystroem' or 'rbf_sampler')
            n_components: Number of kernel components
            gamma: Kernel coefficient ('scale', 'auto', or float)
            learning_rate: Learning rate schedule
            eta0: Initial learning rate
            random_state: Random seed
            max_iter: Maximum iterations for SGD
            tol: Tolerance for convergence
        """
        self.nu = nu
        self.kernel_approx = kernel_approx
        self.n_components = n_components
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol

        self._kernel_transformer = None
        self._sgd_classifier = None
        self._scaler = None
        self._is_fitted = False
        self._n_features = None

    def _init_models(self, X: np.ndarray) -> None:
        """Initialize internal models on first fit."""
        from sklearn.kernel_approximation import Nystroem, RBFSampler
        from sklearn.linear_model import SGDClassifier

        self._n_features = X.shape[1]

        # Compute gamma if needed
        gamma = self.gamma
        if gamma == "scale":
            gamma = 1.0 / (self._n_features * X.var())
        elif gamma == "auto":
            gamma = 1.0 / self._n_features

        # Initialize kernel approximation
        if self.kernel_approx == "nystroem":
            self._kernel_transformer = Nystroem(
                kernel="rbf",
                gamma=gamma,
                n_components=min(self.n_components, len(X)),
                random_state=self.random_state,
            )
        else:
            self._kernel_transformer = RBFSampler(
                gamma=gamma,
                n_components=self.n_components,
                random_state=self.random_state,
            )

        # Initialize SGD classifier
        # Use hinge loss for SVM-like behavior
        self._sgd_classifier = SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=0.0001,
            learning_rate=self.learning_rate,
            eta0=self.eta0,
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=self.tol,
            warm_start=True,
        )

        # Initialize online scaler
        self._scaler = OnlineScaler()

    def partial_fit(self, X: np.ndarray) -> "SGDOneClassSVM":
        """Incrementally fit the model on a batch.

        For One-Class SVM, we generate synthetic outliers and train
        a binary classifier to separate normal data from outliers.

        Args:
            X: Training data (n_samples, n_features)

        Returns:
            self
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Initialize on first call
        if not self._is_fitted:
            self._init_models(X)
            # Fit kernel transformer on first batch
            self._kernel_transformer.fit(X)

        # Update scaler
        self._scaler.partial_fit(X)
        X_scaled = self._scaler.transform(X)

        # Transform to kernel space
        X_kernel = self._kernel_transformer.transform(X_scaled)

        # Generate synthetic outliers
        n_outliers = max(1, int(len(X) * self.nu / (1 - self.nu)))
        outliers = self._generate_outliers(X_scaled, n_outliers)
        outliers_kernel = self._kernel_transformer.transform(outliers)

        # Combine normal and outliers
        X_combined = np.vstack([X_kernel, outliers_kernel])
        y_combined = np.array([1] * len(X) + [-1] * n_outliers)

        # Partial fit SGD classifier
        self._sgd_classifier.partial_fit(X_combined, y_combined, classes=[-1, 1])
        self._is_fitted = True

        return self

    def _generate_outliers(self, X: np.ndarray, n_outliers: int) -> np.ndarray:
        """Generate synthetic outliers for training.

        Uses uniform sampling in an expanded bounding box around the data.
        """
        rng = np.random.default_rng(self.random_state)

        # Expand bounding box
        min_vals = X.min(axis=0)
        max_vals = X.max(axis=0)
        ranges = max_vals - min_vals
        ranges = np.where(ranges == 0, 1, ranges)

        # Sample from expanded box
        expansion = 1.5
        outliers = rng.uniform(
            min_vals - expansion * ranges,
            max_vals + expansion * ranges,
            size=(n_outliers, X.shape[1]),
        )

        return outliers

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict if samples are outliers.

        Args:
            X: Test data (n_samples, n_features)

        Returns:
            Array of predictions: 1 for normal, -1 for outlier
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call partial_fit first.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = self._scaler.transform(X)
        X_kernel = self._kernel_transformer.transform(X_scaled)

        return self._sgd_classifier.predict(X_kernel)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function values.

        Args:
            X: Test data

        Returns:
            Decision function values (positive = normal, negative = outlier)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call partial_fit first.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = self._scaler.transform(X)
        X_kernel = self._kernel_transformer.transform(X_scaled)

        return self._sgd_classifier.decision_function(X_kernel)

    def fit(self, X: np.ndarray) -> "SGDOneClassSVM":
        """Fit the model on entire dataset at once.

        For compatibility with sklearn API.
        """
        return self.partial_fit(X)


class IncrementalMahalanobis:
    """Incremental Mahalanobis distance computation.

    Maintains running mean and covariance matrix for computing
    Mahalanobis distances without storing all data.

    Memory: O(d²) for d features instead of O(n × d) for n samples.

    Example:
        detector = IncrementalMahalanobis()
        for batch in training_data:
            detector.partial_fit(batch)
        distances = detector.mahalanobis(test_data)
    """

    def __init__(self, regularization: float = 1e-6):
        """Initialize detector.

        Args:
            regularization: Regularization for covariance inversion
        """
        self.regularization = regularization
        self._n_samples = 0
        self._mean = None
        self._cov_sum = None
        self._inv_cov = None

    def partial_fit(self, X: np.ndarray) -> "IncrementalMahalanobis":
        """Update with new batch.

        Args:
            X: Data batch (n_samples, n_features)

        Returns:
            self
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_batch = len(X)
        batch_mean = X.mean(axis=0)

        if self._n_samples == 0:
            self._mean = batch_mean
            self._cov_sum = np.zeros((X.shape[1], X.shape[1]))
        else:
            # Update mean
            total = self._n_samples + n_batch
            self._mean = (self._n_samples * self._mean + n_batch * batch_mean) / total

        # Update covariance sum
        centered = X - self._mean
        self._cov_sum += centered.T @ centered
        self._n_samples += n_batch

        # Invalidate cached inverse
        self._inv_cov = None

        return self

    @property
    def covariance(self) -> np.ndarray:
        """Get current covariance matrix."""
        if self._n_samples < 2:
            raise RuntimeError("Need at least 2 samples for covariance")
        return self._cov_sum / (self._n_samples - 1)

    def _compute_inverse_covariance(self) -> np.ndarray:
        """Compute and cache inverse covariance."""
        if self._inv_cov is None:
            cov = self.covariance
            # Add regularization
            cov_reg = cov + self.regularization * np.eye(cov.shape[0])
            self._inv_cov = np.linalg.inv(cov_reg)
        return self._inv_cov

    def mahalanobis(self, X: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distances.

        Args:
            X: Test data (n_samples, n_features)

        Returns:
            Array of Mahalanobis distances
        """
        if self._n_samples < 2:
            raise RuntimeError("Model not fitted with enough samples")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        inv_cov = self._compute_inverse_covariance()
        centered = X - self._mean

        # Mahalanobis distance: sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))
        left = centered @ inv_cov
        distances = np.sqrt(np.sum(left * centered, axis=1))

        return distances

    def predict(self, X: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Predict outliers based on Mahalanobis distance.

        Args:
            X: Test data
            threshold: Distance threshold for outlier detection

        Returns:
            Array of predictions: 1 for normal, -1 for outlier
        """
        distances = self.mahalanobis(X)
        return np.where(distances > threshold, -1, 1)


class SGDOnlineMixin:
    """Mixin providing SGD-based online learning capabilities.

    This mixin enables memory-efficient training of models that
    traditionally require full data loading (SVM, covariance-based methods).

    Example:
        class MemoryEfficientSVMValidator(AnomalyValidator, SGDOnlineMixin):
            def validate(self, lf):
                # Create online SVM
                model = self.create_online_svm(nu=0.05)

                # Stream training data
                for chunk in self.iterate_training_chunks(lf):
                    model.partial_fit(chunk)

                # Predict
                predictions = model.predict(current_data)
    """

    _online_config: OnlineLearnerConfig = None
    _online_models: dict[str, IncrementalModel] = None

    def get_online_config(self) -> OnlineLearnerConfig:
        """Get online learning configuration."""
        if self._online_config is None:
            self._online_config = OnlineLearnerConfig()
        return self._online_config

    def set_online_config(self, config: OnlineLearnerConfig) -> None:
        """Set online learning configuration."""
        self._online_config = config

    def create_online_svm(
        self,
        nu: float | None = None,
        n_components: int | None = None,
        **kwargs: Any,
    ) -> SGDOneClassSVM:
        """Create SGD-based One-Class SVM.

        Args:
            nu: Upper bound on outlier fraction
            n_components: Number of kernel components
            **kwargs: Additional parameters

        Returns:
            SGDOneClassSVM instance
        """
        config = self.get_online_config()

        return SGDOneClassSVM(
            nu=nu or config.nu,
            n_components=n_components or config.n_components,
            kernel_approx=config.kernel_approx,
            learning_rate=config.learning_rate_schedule,
            eta0=config.learning_rate,
            random_state=config.random_state,
            tol=config.tol,
            **kwargs,
        )

    def create_online_scaler(self) -> OnlineScaler:
        """Create online standardization scaler."""
        return OnlineScaler()

    def create_online_statistics(self, n_features: int) -> OnlineStatistics:
        """Create online statistics tracker."""
        return OnlineStatistics(n_features)

    def create_mahalanobis_detector(
        self,
        regularization: float = 1e-6,
    ) -> IncrementalMahalanobis:
        """Create incremental Mahalanobis distance detector."""
        return IncrementalMahalanobis(regularization=regularization)

    def train_incrementally(
        self,
        lf: "pl.LazyFrame",
        columns: list[str],
        model: IncrementalModel,
        n_iterations: int | None = None,
    ) -> IncrementalModel:
        """Train model incrementally on streaming data.

        Args:
            lf: Input LazyFrame
            columns: Columns to use
            model: Incremental model with partial_fit method
            n_iterations: Number of passes through data

        Returns:
            Trained model
        """
        from truthound.validators.memory.base import DataChunker

        config = self.get_online_config()
        n_iterations = n_iterations or config.n_iterations

        chunker = DataChunker(
            chunk_size=config.batch_size,
            columns=columns,
            drop_nulls=True,
        )

        for iteration in range(n_iterations):
            for chunk_arr in chunker.iterate(lf, as_numpy=True):
                model.partial_fit(chunk_arr)

            if hasattr(self, "logger"):
                self.logger.debug(f"Completed iteration {iteration + 1}/{n_iterations}")

        return model
