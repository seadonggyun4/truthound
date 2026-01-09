"""Isolation Forest anomaly detection.

Implements the Isolation Forest algorithm for anomaly detection.
This is a pure Python implementation that doesn't require scikit-learn.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from math import log
from typing import Any

import polars as pl

from truthound.ml.base import (
    AnomalyConfig,
    AnomalyDetector,
    AnomalyResult,
    AnomalyScore,
    AnomalyType,
    InsufficientDataError,
    ModelInfo,
    ModelNotTrainedError,
    ModelState,
    ModelTrainingError,
    ModelType,
    register_model,
)


@dataclass
class IsolationForestConfig(AnomalyConfig):
    """Configuration for Isolation Forest.

    Attributes:
        n_estimators: Number of isolation trees
        max_samples: Samples per tree (None = min(256, n_samples))
        max_depth: Maximum tree depth (None = auto based on max_samples)
        bootstrap: Whether to use bootstrap sampling
    """

    n_estimators: int = 100
    max_samples: int | None = 256
    max_depth: int | None = None
    bootstrap: bool = True
    columns: list[str] | None = None


class IsolationTreeNode:
    """Node in an Isolation Tree."""

    __slots__ = [
        "split_feature",
        "split_value",
        "left",
        "right",
        "size",
        "is_leaf",
    ]

    def __init__(
        self,
        split_feature: int | None = None,
        split_value: float | None = None,
        left: "IsolationTreeNode | None" = None,
        right: "IsolationTreeNode | None" = None,
        size: int = 0,
    ):
        self.split_feature = split_feature
        self.split_value = split_value
        self.left = left
        self.right = right
        self.size = size
        self.is_leaf = split_feature is None


class IsolationTree:
    """A single Isolation Tree.

    Isolates observations by randomly selecting a feature
    and then randomly selecting a split value.
    """

    def __init__(self, max_depth: int, random_state: random.Random):
        self.max_depth = max_depth
        self.random = random_state
        self.root: IsolationTreeNode | None = None
        self.n_features: int = 0

    def fit(self, data: list[list[float]]) -> None:
        """Build the isolation tree.

        Args:
            data: List of samples, each sample is a list of feature values
        """
        if not data:
            return

        self.n_features = len(data[0])
        self.root = self._build_tree(data, depth=0)

    def _build_tree(
        self,
        data: list[list[float]],
        depth: int,
    ) -> IsolationTreeNode:
        """Recursively build tree nodes."""
        n_samples = len(data)

        # Terminal conditions
        if depth >= self.max_depth or n_samples <= 1:
            return IsolationTreeNode(size=n_samples)

        # Check if all values are the same
        if self._all_same(data):
            return IsolationTreeNode(size=n_samples)

        # Randomly select feature
        feature_idx = self.random.randint(0, self.n_features - 1)

        # Get min/max for selected feature
        feature_values = [row[feature_idx] for row in data]
        min_val = min(feature_values)
        max_val = max(feature_values)

        if min_val == max_val:
            # Try another feature
            for _ in range(self.n_features):
                feature_idx = self.random.randint(0, self.n_features - 1)
                feature_values = [row[feature_idx] for row in data]
                min_val = min(feature_values)
                max_val = max(feature_values)
                if min_val != max_val:
                    break
            else:
                return IsolationTreeNode(size=n_samples)

        # Random split value
        split_value = self.random.uniform(min_val, max_val)

        # Partition data
        left_data = [row for row in data if row[feature_idx] < split_value]
        right_data = [row for row in data if row[feature_idx] >= split_value]

        # Handle edge cases
        if not left_data or not right_data:
            return IsolationTreeNode(size=n_samples)

        return IsolationTreeNode(
            split_feature=feature_idx,
            split_value=split_value,
            left=self._build_tree(left_data, depth + 1),
            right=self._build_tree(right_data, depth + 1),
            size=n_samples,
        )

    def _all_same(self, data: list[list[float]]) -> bool:
        """Check if all samples are identical."""
        if len(data) <= 1:
            return True
        first = data[0]
        return all(row == first for row in data[1:])

    def path_length(self, sample: list[float]) -> float:
        """Compute path length for a sample.

        Args:
            sample: Feature values for a single sample

        Returns:
            Path length (depth) to isolate the sample
        """
        if self.root is None:
            return 0.0

        return self._traverse(sample, self.root, depth=0)

    def _traverse(
        self,
        sample: list[float],
        node: IsolationTreeNode,
        depth: int,
    ) -> float:
        """Traverse tree and compute path length."""
        if node.is_leaf:
            # Add expected path length for remaining samples
            return depth + self._average_path_length(node.size)

        if sample[node.split_feature] < node.split_value:  # type: ignore
            return self._traverse(sample, node.left, depth + 1)  # type: ignore
        else:
            return self._traverse(sample, node.right, depth + 1)  # type: ignore

    @staticmethod
    def _average_path_length(n: int) -> float:
        """Compute average path length for BST with n samples.

        This is the expected path length to isolate n samples.
        """
        if n <= 1:
            return 0.0
        if n == 2:
            return 1.0
        # Harmonic number approximation: H(n-1) ≈ ln(n-1) + 0.5772
        return 2.0 * (log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)


@register_model("isolation_forest")
class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest anomaly detector.

    Detects anomalies by measuring how quickly observations are isolated
    in random partitions. Anomalies are isolated faster (shorter path).

    This is a pure Python implementation for portability.
    For production with large datasets, consider using scikit-learn.

    Example:
        >>> detector = IsolationForestDetector(contamination=0.1)
        >>> detector.fit(train_data)
        >>> result = detector.predict(test_data)
    """

    def __init__(self, config: IsolationForestConfig | None = None, **kwargs: Any):
        super().__init__(config, **kwargs)
        self._trees: list[IsolationTree] = []
        self._columns: list[str] = []
        self._avg_path_length: float = 0.0

    @property
    def config(self) -> IsolationForestConfig:
        return self._config  # type: ignore

    def _default_config(self) -> IsolationForestConfig:
        return IsolationForestConfig()

    def _get_model_name(self) -> str:
        return "isolation_forest"

    def _get_description(self) -> str:
        return "Isolation Forest for unsupervised anomaly detection"

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name=self._get_model_name(),
            version="1.0.0",
            model_type=ModelType.ANOMALY_DETECTOR,
            description=self._get_description(),
            min_samples_required=10,
            tags=("ensemble", "tree-based", "unsupervised"),
        )

    def fit(self, data: pl.LazyFrame) -> None:
        """Train the Isolation Forest.

        Args:
            data: Training data
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

            if not self._columns:
                raise InsufficientDataError(
                    "No numeric columns found",
                    model_name=self.info.name,
                )

            # Convert to list of lists for training
            training_data = self._to_matrix(df)
            n_samples = len(training_data)

            # Determine parameters
            max_samples = self.config.max_samples
            if max_samples is None:
                max_samples = min(256, n_samples)
            else:
                max_samples = min(max_samples, n_samples)

            max_depth = self.config.max_depth
            if max_depth is None:
                # Recommended: ceil(log2(max_samples))
                max_depth = int(log(max_samples) / log(2)) + 1

            # Store average path length for normalization
            self._avg_path_length = IsolationTree._average_path_length(max_samples)

            # Build trees
            self._trees = []
            rng = random.Random(self.config.random_seed)

            for _ in range(self.config.n_estimators):
                tree = IsolationTree(max_depth=max_depth, random_state=rng)

                # Sample data for this tree
                if self.config.bootstrap:
                    sample_indices = [
                        rng.randint(0, n_samples - 1)
                        for _ in range(max_samples)
                    ]
                else:
                    sample_indices = rng.sample(range(n_samples), max_samples)

                tree_data = [training_data[i] for i in sample_indices]
                tree.fit(tree_data)
                self._trees.append(tree)

            self._training_samples = row_count
            self._trained_at = datetime.now()
            self._state = ModelState.TRAINED

        except Exception as e:
            self._state = ModelState.ERROR
            self._error = e
            if isinstance(e, (InsufficientDataError, ModelTrainingError)):
                raise
            raise ModelTrainingError(
                f"Failed to train Isolation Forest: {e}",
                model_name=self.info.name,
            ) from e

    def _to_matrix(self, df: pl.DataFrame) -> list[list[float]]:
        """Convert DataFrame to list of lists."""
        result = []
        for row in df.select(self._columns).iter_rows():
            # Replace None with 0.0
            result.append([float(v) if v is not None else 0.0 for v in row])
        return result

    def score(self, data: pl.LazyFrame) -> pl.Series:
        """Compute anomaly scores.

        Lower path length = more anomalous.
        Score is normalized to [0, 1] where 1 = most anomalous.

        The standard Isolation Forest score formula is: s(x, n) = 2^(-E(h(x)) / c(n))
        - Score close to 1 indicates anomaly
        - Score close to 0.5 indicates normal
        - Score close to 0 indicates very normal (long path)

        Args:
            data: Data to score

        Returns:
            Series of anomaly scores
        """
        if not self.is_trained:
            raise ModelNotTrainedError(
                "Model must be trained before scoring",
                model_name=self.info.name,
            )

        df = data.collect()
        test_data = self._to_matrix(df)

        scores = []
        for sample in test_data:
            # Average path length across all trees
            avg_path = sum(
                tree.path_length(sample) for tree in self._trees
            ) / len(self._trees)

            # Anomaly score: 2^(-avg_path / c(n))
            # Where c(n) is average path length for n samples
            if self._avg_path_length == 0:
                score = 0.5
            else:
                score = 2 ** (-avg_path / self._avg_path_length)

            scores.append(score)

        return pl.Series("anomaly_score", scores)

    def _get_threshold(self) -> float:
        """Get the threshold for anomaly classification.

        For Isolation Forest, scores are computed as 2^(-E(h(x))/c(n)):
        - Scores close to 1 indicate anomalies (short path lengths)
        - Scores close to 0.5 indicate normal points
        - Scores close to 0 indicate very normal points (long path lengths)

        The threshold is set based on contamination:
        - contamination=0.1 means top 10% highest scores are anomalies
        - A score > 0.5 generally indicates more anomalous than average
        - We use 0.5 + (1-contamination) * 0.5 to scale threshold appropriately

        For contamination=0.1: threshold = 0.5 + 0.9 * 0.5 = 0.95 (too strict)
        Instead, we use a more practical approach: threshold = 0.5 + contamination
        For contamination=0.1: threshold = 0.6 (reasonable for IF scores)
        """
        if self.config.score_threshold is not None:
            return self.config.score_threshold
        # Use contamination to set a practical threshold
        # Scores above 0.5 are more anomalous than average
        # contamination of 0.1 -> threshold of 0.6
        # contamination of 0.3 -> threshold of 0.55 (lower, catches more)
        # contamination of 0.01 -> threshold of 0.65 (higher, catches fewer)
        return 0.5 + (0.1 + (1 - self.config.contamination) * 0.05)

    def _serialize(self) -> dict[str, Any]:
        base = super()._serialize()
        base["columns"] = self._columns
        base["avg_path_length"] = self._avg_path_length
        base["n_trees"] = len(self._trees)
        # Note: Full tree serialization would require more complex logic
        return base

    def _deserialize(self, data: dict[str, Any]) -> None:
        super()._deserialize(data)
        self._columns = data.get("columns", [])
        self._avg_path_length = data.get("avg_path_length", 0.0)
        # Trees need to be retrained from data
        self._trees = []

    def get_feature_importance(self) -> dict[str, float]:
        """Compute feature importance based on split frequency.

        Returns:
            Dict mapping feature name to importance score
        """
        if not self.is_trained:
            return {}

        importance = {col: 0.0 for col in self._columns}
        total_splits = 0

        def count_splits(node: IsolationTreeNode) -> None:
            nonlocal total_splits
            if node.is_leaf:
                return
            if node.split_feature is not None:
                importance[self._columns[node.split_feature]] += 1
                total_splits += 1
            if node.left:
                count_splits(node.left)
            if node.right:
                count_splits(node.right)

        for tree in self._trees:
            if tree.root:
                count_splits(tree.root)

        # Normalize
        if total_splits > 0:
            for col in importance:
                importance[col] /= total_splits

        return importance
