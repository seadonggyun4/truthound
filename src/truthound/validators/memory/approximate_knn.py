"""Approximate k-NN algorithms for memory-efficient neighbor-based methods.

This module provides approximate nearest neighbor implementations that
scale to large datasets where exact k-NN is infeasible due to O(n²) memory.

Supported Backends:
    - BallTree: sklearn's BallTree (no extra dependencies)
    - Annoy: Spotify's Annoy library (optional, very fast)
    - HNSW: Hierarchical Navigable Small World (optional, best recall)
    - Faiss: Facebook's Faiss library (optional, GPU support)

Memory Complexity:
    - Exact k-NN: O(n²) for distance matrix
    - BallTree: O(n) with O(log n) query time
    - Annoy: O(n × n_trees) with O(log n) query time
    - HNSW: O(n × M) with O(log n) query time

Usage:
    class MyValidator(AnomalyValidator, ApproximateKNNMixin):
        def validate(self, lf):
            # Build index
            self.build_approximate_index(data, backend="balltree")

            # Find neighbors
            neighbors, distances = self.find_approximate_neighbors(query, k=20)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, TYPE_CHECKING
import warnings

import numpy as np

if TYPE_CHECKING:
    from sklearn.neighbors import BallTree, KDTree


class KNNBackend(Enum):
    """Available k-NN backend implementations."""

    BALLTREE = auto()  # sklearn BallTree (default, no extra deps)
    KDTREE = auto()  # sklearn KDTree (faster for low dimensions)
    ANNOY = auto()  # Spotify Annoy (very fast, approximate)
    HNSW = auto()  # hnswlib (best recall/speed tradeoff)
    FAISS = auto()  # Facebook Faiss (GPU support)
    EXACT = auto()  # Exact brute-force (for small datasets)


@dataclass
class ApproximateNeighborResult:
    """Result from approximate neighbor search.

    Attributes:
        indices: Neighbor indices (n_queries, k)
        distances: Neighbor distances (n_queries, k)
        is_approximate: Whether results are approximate
        recall_estimate: Estimated recall (1.0 for exact)
    """

    indices: np.ndarray
    distances: np.ndarray
    is_approximate: bool = True
    recall_estimate: float = 0.95


def _check_annoy_available() -> bool:
    """Check if Annoy is available."""
    try:
        import annoy  # noqa: F401
        return True
    except ImportError:
        return False


def _check_hnswlib_available() -> bool:
    """Check if hnswlib is available."""
    try:
        import hnswlib  # noqa: F401
        return True
    except ImportError:
        return False


def _check_faiss_available() -> bool:
    """Check if Faiss is available."""
    try:
        import faiss  # noqa: F401
        return True
    except ImportError:
        return False


class ApproximateKNNMixin:
    """Mixin providing approximate k-NN functionality.

    This mixin enables memory-efficient neighbor-based algorithms like
    LOF and DBSCAN to scale to large datasets by using approximate
    nearest neighbor indices instead of exact distance matrices.

    The mixin automatically selects the best available backend based on
    installed libraries and dataset characteristics.

    Example:
        class MemoryEfficientLOF(AnomalyValidator, ApproximateKNNMixin):
            def validate(self, lf):
                data, total, sampled = self.smart_sample(lf, columns)

                # Build approximate index
                self.build_approximate_index(data)

                # Compute LOF using approximate neighbors
                lof_scores = self._compute_lof_approximate(data, k=20)
                ...
    """

    # Index storage
    _knn_index: Any = None
    _knn_backend: KNNBackend | None = None
    _knn_data: np.ndarray | None = None  # Only for BallTree/KDTree

    def get_best_backend(
        self,
        n_samples: int,
        n_features: int,
        prefer_exact: bool = False,
    ) -> KNNBackend:
        """Select the best available k-NN backend.

        Args:
            n_samples: Number of data points
            n_features: Number of dimensions
            prefer_exact: If True, prefer exact methods for small datasets

        Returns:
            Best available backend
        """
        # For small datasets, exact is fine
        if prefer_exact and n_samples < 5000:
            return KNNBackend.EXACT

        # Check available backends in order of preference
        if _check_faiss_available():
            return KNNBackend.FAISS
        if _check_hnswlib_available():
            return KNNBackend.HNSW
        if _check_annoy_available():
            return KNNBackend.ANNOY

        # Fallback to sklearn
        if n_features <= 20:
            return KNNBackend.KDTREE
        return KNNBackend.BALLTREE

    def build_approximate_index(
        self,
        data: np.ndarray,
        backend: KNNBackend | str | None = None,
        metric: str = "euclidean",
        **kwargs: Any,
    ) -> None:
        """Build approximate nearest neighbor index.

        Args:
            data: Training data (n_samples, n_features)
            backend: Backend to use (auto-detected if None)
            metric: Distance metric
            **kwargs: Backend-specific parameters
        """
        n_samples, n_features = data.shape

        # Select backend
        if backend is None:
            backend = self.get_best_backend(n_samples, n_features)
        elif isinstance(backend, str):
            backend = KNNBackend[backend.upper()]

        self._knn_backend = backend

        # Build index based on backend
        if backend == KNNBackend.BALLTREE:
            self._build_balltree(data, metric, **kwargs)
        elif backend == KNNBackend.KDTREE:
            self._build_kdtree(data, metric, **kwargs)
        elif backend == KNNBackend.ANNOY:
            self._build_annoy(data, metric, **kwargs)
        elif backend == KNNBackend.HNSW:
            self._build_hnsw(data, metric, **kwargs)
        elif backend == KNNBackend.FAISS:
            self._build_faiss(data, metric, **kwargs)
        elif backend == KNNBackend.EXACT:
            # Just store data for brute force
            self._knn_data = data
        else:
            raise ValueError(f"Unknown backend: {backend}")

        if hasattr(self, "logger"):
            self.logger.debug(
                f"Built {backend.name} index: {n_samples} samples, {n_features} features"
            )

    def _build_balltree(
        self,
        data: np.ndarray,
        metric: str,
        leaf_size: int = 40,
        **kwargs: Any,
    ) -> None:
        """Build sklearn BallTree index."""
        from sklearn.neighbors import BallTree

        # Map common metric names
        metric_map = {"euclidean": "euclidean", "manhattan": "manhattan", "l2": "euclidean", "l1": "manhattan"}
        metric = metric_map.get(metric, metric)

        self._knn_index = BallTree(data, leaf_size=leaf_size, metric=metric)
        self._knn_data = data

    def _build_kdtree(
        self,
        data: np.ndarray,
        metric: str,
        leaf_size: int = 40,
        **kwargs: Any,
    ) -> None:
        """Build sklearn KDTree index."""
        from sklearn.neighbors import KDTree

        if metric not in ("euclidean", "l2", "minkowski"):
            warnings.warn(f"KDTree only supports Minkowski metrics, falling back to euclidean")

        self._knn_index = KDTree(data, leaf_size=leaf_size, metric="euclidean")
        self._knn_data = data

    def _build_annoy(
        self,
        data: np.ndarray,
        metric: str,
        n_trees: int = 50,
        **kwargs: Any,
    ) -> None:
        """Build Annoy index."""
        import annoy

        n_samples, n_features = data.shape

        # Map metric names
        metric_map = {"euclidean": "euclidean", "manhattan": "manhattan", "l2": "euclidean", "cosine": "angular"}
        annoy_metric = metric_map.get(metric, "euclidean")

        index = annoy.AnnoyIndex(n_features, annoy_metric)

        for i in range(n_samples):
            index.add_item(i, data[i])

        index.build(n_trees)
        self._knn_index = index
        self._knn_data = None  # Annoy stores its own data

    def _build_hnsw(
        self,
        data: np.ndarray,
        metric: str,
        M: int = 16,
        ef_construction: int = 200,
        **kwargs: Any,
    ) -> None:
        """Build hnswlib index."""
        import hnswlib

        n_samples, n_features = data.shape

        # Map metric names
        metric_map = {"euclidean": "l2", "l2": "l2", "cosine": "cosine", "ip": "ip"}
        hnsw_metric = metric_map.get(metric, "l2")

        index = hnswlib.Index(space=hnsw_metric, dim=n_features)
        index.init_index(max_elements=n_samples, ef_construction=ef_construction, M=M)
        index.add_items(data, np.arange(n_samples))

        self._knn_index = index
        self._knn_data = None

    def _build_faiss(
        self,
        data: np.ndarray,
        metric: str,
        nlist: int = 100,
        use_gpu: bool = False,
        **kwargs: Any,
    ) -> None:
        """Build Faiss index."""
        import faiss

        n_samples, n_features = data.shape

        # Ensure data is float32 (Faiss requirement)
        data = data.astype(np.float32)

        # Choose index type based on dataset size
        if n_samples < 10000:
            # Flat index for small datasets
            if metric == "cosine":
                faiss.normalize_L2(data)
            index = faiss.IndexFlatL2(n_features)
        else:
            # IVF index for larger datasets
            quantizer = faiss.IndexFlatL2(n_features)
            nlist = min(nlist, n_samples // 10)
            index = faiss.IndexIVFFlat(quantizer, n_features, nlist)
            index.train(data)

        index.add(data)

        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception:
                warnings.warn("GPU not available for Faiss, using CPU")

        self._knn_index = index
        self._knn_data = data  # Keep for reference

    def find_approximate_neighbors(
        self,
        query: np.ndarray,
        k: int,
        ef_search: int = 100,
    ) -> ApproximateNeighborResult:
        """Find k approximate nearest neighbors.

        Args:
            query: Query points (n_queries, n_features) or (n_features,)
            k: Number of neighbors
            ef_search: Search effort (for HNSW)

        Returns:
            ApproximateNeighborResult with indices and distances
        """
        if self._knn_index is None and self._knn_data is None:
            raise RuntimeError("Index not built. Call build_approximate_index first.")

        # Handle single query
        if query.ndim == 1:
            query = query.reshape(1, -1)

        backend = self._knn_backend

        if backend == KNNBackend.BALLTREE or backend == KNNBackend.KDTREE:
            distances, indices = self._knn_index.query(query, k=k)
            return ApproximateNeighborResult(
                indices=indices,
                distances=distances,
                is_approximate=False,
                recall_estimate=1.0,
            )

        elif backend == KNNBackend.ANNOY:
            n_queries = len(query)
            indices = np.zeros((n_queries, k), dtype=np.int64)
            distances = np.zeros((n_queries, k), dtype=np.float64)

            for i in range(n_queries):
                idx, dist = self._knn_index.get_nns_by_vector(
                    query[i], k, include_distances=True
                )
                indices[i] = idx
                distances[i] = dist

            return ApproximateNeighborResult(
                indices=indices,
                distances=distances,
                is_approximate=True,
                recall_estimate=0.95,
            )

        elif backend == KNNBackend.HNSW:
            self._knn_index.set_ef(ef_search)
            indices, distances = self._knn_index.knn_query(query, k=k)

            return ApproximateNeighborResult(
                indices=indices,
                distances=np.sqrt(distances),  # hnswlib returns squared distances
                is_approximate=True,
                recall_estimate=0.98,
            )

        elif backend == KNNBackend.FAISS:
            query = query.astype(np.float32)
            distances, indices = self._knn_index.search(query, k)

            return ApproximateNeighborResult(
                indices=indices,
                distances=np.sqrt(distances),  # Faiss returns squared L2
                is_approximate=True,
                recall_estimate=0.99,
            )

        elif backend == KNNBackend.EXACT:
            # Brute force for small datasets
            from sklearn.neighbors import NearestNeighbors

            nn = NearestNeighbors(n_neighbors=k, algorithm="brute")
            nn.fit(self._knn_data)
            distances, indices = nn.kneighbors(query)

            return ApproximateNeighborResult(
                indices=indices,
                distances=distances,
                is_approximate=False,
                recall_estimate=1.0,
            )

        raise RuntimeError(f"Unknown backend: {backend}")

    def compute_local_reachability_density(
        self,
        data: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """Compute Local Reachability Density (LRD) for LOF.

        Memory-efficient implementation using approximate k-NN.

        Args:
            data: Data points
            k: Number of neighbors

        Returns:
            Array of LRD values
        """
        n_samples = len(data)

        # Find k-neighbors for all points
        result = self.find_approximate_neighbors(data, k=k + 1)  # +1 to exclude self

        # Remove self from neighbors
        indices = result.indices[:, 1:]  # Skip first (self)
        distances = result.distances[:, 1:]

        # Compute reachability distances
        # reach_dist(p, o) = max(k-dist(o), d(p, o))
        k_distances = distances[:, -1]  # k-th distance for each point

        reach_dists = np.zeros((n_samples, k))
        for i in range(n_samples):
            for j in range(k):
                neighbor_idx = indices[i, j]
                reach_dists[i, j] = max(k_distances[neighbor_idx], distances[i, j])

        # Compute LRD
        # LRD(p) = 1 / (mean reachability distance to neighbors)
        mean_reach_dists = reach_dists.mean(axis=1)
        lrd = np.where(mean_reach_dists > 0, 1.0 / mean_reach_dists, 0.0)

        return lrd

    def compute_local_outlier_factor(
        self,
        data: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """Compute Local Outlier Factor (LOF) scores.

        Memory-efficient implementation using approximate k-NN.

        Args:
            data: Data points
            k: Number of neighbors

        Returns:
            Array of LOF scores (higher = more outlier)
        """
        n_samples = len(data)

        # Build index if not already built
        if self._knn_index is None:
            self.build_approximate_index(data)

        # Compute LRD for all points
        lrd = self.compute_local_reachability_density(data, k)

        # Find neighbors
        result = self.find_approximate_neighbors(data, k=k + 1)
        indices = result.indices[:, 1:]  # Skip self

        # Compute LOF
        # LOF(p) = mean(LRD(neighbors)) / LRD(p)
        lof_scores = np.zeros(n_samples)
        for i in range(n_samples):
            neighbor_lrds = lrd[indices[i]]
            if lrd[i] > 0:
                lof_scores[i] = neighbor_lrds.mean() / lrd[i]
            else:
                lof_scores[i] = 1.0  # No local density

        return lof_scores

    def clear_index(self) -> None:
        """Clear the k-NN index to free memory."""
        self._knn_index = None
        self._knn_data = None
        self._knn_backend = None
