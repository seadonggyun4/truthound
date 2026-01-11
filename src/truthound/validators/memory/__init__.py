"""Memory-efficient validation abstractions.

This module provides extensible mixins and base classes for handling large datasets
with constrained memory. Each mixin addresses specific memory bottleneck patterns.

Architecture Overview:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    MemoryEfficientMixin                         │
    │  (Base: sampling, batching, memory estimation, auto-config)     │
    └─────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
    ┌─────────────┐         ┌─────────────┐         ┌─────────────────┐
    │ApproximateKNN│         │  SGDOnline  │         │ StreamingECDF   │
    │   Mixin     │         │   Mixin     │         │    Mixin        │
    │(LOF,DBSCAN) │         │ (SVM,Linear)│         │ (KS,Stats tests)│
    └─────────────┘         └─────────────┘         └─────────────────┘

Memory Problem → Solution Mapping:
    - IsolationForest: df.to_numpy() full load → Sampling + Batch scoring
    - LOF: Full data + k-NN O(n²) → Approximate k-NN (Ball Tree, Annoy)
    - OneClassSVM: Full fit O(n²-n³) → SGD-based online learning
    - KSTest: Two datasets loaded → Streaming ECDF computation
    - PSI: Histogram computation → Already efficient (reference only)

Usage:
    from truthound.validators.memory import (
        MemoryEfficientMixin,
        ApproximateKNNMixin,
        SGDOnlineMixin,
        StreamingECDFMixin,
        MemoryConfig,
        estimate_memory_usage,
    )

    class MyValidator(AnomalyValidator, MemoryEfficientMixin, ApproximateKNNMixin):
        def validate(self, lf):
            # Auto-configure based on data size
            self.auto_configure_memory(lf)

            # Use approximate k-NN for neighbor-based methods
            neighbors = self.find_approximate_neighbors(data, k=20)
            ...
"""

from truthound.validators.memory.base import (
    MemoryEfficientMixin,
    MemoryConfig,
    MemoryStrategy,
    estimate_memory_usage,
    get_available_memory,
    DataChunker,
)
from truthound.validators.memory.approximate_knn import (
    ApproximateKNNMixin,
    KNNBackend,
    ApproximateNeighborResult,
)
from truthound.validators.memory.sgd_online import (
    SGDOnlineMixin,
    OnlineLearnerConfig,
    IncrementalModel,
    SGDOneClassSVM,
    OnlineScaler,
    OnlineStatistics,
    IncrementalMahalanobis,
)
from truthound.validators.memory.streaming_ecdf import (
    StreamingECDFMixin,
    StreamingECDF,
    StreamingStatistics,
    TDigest,
)

__all__ = [
    # Base memory utilities
    "MemoryEfficientMixin",
    "MemoryConfig",
    "MemoryStrategy",
    "estimate_memory_usage",
    "get_available_memory",
    "DataChunker",
    # Approximate k-NN
    "ApproximateKNNMixin",
    "KNNBackend",
    "ApproximateNeighborResult",
    # SGD/Online learning
    "SGDOnlineMixin",
    "OnlineLearnerConfig",
    "IncrementalModel",
    "SGDOneClassSVM",
    "OnlineScaler",
    "OnlineStatistics",
    "IncrementalMahalanobis",
    # Streaming ECDF
    "StreamingECDFMixin",
    "StreamingECDF",
    "StreamingStatistics",
    "TDigest",
]
