"""Optimization abstractions for validator performance.

This module provides reusable mixins for optimizing common validation patterns:

Performance Improvements:
    - GraphTraversalMixin: Tarjan's SCC O(V+E), iterative DFS
    - BatchCovarianceMixin: Incremental covariance O(n), Woodbury updates
    - VectorizedGeoMixin: SIMD-friendly Haversine, batch processing
    - LazyAggregationMixin: Polars lazy evaluation, predicate pushdown

Usage:
    from truthound.validators.optimization import (
        GraphTraversalMixin,
        BatchCovarianceMixin,
        VectorizedGeoMixin,
        LazyAggregationMixin,
    )
"""

from truthound.validators.optimization.graph import (
    GraphTraversalMixin,
    TarjanSCC,
    IterativeDFS,
    TopologicalSort,
    CycleInfo,
)
from truthound.validators.optimization.covariance import (
    BatchCovarianceMixin,
    IncrementalCovariance,
    WoodburyCovariance,
    RobustCovarianceEstimator,
)
from truthound.validators.optimization.geo import (
    VectorizedGeoMixin,
    SpatialIndexMixin,
    BoundingBox,
    DistanceUnit,
    EARTH_RADIUS,
)
from truthound.validators.optimization.aggregation import (
    LazyAggregationMixin,
    AggregationResult,
    AggregationExpressionBuilder,
    JoinStrategy,
)

__all__ = [
    # Graph algorithms
    "GraphTraversalMixin",
    "TarjanSCC",
    "IterativeDFS",
    "TopologicalSort",
    "CycleInfo",
    # Covariance optimization
    "BatchCovarianceMixin",
    "IncrementalCovariance",
    "WoodburyCovariance",
    "RobustCovarianceEstimator",
    # Geo optimization
    "VectorizedGeoMixin",
    "SpatialIndexMixin",
    "BoundingBox",
    "DistanceUnit",
    "EARTH_RADIUS",
    # Aggregation optimization
    "LazyAggregationMixin",
    "AggregationResult",
    "AggregationExpressionBuilder",
    "JoinStrategy",
]
