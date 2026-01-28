"""Probabilistic data structures for enterprise-scale data analysis.

This module provides memory-efficient probabilistic data structures that enable
O(1) memory footprint for aggregation operations on datasets of any size.

Key Data Structures:
    - HyperLogLog: Cardinality (distinct count) estimation
    - CountMinSketch: Frequency estimation for heavy hitters detection
    - BloomFilter: Set membership testing

Design Principles:
    - Protocol-based design for extensibility
    - Mergeable structures for distributed processing
    - Configurable accuracy vs memory tradeoffs
    - Thread-safe implementations

Usage:
    from truthound.profiler.sketches import (
        HyperLogLog,
        CountMinSketch,
        BloomFilter,
        create_sketch,
    )

    # Cardinality estimation
    hll = HyperLogLog(precision=12)
    for value in data:
        hll.add(value)
    distinct_count = hll.estimate()

    # Frequency estimation
    cms = CountMinSketch(width=2000, depth=5)
    for value in data:
        cms.add(value)
    freq = cms.estimate(some_value)

    # Set membership
    bf = BloomFilter(capacity=1_000_000, error_rate=0.01)
    for value in data:
        bf.add(value)
    if bf.contains(query_value):
        print("Possibly in set")
"""

from truthound.profiler.sketches.protocols import (
    Sketch,
    MergeableSketch,
    CardinalityEstimator,
    FrequencyEstimator,
    MembershipTester,
    SketchConfig,
)
from truthound.profiler.sketches.hyperloglog import HyperLogLog
from truthound.profiler.sketches.countmin import CountMinSketch
from truthound.profiler.sketches.bloom import BloomFilter
from truthound.profiler.sketches.factory import (
    create_sketch,
    SketchType,
    SketchFactory,
)

__all__ = [
    # Protocols
    "Sketch",
    "MergeableSketch",
    "CardinalityEstimator",
    "FrequencyEstimator",
    "MembershipTester",
    "SketchConfig",
    # Implementations
    "HyperLogLog",
    "CountMinSketch",
    "BloomFilter",
    # Factory
    "create_sketch",
    "SketchType",
    "SketchFactory",
]
