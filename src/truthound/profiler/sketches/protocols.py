"""Protocol definitions for probabilistic data structures.

This module defines the interfaces that all probabilistic data structures
(sketches) must implement, enabling consistent usage and interoperability.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")
S = TypeVar("S", bound="Sketch")


@dataclass(frozen=True)
class SketchConfig:
    """Base configuration for sketch data structures.

    Attributes:
        seed: Random seed for hash functions (reproducibility)
        name: Optional name for identification
    """

    seed: int = 42
    name: str = ""


@runtime_checkable
class Sketch(Protocol):
    """Base protocol for all probabilistic data structures.

    All sketch implementations must provide:
    - add(): Add a single element
    - add_batch(): Efficiently add multiple elements
    - memory_bytes(): Report memory usage
    - clear(): Reset the sketch
    """

    @abstractmethod
    def add(self, value: Any) -> None:
        """Add a value to the sketch."""
        ...

    @abstractmethod
    def add_batch(self, values: list[Any]) -> None:
        """Add multiple values efficiently."""
        ...

    @abstractmethod
    def memory_bytes(self) -> int:
        """Return memory usage in bytes."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Reset the sketch to initial state."""
        ...


@runtime_checkable
class MergeableSketch(Sketch, Protocol[S]):
    """Protocol for sketches that can be merged for distributed processing.

    Enables parallel computation by merging partial results.
    """

    @abstractmethod
    def merge(self, other: S) -> S:
        """Merge another sketch into a new sketch.

        Args:
            other: Another sketch of the same type

        Returns:
            New sketch containing merged data

        Raises:
            ValueError: If sketches have incompatible configurations
        """
        ...

    @abstractmethod
    def merge_inplace(self, other: S) -> None:
        """Merge another sketch into this sketch in-place.

        More memory-efficient than merge() when the original is not needed.

        Args:
            other: Another sketch of the same type

        Raises:
            ValueError: If sketches have incompatible configurations
        """
        ...


@runtime_checkable
class CardinalityEstimator(Protocol):
    """Protocol for structures that estimate distinct element count.

    Examples: HyperLogLog, Linear Counting, LogLog
    """

    @abstractmethod
    def estimate(self) -> int:
        """Estimate the number of distinct elements.

        Returns:
            Estimated cardinality (distinct count)
        """
        ...

    @abstractmethod
    def standard_error(self) -> float:
        """Return the standard error rate.

        Returns:
            Standard error as a ratio (e.g., 0.01 = 1% error)
        """
        ...


@runtime_checkable
class FrequencyEstimator(Protocol):
    """Protocol for structures that estimate element frequencies.

    Examples: Count-Min Sketch, Count Sketch
    """

    @abstractmethod
    def estimate_frequency(self, value: Any) -> int:
        """Estimate the frequency of a specific value.

        Args:
            value: The value to estimate frequency for

        Returns:
            Estimated count (may be overestimate)
        """
        ...

    @abstractmethod
    def get_heavy_hitters(self, threshold: float) -> list[tuple[Any, int]]:
        """Get elements exceeding a frequency threshold.

        Args:
            threshold: Minimum frequency ratio (0.0-1.0) to be considered heavy

        Returns:
            List of (value, estimated_count) tuples
        """
        ...


@runtime_checkable
class MembershipTester(Protocol):
    """Protocol for structures that test set membership.

    Examples: Bloom Filter, Cuckoo Filter
    """

    @abstractmethod
    def contains(self, value: Any) -> bool:
        """Test if a value might be in the set.

        Returns:
            True if possibly present, False if definitely absent
        """
        ...

    @abstractmethod
    def false_positive_rate(self) -> float:
        """Return the current false positive rate.

        Returns:
            Estimated false positive probability
        """
        ...


@dataclass(frozen=True)
class SketchMetrics:
    """Metrics about a sketch's state and accuracy.

    Attributes:
        elements_added: Total elements added
        memory_bytes: Current memory usage
        estimated_error: Estimated error rate
        fill_ratio: How full the structure is (if applicable)
    """

    elements_added: int = 0
    memory_bytes: int = 0
    estimated_error: float = 0.0
    fill_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "elements_added": self.elements_added,
            "memory_bytes": self.memory_bytes,
            "estimated_error": self.estimated_error,
            "fill_ratio": self.fill_ratio,
        }


@dataclass(frozen=True)
class HyperLogLogConfig(SketchConfig):
    """Configuration for HyperLogLog cardinality estimator.

    Attributes:
        precision: Number of bits for register indexing (4-18)
                   Higher = more accuracy, more memory
                   Memory = 2^precision bytes
    """

    precision: int = 12

    def __post_init__(self) -> None:
        if not 4 <= self.precision <= 18:
            raise ValueError(f"precision must be 4-18, got {self.precision}")

    @property
    def num_registers(self) -> int:
        return 1 << self.precision

    @property
    def expected_error(self) -> float:
        """Expected standard error based on precision."""
        return 1.04 / (self.num_registers**0.5)

    @classmethod
    def for_error_rate(cls, target_error: float, seed: int = 42) -> "HyperLogLogConfig":
        """Create config that achieves target error rate.

        Args:
            target_error: Desired standard error (e.g., 0.01 for 1%)
            seed: Random seed

        Returns:
            HyperLogLogConfig with appropriate precision
        """
        import math

        # Error = 1.04 / sqrt(m), so m = (1.04 / error)^2
        required_registers = (1.04 / target_error) ** 2
        precision = max(4, min(18, int(math.ceil(math.log2(required_registers)))))
        return cls(precision=precision, seed=seed)


@dataclass(frozen=True)
class CountMinSketchConfig(SketchConfig):
    """Configuration for Count-Min Sketch frequency estimator.

    Attributes:
        width: Number of counters per row (affects accuracy)
        depth: Number of hash functions/rows (affects confidence)
    """

    width: int = 2000
    depth: int = 5

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if self.depth <= 0:
            raise ValueError(f"depth must be positive, got {self.depth}")

    @property
    def expected_error(self) -> float:
        """Expected error rate: e/width."""
        import math

        return math.e / self.width

    @property
    def confidence(self) -> float:
        """Confidence level: 1 - (1/e)^depth."""
        import math

        return 1.0 - math.exp(-self.depth)

    @classmethod
    def for_error_and_confidence(
        cls,
        epsilon: float = 0.001,
        delta: float = 0.01,
        seed: int = 42,
    ) -> "CountMinSketchConfig":
        """Create config for target error rate and confidence.

        Args:
            epsilon: Maximum overestimate error (as ratio of total count)
            delta: Probability of exceeding epsilon error
            seed: Random seed

        Returns:
            CountMinSketchConfig with appropriate dimensions
        """
        import math

        width = int(math.ceil(math.e / epsilon))
        depth = int(math.ceil(math.log(1.0 / delta)))
        return cls(width=width, depth=depth, seed=seed)


@dataclass(frozen=True)
class BloomFilterConfig(SketchConfig):
    """Configuration for Bloom Filter membership tester.

    Attributes:
        capacity: Expected number of elements
        error_rate: Target false positive rate
    """

    capacity: int = 1_000_000
    error_rate: float = 0.01

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {self.capacity}")
        if not 0 < self.error_rate < 1:
            raise ValueError(f"error_rate must be (0, 1), got {self.error_rate}")

    @property
    def optimal_size(self) -> int:
        """Optimal bit array size: -n*ln(p) / (ln(2)^2)."""
        import math

        return int(-self.capacity * math.log(self.error_rate) / (math.log(2) ** 2))

    @property
    def optimal_hash_count(self) -> int:
        """Optimal number of hash functions: (m/n) * ln(2)."""
        import math

        m = self.optimal_size
        return max(1, int((m / self.capacity) * math.log(2)))
