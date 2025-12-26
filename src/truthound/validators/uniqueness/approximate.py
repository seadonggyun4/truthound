"""Approximate uniqueness validators using probabilistic algorithms.

This module provides memory-efficient alternatives to exact uniqueness counting
for very large datasets where exact counts are not critical.

Key Features:
- HyperLogLog: O(1) memory for cardinality estimation
- Streaming compatible: Can process data in chunks
- Configurable precision vs memory tradeoff
"""

from typing import Any
import hashlib
import struct

import polars as pl

from truthound.types import Severity
from truthound.validators.base import (
    ValidationIssue,
    Validator,
    StreamingValidatorMixin,
)
from truthound.validators.registry import register_validator


class HyperLogLog:
    """HyperLogLog cardinality estimator.

    A probabilistic data structure for estimating the number of distinct
    elements in a multiset. Uses O(m) memory where m is the number of
    registers, regardless of the data size.

    Precision vs Memory:
    - precision=10: 1KB memory, ~1.04% standard error
    - precision=12: 4KB memory, ~0.65% standard error (default)
    - precision=14: 16KB memory, ~0.41% standard error
    - precision=16: 64KB memory, ~0.26% standard error

    Reference:
        Flajolet, P., et al. "HyperLogLog: the analysis of a near-optimal
        cardinality estimation algorithm." (2007)
    """

    def __init__(self, precision: int = 12):
        """Initialize HyperLogLog estimator.

        Args:
            precision: Number of bits for bucket indexing (4-16).
                       Higher = more accuracy, more memory.
        """
        if not 4 <= precision <= 16:
            raise ValueError(f"precision must be between 4 and 16, got {precision}")

        self.precision = precision
        self.num_registers = 1 << precision  # 2^precision
        self.registers = [0] * self.num_registers

        # Alpha constant for bias correction
        if self.num_registers == 16:
            self.alpha = 0.673
        elif self.num_registers == 32:
            self.alpha = 0.697
        elif self.num_registers == 64:
            self.alpha = 0.709
        else:
            self.alpha = 0.7213 / (1 + 1.079 / self.num_registers)

    def _hash(self, value: Any) -> int:
        """Hash a value to a 64-bit integer."""
        # Convert to string and hash
        value_bytes = str(value).encode("utf-8")
        hash_bytes = hashlib.md5(value_bytes).digest()[:8]
        return struct.unpack("<Q", hash_bytes)[0]

    def _rho(self, value: int, max_bits: int) -> int:
        """Find position of rightmost 1-bit (1-indexed).

        Returns the position of the first 1-bit from the right (LSB).
        This is equivalent to counting trailing zeros + 1.
        If value is 0, returns max_bits + 1.

        Example: value=0b1000 -> trailing zeros=3, rho=4
        """
        if value == 0:
            return max_bits + 1

        # Count trailing zeros + 1 (position of first 1-bit from right)
        count = 1
        while (value & 1) == 0 and count <= max_bits:
            count += 1
            value >>= 1

        return count

    def add(self, value: Any) -> None:
        """Add a value to the estimator."""
        hash_value = self._hash(value)

        # Use first 'precision' bits for bucket index
        bucket_index = hash_value >> (64 - self.precision)
        bucket_index = bucket_index & (self.num_registers - 1)  # Ensure valid index

        # Use remaining bits for rho calculation
        remaining_bits = 64 - self.precision
        remaining = hash_value & ((1 << remaining_bits) - 1)
        rho = self._rho(remaining, remaining_bits)

        # Update register with max
        self.registers[bucket_index] = max(self.registers[bucket_index], rho)

    def add_batch(self, values: list[Any]) -> None:
        """Add multiple values efficiently."""
        for value in values:
            if value is not None:
                self.add(value)

    def estimate(self) -> int:
        """Estimate the cardinality (number of distinct elements)."""
        import math

        # Raw estimate using harmonic mean
        indicator = sum(2.0 ** (-r) for r in self.registers)
        raw_estimate = self.alpha * (self.num_registers ** 2) / indicator

        # Small range correction using Linear Counting
        # When estimate is small, many registers are still 0
        # Use: m * ln(m / V) where V = number of zero registers
        if raw_estimate <= 2.5 * self.num_registers:
            zeros = self.registers.count(0)
            if zeros > 0:
                # Linear counting formula
                linear_count = self.num_registers * math.log(
                    self.num_registers / zeros
                )
                return int(linear_count)

        # Large range correction (for 32-bit hashes only)
        # Not needed for 64-bit hashes

        return int(raw_estimate)

    def merge(self, other: "HyperLogLog") -> "HyperLogLog":
        """Merge another HyperLogLog into this one."""
        if self.precision != other.precision:
            raise ValueError("Cannot merge HyperLogLog with different precision")

        merged = HyperLogLog(self.precision)
        merged.registers = [
            max(a, b) for a, b in zip(self.registers, other.registers)
        ]
        return merged

    def standard_error(self) -> float:
        """Return the standard error for this precision."""
        return 1.04 / (self.num_registers ** 0.5)

    def memory_bytes(self) -> int:
        """Return memory usage in bytes."""
        return self.num_registers  # Each register is 1 byte


@register_validator
class ApproximateDistinctCountValidator(Validator, StreamingValidatorMixin):
    """Validates distinct count using HyperLogLog approximation.

    Memory-efficient alternative to DistinctCountValidator for very large
    datasets. Uses HyperLogLog algorithm with configurable precision.

    Memory Usage by Precision:
    - precision=10: ~1KB per column
    - precision=12: ~4KB per column (default, ~0.65% error)
    - precision=14: ~16KB per column
    - precision=16: ~64KB per column (~0.26% error)

    Example:
        # Check approximate distinct count
        validator = ApproximateDistinctCountValidator(
            min_count=1000,
            max_count=100000,
            precision=12,  # ~0.65% standard error
        )

        # For higher accuracy
        validator = ApproximateDistinctCountValidator(
            min_count=1000,
            precision=14,  # ~0.41% standard error
        )
    """

    name = "approximate_distinct_count"
    category = "uniqueness"

    def __init__(
        self,
        min_count: int | None = None,
        max_count: int | None = None,
        precision: int = 12,
        tolerance: float | None = None,
        **kwargs: Any,
    ):
        """Initialize the validator.

        Args:
            min_count: Minimum expected distinct count
            max_count: Maximum expected distinct count
            precision: HyperLogLog precision (4-16, default 12)
            tolerance: Error tolerance multiplier for threshold comparison.
                       If None, uses 2x standard error. E.g., tolerance=2.0
                       means accept if within 2 standard errors.
            **kwargs: Additional validator configuration
        """
        super().__init__(**kwargs)
        self.min_count = min_count
        self.max_count = max_count
        self.precision = precision
        self.tolerance = tolerance

        # Validate precision
        if not 4 <= precision <= 16:
            raise ValueError(f"precision must be between 4 and 16, got {precision}")

    def _get_error_margin(self, estimate: int) -> int:
        """Calculate error margin based on precision and tolerance."""
        hll = HyperLogLog(self.precision)
        std_error = hll.standard_error()
        multiplier = self.tolerance if self.tolerance else 2.0
        return int(estimate * std_error * multiplier)

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_target_columns(lf)

        if not columns:
            return issues

        # Collect data and build HLL for each column
        df = lf.collect()

        for col in columns:
            hll = HyperLogLog(self.precision)
            col_data = df.get_column(col).drop_nulls().to_list()

            if not col_data:
                continue

            hll.add_batch(col_data)
            estimate = hll.estimate()
            error_margin = self._get_error_margin(estimate)
            std_error_pct = hll.standard_error() * 100

            # Check min count
            if self.min_count is not None:
                # Account for error margin
                if estimate + error_margin < self.min_count:
                    issues.append(
                        ValidationIssue(
                            column=col,
                            issue_type="approximate_distinct_count_low",
                            count=max(0, self.min_count - estimate),
                            severity=Severity.MEDIUM,
                            details=(
                                f"Approx distinct count ~{estimate:,} "
                                f"(±{std_error_pct:.1f}%) < min {self.min_count:,}"
                            ),
                            expected=f">= {self.min_count:,}",
                            actual=f"~{estimate:,} (±{error_margin:,})",
                        )
                    )

            # Check max count
            if self.max_count is not None:
                if estimate - error_margin > self.max_count:
                    issues.append(
                        ValidationIssue(
                            column=col,
                            issue_type="approximate_distinct_count_high",
                            count=estimate - self.max_count,
                            severity=Severity.MEDIUM,
                            details=(
                                f"Approx distinct count ~{estimate:,} "
                                f"(±{std_error_pct:.1f}%) > max {self.max_count:,}"
                            ),
                            expected=f"<= {self.max_count:,}",
                            actual=f"~{estimate:,} (±{error_margin:,})",
                        )
                    )

        return issues


@register_validator
class ApproximateUniqueRatioValidator(Validator, StreamingValidatorMixin):
    """Validates uniqueness ratio using HyperLogLog approximation.

    Memory-efficient alternative to UniqueRatioValidator for large datasets.

    Example:
        # Check if at least 90% of values are unique
        validator = ApproximateUniqueRatioValidator(
            min_ratio=0.9,
            precision=12,
        )
    """

    name = "approximate_unique_ratio"
    category = "uniqueness"

    def __init__(
        self,
        min_ratio: float | None = None,
        max_ratio: float | None = None,
        precision: int = 12,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.precision = precision

        if not 4 <= precision <= 16:
            raise ValueError(f"precision must be between 4 and 16, got {precision}")

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_target_columns(lf)

        if not columns:
            return issues

        df = lf.collect()

        for col in columns:
            hll = HyperLogLog(self.precision)
            col_data = df.get_column(col).drop_nulls().to_list()

            if not col_data:
                continue

            total_count = len(col_data)
            hll.add_batch(col_data)
            unique_estimate = hll.estimate()

            ratio = unique_estimate / total_count if total_count > 0 else 0
            std_error_pct = hll.standard_error() * 100

            if self.min_ratio is not None and ratio < self.min_ratio:
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="approximate_unique_ratio_low",
                        count=int((self.min_ratio - ratio) * total_count),
                        severity=Severity.MEDIUM,
                        details=(
                            f"Approx unique ratio ~{ratio:.1%} "
                            f"(±{std_error_pct:.1f}%) < min {self.min_ratio:.1%}"
                        ),
                        expected=f">= {self.min_ratio:.1%}",
                        actual=f"~{ratio:.1%}",
                    )
                )

            if self.max_ratio is not None and ratio > self.max_ratio:
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="approximate_unique_ratio_high",
                        count=int((ratio - self.max_ratio) * total_count),
                        severity=Severity.MEDIUM,
                        details=(
                            f"Approx unique ratio ~{ratio:.1%} "
                            f"(±{std_error_pct:.1f}%) > max {self.max_ratio:.1%}"
                        ),
                        expected=f"<= {self.max_ratio:.1%}",
                        actual=f"~{ratio:.1%}",
                    )
                )

        return issues


@register_validator
class StreamingDistinctCountValidator(Validator, StreamingValidatorMixin):
    """Validates distinct count with streaming support for very large datasets.

    Processes data in chunks to limit memory usage while using HyperLogLog
    for cardinality estimation.

    Example:
        # Process in 100K row chunks
        validator = StreamingDistinctCountValidator(
            min_count=10000,
            chunk_size=100_000,
        )
    """

    name = "streaming_distinct_count"
    category = "uniqueness"

    def __init__(
        self,
        min_count: int | None = None,
        max_count: int | None = None,
        precision: int = 12,
        chunk_size: int = 100_000,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.min_count = min_count
        self.max_count = max_count
        self.precision = precision
        self.chunk_size = chunk_size

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_target_columns(lf)

        if not columns:
            return issues

        total_rows = lf.select(pl.len()).collect().item()
        if total_rows == 0:
            return issues

        # Initialize HLL for each column
        hlls: dict[str, HyperLogLog] = {
            col: HyperLogLog(self.precision) for col in columns
        }

        # Process in chunks
        for offset in range(0, total_rows, self.chunk_size):
            chunk_df = lf.slice(offset, self.chunk_size).collect()

            for col in columns:
                col_data = chunk_df.get_column(col).drop_nulls().to_list()
                hlls[col].add_batch(col_data)

        # Check estimates
        for col in columns:
            estimate = hlls[col].estimate()
            std_error_pct = hlls[col].standard_error() * 100

            if self.min_count is not None and estimate < self.min_count:
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="streaming_distinct_count_low",
                        count=self.min_count - estimate,
                        severity=Severity.MEDIUM,
                        details=(
                            f"Streaming approx distinct ~{estimate:,} "
                            f"(±{std_error_pct:.1f}%) < min {self.min_count:,}"
                        ),
                        expected=f">= {self.min_count:,}",
                        actual=f"~{estimate:,}",
                    )
                )

            if self.max_count is not None and estimate > self.max_count:
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="streaming_distinct_count_high",
                        count=estimate - self.max_count,
                        severity=Severity.MEDIUM,
                        details=(
                            f"Streaming approx distinct ~{estimate:,} "
                            f"(±{std_error_pct:.1f}%) > max {self.max_count:,}"
                        ),
                        expected=f"<= {self.max_count:,}",
                        actual=f"~{estimate:,}",
                    )
                )

        return issues
