"""Tests for approximate (HyperLogLog) validators."""

import pytest
import polars as pl

from truthound.validators.uniqueness.approximate import (
    HyperLogLog,
    ApproximateDistinctCountValidator,
    ApproximateUniqueRatioValidator,
    StreamingDistinctCountValidator,
)


class TestHyperLogLog:
    """Tests for HyperLogLog implementation."""

    def test_basic_cardinality(self):
        """Should estimate cardinality with reasonable accuracy."""
        hll = HyperLogLog(precision=12)

        # Add 10,000 unique values
        for i in range(10_000):
            hll.add(f"value_{i}")

        estimate = hll.estimate()

        # With precision=12, standard error is ~0.65%
        # Allow 5% error for testing
        assert 9500 <= estimate <= 10500

    def test_duplicate_values(self):
        """Should handle duplicate values correctly."""
        hll = HyperLogLog(precision=12)

        # Add same value 1000 times
        for _ in range(1000):
            hll.add("same_value")

        estimate = hll.estimate()

        # Should estimate close to 1
        assert estimate <= 3  # Some error expected

    def test_add_batch(self):
        """Should add multiple values efficiently."""
        hll = HyperLogLog(precision=12)

        values = [f"value_{i}" for i in range(5000)]
        hll.add_batch(values)

        estimate = hll.estimate()
        assert 4500 <= estimate <= 5500

    def test_merge(self):
        """Should merge two HLLs correctly."""
        hll1 = HyperLogLog(precision=12)
        hll2 = HyperLogLog(precision=12)

        # Add 1000 values to each, with some overlap
        for i in range(1000):
            hll1.add(f"value_{i}")

        for i in range(500, 1500):  # 500 overlap
            hll2.add(f"value_{i}")

        merged = hll1.merge(hll2)
        estimate = merged.estimate()

        # Expected unique: 1500
        assert 1350 <= estimate <= 1650

    def test_merge_different_precision_fails(self):
        """Should fail when merging HLLs with different precision."""
        hll1 = HyperLogLog(precision=10)
        hll2 = HyperLogLog(precision=12)

        with pytest.raises(ValueError):
            hll1.merge(hll2)

    def test_invalid_precision(self):
        """Should reject invalid precision values."""
        with pytest.raises(ValueError):
            HyperLogLog(precision=3)

        with pytest.raises(ValueError):
            HyperLogLog(precision=17)

    def test_memory_bytes(self):
        """Should report correct memory usage."""
        hll = HyperLogLog(precision=12)
        assert hll.memory_bytes() == 4096  # 2^12 registers

    def test_standard_error(self):
        """Should report correct standard error."""
        hll = HyperLogLog(precision=12)
        error = hll.standard_error()

        # 1.04 / sqrt(2^12) ≈ 0.0162
        assert 0.015 <= error <= 0.02


class TestApproximateDistinctCountValidator:
    """Tests for ApproximateDistinctCountValidator."""

    def test_detects_low_distinct_count(self):
        """Should detect when distinct count is below minimum."""
        lf = pl.LazyFrame({
            "values": ["a", "b", "c", "a", "b"],  # 3 distinct
        })

        validator = ApproximateDistinctCountValidator(
            min_count=10,
            precision=12,
        )
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert "approximate_distinct_count_low" in issues[0].issue_type

    def test_detects_high_distinct_count(self):
        """Should detect when distinct count exceeds maximum."""
        lf = pl.LazyFrame({
            "values": [f"value_{i}" for i in range(1000)],
        })

        validator = ApproximateDistinctCountValidator(
            max_count=500,
            precision=12,
        )
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert "approximate_distinct_count_high" in issues[0].issue_type

    def test_passes_within_range(self):
        """Should pass when distinct count is within range."""
        lf = pl.LazyFrame({
            "values": [f"value_{i}" for i in range(500)],
        })

        validator = ApproximateDistinctCountValidator(
            min_count=400,
            max_count=600,
            precision=12,
        )
        issues = validator.validate(lf)

        assert len(issues) == 0

    def test_empty_dataframe(self):
        """Should handle empty dataframe."""
        lf = pl.LazyFrame({"values": []})

        validator = ApproximateDistinctCountValidator(min_count=10)
        issues = validator.validate(lf)

        assert len(issues) == 0

    def test_invalid_precision(self):
        """Should reject invalid precision."""
        with pytest.raises(ValueError):
            ApproximateDistinctCountValidator(min_count=10, precision=3)


class TestApproximateUniqueRatioValidator:
    """Tests for ApproximateUniqueRatioValidator."""

    def test_detects_low_ratio(self):
        """Should detect when unique ratio is too low."""
        # 100 values, only 10 unique (10% unique)
        values = [f"value_{i % 10}" for i in range(100)]
        lf = pl.LazyFrame({"values": values})

        validator = ApproximateUniqueRatioValidator(min_ratio=0.5)
        issues = validator.validate(lf)

        assert len(issues) == 1

    def test_detects_high_ratio(self):
        """Should detect when unique ratio is too high."""
        # 100% unique
        lf = pl.LazyFrame({
            "values": [f"value_{i}" for i in range(100)],
        })

        validator = ApproximateUniqueRatioValidator(max_ratio=0.5)
        issues = validator.validate(lf)

        assert len(issues) == 1


class TestStreamingDistinctCountValidator:
    """Tests for StreamingDistinctCountValidator."""

    def test_streaming_with_chunks(self):
        """Should process data in chunks and aggregate."""
        lf = pl.LazyFrame({
            "values": [f"value_{i}" for i in range(10_000)],
        })

        validator = StreamingDistinctCountValidator(
            min_count=9000,
            chunk_size=1000,  # 10 chunks
            precision=12,
        )
        issues = validator.validate(lf)

        # Should pass - 10,000 unique values
        assert len(issues) == 0

    def test_streaming_detects_violation(self):
        """Should detect violations in streaming mode."""
        lf = pl.LazyFrame({
            "values": [f"value_{i % 100}" for i in range(10_000)],  # Only 100 unique
        })

        validator = StreamingDistinctCountValidator(
            min_count=500,
            chunk_size=1000,
        )
        issues = validator.validate(lf)

        assert len(issues) == 1
