"""Tests for streaming validators."""

import pytest
import polars as pl

from truthound.validators.streaming.base import (
    StreamingValidator,
    StreamingValidationPipeline,
    StreamingState,
)
from truthound.validators.streaming.completeness import (
    StreamingNullValidator,
    StreamingCompletenessValidator,
    StreamingNaNValidator,
)
from truthound.validators.streaming.range import (
    StreamingRangeValidator,
    StreamingOutlierValidator,
    StreamingPositiveValidator,
)


class TestStreamingState:
    """Tests for StreamingState."""

    def test_update_column_stat_sum(self):
        """Should aggregate with sum."""
        state = StreamingState()

        state.update_column_stat("col1", "count", 10, "sum")
        state.update_column_stat("col1", "count", 20, "sum")

        assert state.get_column_stat("col1", "count") == 30

    def test_update_column_stat_max(self):
        """Should aggregate with max."""
        state = StreamingState()

        state.update_column_stat("col1", "max_val", 10, "max")
        state.update_column_stat("col1", "max_val", 5, "max")
        state.update_column_stat("col1", "max_val", 20, "max")

        assert state.get_column_stat("col1", "max_val") == 20

    def test_update_column_stat_min(self):
        """Should aggregate with min."""
        state = StreamingState()

        state.update_column_stat("col1", "min_val", 10, "min")
        state.update_column_stat("col1", "min_val", 5, "min")
        state.update_column_stat("col1", "min_val", 20, "min")

        assert state.get_column_stat("col1", "min_val") == 5

    def test_get_nonexistent_stat(self):
        """Should return default for nonexistent stat."""
        state = StreamingState()

        assert state.get_column_stat("col1", "count", 0) == 0
        assert state.get_column_stat("col1", "count") is None


class TestStreamingNullValidator:
    """Tests for StreamingNullValidator."""

    def test_counts_nulls_across_chunks(self):
        """Should count nulls correctly across chunks."""
        # Create data that will be split into chunks
        values = [None if i % 10 == 0 else 1.0 for i in range(1000)]
        lf = pl.LazyFrame({"values": values})

        validator = StreamingNullValidator(chunk_size=100)
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].count == 100  # 10% of 1000

    def test_empty_dataframe(self):
        """Should handle empty dataframe."""
        lf = pl.LazyFrame({"values": []})

        validator = StreamingNullValidator()
        issues = validator.validate(lf)

        assert len(issues) == 0

    def test_no_nulls(self):
        """Should pass when no nulls."""
        lf = pl.LazyFrame({"values": [1.0, 2.0, 3.0] * 100})

        validator = StreamingNullValidator(chunk_size=50)
        issues = validator.validate(lf)

        assert len(issues) == 0


class TestStreamingCompletenessValidator:
    """Tests for StreamingCompletenessValidator."""

    def test_completeness_below_threshold(self):
        """Should detect when completeness is below threshold."""
        # 80% non-null
        values = [None if i % 5 == 0 else 1.0 for i in range(500)]
        lf = pl.LazyFrame({"values": values})

        validator = StreamingCompletenessValidator(
            min_ratio=0.9,
            chunk_size=100,
        )
        issues = validator.validate(lf)

        assert len(issues) == 1

    def test_completeness_above_threshold(self):
        """Should pass when completeness is above threshold."""
        # 98% non-null
        values = [None if i % 50 == 0 else 1.0 for i in range(500)]
        lf = pl.LazyFrame({"values": values})

        validator = StreamingCompletenessValidator(
            min_ratio=0.95,
            chunk_size=100,
        )
        issues = validator.validate(lf)

        assert len(issues) == 0


class TestStreamingRangeValidator:
    """Tests for StreamingRangeValidator."""

    def test_counts_out_of_range_across_chunks(self):
        """Should count out-of-range values across chunks."""
        # Values 0-999, range 100-900
        lf = pl.LazyFrame({"values": list(range(1000))})

        validator = StreamingRangeValidator(
            min_value=100,
            max_value=900,
            chunk_size=200,
        )
        issues = validator.validate(lf)

        assert len(issues) == 1
        # 0-99 and 901-999 = 199 out of range
        assert issues[0].count == 199

    def test_all_in_range(self):
        """Should pass when all values in range."""
        lf = pl.LazyFrame({"values": list(range(50, 100))})

        validator = StreamingRangeValidator(
            min_value=0,
            max_value=100,
            chunk_size=10,
        )
        issues = validator.validate(lf)

        assert len(issues) == 0


class TestStreamingOutlierValidator:
    """Tests for StreamingOutlierValidator."""

    def test_detects_outliers_in_streaming(self):
        """Should detect outliers using IQR in streaming mode."""
        # Normal distribution with outliers
        import random
        random.seed(42)

        values = [random.gauss(50, 10) for _ in range(900)]
        values.extend([200, -100, 300, -150, 250])  # Outliers
        lf = pl.LazyFrame({"values": values})

        validator = StreamingOutlierValidator(
            iqr_multiplier=1.5,
            chunk_size=200,
        )
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].count >= 5  # At least our added outliers


class TestStreamingPositiveValidator:
    """Tests for StreamingPositiveValidator."""

    def test_counts_non_positive_across_chunks(self):
        """Should count non-positive values across chunks."""
        values = list(range(-50, 100))  # -50 to 99
        lf = pl.LazyFrame({"values": values})

        validator = StreamingPositiveValidator(chunk_size=30)
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].count == 51  # -50 to 0 inclusive


class TestStreamingValidationPipeline:
    """Tests for StreamingValidationPipeline."""

    def test_runs_multiple_validators_in_single_pass(self):
        """Should run multiple validators efficiently."""
        # Data with various issues
        values = [
            None if i % 10 == 0 else float(i) for i in range(-50, 450)
        ]
        lf = pl.LazyFrame({"values": values})

        pipeline = StreamingValidationPipeline(
            validators=[
                StreamingNullValidator(),
                StreamingPositiveValidator(),
            ],
            chunk_size=100,
        )

        issues = pipeline.validate(lf)

        # Should find both null and non-positive issues
        issue_types = {i.issue_type for i in issues}
        assert "null" in issue_types
        assert "not_positive" in issue_types

    def test_add_validator(self):
        """Should allow adding validators."""
        pipeline = StreamingValidationPipeline(
            validators=[StreamingNullValidator()],
            chunk_size=100,
        )

        pipeline.add_validator(StreamingPositiveValidator())

        assert len(pipeline.validators) == 2

    def test_empty_dataframe(self):
        """Should handle empty dataframe."""
        lf = pl.LazyFrame({"values": []})

        pipeline = StreamingValidationPipeline(
            validators=[
                StreamingNullValidator(),
                StreamingRangeValidator(min_value=0, max_value=100),
            ],
        )

        issues = pipeline.validate(lf)
        assert len(issues) == 0
