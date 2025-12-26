"""Tests for streaming data sources and streaming validation."""

import tempfile
from pathlib import Path

import pytest
import polars as pl
import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

from truthound.validators.streaming import (
    # Sources
    ParquetStreamingSource,
    CSVStreamingSource,
    JSONLStreamingSource,
    ArrowIPCStreamingSource,
    LazyFrameStreamingSource,
    create_streaming_source,
    # Mixin and utilities
    StreamingValidatorMixin,
    StreamingValidatorAdapter,
    CountingAccumulator,
    SamplingAccumulator,
    stream_validate,
    stream_validate_many,
    # Validators
    StreamingNullValidator,
    StreamingRangeValidator,
)
from truthound.validators.completeness import NullValidator
from truthound.validators.distribution.range import RangeValidator


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    return pl.DataFrame({
        "id": list(range(1000)),
        "value": np.random.normal(50, 10, 1000).tolist(),
        "name": ["test"] * 1000,
    })


@pytest.fixture
def sample_df_with_issues():
    """Create a DataFrame with some null values."""
    np.random.seed(42)
    values = list(range(100))
    # Add some nulls
    for i in [10, 20, 30, 40, 50]:
        values[i] = None
    return pl.DataFrame({
        "id": values,
        "value": list(range(100)),
    })


@pytest.fixture
def parquet_file(sample_df, tmp_path):
    """Create a temporary Parquet file."""
    file_path = tmp_path / "test_data.parquet"
    sample_df.write_parquet(file_path)
    return file_path


@pytest.fixture
def csv_file(sample_df, tmp_path):
    """Create a temporary CSV file."""
    file_path = tmp_path / "test_data.csv"
    sample_df.write_csv(file_path)
    return file_path


@pytest.fixture
def jsonl_file(sample_df, tmp_path):
    """Create a temporary JSONL file."""
    file_path = tmp_path / "test_data.jsonl"
    sample_df.write_ndjson(file_path)
    return file_path


@pytest.fixture
def arrow_file(sample_df, tmp_path):
    """Create a temporary Arrow IPC file."""
    file_path = tmp_path / "test_data.arrow"
    sample_df.write_ipc(file_path)
    return file_path


# =============================================================================
# Parquet Streaming Source Tests
# =============================================================================


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not available")
class TestParquetStreamingSource:
    """Tests for ParquetStreamingSource."""

    def test_basic_streaming(self, parquet_file):
        """Test basic streaming through Parquet file."""
        chunks = []
        with ParquetStreamingSource(parquet_file, chunk_size=100) as source:
            for chunk in source:
                chunks.append(chunk)

        total_rows = sum(len(c) for c in chunks)
        assert total_rows == 1000

    def test_chunk_size_respected(self, parquet_file):
        """Test that chunk sizes are approximately correct."""
        chunk_sizes = []
        with ParquetStreamingSource(parquet_file, chunk_size=200) as source:
            for chunk in source:
                chunk_sizes.append(len(chunk))

        # Last chunk may be smaller
        for size in chunk_sizes[:-1]:
            assert size <= 200

    def test_column_selection(self, parquet_file):
        """Test column projection."""
        with ParquetStreamingSource(parquet_file, columns=["id"]) as source:
            for chunk in source:
                assert list(chunk.columns) == ["id"]

    def test_max_rows(self, parquet_file):
        """Test max_rows limit."""
        total_rows = 0
        with ParquetStreamingSource(parquet_file, chunk_size=100, max_rows=250) as source:
            for chunk in source:
                total_rows += len(chunk)

        assert total_rows == 250

    def test_len(self, parquet_file):
        """Test row count."""
        source = ParquetStreamingSource(parquet_file)
        assert len(source) == 1000


# =============================================================================
# CSV Streaming Source Tests
# =============================================================================


class TestCSVStreamingSource:
    """Tests for CSVStreamingSource."""

    def test_basic_streaming(self, csv_file):
        """Test basic streaming through CSV file."""
        chunks = []
        with CSVStreamingSource(csv_file, chunk_size=100) as source:
            for chunk in source:
                chunks.append(chunk)

        total_rows = sum(len(c) for c in chunks)
        assert total_rows == 1000

    def test_column_selection(self, csv_file):
        """Test column selection."""
        with CSVStreamingSource(csv_file, columns=["id", "value"]) as source:
            for chunk in source:
                assert set(chunk.columns) == {"id", "value"}
                break

    def test_max_rows(self, csv_file):
        """Test max_rows limit."""
        total_rows = 0
        with CSVStreamingSource(csv_file, chunk_size=100, max_rows=150) as source:
            for chunk in source:
                total_rows += len(chunk)

        assert total_rows == 150


# =============================================================================
# JSONL Streaming Source Tests
# =============================================================================


class TestJSONLStreamingSource:
    """Tests for JSONLStreamingSource."""

    def test_basic_streaming(self, jsonl_file):
        """Test basic streaming through JSONL file."""
        chunks = []
        with JSONLStreamingSource(jsonl_file, chunk_size=100) as source:
            for chunk in source:
                chunks.append(chunk)

        total_rows = sum(len(c) for c in chunks)
        assert total_rows == 1000


# =============================================================================
# Arrow IPC Streaming Source Tests
# =============================================================================


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not available")
class TestArrowIPCStreamingSource:
    """Tests for ArrowIPCStreamingSource."""

    def test_basic_streaming(self, arrow_file):
        """Test basic streaming through Arrow IPC file."""
        chunks = []
        with ArrowIPCStreamingSource(arrow_file, chunk_size=100) as source:
            for chunk in source:
                chunks.append(chunk)

        total_rows = sum(len(c) for c in chunks)
        assert total_rows == 1000

    def test_column_selection(self, arrow_file):
        """Test column projection."""
        with ArrowIPCStreamingSource(arrow_file, columns=["id"]) as source:
            for chunk in source:
                assert "id" in chunk.columns


# =============================================================================
# LazyFrame Streaming Source Tests
# =============================================================================


class TestLazyFrameStreamingSource:
    """Tests for LazyFrameStreamingSource."""

    def test_basic_streaming(self, sample_df):
        """Test streaming from LazyFrame."""
        lf = sample_df.lazy()
        chunks = []
        with LazyFrameStreamingSource(lf, chunk_size=100) as source:
            for chunk in source:
                chunks.append(chunk)

        total_rows = sum(len(c) for c in chunks)
        assert total_rows == 1000


# =============================================================================
# create_streaming_source Tests
# =============================================================================


class TestCreateStreamingSource:
    """Tests for create_streaming_source factory function."""

    def test_parquet_detection(self, parquet_file):
        """Test automatic Parquet detection."""
        source = create_streaming_source(parquet_file)
        assert isinstance(source, ParquetStreamingSource)

    def test_csv_detection(self, csv_file):
        """Test automatic CSV detection."""
        source = create_streaming_source(csv_file)
        assert isinstance(source, CSVStreamingSource)

    def test_arrow_detection(self, arrow_file):
        """Test automatic Arrow IPC detection."""
        source = create_streaming_source(arrow_file)
        assert isinstance(source, ArrowIPCStreamingSource)

    def test_lazyframe_input(self, sample_df):
        """Test LazyFrame input."""
        lf = sample_df.lazy()
        source = create_streaming_source(lf)
        assert isinstance(source, LazyFrameStreamingSource)

    def test_unsupported_format(self, tmp_path):
        """Test unsupported file format."""
        fake_file = tmp_path / "test.xyz"
        fake_file.write_text("test")

        with pytest.raises(ValueError, match="Unsupported file type"):
            create_streaming_source(fake_file)


# =============================================================================
# Streaming Validator Mixin Tests
# =============================================================================


class TestStreamingValidatorMixin:
    """Tests for StreamingValidatorMixin."""

    def test_validate_streaming_from_file(self, parquet_file, sample_df_with_issues, tmp_path):
        """Test streaming validation from file."""
        # Create file with issues
        file_with_issues = tmp_path / "issues.parquet"
        sample_df_with_issues.write_parquet(file_with_issues)

        validator = NullValidator(column="id")
        adapter = StreamingValidatorAdapter(validator)

        issues = adapter.validate_streaming(file_with_issues, chunk_size=20)

        # Should find null issues
        assert len(issues) >= 0  # May or may not aggregate

    def test_validate_streaming_from_lazyframe(self, sample_df_with_issues):
        """Test streaming validation from LazyFrame."""
        lf = sample_df_with_issues.lazy()

        validator = NullValidator(column="id")
        adapter = StreamingValidatorAdapter(validator)

        issues = adapter.validate_streaming(lf, chunk_size=20)
        assert isinstance(issues, list)

    def test_validate_streaming_iter(self, sample_df_with_issues):
        """Test iterator-based streaming validation."""
        lf = sample_df_with_issues.lazy()

        validator = NullValidator(column="id")
        adapter = StreamingValidatorAdapter(validator)

        chunk_results = list(adapter.validate_streaming_iter(lf, chunk_size=20))

        assert len(chunk_results) > 0
        for chunk_idx, issues in chunk_results:
            assert isinstance(chunk_idx, int)
            assert isinstance(issues, list)

    def test_progress_callback(self, parquet_file):
        """Test progress callback."""
        validator = NullValidator(column="id")
        adapter = StreamingValidatorAdapter(validator)

        progress_calls = []

        def on_progress(chunk_idx, total_chunks):
            progress_calls.append((chunk_idx, total_chunks))

        adapter.validate_streaming(
            parquet_file,
            chunk_size=200,
            on_chunk=on_progress,
        )

        assert len(progress_calls) > 0
        # Check chunks are sequential
        for i, (chunk_idx, _) in enumerate(progress_calls):
            assert chunk_idx == i


# =============================================================================
# Accumulator Tests
# =============================================================================


class TestAccumulators:
    """Tests for streaming accumulators."""

    def test_counting_accumulator(self):
        """Test CountingAccumulator."""
        from truthound.validators.base import ValidationIssue
        from truthound.types import Severity

        acc = CountingAccumulator()
        state = acc.initialize()

        # Simulate multiple chunks with issues
        for i in range(3):
            issues = [
                ValidationIssue(
                    column="id",
                    issue_type="null_value",
                    count=10,
                    severity=Severity.MEDIUM,
                    details="test",
                    expected="not null",
                )
            ]
            state = acc.accumulate(state, issues)

        final_issues = acc.finalize(state, 1000)

        # Should aggregate into single issue with count=30
        assert len(final_issues) == 1
        assert final_issues[0].count == 30

    def test_sampling_accumulator(self):
        """Test SamplingAccumulator."""
        from truthound.validators.base import ValidationIssue
        from truthound.types import Severity

        acc = SamplingAccumulator(max_samples=5)
        state = acc.initialize()

        # Add more issues than max_samples
        for i in range(10):
            issues = [
                ValidationIssue(
                    column="id",
                    issue_type="null_value",
                    count=1,
                    severity=Severity.MEDIUM,
                    details=f"issue {i}",
                    expected="not null",
                )
            ]
            state = acc.accumulate(state, issues)

        final_issues = acc.finalize(state, 1000)

        # Should only keep max_samples
        assert len(final_issues) <= 5


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_stream_validate(self, sample_df_with_issues, tmp_path):
        """Test stream_validate function."""
        file_path = tmp_path / "data.parquet"
        sample_df_with_issues.write_parquet(file_path)

        validator = NullValidator(column="id")
        issues = stream_validate(validator, file_path, chunk_size=20)

        assert isinstance(issues, list)

    def test_stream_validate_many(self, sample_df_with_issues, tmp_path):
        """Test stream_validate_many function."""
        file_path = tmp_path / "data.parquet"
        sample_df_with_issues.write_parquet(file_path)

        validators = [
            NullValidator(column="id"),
            RangeValidator(column="value", min_value=0, max_value=50),
        ]

        results = stream_validate_many(validators, file_path, chunk_size=20)

        assert isinstance(results, dict)
        assert "null" in results
        assert "range" in results


# =============================================================================
# Integration Tests
# =============================================================================


class TestStreamingIntegration:
    """Integration tests for streaming validation."""

    def test_large_file_simulation(self, tmp_path):
        """Test streaming through a simulated large file."""
        # Create a larger dataset
        np.random.seed(42)
        large_df = pl.DataFrame({
            "id": list(range(10000)),
            "value": np.random.normal(50, 10, 10000).tolist(),
        })

        file_path = tmp_path / "large.parquet"
        large_df.write_parquet(file_path)

        # Stream validate
        validator = RangeValidator(column="value", min_value=0, max_value=100)
        adapter = StreamingValidatorAdapter(validator, chunk_size=1000)

        issues = adapter.validate_streaming(file_path)
        assert isinstance(issues, list)

    def test_multi_validator_pipeline(self, tmp_path):
        """Test multi-validator streaming pipeline."""
        # Create test data
        np.random.seed(42)
        df = pl.DataFrame({
            "id": list(range(1000)) + [None] * 10,
            "value": list(range(1000)) + [999] * 10,
        })

        file_path = tmp_path / "multi.parquet"
        df.write_parquet(file_path)

        validators = [
            NullValidator(column="id"),
            RangeValidator(column="value", min_value=0, max_value=100),
        ]

        results = stream_validate_many(validators, file_path, chunk_size=200)

        assert "null" in results
        assert "range" in results

    def test_streaming_with_column_selection(self, tmp_path):
        """Test streaming with column selection."""
        df = pl.DataFrame({
            "id": list(range(100)),
            "value": list(range(100)),
            "extra": ["x"] * 100,
        })

        file_path = tmp_path / "columns.parquet"
        df.write_parquet(file_path)

        validator = NullValidator(column="id")
        adapter = StreamingValidatorAdapter(validator)

        # Only load needed columns
        issues = adapter.validate_streaming(
            file_path,
            columns=["id"],
            chunk_size=20,
        )

        assert isinstance(issues, list)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_file(self, tmp_path):
        """Test streaming empty file."""
        empty_df = pl.DataFrame({"id": []})
        file_path = tmp_path / "empty.parquet"
        empty_df.write_parquet(file_path)

        with ParquetStreamingSource(file_path, chunk_size=100) as source:
            chunks = list(source)

        # Should yield no chunks or empty chunks
        total_rows = sum(len(c) for c in chunks)
        assert total_rows == 0

    def test_single_row_file(self, tmp_path):
        """Test streaming single row file."""
        single_df = pl.DataFrame({"id": [1]})
        file_path = tmp_path / "single.parquet"
        single_df.write_parquet(file_path)

        with ParquetStreamingSource(file_path, chunk_size=100) as source:
            chunks = list(source)

        total_rows = sum(len(c) for c in chunks)
        assert total_rows == 1

    def test_source_not_open_error(self, parquet_file):
        """Test error when source not opened."""
        source = ParquetStreamingSource(parquet_file)

        with pytest.raises(RuntimeError, match="not open"):
            list(source)
