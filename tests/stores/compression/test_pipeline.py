"""Tests for compression pipeline."""

import pytest

from truthound.stores.compression import (
    CompressionAlgorithm,
    CompressionPipeline,
    PipelineStage,
    PipelineResult,
    PipelineMetrics,
    Transform,
    IdentityTransform,
    DeduplicationTransform,
    DeltaEncodingTransform,
    RunLengthTransform,
    PipelineError,
    TransformError,
    StageType,
    get_compressor,
    create_text_pipeline,
    create_json_pipeline,
    create_binary_pipeline,
    create_timeseries_pipeline,
)


class TestIdentityTransform:
    """Tests for IdentityTransform."""

    def test_apply_returns_same(self):
        """Test that identity returns same data."""
        transform = IdentityTransform()
        data = b"test data"

        result = transform.apply(data)
        assert result == data

    def test_reverse_returns_same(self):
        """Test that reverse returns same data."""
        transform = IdentityTransform()
        data = b"test data"

        result = transform.reverse(data)
        assert result == data

    def test_name(self):
        """Test transform name."""
        transform = IdentityTransform()
        assert transform.name == "identity"


class TestDeduplicationTransform:
    """Tests for DeduplicationTransform."""

    def test_basic_dedup(self):
        """Test basic deduplication."""
        transform = DeduplicationTransform(block_size=4)
        # Create data with repeated 4-byte blocks
        data = b"AAAA" * 10 + b"BBBB" * 5 + b"AAAA" * 5

        compressed = transform.apply(data)
        decompressed = transform.reverse(compressed)

        assert decompressed == data

    def test_dedup_with_duplicates(self):
        """Test dedup effectiveness on duplicate data."""
        transform = DeduplicationTransform(block_size=16)
        # Highly repetitive data
        block = b"A" * 16
        data = block * 100

        compressed = transform.apply(data)
        decompressed = transform.reverse(compressed)

        assert decompressed == data
        # Should be significantly smaller
        assert len(compressed) < len(data) * 0.5

    def test_dedup_small_data(self):
        """Test dedup with small data (passthrough)."""
        transform = DeduplicationTransform(block_size=16)
        data = b"small"

        compressed = transform.apply(data)
        decompressed = transform.reverse(compressed)

        assert decompressed == data

    def test_dedup_no_duplicates(self):
        """Test dedup with unique blocks."""
        transform = DeduplicationTransform(block_size=4)
        data = bytes(range(256))[:64]  # All unique 4-byte blocks

        compressed = transform.apply(data)
        decompressed = transform.reverse(compressed)

        assert decompressed == data

    def test_name(self):
        """Test transform name."""
        transform = DeduplicationTransform()
        assert transform.name == "deduplication"

    def test_get_stats(self):
        """Test statistics retrieval."""
        transform = DeduplicationTransform(block_size=4)
        data = b"AAAA" * 10

        transform.apply(data)
        stats = transform.get_stats()

        assert "name" in stats
        assert "dedup_ratio" in stats or "original_blocks" in stats


class TestDeltaEncodingTransform:
    """Tests for DeltaEncodingTransform."""

    def test_basic_delta(self):
        """Test basic delta encoding."""
        transform = DeltaEncodingTransform()
        data = bytes([10, 12, 14, 16, 18, 20])

        encoded = transform.apply(data)
        decoded = transform.reverse(encoded)

        assert decoded == data

    def test_delta_with_sequence(self):
        """Test delta with sequential data."""
        transform = DeltaEncodingTransform()
        data = bytes(range(100))

        encoded = transform.apply(data)
        decoded = transform.reverse(encoded)

        assert decoded == data

    def test_delta_small_data(self):
        """Test delta with small data."""
        transform = DeltaEncodingTransform()
        data = b"X"

        encoded = transform.apply(data)
        decoded = transform.reverse(encoded)

        assert decoded == data

    def test_delta_empty_data(self):
        """Test delta with empty data."""
        transform = DeltaEncodingTransform()
        data = b""

        encoded = transform.apply(data)
        decoded = transform.reverse(encoded)

        assert decoded == data

    def test_name(self):
        """Test transform name."""
        transform = DeltaEncodingTransform()
        assert transform.name == "delta_encoding"


class TestRunLengthTransform:
    """Tests for RunLengthTransform."""

    def test_basic_rle(self):
        """Test basic run-length encoding."""
        transform = RunLengthTransform(min_run=2)
        data = b"AAAABBCCCCCC"

        encoded = transform.apply(data)
        decoded = transform.reverse(encoded)

        assert decoded == data

    def test_rle_with_runs(self):
        """Test RLE with long runs."""
        transform = RunLengthTransform(min_run=4)
        data = b"A" * 100 + b"B" * 50 + b"C" * 25

        encoded = transform.apply(data)
        decoded = transform.reverse(encoded)

        assert decoded == data
        # Should be significantly smaller
        assert len(encoded) < len(data) * 0.5

    def test_rle_no_runs(self):
        """Test RLE with no runs."""
        transform = RunLengthTransform(min_run=4)
        data = b"ABCDEFGHIJ"

        encoded = transform.apply(data)
        decoded = transform.reverse(encoded)

        assert decoded == data

    def test_rle_small_data(self):
        """Test RLE with small data."""
        transform = RunLengthTransform()
        data = b"X"

        encoded = transform.apply(data)
        decoded = transform.reverse(encoded)

        assert decoded == data

    def test_name(self):
        """Test transform name."""
        transform = RunLengthTransform()
        assert transform.name == "run_length"


class TestPipelineMetrics:
    """Tests for PipelineMetrics."""

    def test_initial_values(self):
        """Test initial metric values."""
        metrics = PipelineMetrics()

        assert metrics.total_time_ms == 0.0
        assert metrics.input_size == 0
        assert metrics.output_size == 0

    def test_add_stage_metric(self):
        """Test adding stage metrics."""
        metrics = PipelineMetrics()
        metrics.add_stage_metric("compress", 1000, 250, 10.5)

        assert "compress" in metrics.stage_metrics
        assert metrics.stage_metrics["compress"]["input_size"] == 1000
        assert metrics.stage_metrics["compress"]["output_size"] == 250

    def test_update_totals(self):
        """Test updating totals."""
        metrics = PipelineMetrics()
        metrics.input_size = 1000
        metrics.output_size = 250
        metrics.update_totals()

        assert metrics.overall_ratio == 4.0

    def test_to_dict(self):
        """Test dictionary conversion."""
        metrics = PipelineMetrics()
        metrics.input_size = 1000
        metrics.output_size = 250
        metrics.total_time_ms = 15.5
        metrics.update_totals()

        data = metrics.to_dict()

        assert data["input_size"] == 1000
        assert data["output_size"] == 250
        assert data["overall_ratio"] == 4.0


class TestPipelineStage:
    """Tests for PipelineStage."""

    def test_transform_stage(self):
        """Test transform stage processing."""
        transform = IdentityTransform()
        stage = PipelineStage(
            name="identity",
            stage_type=StageType.TRANSFORM,
            processor=transform,
        )

        data = b"test data"
        result = stage.process(data)

        assert result == data

    def test_compress_stage(self):
        """Test compression stage processing."""
        compressor = get_compressor(CompressionAlgorithm.GZIP)
        stage = PipelineStage(
            name="compress",
            stage_type=StageType.COMPRESS,
            processor=compressor,
        )

        data = b"test data " * 100
        result = stage.process(data)

        assert len(result) < len(data)

    def test_disabled_stage(self):
        """Test disabled stage."""
        transform = DeltaEncodingTransform()
        stage = PipelineStage(
            name="delta",
            stage_type=StageType.TRANSFORM,
            processor=transform,
            enabled=False,
        )

        data = b"test data"
        result = stage.process(data)

        # Disabled, should return unchanged
        assert result == data

    def test_reverse_transform(self):
        """Test reversing transform stage."""
        transform = DeltaEncodingTransform()
        stage = PipelineStage(
            name="delta",
            stage_type=StageType.TRANSFORM,
            processor=transform,
        )

        data = bytes([10, 20, 30, 40])
        processed = stage.process(data)
        reversed_data = stage.reverse(processed)

        assert reversed_data == data

    def test_checksum_stage(self):
        """Test checksum stage."""
        stage = PipelineStage(
            name="checksum",
            stage_type=StageType.CHECKSUM,
            processor=None,
        )

        data = b"test data"
        with_checksum = stage.process(data)

        # Should be longer (checksum added)
        assert len(with_checksum) > len(data)

        # Should verify and return original
        verified = stage.reverse(with_checksum)
        assert verified == data


class TestCompressionPipeline:
    """Tests for CompressionPipeline."""

    def test_empty_pipeline(self):
        """Test empty pipeline."""
        pipeline = CompressionPipeline("empty")
        data = b"test data"

        result = pipeline.process(data)

        assert result.data == data
        assert result.stage_order == []

    def test_single_compression(self):
        """Test pipeline with single compression."""
        compressor = get_compressor(CompressionAlgorithm.GZIP)
        pipeline = CompressionPipeline("single").add_compression(compressor)

        data = b"test data " * 100
        result = pipeline.process(data)

        assert len(result.data) < len(data)
        assert len(result.stage_order) == 1

    def test_transform_then_compress(self):
        """Test transform followed by compression."""
        pipeline = (
            CompressionPipeline("combined")
            .add_transform(DeduplicationTransform(block_size=8))
            .add_compression(get_compressor(CompressionAlgorithm.GZIP))
        )

        data = b"AAAAAAAA" * 100  # Highly compressible
        result = pipeline.process(data)

        assert len(result.data) < len(data)
        assert len(result.stage_order) == 2

    def test_process_and_reverse(self):
        """Test processing and reversing."""
        pipeline = (
            CompressionPipeline("reversible")
            .add_transform(DeltaEncodingTransform())
            .add_compression(get_compressor(CompressionAlgorithm.GZIP))
        )

        data = bytes(range(100))
        result = pipeline.process(data)
        reversed_data = pipeline.reverse(result.data)

        assert reversed_data == data

    def test_enable_disable_stage(self):
        """Test enabling/disabling stages."""
        pipeline = (
            CompressionPipeline("toggle")
            .add_transform(DeltaEncodingTransform(), name="delta")
            .add_compression(get_compressor(CompressionAlgorithm.GZIP))
        )

        data = bytes(range(50))

        # Process with all stages
        result1 = pipeline.process(data)

        # Disable delta
        pipeline.enable_stage("delta", False)
        result2 = pipeline.process(data)

        # Results should differ
        assert result1.data != result2.data

    def test_remove_stage(self):
        """Test removing a stage."""
        pipeline = (
            CompressionPipeline("remove")
            .add_transform(DeltaEncodingTransform(), name="delta")
            .add_compression(get_compressor(CompressionAlgorithm.GZIP), name="gzip")
        )

        assert len(pipeline.stages) == 2

        pipeline.remove_stage("delta")

        assert len(pipeline.stages) == 1
        assert pipeline.stages[0].name == "gzip"

    def test_clone_pipeline(self):
        """Test cloning a pipeline."""
        original = (
            CompressionPipeline("original")
            .add_compression(get_compressor(CompressionAlgorithm.GZIP))
        )

        cloned = original.clone("cloned")

        assert cloned.name == "cloned"
        assert len(cloned.stages) == len(original.stages)

        # Modifying clone shouldn't affect original
        cloned.add_transform(IdentityTransform())
        assert len(cloned.stages) != len(original.stages)

    def test_metrics_tracking(self):
        """Test that metrics are tracked."""
        pipeline = (
            CompressionPipeline("metrics")
            .add_compression(get_compressor(CompressionAlgorithm.GZIP))
        )

        data = b"test data " * 100
        result = pipeline.process(data)

        assert result.metrics.input_size == len(data)
        assert result.metrics.output_size == len(result.data)
        assert result.metrics.total_time_ms > 0

    def test_config_snapshot(self):
        """Test configuration snapshot."""
        pipeline = (
            CompressionPipeline("snapshot")
            .add_transform(IdentityTransform(), name="id")
            .add_compression(get_compressor(CompressionAlgorithm.GZIP), name="gzip")
        )

        result = pipeline.process(b"test")

        assert result.config_snapshot["name"] == "snapshot"
        assert len(result.config_snapshot["stages"]) == 2


class TestPrebuiltPipelines:
    """Tests for pre-built pipelines."""

    def test_text_pipeline(self):
        """Test text pipeline."""
        pipeline = create_text_pipeline()
        data = b"Hello, World! " * 100

        result = pipeline.process(data)
        reversed_data = pipeline.reverse(result.data)

        assert reversed_data == data
        assert len(result.data) < len(data)

    def test_json_pipeline(self):
        """Test JSON pipeline."""
        pipeline = create_json_pipeline()
        data = b'{"key": "value"}' * 100

        result = pipeline.process(data)
        reversed_data = pipeline.reverse(result.data)

        assert reversed_data == data

    def test_binary_pipeline(self):
        """Test binary pipeline."""
        pipeline = create_binary_pipeline()
        data = b"\x00" * 500 + b"\xFF" * 500

        result = pipeline.process(data)
        reversed_data = pipeline.reverse(result.data)

        assert reversed_data == data

    def test_timeseries_pipeline(self):
        """Test timeseries pipeline."""
        pipeline = create_timeseries_pipeline()
        data = bytes(range(200))

        result = pipeline.process(data)
        reversed_data = pipeline.reverse(result.data)

        assert reversed_data == data


class TestPipelineErrors:
    """Tests for pipeline error handling."""

    def test_transform_error_propagation(self):
        """Test that transform errors propagate."""

        class FailingTransform(Transform):
            @property
            def name(self):
                return "failing"

            def apply(self, data):
                raise ValueError("Intentional failure")

            def reverse(self, data):
                return data

        pipeline = CompressionPipeline("failing").add_transform(FailingTransform())

        with pytest.raises(PipelineError):
            pipeline.process(b"test")

    def test_invalid_checksum(self):
        """Test invalid checksum detection."""
        stage = PipelineStage(
            name="checksum",
            stage_type=StageType.CHECKSUM,
            processor=None,
        )

        data = b"test data"
        with_checksum = stage.process(data)

        # Corrupt the data
        corrupted = with_checksum[:32] + b"X" + with_checksum[33:]

        with pytest.raises(PipelineError, match="Checksum"):
            stage.reverse(corrupted)
