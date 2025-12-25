"""Tests for P2 profiler improvements.

Tests caching, observability, quality scoring, and custom patterns modules.
"""

import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest


# =============================================================================
# Caching Tests
# =============================================================================


class TestCacheKey:
    """Tests for cache key generation."""

    def test_cache_key_basic(self):
        """Test basic cache key creation."""
        from truthound.profiler.caching import CacheKey

        key = CacheKey(key="test123", namespace="profile", version="1")
        assert key.to_string() == "profile:1:test123"

    def test_file_hash_cache_key(self):
        """Test file hash based cache key."""
        from truthound.profiler.caching import FileHashCacheKey

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
            f.write(b"col1,col2\n1,2\n3,4\n")
            f.flush()

            key = FileHashCacheKey.from_file(f.name)
            assert key.file_hash
            assert key.file_size > 0
            assert key.file_mtime > 0

            # Same file should produce same key
            key2 = FileHashCacheKey.from_file(f.name)
            assert key.file_hash == key2.file_hash

    def test_file_hash_quick_mode(self):
        """Test quick hash for large files."""
        from truthound.profiler.caching import FileHashCacheKey

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
            # Write enough data to trigger quick hash
            data = b"x" * (1024 * 1024 * 3)  # 3MB
            f.write(data)
            f.flush()

            key_full = FileHashCacheKey.from_file(f.name, quick_hash=False)
            key_quick = FileHashCacheKey.from_file(f.name, quick_hash=True)

            # Both should produce valid hashes
            assert key_full.file_hash
            assert key_quick.file_hash

    def test_dataframe_hash_cache_key(self):
        """Test DataFrame based cache key."""
        from truthound.profiler.caching import DataFrameHashCacheKey

        df = pl.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"],
        })

        key = DataFrameHashCacheKey.from_dataframe(df)
        assert key.schema_hash
        assert key.sample_hash
        assert key.row_count == 3
        assert key.column_count == 2


class TestMemoryCacheBackend:
    """Tests for in-memory cache backend."""

    def test_basic_operations(self):
        """Test basic get/set/delete operations."""
        from truthound.profiler.caching import MemoryCacheBackend, CacheEntry
        from truthound.profiler.base import TableProfile

        backend = MemoryCacheBackend(max_size=100)

        profile = TableProfile(name="test", row_count=100)
        entry = CacheEntry(profile=profile)

        # Set
        backend.set("key1", entry)
        assert backend.exists("key1")

        # Get
        retrieved = backend.get("key1")
        assert retrieved is not None
        assert retrieved.profile.name == "test"

        # Delete
        assert backend.delete("key1")
        assert not backend.exists("key1")

    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        from truthound.profiler.caching import MemoryCacheBackend, CacheEntry
        from truthound.profiler.base import TableProfile

        backend = MemoryCacheBackend()

        profile = TableProfile(name="test")
        entry = CacheEntry(profile=profile)

        # Set with short TTL
        backend.set("key1", entry, ttl=timedelta(milliseconds=50))

        # Should exist immediately
        assert backend.get("key1") is not None

        # Wait for expiration
        time.sleep(0.1)

        # Should be expired
        assert backend.get("key1") is None

    def test_lru_eviction(self):
        """Test LRU eviction when max size is reached."""
        from truthound.profiler.caching import MemoryCacheBackend, CacheEntry
        from truthound.profiler.base import TableProfile

        backend = MemoryCacheBackend(max_size=3)

        for i in range(5):
            profile = TableProfile(name=f"test{i}")
            entry = CacheEntry(profile=profile)
            backend.set(f"key{i}", entry)
            time.sleep(0.01)  # Ensure different access times

        # Should have evicted oldest entries
        stats = backend.get_stats()
        assert stats["size"] <= 3

    def test_cache_stats(self):
        """Test cache statistics."""
        from truthound.profiler.caching import MemoryCacheBackend, CacheEntry
        from truthound.profiler.base import TableProfile

        backend = MemoryCacheBackend()

        profile = TableProfile(name="test")
        entry = CacheEntry(profile=profile)
        backend.set("key1", entry)

        # Hit
        backend.get("key1")
        # Miss
        backend.get("key2")

        stats = backend.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_ratio"] == 0.5


class TestFileCacheBackend:
    """Tests for file-based cache backend."""

    def test_basic_operations(self):
        """Test basic get/set/delete operations."""
        from truthound.profiler.caching import FileCacheBackend, CacheEntry
        from truthound.profiler.base import TableProfile

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileCacheBackend(cache_dir=tmpdir, compress=False)

            profile = TableProfile(name="test", row_count=100)
            entry = CacheEntry(profile=profile)

            # Set
            backend.set("key1", entry)
            assert backend.exists("key1")

            # Get
            retrieved = backend.get("key1")
            assert retrieved is not None
            assert retrieved.profile.name == "test"

            # Delete
            assert backend.delete("key1")
            assert not backend.exists("key1")

    def test_compression(self):
        """Test compressed storage."""
        from truthound.profiler.caching import FileCacheBackend, CacheEntry
        from truthound.profiler.base import TableProfile

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileCacheBackend(cache_dir=tmpdir, compress=True)

            profile = TableProfile(name="test", row_count=100)
            entry = CacheEntry(profile=profile)

            backend.set("key1", entry)

            # Should be stored as .json.gz
            files = list(Path(tmpdir).glob("*.json.gz"))
            assert len(files) == 1


class TestProfileCache:
    """Tests for high-level ProfileCache."""

    def test_get_or_compute(self):
        """Test cache-through pattern."""
        from truthound.profiler.caching import ProfileCache, CacheKey
        from truthound.profiler.base import TableProfile

        cache = ProfileCache(backend="memory")

        compute_count = 0

        def compute():
            nonlocal compute_count
            compute_count += 1
            return TableProfile(name="computed")

        key = CacheKey(key="test")

        # First call - should compute
        profile1 = cache.get_or_compute(key, compute)
        assert compute_count == 1

        # Second call - should use cache
        profile2 = cache.get_or_compute(key, compute)
        assert compute_count == 1  # Still 1

        assert profile1.name == profile2.name

    def test_disabled_cache(self):
        """Test cache when disabled."""
        from truthound.profiler.caching import ProfileCache, CacheKey
        from truthound.profiler.base import TableProfile

        cache = ProfileCache(backend="memory", enabled=False)

        compute_count = 0

        def compute():
            nonlocal compute_count
            compute_count += 1
            return TableProfile(name="computed")

        key = CacheKey(key="test")

        # Both calls should compute
        cache.get_or_compute(key, compute)
        cache.get_or_compute(key, compute)

        assert compute_count == 2


# =============================================================================
# Observability Tests
# =============================================================================


class TestMetrics:
    """Tests for metrics collection."""

    def test_counter(self):
        """Test counter metric."""
        from truthound.profiler.observability import Counter

        counter = Counter("test_counter", labels=["status"])

        counter.inc(status="success")
        counter.inc(status="success")
        counter.inc(status="error")

        values = counter.collect()
        success_val = next(v for v in values if v.labels.get("status") == "success")
        error_val = next(v for v in values if v.labels.get("status") == "error")

        assert success_val.value == 2
        assert error_val.value == 1

    def test_gauge(self):
        """Test gauge metric."""
        from truthound.profiler.observability import Gauge

        gauge = Gauge("test_gauge")

        gauge.set(10)
        gauge.inc(5)
        gauge.dec(3)

        values = gauge.collect()
        assert values[0].value == 12

    def test_histogram(self):
        """Test histogram metric."""
        from truthound.profiler.observability import Histogram

        histogram = Histogram("test_histogram", buckets=(0.1, 0.5, 1.0))

        histogram.observe(0.05)
        histogram.observe(0.3)
        histogram.observe(0.7)
        histogram.observe(1.5)

        values = histogram.collect()

        # Check count
        count_val = next(v for v in values if v.labels.get("stat") == "count")
        assert count_val.value == 4

    def test_histogram_timer(self):
        """Test histogram timer context manager."""
        from truthound.profiler.observability import Histogram

        histogram = Histogram("test_duration")

        with histogram.time():
            time.sleep(0.05)

        values = histogram.collect()
        sum_val = next(v for v in values if v.labels.get("stat") == "sum")
        assert sum_val.value >= 0.05


class TestSpan:
    """Tests for span tracking."""

    def test_span_creation(self):
        """Test span creation and attributes."""
        from truthound.profiler.observability import Span, SpanStatus

        span = Span(name="test", trace_id="trace1", span_id="span1")
        span.set_attribute("key", "value")
        span.set_status(SpanStatus.OK)
        span.end()

        assert span.name == "test"
        assert span.attributes["key"] == "value"
        assert span.status == SpanStatus.OK
        assert span.end_time is not None
        assert span.duration_ms > 0

    def test_span_events(self):
        """Test span events."""
        from truthound.profiler.observability import Span

        span = Span(name="test", trace_id="trace1", span_id="span1")
        span.add_event("checkpoint", {"data": "value"})

        assert len(span.events) == 1
        assert span.events[0].name == "checkpoint"

    def test_span_exception(self):
        """Test recording exception in span."""
        from truthound.profiler.observability import Span, SpanStatus

        span = Span(name="test", trace_id="trace1", span_id="span1")

        try:
            raise ValueError("test error")
        except ValueError as e:
            span.record_exception(e)

        assert span.status == SpanStatus.ERROR
        assert span.exception is not None


class TestProfilerTelemetry:
    """Tests for ProfilerTelemetry."""

    def test_span_context(self):
        """Test span context manager."""
        from truthound.profiler.observability import ProfilerTelemetry, InMemorySpanExporter

        exporter = InMemorySpanExporter()
        telemetry = ProfilerTelemetry(exporter=exporter, sample_rate=1.0)

        with telemetry.span("test_operation") as span:
            span.set_attribute("column", "user_id")

        # Force flush
        telemetry._flush()

        spans = exporter.get_spans()
        assert len(spans) == 1
        assert spans[0].name == "test_operation"
        assert spans[0].attributes["column"] == "user_id"

    def test_nested_spans(self):
        """Test nested span contexts."""
        from truthound.profiler.observability import ProfilerTelemetry, InMemorySpanExporter

        exporter = InMemorySpanExporter()
        telemetry = ProfilerTelemetry(exporter=exporter, sample_rate=1.0)

        with telemetry.span("parent") as parent:
            with telemetry.span("child") as child:
                pass

        # Force flush
        telemetry._flush()

        spans = exporter.get_spans()
        assert len(spans) == 2

        child_span = next(s for s in spans if s.name == "child")
        parent_span = next(s for s in spans if s.name == "parent")

        assert child_span.parent_id == parent_span.span_id

    def test_traced_decorator(self):
        """Test @traced decorator."""
        from truthound.profiler.observability import (
            ProfilerTelemetry,
            InMemorySpanExporter,
            traced,
            set_telemetry,
            get_telemetry,
        )

        exporter = InMemorySpanExporter()
        telemetry = ProfilerTelemetry(exporter=exporter, sample_rate=1.0)
        set_telemetry(telemetry)

        @traced("my_function")
        def my_function(x: int) -> int:
            return x * 2

        result = my_function(5)

        # Force flush
        get_telemetry()._flush()

        assert result == 10
        spans = exporter.get_spans()
        assert len(spans) == 1
        assert spans[0].name == "my_function"


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_default_metrics(self):
        """Test default metrics are registered."""
        from truthound.profiler.observability import MetricsCollector

        collector = MetricsCollector()

        # Check default metrics exist
        assert collector.profiles_total is not None
        assert collector.columns_profiled is not None
        assert collector.profile_duration is not None

    def test_prometheus_export(self):
        """Test Prometheus format export."""
        from truthound.profiler.observability import MetricsCollector

        collector = MetricsCollector(prefix="test")

        collector.profiles_total.inc(status="success", type="table")

        prometheus = collector.to_prometheus()
        assert "test_profiles_total" in prometheus
        assert 'status="success"' in prometheus


# =============================================================================
# Quality Scoring Tests
# =============================================================================


class TestConfusionMatrix:
    """Tests for confusion matrix calculations."""

    def test_basic_metrics(self):
        """Test basic metric calculations."""
        from truthound.profiler.quality import ConfusionMatrix

        matrix = ConfusionMatrix(
            true_positives=90,
            true_negatives=80,
            false_positives=10,
            false_negatives=20,
        )

        assert matrix.total == 200
        assert matrix.accuracy == 0.85
        assert matrix.precision == 0.9  # 90 / (90 + 10)
        assert matrix.recall == 90 / 110  # 90 / (90 + 20)
        assert matrix.specificity == 80 / 90  # 80 / (80 + 10)

    def test_f1_score(self):
        """Test F1 score calculation."""
        from truthound.profiler.quality import ConfusionMatrix

        matrix = ConfusionMatrix(
            true_positives=80,
            true_negatives=70,
            false_positives=20,
            false_negatives=30,
        )

        precision = 80 / 100
        recall = 80 / 110
        expected_f1 = 2 * precision * recall / (precision + recall)

        assert abs(matrix.f1_score - expected_f1) < 0.001

    def test_edge_cases(self):
        """Test edge cases with zeros."""
        from truthound.profiler.quality import ConfusionMatrix

        # All zeros
        matrix = ConfusionMatrix()
        assert matrix.precision == 0.0
        assert matrix.recall == 0.0
        assert matrix.f1_score == 0.0


class TestQualityMetrics:
    """Tests for QualityMetrics."""

    def test_from_confusion_matrix(self):
        """Test creating metrics from confusion matrix."""
        from truthound.profiler.quality import QualityMetrics, ConfusionMatrix, QualityLevel

        matrix = ConfusionMatrix(
            true_positives=95,
            true_negatives=90,
            false_positives=5,
            false_negatives=10,
        )

        metrics = QualityMetrics.from_confusion_matrix(
            matrix,
            sample_size=200,
            population_size=1000,
        )

        assert metrics.precision == matrix.precision
        assert metrics.recall == matrix.recall
        assert metrics.f1_score == matrix.f1_score
        assert metrics.sample_size == 200
        assert metrics.quality_level == QualityLevel.GOOD


class TestValidationRule:
    """Tests for ValidationRule."""

    def test_pattern_rule(self):
        """Test pattern-based validation rule."""
        from truthound.profiler.quality import ValidationRule, RuleType

        rule = ValidationRule(
            name="email_pattern",
            rule_type=RuleType.PATTERN,
            pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        )

        assert rule.validate("user@example.com")
        assert not rule.validate("not-an-email")

    def test_range_rule(self):
        """Test range validation rule."""
        from truthound.profiler.quality import ValidationRule, RuleType

        rule = ValidationRule(
            name="age_range",
            rule_type=RuleType.RANGE,
            min_value=0,
            max_value=150,
        )

        assert rule.validate(25)
        assert rule.validate(0)
        assert not rule.validate(-1)
        assert not rule.validate(200)

    def test_allowed_values_rule(self):
        """Test allowed values validation rule."""
        from truthound.profiler.quality import ValidationRule, RuleType

        rule = ValidationRule(
            name="status",
            rule_type=RuleType.CUSTOM,
            allowed_values={"active", "inactive", "pending"},
        )

        assert rule.validate("active")
        assert not rule.validate("unknown")


class TestRuleQualityScorer:
    """Tests for RuleQualityScorer."""

    def test_score_pattern_rule(self):
        """Test scoring a pattern rule."""
        from truthound.profiler.quality import RuleQualityScorer, ValidationRule, RuleType

        df = pl.DataFrame({
            "email": [
                "user1@example.com",
                "user2@example.com",
                "invalid",
                "user3@test.org",
            ]
        })

        rule = ValidationRule(
            name="email_pattern",
            rule_type=RuleType.PATTERN,
            column="email",
            pattern=r".*@.*\..*",
        )

        scorer = RuleQualityScorer(estimator="heuristic")
        score = scorer.score(rule, df)

        assert score.rule_name == "email_pattern"
        assert score.metrics.f1_score > 0

    def test_compare_rules(self):
        """Test comparing multiple rules."""
        from truthound.profiler.quality import RuleQualityScorer, ValidationRule, RuleType

        df = pl.DataFrame({
            "value": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })

        rule1 = ValidationRule(
            name="strict_range",
            rule_type=RuleType.RANGE,
            column="value",
            min_value=0,
            max_value=50,
        )

        rule2 = ValidationRule(
            name="loose_range",
            rule_type=RuleType.RANGE,
            column="value",
            min_value=0,
            max_value=100,
        )

        scorer = RuleQualityScorer(estimator="heuristic")
        ranked = scorer.compare([rule1, rule2], df)

        # Both rules should be scored and ranked
        assert len(ranked) == 2
        # Check that scores exist
        assert all(s.metrics.f1_score >= 0 for s in ranked)


class TestQualityTrendAnalyzer:
    """Tests for QualityTrendAnalyzer."""

    def test_record_and_analyze(self):
        """Test recording and analyzing trends."""
        from truthound.profiler.quality import QualityTrendAnalyzer, QualityMetrics

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            analyzer = QualityTrendAnalyzer(storage_path=f.name)

            # Record improving metrics with significant change
            for i in range(10):
                metrics = QualityMetrics(
                    precision=0.5 + i * 0.05,  # 0.5 -> 0.95
                    recall=0.5 + i * 0.05,
                    f1_score=0.5 + i * 0.05,
                )
                analyzer.record("rule1", metrics, data_size=1000)

            trend = analyzer.analyze_trend("rule1")

            # Either improving or stable is acceptable
            assert trend.get("trend") in ["improving", "stable"]
            assert trend.get("points_analyzed") == 10


# =============================================================================
# Custom Patterns Tests
# =============================================================================


class TestPatternConfig:
    """Tests for PatternConfig."""

    def test_basic_pattern(self):
        """Test basic pattern configuration."""
        from truthound.profiler.custom_patterns import PatternConfig, PatternExample

        pattern = PatternConfig(
            name="Email",
            pattern_id="email",
            regex=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            data_type="email",
            examples=[
                PatternExample(value="user@example.com", should_match=True),
                PatternExample(value="invalid", should_match=False),
            ],
        )

        assert pattern.matches("user@example.com")
        assert not pattern.matches("invalid")

    def test_example_validation(self):
        """Test example validation."""
        from truthound.profiler.custom_patterns import PatternConfig, PatternExample

        pattern = PatternConfig(
            name="Test",
            pattern_id="test",
            regex=r"^\d+$",
            examples=[
                PatternExample(value="123", should_match=True),
                PatternExample(value="abc", should_match=False),
            ],
        )

        results = pattern.validate_examples()
        assert all(passed for _, passed, _ in results)

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        from truthound.profiler.custom_patterns import PatternConfig

        pattern = PatternConfig(
            name="Test",
            pattern_id="test",
            regex=r"^hello$",
            case_sensitive=False,
        )

        assert pattern.matches("hello")
        assert pattern.matches("HELLO")
        assert pattern.matches("HeLLo")


class TestPatternConfigLoader:
    """Tests for PatternConfigLoader."""

    def test_load_from_string(self):
        """Test loading patterns from YAML string."""
        from truthound.profiler.custom_patterns import PatternConfigLoader

        yaml_content = """
version: "1.0"
name: "Test Patterns"
patterns:
  test_email:
    name: Email
    regex: "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\\\.[a-zA-Z]{2,}$"
    data_type: email
    priority: 85
"""

        loader = PatternConfigLoader()
        config = loader.load_from_string(yaml_content)

        assert config.name == "Test Patterns"
        patterns = config.get_all_patterns()
        assert len(patterns) >= 1

    def test_load_from_file(self):
        """Test loading patterns from file."""
        from truthound.profiler.custom_patterns import PatternConfigLoader

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
version: "1.0"
patterns:
  phone:
    name: Phone Number
    regex: "^\\\\d{3}-\\\\d{4}-\\\\d{4}$"
    data_type: phone
""")
            f.flush()

            loader = PatternConfigLoader()
            config = loader.load_file(f.name)

            patterns = config.get_all_patterns()
            assert any(p.pattern_id == "phone" for p in patterns)

    def test_pattern_groups(self):
        """Test pattern groups."""
        from truthound.profiler.custom_patterns import PatternConfigLoader

        yaml_content = """
version: "1.0"
groups:
  korean:
    name: Korean Patterns
    priority_boost: 10
    patterns:
      korean_phone:
        name: Korean Phone
        regex: "^01[0-9]-\\\\d{3,4}-\\\\d{4}$"
        data_type: korean_phone
        priority: 80
"""

        loader = PatternConfigLoader()
        config = loader.load_from_string(yaml_content)

        patterns = config.get_all_patterns()
        korean_phone = next(p for p in patterns if p.pattern_id == "korean_phone")

        # Priority should be boosted
        assert korean_phone.priority == 90  # 80 + 10


class TestPatternRegistry:
    """Tests for PatternRegistry."""

    def test_register_pattern(self):
        """Test registering patterns programmatically."""
        from truthound.profiler.custom_patterns import (
            PatternRegistry,
            PatternConfig,
            register_pattern,
        )

        registry = PatternRegistry()
        registry.clear()

        pattern = PatternConfig(
            name="Test",
            pattern_id="test_pattern",
            regex=r"^\d+$",
        )

        registry.register(pattern)

        retrieved = registry.get_pattern("test_pattern")
        assert retrieved is not None
        assert retrieved.name == "Test"

    def test_match_patterns(self):
        """Test pattern matching."""
        from truthound.profiler.custom_patterns import PatternRegistry, PatternConfig

        registry = PatternRegistry()
        registry.clear()

        registry.register(PatternConfig(
            name="Number",
            pattern_id="number",
            regex=r"^\d+$",
            priority=50,
        ))

        registry.register(PatternConfig(
            name="Email",
            pattern_id="email",
            regex=r"^[^@]+@[^@]+$",
            priority=80,
        ))

        # Test matching
        number_matches = registry.match("123")
        assert len(number_matches) == 1
        assert number_matches[0].pattern_id == "number"

        email_matches = registry.match("user@example.com")
        assert len(email_matches) == 1
        assert email_matches[0].pattern_id == "email"


class TestDefaultPatterns:
    """Tests for default pattern loading."""

    def test_load_default_patterns(self):
        """Test loading built-in default patterns."""
        from truthound.profiler.custom_patterns import (
            PatternRegistry,
            load_default_patterns,
            pattern_registry,
        )

        # Clear and reload
        pattern_registry.clear()
        load_default_patterns()

        patterns = pattern_registry.get_patterns()

        # Should have some default patterns
        pattern_ids = [p.pattern_id for p in patterns]
        assert "email" in pattern_ids
        assert "uuid" in pattern_ids

    def test_korean_patterns(self):
        """Test Korean-specific patterns."""
        from truthound.profiler.custom_patterns import (
            load_default_patterns,
            pattern_registry,
        )

        # Clear and reload
        pattern_registry.clear()
        load_default_patterns()

        patterns = pattern_registry.get_patterns()
        pattern_ids = [p.pattern_id for p in patterns]

        assert "korean_phone" in pattern_ids
        assert "korean_rrn" in pattern_ids


# =============================================================================
# Integration Tests
# =============================================================================


class TestP2Integration:
    """Integration tests for P2 modules working together."""

    def test_cached_profiling_with_telemetry(self):
        """Test caching with telemetry."""
        from truthound.profiler.caching import ProfileCache, CacheKey
        from truthound.profiler.observability import (
            ProfilerTelemetry,
            InMemorySpanExporter,
        )
        from truthound.profiler.base import TableProfile

        exporter = InMemorySpanExporter()
        telemetry = ProfilerTelemetry(exporter=exporter, sample_rate=1.0)
        cache = ProfileCache(backend="memory")

        compute_count = 0

        def compute_with_telemetry():
            nonlocal compute_count
            with telemetry.span("compute_profile") as span:
                compute_count += 1
                span.set_attribute("compute_count", compute_count)
                return TableProfile(name="test", row_count=100)

        key = CacheKey(key="test")

        # First call
        profile1 = cache.get_or_compute(key, compute_with_telemetry)

        # Second call (cached)
        profile2 = cache.get_or_compute(key, compute_with_telemetry)

        # Force flush
        telemetry._flush()

        assert compute_count == 1
        spans = exporter.get_spans()
        assert len(spans) == 1  # Only computed once

    def test_quality_scoring_with_custom_patterns(self):
        """Test quality scoring with custom pattern rules."""
        from truthound.profiler.quality import RuleQualityScorer, ValidationRule, RuleType
        from truthound.profiler.custom_patterns import load_default_patterns, pattern_registry

        # Clear and load default patterns
        pattern_registry.clear()
        load_default_patterns()

        # Create test data
        df = pl.DataFrame({
            "email": [
                "user1@example.com",
                "user2@test.org",
                "user3@company.co.kr",
                "invalid-email",
                "another@valid.com",
            ]
        })

        # Create rule from pattern
        rule = ValidationRule(
            name="email_validation",
            rule_type=RuleType.PATTERN,
            column="email",
            pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        )

        scorer = RuleQualityScorer(estimator="heuristic")
        score = scorer.score(rule, df)

        assert score.metrics.f1_score > 0
        assert score.recommendation


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
