"""Tests for PHASE 3: Metric Deduplication.

Covers:
    - MetricKey identity and hashing
    - SharedMetricStore basic operations
    - SharedMetricStore thread safety
    - CommonMetrics expression library
    - metric_key_to_expr resolution
    - Validator.get_required_metrics() declarations
    - ExpressionBatchExecutor metric precomputation
    - End-to-end deduplication correctness
    - Backward compatibility (no metric_store path)
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import polars as pl
import pytest

from truthound.validators.metrics import (
    CommonMetrics,
    MetricKey,
    MetricStoreStats,
    SharedMetricStore,
    metric_key_to_expr,
)


# ============================================================================
# 1. MetricKey Tests
# ============================================================================


class TestMetricKey:
    """Tests for MetricKey identity, equality, and hashing."""

    def test_basic_identity(self):
        k1 = MetricKey("null_count", "col_a")
        k2 = MetricKey("null_count", "col_a")
        assert k1 == k2
        assert hash(k1) == hash(k2)

    def test_different_metric_name(self):
        k1 = MetricKey("null_count", "col_a")
        k2 = MetricKey("non_null_count", "col_a")
        assert k1 != k2

    def test_different_column(self):
        k1 = MetricKey("null_count", "col_a")
        k2 = MetricKey("null_count", "col_b")
        assert k1 != k2

    def test_table_level_metric(self):
        k = MetricKey("row_count", None)
        assert k.column is None
        assert k.kwargs_hash == ""

    def test_create_factory_without_kwargs(self):
        k = MetricKey.create("null_count", "col_a")
        assert k.kwargs_hash == ""
        assert k == MetricKey("null_count", "col_a")

    def test_create_factory_with_kwargs(self):
        k1 = MetricKey.create("quantile", "col_a", q=0.5)
        k2 = MetricKey.create("quantile", "col_a", q=0.5)
        k3 = MetricKey.create("quantile", "col_a", q=0.9)
        assert k1 == k2
        assert k1 != k3
        assert k1.kwargs_hash != ""

    def test_frozen_dataclass(self):
        k = MetricKey("null_count", "col_a")
        with pytest.raises(AttributeError):
            k.metric_name = "other"  # type: ignore

    def test_usable_as_dict_key(self):
        k1 = MetricKey("null_count", "col_a")
        k2 = MetricKey("null_count", "col_a")
        d = {k1: 42}
        assert d[k2] == 42

    def test_repr(self):
        k = MetricKey("null_count", "col_a")
        assert "null_count" in repr(k)


# ============================================================================
# 2. SharedMetricStore Tests
# ============================================================================


class TestSharedMetricStore:
    """Tests for SharedMetricStore cache operations."""

    def test_put_get(self):
        store = SharedMetricStore()
        key = MetricKey("row_count")
        store.put(key, 1000)
        assert store.get(key) == 1000

    def test_get_miss(self):
        store = SharedMetricStore()
        key = MetricKey("row_count")
        assert store.get(key) is None

    def test_get_or_compute_first_call(self):
        store = SharedMetricStore()
        key = MetricKey("row_count")
        value = store.get_or_compute(key, lambda: 42)
        assert value == 42
        assert store.get(key) == 42

    def test_get_or_compute_cached(self):
        store = SharedMetricStore()
        key = MetricKey("row_count")
        store.put(key, 100)
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return 999

        value = store.get_or_compute(key, compute)
        assert value == 100
        assert call_count == 0  # compute_fn was NOT called

    def test_put_many(self):
        store = SharedMetricStore()
        keys = {
            MetricKey("null_count", "a"): 5,
            MetricKey("null_count", "b"): 10,
            MetricKey("row_count"): 100,
        }
        store.put_many(keys)
        assert store.get(MetricKey("null_count", "a")) == 5
        assert store.get(MetricKey("row_count")) == 100
        assert len(store) == 3

    def test_get_many(self):
        store = SharedMetricStore()
        store.put(MetricKey("a"), 1)
        store.put(MetricKey("b"), 2)
        result = store.get_many([MetricKey("a"), MetricKey("b"), MetricKey("c")])
        assert MetricKey("a") in result
        assert MetricKey("b") in result
        assert MetricKey("c") not in result

    def test_missing_keys(self):
        store = SharedMetricStore()
        store.put(MetricKey("a"), 1)
        missing = store.missing_keys([MetricKey("a"), MetricKey("b")])
        assert missing == [MetricKey("b")]

    def test_clear(self):
        store = SharedMetricStore()
        store.put(MetricKey("a"), 1)
        store.clear()
        assert len(store) == 0
        assert store.get(MetricKey("a")) is None

    def test_contains(self):
        store = SharedMetricStore()
        key = MetricKey("a")
        assert key not in store
        store.put(key, 1)
        assert key in store

    def test_stats_tracking(self):
        store = SharedMetricStore()
        key = MetricKey("a")
        store.put(key, 1)
        store.get(key)  # hit
        store.get(MetricKey("b"))  # miss

        stats = store.get_stats_dict()
        assert stats["total_lookups"] == 2
        assert stats["cache_hits"] == 1
        assert stats["metrics_computed"] == 1
        assert stats["hit_ratio"] == 0.5

    def test_repr(self):
        store = SharedMetricStore()
        store.put(MetricKey("a"), 1)
        r = repr(store)
        assert "size=1" in r


class TestSharedMetricStoreThreadSafety:
    """Thread-safety tests for SharedMetricStore."""

    def test_concurrent_put_get(self):
        store = SharedMetricStore()
        errors: list[str] = []

        def writer(idx: int):
            key = MetricKey("metric", f"col_{idx}")
            store.put(key, idx * 10)

        def reader(idx: int):
            key = MetricKey("metric", f"col_{idx}")
            val = store.get(key)
            if val is not None and val != idx * 10:
                errors.append(f"col_{idx}: expected {idx * 10}, got {val}")

        with ThreadPoolExecutor(max_workers=8) as executor:
            # Write
            list(executor.map(writer, range(50)))
            # Read
            list(executor.map(reader, range(50)))

        assert errors == []

    def test_concurrent_get_or_compute(self):
        """Multiple threads requesting the same key — only one should compute."""
        store = SharedMetricStore()
        key = MetricKey("expensive_metric")
        compute_count = 0
        lock = threading.Lock()

        def compute():
            nonlocal compute_count
            with lock:
                compute_count += 1
            import time
            time.sleep(0.01)  # Simulate expensive computation
            return 42

        results: list[Any] = []

        def worker():
            val = store.get_or_compute(key, compute)
            results.append(val)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r == 42 for r in results)
        assert compute_count == 1  # Computed exactly once


# ============================================================================
# 3. CommonMetrics Tests
# ============================================================================


class TestCommonMetrics:
    """Tests for CommonMetrics expression library."""

    @pytest.fixture
    def sample_lf(self) -> pl.LazyFrame:
        return pl.DataFrame({
            "a": [1, 2, None, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "c": ["x", "y", "x", None, "z"],
        }).lazy()

    def test_row_count(self, sample_lf: pl.LazyFrame):
        key, expr = CommonMetrics.row_count()
        result = sample_lf.select(expr).collect()
        assert result.item() == 5
        assert key == MetricKey("row_count", None)

    def test_null_count(self, sample_lf: pl.LazyFrame):
        key, expr = CommonMetrics.null_count("a")
        result = sample_lf.select(expr).collect()
        assert result.item() == 1
        assert key == MetricKey("null_count", "a")

    def test_non_null_count(self, sample_lf: pl.LazyFrame):
        key, expr = CommonMetrics.non_null_count("a")
        result = sample_lf.select(expr).collect()
        assert result.item() == 4

    def test_n_unique(self, sample_lf: pl.LazyFrame):
        key, expr = CommonMetrics.n_unique("c")
        result = sample_lf.select(expr).collect()
        assert result.item() == 4  # "x", "y", "z", null

    def test_mean(self, sample_lf: pl.LazyFrame):
        key, expr = CommonMetrics.mean("b")
        result = sample_lf.select(expr).collect()
        assert result.item() == 30.0

    def test_std(self, sample_lf: pl.LazyFrame):
        key, expr = CommonMetrics.std("b")
        result = sample_lf.select(expr).collect()
        assert result.item() is not None

    def test_min_max(self, sample_lf: pl.LazyFrame):
        _, min_expr = CommonMetrics.min("b")
        _, max_expr = CommonMetrics.max("b")
        result = sample_lf.select(min_expr, max_expr).collect()
        assert result["_metric_min_b"][0] == 10
        assert result["_metric_max_b"][0] == 50

    def test_sum(self, sample_lf: pl.LazyFrame):
        key, expr = CommonMetrics.sum("b")
        result = sample_lf.select(expr).collect()
        assert result.item() == 150

    def test_quantile(self, sample_lf: pl.LazyFrame):
        key, expr = CommonMetrics.quantile("b", 0.5)
        result = sample_lf.select(expr).collect()
        assert result.item() is not None
        assert key.kwargs_hash != ""

    def test_median(self, sample_lf: pl.LazyFrame):
        key, expr = CommonMetrics.median("b")
        result = sample_lf.select(expr).collect()
        assert result.item() == 30.0

    def test_multiple_metrics_single_collect(self, sample_lf: pl.LazyFrame):
        """All metrics can be computed in a single collect()."""
        _, row_expr = CommonMetrics.row_count()
        _, null_expr = CommonMetrics.null_count("a")
        _, nn_expr = CommonMetrics.non_null_count("a")
        _, mean_expr = CommonMetrics.mean("b")

        result = sample_lf.select(row_expr, null_expr, nn_expr, mean_expr).collect()
        row = result.row(0, named=True)
        assert row["_metric_row_count"] == 5
        assert row["_metric_null_count_a"] == 1
        assert row["_metric_non_null_count_a"] == 4
        assert row["_metric_mean_b"] == 30.0


# ============================================================================
# 4. metric_key_to_expr Tests
# ============================================================================


class TestMetricKeyToExpr:
    """Tests for resolving MetricKey to Polars expressions."""

    def test_row_count(self):
        key = MetricKey("row_count")
        expr = metric_key_to_expr(key)
        assert expr is not None

    def test_null_count(self):
        key = MetricKey("null_count", "col_a")
        expr = metric_key_to_expr(key)
        assert expr is not None

    def test_unknown_metric(self):
        key = MetricKey("totally_custom_metric", "col_a")
        expr = metric_key_to_expr(key)
        assert expr is None

    def test_all_registered_metrics(self):
        """Every CommonMetrics method should be resolvable."""
        lf = pl.DataFrame({"x": [1, 2, 3]}).lazy()

        pairs = [
            CommonMetrics.row_count(),
            CommonMetrics.null_count("x"),
            CommonMetrics.non_null_count("x"),
            CommonMetrics.n_unique("x"),
            CommonMetrics.mean("x"),
            CommonMetrics.std("x"),
            CommonMetrics.min("x"),
            CommonMetrics.max("x"),
            CommonMetrics.sum("x"),
            CommonMetrics.median("x"),
        ]
        for key, _ in pairs:
            resolved = metric_key_to_expr(key)
            assert resolved is not None, f"Failed to resolve: {key}"


# ============================================================================
# 5. Validator get_required_metrics() Tests
# ============================================================================


class TestValidatorRequiredMetrics:
    """Tests that validators properly declare their metric dependencies."""

    @pytest.fixture
    def columns(self) -> list[str]:
        return ["col_a", "col_b"]

    def test_null_validator(self, columns):
        from truthound.validators.completeness.null import NullValidator
        v = NullValidator()
        keys = v.get_required_metrics(columns)
        key_names = {k.metric_name for k in keys}
        assert "row_count" in key_names
        assert "null_count" in key_names
        # Should have row_count + null_count per column
        assert len(keys) == 1 + len(columns)

    def test_not_null_validator(self, columns):
        from truthound.validators.completeness.null import NotNullValidator
        v = NotNullValidator()
        keys = v.get_required_metrics(columns)
        key_names = {k.metric_name for k in keys}
        assert "row_count" in key_names
        assert "null_count" in key_names

    def test_completeness_ratio_validator(self, columns):
        from truthound.validators.completeness.null import CompletenessRatioValidator
        v = CompletenessRatioValidator()
        keys = v.get_required_metrics(columns)
        key_names = {k.metric_name for k in keys}
        assert "row_count" in key_names
        assert "null_count" in key_names
        assert "non_null_count" in key_names

    def test_unique_validator(self, columns):
        from truthound.validators.uniqueness.unique import UniqueValidator
        v = UniqueValidator()
        keys = v.get_required_metrics(columns)
        key_names = {k.metric_name for k in keys}
        assert "row_count" in key_names
        assert "n_unique" in key_names
        assert "non_null_count" in key_names

    def test_unique_ratio_validator(self, columns):
        from truthound.validators.uniqueness.unique import UniqueRatioValidator
        v = UniqueRatioValidator()
        keys = v.get_required_metrics(columns)
        key_names = {k.metric_name for k in keys}
        assert "row_count" in key_names
        assert "n_unique" in key_names

    def test_distinct_count_validator(self, columns):
        from truthound.validators.uniqueness.unique import DistinctCountValidator
        v = DistinctCountValidator()
        keys = v.get_required_metrics(columns)
        key_names = {k.metric_name for k in keys}
        assert "row_count" in key_names
        assert "n_unique" in key_names

    def test_between_validator(self, columns):
        from truthound.validators.distribution.range import BetweenValidator
        v = BetweenValidator(min_value=0, max_value=100)
        keys = v.get_required_metrics(columns)
        key_names = {k.metric_name for k in keys}
        assert "row_count" in key_names
        assert "min" in key_names
        assert "max" in key_names

    def test_base_validator_returns_empty(self):
        """Validators without get_required_metrics override return empty."""
        from truthound.validators.base import Validator

        class DummyValidator(Validator):
            name = "dummy"
            def validate(self, lf):
                return []

        v = DummyValidator()
        assert v.get_required_metrics(["a", "b"]) == []


# ============================================================================
# 6. Metric Deduplication Tests
# ============================================================================


class TestMetricDeduplication:
    """Tests that shared metrics are deduplicated across validators."""

    @pytest.fixture
    def sample_lf(self) -> pl.LazyFrame:
        return pl.DataFrame({
            "name": ["Alice", "Bob", "Alice", None, "Eve"],
            "age": [25, 30, 25, 40, None],
            "score": [85.0, 90.0, 85.0, 70.0, 95.0],
        }).lazy()

    def test_null_and_completeness_share_null_count(self, sample_lf):
        """NullValidator and CompletenessRatioValidator both need null_count."""
        from truthound.validators.completeness.null import (
            NullValidator, CompletenessRatioValidator,
        )

        v1 = NullValidator(columns=["name"])
        v2 = CompletenessRatioValidator(columns=["name"])

        # Both request null_count for "name"
        keys1 = v1.get_required_metrics(["name"])
        keys2 = v2.get_required_metrics(["name"])

        # Find common keys
        set1 = set(keys1)
        set2 = set(keys2)
        shared = set1 & set2

        assert MetricKey("row_count") in shared
        assert MetricKey("null_count", "name") in shared

    def test_unique_and_distinct_share_n_unique(self, sample_lf):
        """UniqueValidator and DistinctCountValidator both need n_unique."""
        from truthound.validators.uniqueness.unique import (
            UniqueValidator, DistinctCountValidator,
        )

        v1 = UniqueValidator(columns=["name"])
        v2 = DistinctCountValidator(columns=["name"])

        keys1 = v1.get_required_metrics(["name"])
        keys2 = v2.get_required_metrics(["name"])

        shared = set(keys1) & set(keys2)
        assert MetricKey("row_count") in shared
        assert MetricKey("n_unique", "name") in shared

    def test_precomputation_deduplicates(self, sample_lf):
        """ExpressionBatchExecutor precomputes shared metrics once."""
        from truthound.validators.completeness.null import NullValidator
        from truthound.validators.base import ExpressionBatchExecutor

        store = SharedMetricStore()
        executor = ExpressionBatchExecutor(metric_store=store)
        executor.add_validator(NullValidator(columns=["name"]))
        executor.add_validator(NullValidator(columns=["age"]))

        # Execute — this should precompute row_count once
        executor.execute(sample_lf, metric_store=store)

        # row_count should be in the store (computed once)
        assert store.get(MetricKey("row_count")) == 5
        # null_count for each column
        assert store.get(MetricKey("null_count", "name")) == 1
        assert store.get(MetricKey("null_count", "age")) == 1


# ============================================================================
# 7. UniqueValidator.validate_with_metrics() Tests
# ============================================================================


class TestUniqueValidatorWithMetrics:
    """Tests for UniqueValidator using SharedMetricStore."""

    @pytest.fixture
    def lf_with_dupes(self) -> pl.LazyFrame:
        return pl.DataFrame({
            "id": [1, 2, 3, 3, 4],
            "name": ["Alice", "Bob", "Charlie", "Charlie", "Eve"],
        }).lazy()

    def test_validate_with_metrics_finds_duplicates(self, lf_with_dupes):
        from truthound.validators.uniqueness.unique import UniqueValidator
        from truthound.validators.metrics import CommonMetrics

        store = SharedMetricStore()
        # Pre-populate metrics
        store.put(CommonMetrics.row_count()[0], 5)
        store.put(CommonMetrics.n_unique("id")[0], 4)
        store.put(CommonMetrics.non_null_count("id")[0], 5)

        v = UniqueValidator(columns=["id"])
        issues = v.validate_with_metrics(lf_with_dupes, store)

        assert len(issues) == 1
        assert issues[0].column == "id"
        assert issues[0].count == 1  # 5 - 4 = 1 duplicate

    def test_validate_with_metrics_no_duplicates(self, lf_with_dupes):
        from truthound.validators.uniqueness.unique import UniqueValidator
        from truthound.validators.metrics import CommonMetrics

        store = SharedMetricStore()
        store.put(CommonMetrics.row_count()[0], 5)
        store.put(CommonMetrics.n_unique("name")[0], 5)
        store.put(CommonMetrics.non_null_count("name")[0], 5)

        v = UniqueValidator(columns=["name"])
        issues = v.validate_with_metrics(lf_with_dupes, store)
        assert len(issues) == 0

    def test_validate_with_metrics_fallback(self, lf_with_dupes):
        """Falls back to validate() when store is empty."""
        from truthound.validators.uniqueness.unique import UniqueValidator

        store = SharedMetricStore()
        v = UniqueValidator(columns=["id"])
        issues = v.validate_with_metrics(lf_with_dupes, store)
        # Should still work via fallback
        assert len(issues) == 1

    def test_validate_with_metrics_matches_validate(self, lf_with_dupes):
        """validate_with_metrics produces same results as validate."""
        from truthound.validators.uniqueness.unique import UniqueValidator
        from truthound.validators.metrics import CommonMetrics

        # Populate store with *actual* metrics from the data
        # id: [1, 2, 3, 3, 4] → 4 unique, 5 non-null
        # name: ["Alice", "Bob", "Charlie", "Charlie", "Eve"] → 4 unique, 5 non-null
        store = SharedMetricStore()
        store.put(CommonMetrics.row_count()[0], 5)
        store.put(CommonMetrics.n_unique("id")[0], 4)
        store.put(CommonMetrics.non_null_count("id")[0], 5)
        store.put(CommonMetrics.n_unique("name")[0], 4)
        store.put(CommonMetrics.non_null_count("name")[0], 5)

        v = UniqueValidator(columns=["id", "name"])
        issues_with = v.validate_with_metrics(lf_with_dupes, store)
        issues_without = v.validate(lf_with_dupes)

        assert len(issues_with) == len(issues_without)
        for iw, iwo in zip(
            sorted(issues_with, key=lambda x: x.column),
            sorted(issues_without, key=lambda x: x.column),
        ):
            assert iw.column == iwo.column
            assert iw.count == iwo.count
            assert iw.severity == iwo.severity


# ============================================================================
# 8. End-to-End Integration Tests
# ============================================================================


class TestEndToEndDeduplication:
    """End-to-end tests verifying deduplication correctness."""

    @pytest.fixture
    def mixed_data_lf(self) -> pl.LazyFrame:
        return pl.DataFrame({
            "id": list(range(100)),
            "value": [float(i % 10) for i in range(100)],
            "category": [f"cat_{i % 5}" for i in range(100)],
            "nullable": [None if i % 7 == 0 else i for i in range(100)],
        }).lazy()

    def test_batch_executor_with_store(self, mixed_data_lf):
        """Full batch execution with metric store — results match no-store path."""
        from truthound.validators.completeness.null import NullValidator
        from truthound.validators.base import ExpressionBatchExecutor

        # Without store
        exec1 = ExpressionBatchExecutor()
        exec1.add_validator(NullValidator())
        issues_no_store = exec1.execute(mixed_data_lf)

        # With store
        store = SharedMetricStore()
        exec2 = ExpressionBatchExecutor(metric_store=store)
        exec2.add_validator(NullValidator())
        issues_with_store = exec2.execute(mixed_data_lf)

        # Same results
        assert len(issues_no_store) == len(issues_with_store)
        for i1, i2 in zip(
            sorted(issues_no_store, key=lambda x: x.column),
            sorted(issues_with_store, key=lambda x: x.column),
        ):
            assert i1.column == i2.column
            assert i1.count == i2.count
            assert i1.severity == i2.severity

    def test_store_is_populated_after_execution(self, mixed_data_lf):
        """After execution, the store contains the precomputed metrics."""
        from truthound.validators.completeness.null import NullValidator
        from truthound.validators.base import ExpressionBatchExecutor

        store = SharedMetricStore()
        executor = ExpressionBatchExecutor(metric_store=store)
        executor.add_validator(NullValidator(columns=["nullable"]))
        executor.execute(mixed_data_lf)

        assert store.get(MetricKey("row_count")) == 100
        assert store.get(MetricKey("null_count", "nullable")) is not None

    def test_backward_compatibility_no_store(self, mixed_data_lf):
        """ExpressionBatchExecutor works without any metric_store (backward compat)."""
        from truthound.validators.completeness.null import NullValidator
        from truthound.validators.base import ExpressionBatchExecutor

        executor = ExpressionBatchExecutor()  # No store
        executor.add_validator(NullValidator())
        issues = executor.execute(mixed_data_lf)
        # Should work fine
        assert isinstance(issues, list)


# ============================================================================
# 9. MetricStoreStats Tests
# ============================================================================


class TestMetricStoreStats:
    """Tests for performance counters."""

    def test_initial_state(self):
        stats = MetricStoreStats()
        assert stats.total_lookups == 0
        assert stats.cache_hits == 0
        assert stats.hit_ratio == 0.0

    def test_hit_ratio_calculation(self):
        stats = MetricStoreStats(total_lookups=10, cache_hits=7)
        assert stats.hit_ratio == 0.7

    def test_to_dict(self):
        stats = MetricStoreStats(total_lookups=4, cache_hits=2, metrics_computed=3)
        d = stats.to_dict()
        assert d["total_lookups"] == 4
        assert d["cache_hits"] == 2
        assert d["hit_ratio"] == 0.5
        assert d["metrics_computed"] == 3
