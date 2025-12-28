"""Tests for Prometheus metrics."""

from __future__ import annotations

import threading
import time

import pytest

from truthound.stores.observability.config import MetricsConfig
from truthound.stores.observability.metrics import (
    InMemoryMetricsBackend,
    MetricValue,
    MetricsRegistry,
    PrometheusMetricsBackend,
    StoreMetrics,
)


class TestMetricValue:
    """Tests for MetricValue dataclass."""

    def test_label_key_empty(self) -> None:
        mv = MetricValue(name="test", value=1.0)
        assert mv.label_key() == ""

    def test_label_key_single(self) -> None:
        mv = MetricValue(name="test", value=1.0, labels={"env": "prod"})
        assert mv.label_key() == 'env="prod"'

    def test_label_key_multiple_sorted(self) -> None:
        mv = MetricValue(name="test", value=1.0, labels={"z": "1", "a": "2"})
        assert mv.label_key() == 'a="2",z="1"'


class TestInMemoryMetricsBackend:
    """Tests for InMemoryMetricsBackend."""

    def test_counter_increment(self) -> None:
        backend = InMemoryMetricsBackend()
        backend.counter("requests_total")
        backend.counter("requests_total")
        backend.counter("requests_total", 3)

        assert backend.get_counter("requests_total") == 5.0

    def test_counter_with_labels(self) -> None:
        backend = InMemoryMetricsBackend()
        backend.counter("requests_total", labels={"method": "GET"})
        backend.counter("requests_total", labels={"method": "POST"})
        backend.counter("requests_total", labels={"method": "GET"})

        assert backend.get_counter("requests_total", {"method": "GET"}) == 2.0
        assert backend.get_counter("requests_total", {"method": "POST"}) == 1.0

    def test_gauge_set(self) -> None:
        backend = InMemoryMetricsBackend()
        backend.gauge("temperature", 25.5)
        assert backend.get_gauge("temperature") == 25.5

        backend.gauge("temperature", 30.0)
        assert backend.get_gauge("temperature") == 30.0

    def test_gauge_with_labels(self) -> None:
        backend = InMemoryMetricsBackend()
        backend.gauge("temperature", 25.5, labels={"location": "room1"})
        backend.gauge("temperature", 30.0, labels={"location": "room2"})

        assert backend.get_gauge("temperature", {"location": "room1"}) == 25.5
        assert backend.get_gauge("temperature", {"location": "room2"}) == 30.0

    def test_histogram_observation(self) -> None:
        config = MetricsConfig(histogram_buckets=[0.1, 0.5, 1.0, 5.0])
        backend = InMemoryMetricsBackend(config)

        backend.histogram("request_duration", 0.05)
        backend.histogram("request_duration", 0.3)
        backend.histogram("request_duration", 0.8)
        backend.histogram("request_duration", 2.0)

        export = backend.export()
        assert "request_duration_bucket" in export
        assert "request_duration_sum" in export
        assert "request_duration_count" in export

    def test_summary_observation(self) -> None:
        backend = InMemoryMetricsBackend()

        for i in range(100):
            backend.summary("response_size", float(i))

        export = backend.export()
        assert "response_size_sum" in export
        assert "response_size_count" in export
        assert 'quantile="0.5"' in export
        assert 'quantile="0.99"' in export

    def test_timer_context_manager(self) -> None:
        backend = InMemoryMetricsBackend()

        with backend.timer("operation_duration"):
            time.sleep(0.01)

        export = backend.export()
        assert "operation_duration" in export

    def test_prefix(self) -> None:
        config = MetricsConfig(prefix="myapp")
        backend = InMemoryMetricsBackend(config)

        backend.counter("requests")
        export = backend.export()

        assert "myapp_requests" in export

    def test_default_labels(self) -> None:
        config = MetricsConfig(labels={"service": "api"})
        backend = InMemoryMetricsBackend(config)

        backend.counter("requests")
        export = backend.export()

        assert 'service="api"' in export

    def test_export_prometheus_format(self) -> None:
        backend = InMemoryMetricsBackend()
        backend.counter("http_requests_total", labels={"method": "GET"})
        backend.gauge("active_connections", 10)

        export = backend.export()

        # Check format
        assert "# TYPE" in export or "http_requests_total" in export
        assert "active_connections 10" in export or "active_connections{" in export

    def test_reset(self) -> None:
        backend = InMemoryMetricsBackend()
        backend.counter("test_counter", 5)
        backend.gauge("test_gauge", 10)

        backend.reset()

        assert backend.get_counter("test_counter") == 0.0
        assert backend.get_gauge("test_gauge") is None


class TestPrometheusMetricsBackend:
    """Tests for PrometheusMetricsBackend."""

    def test_inherits_inmemory(self) -> None:
        backend = PrometheusMetricsBackend()
        backend.counter("test_counter")
        assert backend.get_counter("test_counter") == 1.0

    def test_http_server_not_started_by_default(self) -> None:
        backend = PrometheusMetricsBackend()
        assert backend._server is None

    def test_http_server_start_stop(self) -> None:
        config = MetricsConfig(http_port=19090)  # Use non-standard port
        backend = PrometheusMetricsBackend(config)

        backend.start_http_server()
        assert backend._server is not None

        backend.stop_http_server()
        assert backend._server is None


class TestMetricsRegistry:
    """Tests for MetricsRegistry singleton."""

    def teardown_method(self) -> None:
        MetricsRegistry.reset_instance()

    def test_singleton(self) -> None:
        reg1 = MetricsRegistry()
        reg2 = MetricsRegistry()
        assert reg1 is reg2

    def test_configure(self) -> None:
        config = MetricsConfig(prefix="test")
        registry = MetricsRegistry()
        registry.configure(config)

        assert registry.backend is not None

    def test_get_backend(self) -> None:
        registry = MetricsRegistry()
        backend = registry.get_backend()
        assert backend is not None


class TestStoreMetrics:
    """Tests for StoreMetrics helper class."""

    def test_record_operation(self) -> None:
        backend = InMemoryMetricsBackend()
        metrics = StoreMetrics(backend, store_type="filesystem", store_id="store-1")

        metrics.record_operation("save", 0.1, success=True)

        export = backend.export()
        assert "operations_total" in export
        assert "operation_duration_seconds" in export
        assert "operations_success_total" in export

    def test_record_operation_failure(self) -> None:
        backend = InMemoryMetricsBackend()
        metrics = StoreMetrics(backend, store_type="s3", store_id="store-1")

        metrics.record_operation("save", 0.1, success=False)

        export = backend.export()
        assert "operations_failed_total" in export

    def test_record_save(self) -> None:
        backend = InMemoryMetricsBackend()
        metrics = StoreMetrics(backend, store_type="filesystem", store_id="store-1")

        metrics.record_save(0.05, size_bytes=1024)

        export = backend.export()
        assert "save_size_bytes" in export

    def test_record_get_cache_hit(self) -> None:
        backend = InMemoryMetricsBackend()
        metrics = StoreMetrics(backend, store_type="filesystem", store_id="store-1")

        metrics.record_get(0.01, hit=True)
        metrics.record_get(0.02, hit=False)

        assert backend.get_counter("cache_hits_total", {"store_type": "filesystem", "store_id": "store-1"}) == 1.0
        assert backend.get_counter("cache_misses_total", {"store_type": "filesystem", "store_id": "store-1"}) == 1.0

    def test_record_query(self) -> None:
        backend = InMemoryMetricsBackend()
        metrics = StoreMetrics(backend, store_type="database", store_id="db-1")

        metrics.record_query(0.5, result_count=100)

        export = backend.export()
        assert "query_result_count" in export

    def test_set_item_count(self) -> None:
        backend = InMemoryMetricsBackend()
        metrics = StoreMetrics(backend, store_type="filesystem", store_id="store-1")

        metrics.set_item_count(500)

        assert backend.get_gauge("items_total", {"store_type": "filesystem", "store_id": "store-1"}) == 500

    def test_set_storage_size(self) -> None:
        backend = InMemoryMetricsBackend()
        metrics = StoreMetrics(backend, store_type="s3", store_id="bucket-1")

        metrics.set_storage_size(1024 * 1024 * 100)  # 100MB

        labels = {"store_type": "s3", "store_id": "bucket-1"}
        assert backend.get_gauge("storage_bytes", labels) == 104857600

    def test_record_error(self) -> None:
        backend = InMemoryMetricsBackend()
        metrics = StoreMetrics(backend, store_type="database", store_id="db-1")

        metrics.record_error("ConnectionError")
        metrics.record_error("ConnectionError")
        metrics.record_error("TimeoutError")

        labels = {"store_type": "database", "store_id": "db-1", "error_type": "ConnectionError"}
        assert backend.get_counter("errors_total", labels) == 2.0

    def test_record_batch(self) -> None:
        backend = InMemoryMetricsBackend()
        metrics = StoreMetrics(backend, store_type="filesystem", store_id="store-1")

        metrics.record_batch("save", batch_size=100, duration_seconds=1.5)

        export = backend.export()
        assert "batch_size" in export
        assert "batch_duration_seconds" in export
        assert "batch_operations_total" in export

    def test_record_replication(self) -> None:
        backend = InMemoryMetricsBackend()
        metrics = StoreMetrics(backend, store_type="s3", store_id="primary")

        metrics.record_replication("replica-1", 0.5, success=True)
        metrics.record_replication("replica-1", 0.3, success=False)

        export = backend.export()
        assert "replication_duration_seconds" in export
        assert "replications_total" in export
        assert "replication_errors_total" in export

    def test_set_replication_lag(self) -> None:
        backend = InMemoryMetricsBackend()
        metrics = StoreMetrics(backend, store_type="s3", store_id="primary")

        metrics.set_replication_lag("replica-1", 2.5)

        labels = {"store_type": "s3", "store_id": "primary", "target": "replica-1"}
        assert backend.get_gauge("replication_lag_seconds", labels) == 2.5

    def test_labels_include_store_info(self) -> None:
        backend = InMemoryMetricsBackend()
        metrics = StoreMetrics(backend, store_type="gcs", store_id="bucket-prod")

        metrics.record_operation("save", 0.1)

        export = backend.export()
        assert 'store_type="gcs"' in export
        assert 'store_id="bucket-prod"' in export


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_counter_updates(self) -> None:
        backend = InMemoryMetricsBackend()
        num_threads = 10
        increments_per_thread = 100

        def increment() -> None:
            for _ in range(increments_per_thread):
                backend.counter("concurrent_counter")

        threads = [threading.Thread(target=increment) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert backend.get_counter("concurrent_counter") == num_threads * increments_per_thread

    def test_concurrent_gauge_updates(self) -> None:
        backend = InMemoryMetricsBackend()

        def set_gauge(value: float) -> None:
            for _ in range(100):
                backend.gauge("concurrent_gauge", value)

        threads = [
            threading.Thread(target=set_gauge, args=(float(i),))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Gauge should have some value (last write wins)
        assert backend.get_gauge("concurrent_gauge") is not None
