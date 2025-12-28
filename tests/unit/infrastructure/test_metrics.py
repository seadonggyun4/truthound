"""Tests for enterprise metrics system."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from truthound.infrastructure.metrics import (
    # Core
    MetricsManager,
    MetricsConfig,
    # Validator metrics
    ValidatorMetrics,
    CheckpointMetrics,
    DataSourceMetrics,
    # HTTP server
    MetricsServer,
    # Factory
    get_metrics,
    configure_metrics,
    reset_metrics,
)

from truthound.observability.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
)


class TestMetricsConfig:
    """Tests for MetricsConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = MetricsConfig()

        assert config.enabled is True
        assert config.namespace == "truthound"
        assert config.enable_http is False
        assert config.port == 9090

    def test_from_environment(self):
        """Test loading from environment."""
        with patch.dict("os.environ", {
            "METRICS_ENABLED": "true",
            "SERVICE_NAME": "test-service",
            "ENVIRONMENT": "production",
            "METRICS_HTTP_ENABLED": "true",
            "METRICS_PORT": "8080",
        }):
            config = MetricsConfig.from_environment()

        assert config.enabled is True
        assert config.service == "test-service"
        assert config.environment == "production"
        assert config.enable_http is True
        assert config.port == 8080


class TestValidatorMetrics:
    """Tests for ValidatorMetrics."""

    def setup_method(self):
        """Create fresh registry for each test."""
        self.registry = MetricsRegistry()
        self.metrics = ValidatorMetrics(self.registry, namespace="test")

    def test_record_execution(self):
        """Test recording validator execution."""
        self.metrics.record_execution(
            validator="not_null",
            dataset="users",
            column="email",
            success=True,
            duration_seconds=0.5,
            issues_found=0,
            rows_processed=1000,
        )

        # Check counter
        assert self.metrics.executions.get(
            validator="not_null",
            dataset="users",
            column="email",
            status="success",
        ) == 1

        # Check rows counter
        assert self.metrics.rows_processed.get(
            validator="not_null",
            dataset="users",
        ) == 1000

    def test_record_failure(self):
        """Test recording validation failure."""
        self.metrics.record_execution(
            validator="unique",
            dataset="orders",
            column="id",
            success=False,
            duration_seconds=1.2,
            issues_found=5,
            severity="error",
        )

        assert self.metrics.executions.get(
            validator="unique",
            dataset="orders",
            column="id",
            status="failure",
        ) == 1

        assert self.metrics.issues.get(
            validator="unique",
            dataset="orders",
            column="id",
            severity="error",
        ) == 5

    def test_time_context_manager(self):
        """Test time context manager."""
        with self.metrics.time("test_validator", "test_dataset", "test_column"):
            time.sleep(0.01)

        assert self.metrics.executions.get(
            validator="test_validator",
            dataset="test_dataset",
            column="test_column",
            status="success",
        ) == 1

    def test_time_context_manager_with_exception(self):
        """Test time context manager with exception."""
        with pytest.raises(ValueError):
            with self.metrics.time("failing_validator", "test_dataset"):
                raise ValueError("Test error")

        # Empty column becomes "_all" in metrics
        assert self.metrics.executions.get(
            validator="failing_validator",
            dataset="test_dataset",
            column="_all",
            status="failure",
        ) == 1

    def test_pass_rate_calculation(self):
        """Test pass rate calculation."""
        # 7 successes, 3 failures = 70% pass rate
        for _ in range(7):
            self.metrics.record_execution(
                validator="test",
                dataset="data",
                success=True,
                duration_seconds=0.1,
            )
        for _ in range(3):
            self.metrics.record_execution(
                validator="test",
                dataset="data",
                success=False,
                duration_seconds=0.1,
            )

        rate = self.metrics.get_pass_rate("test", "data")
        assert rate == 0.7


class TestCheckpointMetrics:
    """Tests for CheckpointMetrics."""

    def setup_method(self):
        """Create fresh registry for each test."""
        self.registry = MetricsRegistry()
        self.metrics = CheckpointMetrics(self.registry, namespace="test")

    def test_execution_lifecycle(self):
        """Test checkpoint execution lifecycle."""
        self.metrics.execution_started("daily_check")
        assert self.metrics.running.get() == 1

        time.sleep(0.01)

        self.metrics.execution_completed(
            "daily_check",
            success=True,
            issues=5,
            validators_run=10,
        )

        assert self.metrics.running.get() == 0
        assert self.metrics.executions.get(
            checkpoint="daily_check",
            status="success",
        ) == 1

    def test_track_context_manager(self):
        """Test track context manager."""
        with self.metrics.track("my_checkpoint"):
            time.sleep(0.01)

        assert self.metrics.executions.get(
            checkpoint="my_checkpoint",
            status="success",
        ) == 1

    def test_track_with_failure(self):
        """Test track context manager with failure."""
        with pytest.raises(ValueError):
            with self.metrics.track("failing_checkpoint"):
                raise ValueError("Error")

        assert self.metrics.executions.get(
            checkpoint="failing_checkpoint",
            status="failure",
        ) == 1

    def test_issues_by_severity(self):
        """Test recording issues by severity."""
        self.metrics.execution_started("check")
        self.metrics.execution_completed(
            "check",
            success=True,
            issues_by_severity={
                "warning": 5,
                "error": 2,
                "critical": 1,
            },
        )

        assert self.metrics.issues.get(
            checkpoint="check",
            severity="warning",
        ) == 5
        assert self.metrics.issues.get(
            checkpoint="check",
            severity="error",
        ) == 2


class TestDataSourceMetrics:
    """Tests for DataSourceMetrics."""

    def setup_method(self):
        """Create fresh registry for each test."""
        self.registry = MetricsRegistry()
        self.metrics = DataSourceMetrics(self.registry, namespace="test")

    def test_record_query(self):
        """Test recording datasource query."""
        self.metrics.record_query(
            datasource="postgres",
            operation="read",
            success=True,
            duration_seconds=0.25,
            rows=5000,
            bytes_processed=1024 * 1024,
        )

        assert self.metrics.queries.get(
            datasource="postgres",
            operation="read",
            status="success",
        ) == 1

        assert self.metrics.rows.get(
            datasource="postgres",
            operation="read",
        ) == 5000

    def test_record_failure(self):
        """Test recording query failure."""
        self.metrics.record_query(
            datasource="bigquery",
            operation="scan",
            success=False,
            duration_seconds=5.0,
            error_type="TimeoutError",
        )

        assert self.metrics.queries.get(
            datasource="bigquery",
            operation="scan",
            status="failure",
        ) == 1

        assert self.metrics.errors.get(
            datasource="bigquery",
            error_type="TimeoutError",
        ) == 1

    def test_time_query_context_manager(self):
        """Test time_query context manager."""
        with self.metrics.time_query("mysql", "insert"):
            time.sleep(0.01)

        assert self.metrics.queries.get(
            datasource="mysql",
            operation="insert",
            status="success",
        ) == 1

    def test_update_pool_stats(self):
        """Test updating connection pool stats."""
        self.metrics.update_pool_stats("postgres", size=10, available=7)

        assert self.metrics.pool_size.get(datasource="postgres") == 10
        assert self.metrics.pool_available.get(datasource="postgres") == 7


class TestMetricsManager:
    """Tests for MetricsManager."""

    def setup_method(self):
        """Create fresh manager for each test."""
        self.config = MetricsConfig(enabled=True, namespace="test")
        self.manager = MetricsManager(self.config)

    def teardown_method(self):
        """Clean up."""
        self.manager.stop()

    def test_create_manager(self):
        """Test creating metrics manager."""
        assert self.manager.validator is not None
        assert self.manager.checkpoint is not None
        assert self.manager.datasource is not None
        assert self.manager.system is not None

    def test_export_prometheus(self):
        """Test exporting in Prometheus format."""
        # Record some metrics
        self.manager.validator.record_execution(
            validator="test",
            dataset="data",
            success=True,
            duration_seconds=0.1,
        )

        output = self.manager.export()

        assert "test_validator_executions_total" in output
        assert "test_validator_duration_seconds" in output

    def test_custom_counter(self):
        """Test creating custom counter."""
        counter = self.manager.counter(
            "custom_events",
            "Custom event counter",
            labels=["event_type"],
        )

        counter.inc(event_type="click")
        counter.inc(event_type="click")
        counter.inc(event_type="view")

        output = self.manager.export()
        assert "test_custom_events" in output

    def test_custom_gauge(self):
        """Test creating custom gauge."""
        gauge = self.manager.gauge(
            "queue_size",
            "Current queue size",
            labels=["queue"],
        )

        gauge.set(42, queue="main")

        output = self.manager.export()
        assert "test_queue_size" in output

    def test_custom_histogram(self):
        """Test creating custom histogram."""
        histogram = self.manager.histogram(
            "request_size",
            "Request size in bytes",
            labels=["endpoint"],
            buckets=(100, 1000, 10000),
        )

        histogram.observe(500, endpoint="/api/users")

        output = self.manager.export()
        assert "test_request_size" in output


class TestMetricsServer:
    """Tests for MetricsServer."""

    def test_start_stop_server(self):
        """Test starting and stopping server."""
        registry = MetricsRegistry()
        server = MetricsServer(registry, port=19090)

        server.start()
        assert server.url == "http://localhost:19090/metrics"

        server.stop()

    def test_metrics_endpoint(self):
        """Test metrics endpoint returns data."""
        import urllib.request

        registry = MetricsRegistry()

        # Add a metric
        counter = Counter("test_counter", "Test counter")
        registry.register(counter)
        counter.inc()

        server = MetricsServer(registry, port=19091)
        server.start()

        try:
            time.sleep(0.1)  # Wait for server to start

            response = urllib.request.urlopen("http://localhost:19091/metrics", timeout=5)
            content = response.read().decode("utf-8")

            assert "test_counter" in content
        finally:
            server.stop()


class TestGlobalMetrics:
    """Tests for global metrics functions."""

    def setup_method(self):
        """Reset metrics before each test."""
        reset_metrics()

    def teardown_method(self):
        """Reset after each test."""
        reset_metrics()

    def test_get_metrics(self):
        """Test getting global metrics."""
        manager1 = get_metrics()
        manager2 = get_metrics()

        assert manager1 is manager2

    def test_configure_metrics(self):
        """Test configuring global metrics."""
        manager = configure_metrics(
            service="test-service",
            namespace="custom",
            enable_http=False,
        )

        assert manager.config.service == "test-service"
        assert manager.config.namespace == "custom"

    def test_reset_metrics(self):
        """Test resetting global metrics."""
        manager1 = get_metrics()
        reset_metrics()
        manager2 = get_metrics()

        assert manager1 is not manager2
