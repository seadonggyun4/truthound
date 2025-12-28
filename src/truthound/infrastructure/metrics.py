"""Enterprise Prometheus metrics system for Truthound.

This module extends the base metrics system with enterprise features:
- Full validator metrics exposure
- Checkpoint execution metrics
- DataSource performance metrics
- HTTP endpoint for Prometheus scraping
- Push gateway support
- OpenTelemetry export

Architecture:
    MetricsManager
         |
         +---> ValidatorMetrics (per-validator stats)
         +---> CheckpointMetrics (checkpoint execution)
         +---> DataSourceMetrics (datasource performance)
         +---> SystemMetrics (memory, CPU, etc.)
         |
         v
    MetricsServer (HTTP /metrics endpoint)
         |
         +---> Prometheus scrape
         +---> Push gateway

Usage:
    >>> from truthound.infrastructure.metrics import (
    ...     get_metrics, configure_metrics,
    ...     ValidatorMetrics, CheckpointMetrics,
    ... )
    >>>
    >>> # Configure and start HTTP server
    >>> configure_metrics(
    ...     enable_http=True,
    ...     port=9090,
    ...     service="truthound",
    ... )
    >>>
    >>> # Record validator metrics
    >>> metrics = get_metrics()
    >>> with metrics.validator.time("not_null", "users", "email"):
    ...     run_validation()
    >>>
    >>> # Record checkpoint metrics
    >>> metrics.checkpoint.execution_started("daily_check")
    >>> metrics.checkpoint.execution_completed("daily_check", success=True, issues=5)
"""

from __future__ import annotations

import http.server
import os
import socket
import socketserver
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Iterator

# Re-export base metrics components
from truthound.observability.metrics import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    MetricsRegistry,
    MetricsCollector,
    PrometheusExporter,
    StatsDExporter,
)


# =============================================================================
# Metrics Configuration
# =============================================================================


@dataclass
class MetricsConfig:
    """Metrics configuration.

    Example:
        >>> config = MetricsConfig(
        ...     enabled=True,
        ...     service="truthound",
        ...     environment="production",
        ...     enable_http=True,
        ...     port=9090,
        ...     push_gateway_url="http://pushgateway:9091",
        ... )
    """

    enabled: bool = True
    service: str = ""
    environment: str = ""
    namespace: str = "truthound"

    # HTTP server
    enable_http: bool = False
    host: str = "0.0.0.0"
    port: int = 9090
    path: str = "/metrics"

    # Push gateway
    push_gateway_url: str = ""
    push_interval: float = 15.0
    push_job: str = "truthound"

    # Default labels
    default_labels: dict[str, str] = field(default_factory=dict)

    # Histogram buckets
    latency_buckets: tuple[float, ...] = (
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1,
        0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
    )
    size_buckets: tuple[float, ...] = (
        100, 1000, 10000, 100000, 1000000,
        10000000, 100000000, 1000000000,
    )

    @classmethod
    def from_environment(cls) -> "MetricsConfig":
        """Load from environment variables."""
        return cls(
            enabled=os.getenv("METRICS_ENABLED", "true").lower() == "true",
            service=os.getenv("SERVICE_NAME", "truthound"),
            environment=os.getenv("ENVIRONMENT", "development"),
            enable_http=os.getenv("METRICS_HTTP_ENABLED", "false").lower() == "true",
            port=int(os.getenv("METRICS_PORT", "9090")),
            push_gateway_url=os.getenv("METRICS_PUSH_GATEWAY_URL", ""),
        )


# =============================================================================
# Validator Metrics
# =============================================================================


class ValidatorMetrics:
    """Metrics for validator execution.

    Tracks:
    - Execution count per validator
    - Execution duration (histogram)
    - Pass/fail rates
    - Issue counts
    - Row processing rates
    """

    def __init__(
        self,
        registry: MetricsRegistry,
        namespace: str = "truthound",
        latency_buckets: tuple[float, ...] | None = None,
    ) -> None:
        """Initialize validator metrics.

        Args:
            registry: Metrics registry.
            namespace: Metric namespace prefix.
            latency_buckets: Histogram buckets for latency.
        """
        self._registry = registry
        self._namespace = namespace

        # Validator execution counter
        self.executions = Counter(
            f"{namespace}_validator_executions_total",
            "Total number of validator executions",
            labels=("validator", "dataset", "column", "status"),
        )
        registry.register(self.executions)

        # Validator execution duration
        self.duration = Histogram(
            f"{namespace}_validator_duration_seconds",
            "Validator execution duration in seconds",
            buckets=latency_buckets or (
                0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
            ),
            labels=("validator", "dataset"),
        )
        registry.register(self.duration)

        # Issues found counter
        self.issues = Counter(
            f"{namespace}_validator_issues_total",
            "Total number of issues found by validators",
            labels=("validator", "dataset", "column", "severity"),
        )
        registry.register(self.issues)

        # Rows processed counter
        self.rows_processed = Counter(
            f"{namespace}_validator_rows_processed_total",
            "Total number of rows processed by validators",
            labels=("validator", "dataset"),
        )
        registry.register(self.rows_processed)

        # Active validators gauge
        self.active = Gauge(
            f"{namespace}_validators_active",
            "Number of currently active validators",
            labels=("dataset",),
        )
        registry.register(self.active)

        # Pass rate (calculated)
        self._pass_counts: dict[str, int] = {}
        self._total_counts: dict[str, int] = {}

    def record_execution(
        self,
        validator: str,
        dataset: str,
        column: str = "",
        *,
        success: bool,
        duration_seconds: float,
        issues_found: int = 0,
        rows_processed: int = 0,
        severity: str = "warning",
    ) -> None:
        """Record a validator execution.

        Args:
            validator: Validator name.
            dataset: Dataset name.
            column: Column name (if applicable).
            success: Whether validation passed.
            duration_seconds: Execution duration.
            issues_found: Number of issues found.
            rows_processed: Number of rows processed.
            severity: Issue severity level.
        """
        status = "success" if success else "failure"

        # Increment execution counter
        self.executions.inc(
            validator=validator,
            dataset=dataset,
            column=column or "_all",
            status=status,
        )

        # Record duration
        self.duration.observe(duration_seconds, validator=validator, dataset=dataset)

        # Record issues
        if issues_found > 0:
            self.issues.add(
                issues_found,
                validator=validator,
                dataset=dataset,
                column=column or "_all",
                severity=severity,
            )

        # Record rows
        if rows_processed > 0:
            self.rows_processed.add(
                rows_processed,
                validator=validator,
                dataset=dataset,
            )

        # Update pass rate tracking
        key = f"{validator}:{dataset}"
        self._total_counts[key] = self._total_counts.get(key, 0) + 1
        if success:
            self._pass_counts[key] = self._pass_counts.get(key, 0) + 1

    @contextmanager
    def time(
        self,
        validator: str,
        dataset: str,
        column: str = "",
    ) -> Iterator[None]:
        """Context manager to time validator execution.

        Args:
            validator: Validator name.
            dataset: Dataset name.
            column: Column name.

        Example:
            >>> with metrics.validator.time("not_null", "users", "email"):
            ...     run_validation()
        """
        self.active.inc(dataset=dataset)
        start = time.perf_counter()
        success = True
        issues = 0

        try:
            yield
        except Exception:
            success = False
            raise
        finally:
            duration = time.perf_counter() - start
            self.active.dec(dataset=dataset)
            self.record_execution(
                validator,
                dataset,
                column,
                success=success,
                duration_seconds=duration,
                issues_found=issues,
            )

    def get_pass_rate(self, validator: str, dataset: str) -> float:
        """Get pass rate for a validator.

        Args:
            validator: Validator name.
            dataset: Dataset name.

        Returns:
            Pass rate (0.0 to 1.0).
        """
        key = f"{validator}:{dataset}"
        total = self._total_counts.get(key, 0)
        if total == 0:
            return 1.0
        return self._pass_counts.get(key, 0) / total


# =============================================================================
# Checkpoint Metrics
# =============================================================================


class CheckpointMetrics:
    """Metrics for checkpoint execution.

    Tracks:
    - Checkpoint execution count
    - Execution duration
    - Pass/fail status
    - Issue distribution
    """

    def __init__(
        self,
        registry: MetricsRegistry,
        namespace: str = "truthound",
        latency_buckets: tuple[float, ...] | None = None,
    ) -> None:
        """Initialize checkpoint metrics."""
        self._registry = registry
        self._namespace = namespace
        self._start_times: dict[str, float] = {}

        # Checkpoint executions
        self.executions = Counter(
            f"{namespace}_checkpoint_executions_total",
            "Total number of checkpoint executions",
            labels=("checkpoint", "status"),
        )
        registry.register(self.executions)

        # Checkpoint duration
        self.duration = Histogram(
            f"{namespace}_checkpoint_duration_seconds",
            "Checkpoint execution duration in seconds",
            buckets=latency_buckets or (
                0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0
            ),
            labels=("checkpoint",),
        )
        registry.register(self.duration)

        # Issues by checkpoint
        self.issues = Gauge(
            f"{namespace}_checkpoint_issues",
            "Current number of issues in checkpoint",
            labels=("checkpoint", "severity"),
        )
        registry.register(self.issues)

        # Validators run per checkpoint
        self.validators_run = Counter(
            f"{namespace}_checkpoint_validators_run_total",
            "Total validators run per checkpoint",
            labels=("checkpoint",),
        )
        registry.register(self.validators_run)

        # Last execution timestamp
        self.last_execution = Gauge(
            f"{namespace}_checkpoint_last_execution_timestamp",
            "Timestamp of last checkpoint execution",
            labels=("checkpoint",),
        )
        registry.register(self.last_execution)

        # Currently running checkpoints
        self.running = Gauge(
            f"{namespace}_checkpoints_running",
            "Number of currently running checkpoints",
        )
        registry.register(self.running)

    def execution_started(self, checkpoint: str) -> None:
        """Record checkpoint execution start.

        Args:
            checkpoint: Checkpoint name.
        """
        self._start_times[checkpoint] = time.time()
        self.running.inc()

    def execution_completed(
        self,
        checkpoint: str,
        *,
        success: bool,
        issues: int = 0,
        validators_run: int = 0,
        issues_by_severity: dict[str, int] | None = None,
    ) -> None:
        """Record checkpoint execution completion.

        Args:
            checkpoint: Checkpoint name.
            success: Whether checkpoint passed.
            issues: Total issues found.
            validators_run: Number of validators executed.
            issues_by_severity: Issues broken down by severity.
        """
        self.running.dec()

        status = "success" if success else "failure"
        self.executions.inc(checkpoint=checkpoint, status=status)

        # Record duration
        start_time = self._start_times.pop(checkpoint, time.time())
        duration = time.time() - start_time
        self.duration.observe(duration, checkpoint=checkpoint)

        # Record validators run
        if validators_run > 0:
            self.validators_run.add(validators_run, checkpoint=checkpoint)

        # Record issues by severity
        if issues_by_severity:
            for severity, count in issues_by_severity.items():
                self.issues.set(count, checkpoint=checkpoint, severity=severity)
        else:
            self.issues.set(issues, checkpoint=checkpoint, severity="total")

        # Update last execution time
        self.last_execution.set_to_current_time(checkpoint=checkpoint)

    @contextmanager
    def track(self, checkpoint: str) -> Iterator[None]:
        """Context manager to track checkpoint execution.

        Args:
            checkpoint: Checkpoint name.

        Example:
            >>> with metrics.checkpoint.track("daily_check"):
            ...     run_checkpoint()
        """
        self.execution_started(checkpoint)
        success = True
        try:
            yield
        except Exception:
            success = False
            raise
        finally:
            self.execution_completed(checkpoint, success=success)


# =============================================================================
# DataSource Metrics
# =============================================================================


class DataSourceMetrics:
    """Metrics for data source operations.

    Tracks:
    - Query execution count
    - Query duration
    - Rows read/written
    - Connection pool stats
    - Error rates
    """

    def __init__(
        self,
        registry: MetricsRegistry,
        namespace: str = "truthound",
        latency_buckets: tuple[float, ...] | None = None,
        size_buckets: tuple[float, ...] | None = None,
    ) -> None:
        """Initialize datasource metrics."""
        self._registry = registry
        self._namespace = namespace

        # Query counter
        self.queries = Counter(
            f"{namespace}_datasource_queries_total",
            "Total number of datasource queries",
            labels=("datasource", "operation", "status"),
        )
        registry.register(self.queries)

        # Query duration
        self.query_duration = Histogram(
            f"{namespace}_datasource_query_duration_seconds",
            "Datasource query duration in seconds",
            buckets=latency_buckets or (
                0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0
            ),
            labels=("datasource", "operation"),
        )
        registry.register(self.query_duration)

        # Rows processed
        self.rows = Counter(
            f"{namespace}_datasource_rows_total",
            "Total rows processed by datasource",
            labels=("datasource", "operation"),
        )
        registry.register(self.rows)

        # Bytes processed
        self.bytes = Counter(
            f"{namespace}_datasource_bytes_total",
            "Total bytes processed by datasource",
            labels=("datasource", "operation"),
        )
        registry.register(self.bytes)

        # Connection pool
        self.pool_size = Gauge(
            f"{namespace}_datasource_pool_size",
            "Current connection pool size",
            labels=("datasource",),
        )
        registry.register(self.pool_size)

        self.pool_available = Gauge(
            f"{namespace}_datasource_pool_available",
            "Available connections in pool",
            labels=("datasource",),
        )
        registry.register(self.pool_available)

        # Errors
        self.errors = Counter(
            f"{namespace}_datasource_errors_total",
            "Total datasource errors",
            labels=("datasource", "error_type"),
        )
        registry.register(self.errors)

    def record_query(
        self,
        datasource: str,
        operation: str,
        *,
        success: bool,
        duration_seconds: float,
        rows: int = 0,
        bytes_processed: int = 0,
        error_type: str = "",
    ) -> None:
        """Record a datasource query.

        Args:
            datasource: Datasource name.
            operation: Operation type (read, write, scan, etc).
            success: Whether query succeeded.
            duration_seconds: Query duration.
            rows: Rows processed.
            bytes_processed: Bytes processed.
            error_type: Error type if failed.
        """
        status = "success" if success else "failure"

        self.queries.inc(datasource=datasource, operation=operation, status=status)
        self.query_duration.observe(
            duration_seconds, datasource=datasource, operation=operation
        )

        if rows > 0:
            self.rows.add(rows, datasource=datasource, operation=operation)

        if bytes_processed > 0:
            self.bytes.add(bytes_processed, datasource=datasource, operation=operation)

        if not success and error_type:
            self.errors.inc(datasource=datasource, error_type=error_type)

    @contextmanager
    def time_query(
        self,
        datasource: str,
        operation: str = "query",
    ) -> Iterator[None]:
        """Context manager to time datasource query.

        Args:
            datasource: Datasource name.
            operation: Operation type.
        """
        start = time.perf_counter()
        success = True
        error_type = ""

        try:
            yield
        except Exception as e:
            success = False
            error_type = type(e).__name__
            raise
        finally:
            duration = time.perf_counter() - start
            self.record_query(
                datasource,
                operation,
                success=success,
                duration_seconds=duration,
                error_type=error_type,
            )

    def update_pool_stats(
        self,
        datasource: str,
        size: int,
        available: int,
    ) -> None:
        """Update connection pool statistics.

        Args:
            datasource: Datasource name.
            size: Total pool size.
            available: Available connections.
        """
        self.pool_size.set(size, datasource=datasource)
        self.pool_available.set(available, datasource=datasource)


# =============================================================================
# System Metrics
# =============================================================================


class SystemMetrics:
    """System-level metrics (memory, CPU, etc).

    Tracks:
    - Process memory usage
    - CPU usage
    - File descriptors
    - Thread count
    """

    def __init__(
        self,
        registry: MetricsRegistry,
        namespace: str = "truthound",
    ) -> None:
        """Initialize system metrics."""
        self._registry = registry
        self._namespace = namespace

        # Memory
        self.memory_bytes = Gauge(
            f"{namespace}_process_memory_bytes",
            "Process memory usage in bytes",
            labels=("type",),
        )
        registry.register(self.memory_bytes)

        # CPU
        self.cpu_seconds = Counter(
            f"{namespace}_process_cpu_seconds_total",
            "Total CPU time spent in seconds",
            labels=("mode",),
        )
        registry.register(self.cpu_seconds)

        # File descriptors
        self.open_fds = Gauge(
            f"{namespace}_process_open_fds",
            "Number of open file descriptors",
        )
        registry.register(self.open_fds)

        # Threads
        self.threads = Gauge(
            f"{namespace}_process_threads",
            "Number of threads",
        )
        registry.register(self.threads)

        # Start time
        self.start_time = Gauge(
            f"{namespace}_process_start_time_seconds",
            "Start time of the process (Unix timestamp)",
        )
        registry.register(self.start_time)
        self.start_time.set(time.time())

    def collect(self) -> None:
        """Collect current system metrics."""
        try:
            import resource

            # Memory usage
            usage = resource.getrusage(resource.RUSAGE_SELF)
            self.memory_bytes.set(usage.ru_maxrss * 1024, type="rss")

            # CPU time
            self.cpu_seconds.add(usage.ru_utime, mode="user")
            self.cpu_seconds.add(usage.ru_stime, mode="system")

        except Exception:
            pass

        try:
            # Thread count
            self.threads.set(threading.active_count())
        except Exception:
            pass

        try:
            # Open file descriptors (Linux)
            import os

            fd_dir = f"/proc/{os.getpid()}/fd"
            if os.path.isdir(fd_dir):
                self.open_fds.set(len(os.listdir(fd_dir)))
        except Exception:
            pass


# =============================================================================
# Metrics Server (HTTP)
# =============================================================================


class MetricsServer:
    """HTTP server for Prometheus metrics endpoint.

    Provides /metrics endpoint for Prometheus scraping.

    Example:
        >>> server = MetricsServer(registry, port=9090)
        >>> server.start()
        >>> # ... metrics available at http://localhost:9090/metrics
        >>> server.stop()
    """

    def __init__(
        self,
        registry: MetricsRegistry,
        *,
        host: str = "0.0.0.0",
        port: int = 9090,
        path: str = "/metrics",
    ) -> None:
        """Initialize metrics server.

        Args:
            registry: Metrics registry.
            host: Server host.
            port: Server port.
            path: Metrics endpoint path.
        """
        self._registry = registry
        self._host = host
        self._port = port
        self._path = path
        self._exporter = PrometheusExporter()
        self._server: socketserver.TCPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the metrics server."""
        if self._server:
            return

        registry = self._registry
        exporter = self._exporter
        path = self._path

        class MetricsHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == path:
                    content = exporter.export(registry)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content.encode("utf-8"))
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:
                pass  # Suppress logging

        class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
            allow_reuse_address = True

        self._server = ThreadedTCPServer((self._host, self._port), MetricsHandler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="metrics-server",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the metrics server."""
        if self._server:
            self._server.shutdown()
            self._server = None
            self._thread = None

    @property
    def url(self) -> str:
        """Get the metrics endpoint URL."""
        host = self._host if self._host != "0.0.0.0" else "localhost"
        return f"http://{host}:{self._port}{self._path}"


# =============================================================================
# Push Gateway Support
# =============================================================================


class PushGatewayPusher:
    """Push metrics to Prometheus Push Gateway.

    Example:
        >>> pusher = PushGatewayPusher(
        ...     registry,
        ...     url="http://pushgateway:9091",
        ...     job="truthound",
        ... )
        >>> pusher.start()
    """

    def __init__(
        self,
        registry: MetricsRegistry,
        *,
        url: str,
        job: str = "truthound",
        instance: str = "",
        grouping: dict[str, str] | None = None,
        push_interval: float = 15.0,
    ) -> None:
        """Initialize push gateway pusher.

        Args:
            registry: Metrics registry.
            url: Push gateway URL.
            job: Job name.
            instance: Instance name.
            grouping: Additional grouping labels.
            push_interval: Push interval in seconds.
        """
        self._registry = registry
        self._url = url.rstrip("/")
        self._job = job
        self._instance = instance or socket.gethostname()
        self._grouping = grouping or {}
        self._push_interval = push_interval
        self._exporter = PrometheusExporter()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start pushing metrics."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._push_loop,
            daemon=True,
            name="metrics-pusher",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop pushing metrics."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def push(self) -> bool:
        """Push metrics once.

        Returns:
            True if successful.
        """
        try:
            import urllib.request
            import urllib.error

            # Build URL with grouping
            path_parts = [f"/job/{self._job}"]
            if self._instance:
                path_parts.append(f"/instance/{self._instance}")
            for key, value in self._grouping.items():
                path_parts.append(f"/{key}/{value}")

            url = f"{self._url}/metrics" + "".join(path_parts)

            data = self._exporter.export(self._registry).encode("utf-8")

            request = urllib.request.Request(
                url,
                data=data,
                method="POST",
                headers={"Content-Type": "text/plain"},
            )

            with urllib.request.urlopen(request, timeout=30):
                return True

        except Exception:
            return False

    def _push_loop(self) -> None:
        """Background push loop."""
        while self._running:
            self.push()
            time.sleep(self._push_interval)


# =============================================================================
# Metrics Manager
# =============================================================================


class MetricsManager:
    """Central manager for all metrics.

    Provides access to all metric categories and manages the metrics server.

    Example:
        >>> manager = MetricsManager(config=MetricsConfig(enable_http=True))
        >>> manager.start()
        >>>
        >>> # Record metrics
        >>> manager.validator.record_execution("not_null", "users", success=True, ...)
        >>> manager.checkpoint.execution_started("daily")
        >>>
        >>> # Get Prometheus format
        >>> print(manager.export())
    """

    def __init__(self, config: MetricsConfig | None = None) -> None:
        """Initialize metrics manager.

        Args:
            config: Metrics configuration.
        """
        self._config = config or MetricsConfig()
        self._registry = MetricsRegistry()

        # Initialize metric categories
        self.validator = ValidatorMetrics(
            self._registry,
            namespace=self._config.namespace,
            latency_buckets=self._config.latency_buckets,
        )

        self.checkpoint = CheckpointMetrics(
            self._registry,
            namespace=self._config.namespace,
            latency_buckets=self._config.latency_buckets,
        )

        self.datasource = DataSourceMetrics(
            self._registry,
            namespace=self._config.namespace,
            latency_buckets=self._config.latency_buckets,
            size_buckets=self._config.size_buckets,
        )

        self.system = SystemMetrics(
            self._registry,
            namespace=self._config.namespace,
        )

        # Exporters
        self._prometheus_exporter = PrometheusExporter()

        # HTTP server
        self._server: MetricsServer | None = None

        # Push gateway
        self._pusher: PushGatewayPusher | None = None

    @property
    def registry(self) -> MetricsRegistry:
        """Get the metrics registry."""
        return self._registry

    @property
    def config(self) -> MetricsConfig:
        """Get the configuration."""
        return self._config

    def start(self) -> None:
        """Start metrics server and push gateway (if configured)."""
        if not self._config.enabled:
            return

        # Start HTTP server
        if self._config.enable_http:
            self._server = MetricsServer(
                self._registry,
                host=self._config.host,
                port=self._config.port,
                path=self._config.path,
            )
            self._server.start()

        # Start push gateway
        if self._config.push_gateway_url:
            self._pusher = PushGatewayPusher(
                self._registry,
                url=self._config.push_gateway_url,
                job=self._config.push_job,
                push_interval=self._config.push_interval,
            )
            self._pusher.start()

    def stop(self) -> None:
        """Stop metrics server and push gateway."""
        if self._server:
            self._server.stop()
            self._server = None

        if self._pusher:
            self._pusher.stop()
            self._pusher = None

    def export(self) -> str:
        """Export metrics in Prometheus format.

        Returns:
            Prometheus text format metrics.
        """
        # Collect system metrics before export
        self.system.collect()
        return self._prometheus_exporter.export(self._registry)

    def counter(
        self,
        name: str,
        description: str = "",
        labels: list[str] | tuple[str, ...] = (),
    ) -> Counter:
        """Create a custom counter.

        Args:
            name: Counter name.
            description: Description.
            labels: Label names.

        Returns:
            Counter instance.
        """
        full_name = f"{self._config.namespace}_{name}"
        counter = Counter(full_name, description, labels=labels)
        return self._registry.register(counter)  # type: ignore

    def gauge(
        self,
        name: str,
        description: str = "",
        labels: list[str] | tuple[str, ...] = (),
    ) -> Gauge:
        """Create a custom gauge.

        Args:
            name: Gauge name.
            description: Description.
            labels: Label names.

        Returns:
            Gauge instance.
        """
        full_name = f"{self._config.namespace}_{name}"
        gauge = Gauge(full_name, description, labels=labels)
        return self._registry.register(gauge)  # type: ignore

    def histogram(
        self,
        name: str,
        description: str = "",
        labels: list[str] | tuple[str, ...] = (),
        buckets: tuple[float, ...] | None = None,
    ) -> Histogram:
        """Create a custom histogram.

        Args:
            name: Histogram name.
            description: Description.
            labels: Label names.
            buckets: Bucket boundaries.

        Returns:
            Histogram instance.
        """
        full_name = f"{self._config.namespace}_{name}"
        histogram = Histogram(
            full_name,
            description,
            labels=labels,
            buckets=buckets or self._config.latency_buckets,
        )
        return self._registry.register(histogram)  # type: ignore


# =============================================================================
# Global Metrics Management
# =============================================================================

_global_manager: MetricsManager | None = None
_lock = threading.Lock()


def configure_metrics(
    *,
    enabled: bool = True,
    service: str = "",
    environment: str = "",
    namespace: str = "truthound",
    enable_http: bool = False,
    host: str = "0.0.0.0",
    port: int = 9090,
    push_gateway_url: str = "",
    **kwargs: Any,
) -> MetricsManager:
    """Configure global metrics.

    Args:
        enabled: Enable metrics.
        service: Service name.
        environment: Environment name.
        namespace: Metric namespace prefix.
        enable_http: Enable HTTP server.
        host: HTTP server host.
        port: HTTP server port.
        push_gateway_url: Push gateway URL.
        **kwargs: Additional MetricsConfig parameters.

    Returns:
        Configured MetricsManager.
    """
    global _global_manager

    with _lock:
        if _global_manager:
            _global_manager.stop()

        config = MetricsConfig(
            enabled=enabled,
            service=service,
            environment=environment,
            namespace=namespace,
            enable_http=enable_http,
            host=host,
            port=port,
            push_gateway_url=push_gateway_url,
            **kwargs,
        )

        _global_manager = MetricsManager(config)
        _global_manager.start()

        return _global_manager


def get_metrics() -> MetricsManager:
    """Get the global metrics manager.

    Returns:
        MetricsManager instance.
    """
    global _global_manager

    with _lock:
        if _global_manager is None:
            config = MetricsConfig.from_environment()
            _global_manager = MetricsManager(config)
        return _global_manager


def reset_metrics() -> None:
    """Reset global metrics."""
    global _global_manager

    with _lock:
        if _global_manager:
            _global_manager.stop()
            _global_manager = None
