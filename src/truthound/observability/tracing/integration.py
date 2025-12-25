"""OpenTelemetry integration testing and validation utilities.

This module provides comprehensive testing and validation for OpenTelemetry
integration, including mock exporters, integration validators, and
diagnostic tools for Jaeger/Zipkin connectivity.

Key Features:
- Mock span exporters for testing
- Integration validators for Jaeger/Zipkin
- Diagnostic tools for troubleshooting
- End-to-end tracing verification
- Configuration validation

Example:
    from truthound.observability.tracing.integration import (
        TracingValidator,
        MockSpanExporter,
        validate_jaeger_connection,
    )

    # Validate Jaeger connection
    result = validate_jaeger_connection("http://localhost:14268")
    if result.success:
        print("Jaeger is reachable")

    # Run integration tests
    validator = TracingValidator()
    report = validator.validate_all()
    print(report)
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable
from urllib.parse import urlparse

from truthound.observability.tracing.span import (
    Span,
    SpanContextData,
    SpanKind,
    StatusCode,
)
from truthound.observability.tracing.processor import SpanProcessor
from truthound.observability.tracing.exporter import SpanExporter, ExportResult


# =============================================================================
# Mock Exporter for Testing
# =============================================================================


class MockSpanExporter(SpanExporter):
    """Mock span exporter for testing purposes.

    Collects exported spans in memory for verification.

    Example:
        exporter = MockSpanExporter()
        provider.add_processor(SimpleSpanProcessor(exporter))

        with tracer.start_as_current_span("test"):
            pass

        assert len(exporter.spans) == 1
        assert exporter.spans[0].name == "test"

    Attributes:
        spans: List of exported spans
        export_count: Number of export calls
        last_export_time: Time of last export
    """

    def __init__(self, max_spans: int = 10000):
        """Initialize mock exporter.

        Args:
            max_spans: Maximum spans to keep in memory
        """
        self.max_spans = max_spans
        self.spans: list[Span] = []
        self.export_count = 0
        self.last_export_time: datetime | None = None
        self._lock = threading.Lock()
        self._shutdown = False

    def export(self, spans: list[Span]) -> ExportResult:
        """Export spans to memory.

        Args:
            spans: Spans to export

        Returns:
            ExportResult.SUCCESS
        """
        if self._shutdown:
            return ExportResult.SUCCESS

        with self._lock:
            self.spans.extend(spans)
            self.export_count += 1
            self.last_export_time = datetime.now()

            # Trim if over limit
            if len(self.spans) > self.max_spans:
                self.spans = self.spans[-self.max_spans:]

        return ExportResult.SUCCESS

    def shutdown(self) -> bool:
        """Shutdown exporter."""
        self._shutdown = True
        return True

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush (no-op for mock)."""
        return True

    def clear(self) -> None:
        """Clear collected spans."""
        with self._lock:
            self.spans.clear()
            self.export_count = 0

    def get_span_by_name(self, name: str) -> Span | None:
        """Find span by name.

        Args:
            name: Span name to find

        Returns:
            First matching span or None
        """
        with self._lock:
            for span in self.spans:
                if span.name == name:
                    return span
        return None

    def get_spans_by_name(self, name: str) -> list[Span]:
        """Find all spans with given name.

        Args:
            name: Span name to find

        Returns:
            List of matching spans
        """
        with self._lock:
            return [s for s in self.spans if s.name == name]

    def get_span_by_trace_id(self, trace_id: str) -> list[Span]:
        """Find all spans in a trace.

        Args:
            trace_id: Trace ID to find

        Returns:
            List of spans in the trace
        """
        with self._lock:
            return [
                s for s in self.spans
                if s.context.trace_id == trace_id
            ]

    def get_stats(self) -> dict[str, Any]:
        """Get exporter statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            return {
                "span_count": len(self.spans),
                "export_count": self.export_count,
                "last_export_time": (
                    self.last_export_time.isoformat()
                    if self.last_export_time else None
                ),
            }


# =============================================================================
# Recording Processor for Testing
# =============================================================================


class RecordingSpanProcessor(SpanProcessor):
    """Span processor that records events for testing.

    Records all start and end events for verification.

    Example:
        processor = RecordingSpanProcessor()
        provider.add_processor(processor)

        with tracer.start_as_current_span("test"):
            pass

        assert len(processor.started_spans) == 1
        assert len(processor.ended_spans) == 1
    """

    def __init__(self):
        """Initialize processor."""
        self.started_spans: list[tuple[Span, SpanContextData | None]] = []
        self.ended_spans: list[Span] = []
        self._lock = threading.Lock()

    def on_start(self, span: Span, parent_context: SpanContextData | None) -> None:
        """Record span start.

        Args:
            span: Started span
            parent_context: Parent span context
        """
        with self._lock:
            self.started_spans.append((span, parent_context))

    def on_end(self, span: Span) -> None:
        """Record span end.

        Args:
            span: Ended span
        """
        with self._lock:
            self.ended_spans.append(span)

    def shutdown(self) -> bool:
        """Shutdown processor."""
        return True

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush (no-op)."""
        return True

    def clear(self) -> None:
        """Clear recorded spans."""
        with self._lock:
            self.started_spans.clear()
            self.ended_spans.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get processor statistics."""
        with self._lock:
            return {
                "started_count": len(self.started_spans),
                "ended_count": len(self.ended_spans),
            }


# =============================================================================
# Connection Validators
# =============================================================================


class ConnectionStatus(str, Enum):
    """Status of backend connection."""

    CONNECTED = "connected"
    UNREACHABLE = "unreachable"
    AUTH_FAILED = "auth_failed"
    TIMEOUT = "timeout"
    ERROR = "error"
    NOT_CONFIGURED = "not_configured"


@dataclass
class ConnectionResult:
    """Result of connection validation.

    Attributes:
        status: Connection status
        endpoint: Tested endpoint
        latency_ms: Response latency in milliseconds
        message: Status message
        details: Additional details
    """

    status: ConnectionStatus
    endpoint: str = ""
    latency_ms: float = 0.0
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if connection was successful."""
        return self.status == ConnectionStatus.CONNECTED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "endpoint": self.endpoint,
            "latency_ms": self.latency_ms,
            "message": self.message,
            "success": self.success,
            "details": self.details,
        }


def validate_jaeger_connection(
    endpoint: str,
    timeout: float = 5.0,
) -> ConnectionResult:
    """Validate Jaeger backend connection.

    Supports both HTTP (Thrift) and gRPC endpoints.

    Args:
        endpoint: Jaeger endpoint URL
        timeout: Connection timeout in seconds

    Returns:
        ConnectionResult with status

    Example:
        # HTTP endpoint (Thrift)
        result = validate_jaeger_connection("http://localhost:14268")

        # gRPC endpoint
        result = validate_jaeger_connection("grpc://localhost:14250")
    """
    start_time = time.time()

    try:
        parsed = urlparse(endpoint)
        host = parsed.hostname or "localhost"
        port = parsed.port or (14268 if parsed.scheme == "http" else 14250)

        # Try socket connection first
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)

        try:
            sock.connect((host, port))
            sock.close()
        except socket.timeout:
            return ConnectionResult(
                status=ConnectionStatus.TIMEOUT,
                endpoint=endpoint,
                latency_ms=(time.time() - start_time) * 1000,
                message=f"Connection to {host}:{port} timed out",
            )
        except socket.error as e:
            return ConnectionResult(
                status=ConnectionStatus.UNREACHABLE,
                endpoint=endpoint,
                latency_ms=(time.time() - start_time) * 1000,
                message=f"Cannot connect to {host}:{port}: {e}",
            )

        # For HTTP, try health endpoint
        if parsed.scheme in ("http", "https"):
            try:
                import urllib.request

                health_url = f"{endpoint.rstrip('/')}/health"
                req = urllib.request.Request(health_url, method="GET")
                req.add_header("User-Agent", "truthound-validator/1.0")

                with urllib.request.urlopen(req, timeout=timeout) as response:
                    if response.status == 200:
                        return ConnectionResult(
                            status=ConnectionStatus.CONNECTED,
                            endpoint=endpoint,
                            latency_ms=(time.time() - start_time) * 1000,
                            message="Jaeger is healthy",
                            details={"http_status": response.status},
                        )

            except Exception as e:
                # Health check failed, but socket connected
                return ConnectionResult(
                    status=ConnectionStatus.CONNECTED,
                    endpoint=endpoint,
                    latency_ms=(time.time() - start_time) * 1000,
                    message="Jaeger socket reachable (health check failed)",
                    details={"warning": str(e)},
                )

        return ConnectionResult(
            status=ConnectionStatus.CONNECTED,
            endpoint=endpoint,
            latency_ms=(time.time() - start_time) * 1000,
            message="Jaeger is reachable",
        )

    except Exception as e:
        return ConnectionResult(
            status=ConnectionStatus.ERROR,
            endpoint=endpoint,
            latency_ms=(time.time() - start_time) * 1000,
            message=f"Validation error: {e}",
        )


def validate_zipkin_connection(
    endpoint: str,
    timeout: float = 5.0,
) -> ConnectionResult:
    """Validate Zipkin backend connection.

    Args:
        endpoint: Zipkin endpoint URL (e.g., http://localhost:9411)
        timeout: Connection timeout in seconds

    Returns:
        ConnectionResult with status

    Example:
        result = validate_zipkin_connection("http://localhost:9411")
    """
    start_time = time.time()

    try:
        parsed = urlparse(endpoint)
        host = parsed.hostname or "localhost"
        port = parsed.port or 9411

        # Try HTTP API endpoint
        try:
            import urllib.request

            api_url = f"{endpoint.rstrip('/')}/api/v2/services"
            req = urllib.request.Request(api_url, method="GET")
            req.add_header("User-Agent", "truthound-validator/1.0")
            req.add_header("Accept", "application/json")

            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    return ConnectionResult(
                        status=ConnectionStatus.CONNECTED,
                        endpoint=endpoint,
                        latency_ms=(time.time() - start_time) * 1000,
                        message="Zipkin is healthy",
                        details={"http_status": response.status},
                    )

        except urllib.error.HTTPError as e:
            if e.code == 401:
                return ConnectionResult(
                    status=ConnectionStatus.AUTH_FAILED,
                    endpoint=endpoint,
                    latency_ms=(time.time() - start_time) * 1000,
                    message="Authentication required",
                )
            elif e.code == 404:
                # API might be different version, but service is reachable
                return ConnectionResult(
                    status=ConnectionStatus.CONNECTED,
                    endpoint=endpoint,
                    latency_ms=(time.time() - start_time) * 1000,
                    message="Zipkin reachable (API version mismatch)",
                    details={"http_status": e.code},
                )

        except urllib.error.URLError as e:
            return ConnectionResult(
                status=ConnectionStatus.UNREACHABLE,
                endpoint=endpoint,
                latency_ms=(time.time() - start_time) * 1000,
                message=f"Cannot connect to Zipkin: {e.reason}",
            )

        except Exception as e:
            return ConnectionResult(
                status=ConnectionStatus.ERROR,
                endpoint=endpoint,
                latency_ms=(time.time() - start_time) * 1000,
                message=f"Connection error: {e}",
            )

        # Fallback: try socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)

        try:
            sock.connect((host, port))
            sock.close()
            return ConnectionResult(
                status=ConnectionStatus.CONNECTED,
                endpoint=endpoint,
                latency_ms=(time.time() - start_time) * 1000,
                message="Zipkin socket reachable",
            )
        except socket.timeout:
            return ConnectionResult(
                status=ConnectionStatus.TIMEOUT,
                endpoint=endpoint,
                latency_ms=(time.time() - start_time) * 1000,
                message="Connection timed out",
            )
        except socket.error as e:
            return ConnectionResult(
                status=ConnectionStatus.UNREACHABLE,
                endpoint=endpoint,
                latency_ms=(time.time() - start_time) * 1000,
                message=f"Cannot connect: {e}",
            )

    except Exception as e:
        return ConnectionResult(
            status=ConnectionStatus.ERROR,
            endpoint=endpoint,
            latency_ms=(time.time() - start_time) * 1000,
            message=f"Validation error: {e}",
        )


def validate_otlp_connection(
    endpoint: str,
    timeout: float = 5.0,
    use_tls: bool = False,
) -> ConnectionResult:
    """Validate OTLP (OpenTelemetry Protocol) endpoint connection.

    Args:
        endpoint: OTLP endpoint (e.g., http://localhost:4317)
        timeout: Connection timeout in seconds
        use_tls: Whether to use TLS

    Returns:
        ConnectionResult with status
    """
    start_time = time.time()

    try:
        parsed = urlparse(endpoint)
        host = parsed.hostname or "localhost"
        port = parsed.port or 4317

        # Try socket connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)

        try:
            sock.connect((host, port))
            sock.close()

            return ConnectionResult(
                status=ConnectionStatus.CONNECTED,
                endpoint=endpoint,
                latency_ms=(time.time() - start_time) * 1000,
                message="OTLP endpoint is reachable",
                details={"protocol": "grpc" if port == 4317 else "http"},
            )

        except socket.timeout:
            return ConnectionResult(
                status=ConnectionStatus.TIMEOUT,
                endpoint=endpoint,
                latency_ms=(time.time() - start_time) * 1000,
                message="Connection timed out",
            )

        except socket.error as e:
            return ConnectionResult(
                status=ConnectionStatus.UNREACHABLE,
                endpoint=endpoint,
                latency_ms=(time.time() - start_time) * 1000,
                message=f"Cannot connect: {e}",
            )

    except Exception as e:
        return ConnectionResult(
            status=ConnectionStatus.ERROR,
            endpoint=endpoint,
            latency_ms=(time.time() - start_time) * 1000,
            message=f"Validation error: {e}",
        )


# =============================================================================
# Tracing Validator
# =============================================================================


@dataclass
class ValidationReport:
    """Comprehensive tracing validation report.

    Attributes:
        success: Overall validation success
        provider_valid: TracerProvider is working
        span_creation_valid: Spans can be created
        context_propagation_valid: Context propagation works
        exporter_valid: Exporter is working
        backend_connections: Backend connection results
        errors: List of errors
        warnings: List of warnings
        duration_ms: Total validation duration
    """

    success: bool = True
    provider_valid: bool = False
    span_creation_valid: bool = False
    context_propagation_valid: bool = False
    exporter_valid: bool = False
    backend_connections: dict[str, ConnectionResult] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "provider_valid": self.provider_valid,
            "span_creation_valid": self.span_creation_valid,
            "context_propagation_valid": self.context_propagation_valid,
            "exporter_valid": self.exporter_valid,
            "backend_connections": {
                k: v.to_dict()
                for k, v in self.backend_connections.items()
            },
            "errors": self.errors,
            "warnings": self.warnings,
            "duration_ms": self.duration_ms,
        }

    def __str__(self) -> str:
        """Human-readable report."""
        lines = [
            "OpenTelemetry Tracing Validation Report",
            "=" * 40,
            f"Overall Status: {'PASS' if self.success else 'FAIL'}",
            "",
            "Component Status:",
            f"  Provider:            {'✓' if self.provider_valid else '✗'}",
            f"  Span Creation:       {'✓' if self.span_creation_valid else '✗'}",
            f"  Context Propagation: {'✓' if self.context_propagation_valid else '✗'}",
            f"  Exporter:            {'✓' if self.exporter_valid else '✗'}",
            "",
        ]

        if self.backend_connections:
            lines.append("Backend Connections:")
            for name, result in self.backend_connections.items():
                status = "✓" if result.success else "✗"
                lines.append(f"  {name}: {status} ({result.message})")
            lines.append("")

        if self.errors:
            lines.append("Errors:")
            for err in self.errors:
                lines.append(f"  - {err}")
            lines.append("")

        if self.warnings:
            lines.append("Warnings:")
            for warn in self.warnings:
                lines.append(f"  - {warn}")
            lines.append("")

        lines.append(f"Duration: {self.duration_ms:.2f}ms")

        return "\n".join(lines)


class TracingValidator:
    """Validates OpenTelemetry tracing integration.

    Provides comprehensive validation of tracing setup including
    provider configuration, span creation, context propagation,
    and backend connectivity.

    Example:
        validator = TracingValidator()

        # Validate everything
        report = validator.validate_all()
        print(report)

        # Validate specific components
        result = validator.validate_span_creation()
    """

    def __init__(
        self,
        jaeger_endpoint: str | None = None,
        zipkin_endpoint: str | None = None,
        otlp_endpoint: str | None = None,
    ):
        """Initialize validator.

        Args:
            jaeger_endpoint: Jaeger endpoint URL
            zipkin_endpoint: Zipkin endpoint URL
            otlp_endpoint: OTLP endpoint URL
        """
        self.jaeger_endpoint = jaeger_endpoint
        self.zipkin_endpoint = zipkin_endpoint
        self.otlp_endpoint = otlp_endpoint

    def validate_all(self) -> ValidationReport:
        """Run all validations.

        Returns:
            Comprehensive validation report
        """
        start_time = time.time()
        report = ValidationReport()

        # Validate provider
        try:
            report.provider_valid = self.validate_provider()
        except Exception as e:
            report.errors.append(f"Provider validation error: {e}")

        # Validate span creation
        try:
            report.span_creation_valid = self.validate_span_creation()
        except Exception as e:
            report.errors.append(f"Span creation validation error: {e}")

        # Validate context propagation
        try:
            report.context_propagation_valid = self.validate_context_propagation()
        except Exception as e:
            report.errors.append(f"Context propagation validation error: {e}")

        # Validate exporter
        try:
            report.exporter_valid = self.validate_exporter()
        except Exception as e:
            report.errors.append(f"Exporter validation error: {e}")

        # Validate backend connections
        if self.jaeger_endpoint:
            result = validate_jaeger_connection(self.jaeger_endpoint)
            report.backend_connections["jaeger"] = result
            if not result.success:
                report.warnings.append(f"Jaeger not reachable: {result.message}")

        if self.zipkin_endpoint:
            result = validate_zipkin_connection(self.zipkin_endpoint)
            report.backend_connections["zipkin"] = result
            if not result.success:
                report.warnings.append(f"Zipkin not reachable: {result.message}")

        if self.otlp_endpoint:
            result = validate_otlp_connection(self.otlp_endpoint)
            report.backend_connections["otlp"] = result
            if not result.success:
                report.warnings.append(f"OTLP endpoint not reachable: {result.message}")

        # Overall success
        report.success = (
            report.provider_valid and
            report.span_creation_valid and
            report.context_propagation_valid and
            len(report.errors) == 0
        )

        report.duration_ms = (time.time() - start_time) * 1000
        return report

    def validate_provider(self) -> bool:
        """Validate TracerProvider is working.

        Returns:
            True if provider is valid
        """
        from truthound.observability.tracing.provider import (
            TracerProvider,
            get_tracer_provider,
        )

        # Check global provider exists
        provider = get_tracer_provider()
        if provider is None:
            return False

        # Check we can get a tracer
        tracer = provider.get_tracer("validation-test")
        if tracer is None:
            return False

        return True

    def validate_span_creation(self) -> bool:
        """Validate spans can be created.

        Returns:
            True if span creation works
        """
        from truthound.observability.tracing.provider import (
            TracerProvider,
            Tracer,
        )

        # Create isolated provider for testing
        mock_exporter = MockSpanExporter()
        from truthound.observability.tracing.processor import SimpleSpanProcessor

        provider = TracerProvider()
        provider.add_processor(SimpleSpanProcessor(mock_exporter))

        tracer = provider.get_tracer("validation-test")

        # Create a span
        with tracer.start_as_current_span("test-span") as span:
            span.set_attribute("test.key", "test-value")

        # Verify span was created
        if len(mock_exporter.spans) != 1:
            return False

        exported_span = mock_exporter.spans[0]
        if exported_span.name != "test-span":
            return False

        if exported_span.attributes.get("test.key") != "test-value":
            return False

        return True

    def validate_context_propagation(self) -> bool:
        """Validate context propagation works.

        Returns:
            True if context propagation works
        """
        from truthound.observability.tracing.provider import (
            TracerProvider,
            get_current_span,
        )
        from truthound.observability.tracing.processor import SimpleSpanProcessor

        mock_exporter = MockSpanExporter()
        provider = TracerProvider()
        provider.add_processor(SimpleSpanProcessor(mock_exporter))

        tracer = provider.get_tracer("validation-test")

        # Create parent span
        with tracer.start_as_current_span("parent") as parent:
            parent_trace_id = parent.context.trace_id

            # Create child span
            with tracer.start_as_current_span("child") as child:
                # Verify child has same trace ID
                if child.context.trace_id != parent_trace_id:
                    return False

                # Verify parent is set correctly
                if child.parent is None:
                    return False

                if child.parent.span_id != parent.context.span_id:
                    return False

        # Verify we exported both spans
        if len(mock_exporter.spans) != 2:
            return False

        return True

    def validate_exporter(self) -> bool:
        """Validate exporter functionality.

        Returns:
            True if exporter works
        """
        # Test mock exporter
        exporter = MockSpanExporter()

        from truthound.observability.tracing.span import Span, SpanContextData

        # Create test span
        context = SpanContextData(
            trace_id="0" * 32,
            span_id="0" * 16,
            trace_flags=1,
        )

        span = Span(
            name="test-export",
            context=context,
            kind=SpanKind.INTERNAL,
        )
        span.end()

        # Export
        result = exporter.export([span])
        if result != ExportResult.SUCCESS:
            return False

        if len(exporter.spans) != 1:
            return False

        return True


# =============================================================================
# Diagnostic Tools
# =============================================================================


def diagnose_tracing_setup() -> dict[str, Any]:
    """Run comprehensive tracing diagnostics.

    Returns:
        Diagnostic information dictionary
    """
    diagnostics = {
        "timestamp": datetime.now().isoformat(),
        "python_version": None,
        "truthound_version": None,
        "otel_packages": {},
        "provider_info": {},
        "environment": {},
    }

    # Python version
    import sys
    diagnostics["python_version"] = sys.version

    # Truthound version
    try:
        import truthound
        diagnostics["truthound_version"] = getattr(truthound, "__version__", "unknown")
    except ImportError:
        diagnostics["truthound_version"] = "not installed"

    # OpenTelemetry packages
    otel_packages = [
        "opentelemetry-api",
        "opentelemetry-sdk",
        "opentelemetry-exporter-jaeger",
        "opentelemetry-exporter-zipkin",
        "opentelemetry-exporter-otlp",
    ]

    for pkg in otel_packages:
        try:
            import importlib.metadata
            version = importlib.metadata.version(pkg.replace("-", "_"))
            diagnostics["otel_packages"][pkg] = version
        except Exception:
            diagnostics["otel_packages"][pkg] = "not installed"

    # Provider info
    try:
        from truthound.observability.tracing.provider import get_tracer_provider

        provider = get_tracer_provider()
        diagnostics["provider_info"] = {
            "type": type(provider).__name__,
            "resource": str(provider.resource) if hasattr(provider, "resource") else "N/A",
            "sampler": str(provider.sampler) if hasattr(provider, "sampler") else "N/A",
        }
    except Exception as e:
        diagnostics["provider_info"] = {"error": str(e)}

    # Environment variables
    import os
    otel_env_vars = [
        "OTEL_SERVICE_NAME",
        "OTEL_EXPORTER_JAEGER_ENDPOINT",
        "OTEL_EXPORTER_ZIPKIN_ENDPOINT",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_TRACES_SAMPLER",
        "OTEL_TRACES_EXPORTER",
    ]

    for var in otel_env_vars:
        value = os.environ.get(var)
        if value:
            diagnostics["environment"][var] = value

    return diagnostics


def create_test_trace(
    tracer_name: str = "test-tracer",
    span_count: int = 3,
) -> list[Span]:
    """Create a test trace for verification.

    Args:
        tracer_name: Name for test tracer
        span_count: Number of nested spans to create

    Returns:
        List of created spans
    """
    from truthound.observability.tracing.provider import TracerProvider
    from truthound.observability.tracing.processor import SimpleSpanProcessor

    mock_exporter = MockSpanExporter()
    provider = TracerProvider()
    provider.add_processor(SimpleSpanProcessor(mock_exporter))

    tracer = provider.get_tracer(tracer_name)

    def create_nested_spans(depth: int, current: int = 0):
        if current >= depth:
            return

        with tracer.start_as_current_span(f"span-{current}") as span:
            span.set_attribute("depth", current)
            span.set_attribute("timestamp", datetime.now().isoformat())
            time.sleep(0.001)  # Small delay for realistic timing
            create_nested_spans(depth, current + 1)

    create_nested_spans(span_count)

    return list(mock_exporter.spans)
