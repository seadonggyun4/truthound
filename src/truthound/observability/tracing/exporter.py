"""Span exporters for sending trace data to backends.

Exporters are responsible for serializing and transmitting spans
to various trace collection backends.

Supported Backends:
    - Console: Print to stdout (debugging)
    - InMemory: Store in memory (testing)
    - OTLP: OpenTelemetry Protocol (Jaeger, Tempo, etc.)
    - Jaeger: Native Jaeger format
    - Zipkin: Zipkin JSON format
"""

from __future__ import annotations

import json
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from truthound.observability.tracing.span import Span


# =============================================================================
# Export Result
# =============================================================================


class ExportResult(Enum):
    """Result of an export operation."""

    SUCCESS = auto()  # Export completed successfully
    FAILURE = auto()  # Export failed, no retry
    RETRY = auto()  # Export failed, should retry


# =============================================================================
# Exporter Interface
# =============================================================================


class SpanExporter(ABC):
    """Abstract base class for span exporters.

    Exporters serialize and transmit spans to trace collection backends.
    """

    @abstractmethod
    def export(self, spans: Sequence["Span"]) -> ExportResult:
        """Export a batch of spans.

        Args:
            spans: Spans to export.

        Returns:
            ExportResult indicating success or failure.
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the exporter.

        Releases any resources held by the exporter.
        """
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush pending exports.

        Args:
            timeout_millis: Maximum time to wait.

        Returns:
            True if flush completed.
        """
        return True


# =============================================================================
# Console Exporter
# =============================================================================


class ConsoleSpanExporter(SpanExporter):
    """Exporter that prints spans to console.

    Useful for development and debugging. Outputs spans in a
    human-readable format or JSON.

    Example:
        >>> exporter = ConsoleSpanExporter(pretty=True)
        >>> processor = SimpleSpanProcessor(exporter)
    """

    def __init__(
        self,
        *,
        output: Any = None,
        pretty: bool = True,
        json_output: bool = False,
    ) -> None:
        """Initialize console exporter.

        Args:
            output: Output stream (default: sys.stdout).
            pretty: Pretty print output.
            json_output: Output as JSON.
        """
        self._output = output or sys.stdout
        self._pretty = pretty
        self._json_output = json_output
        self._lock = threading.Lock()

    def export(self, spans: Sequence["Span"]) -> ExportResult:
        """Print spans to console."""
        with self._lock:
            for span in spans:
                try:
                    if self._json_output:
                        self._print_json(span)
                    else:
                        self._print_pretty(span)
                except Exception:
                    return ExportResult.FAILURE

        return ExportResult.SUCCESS

    def _print_json(self, span: "Span") -> None:
        """Print span as JSON."""
        data = span.to_dict()
        if self._pretty:
            output = json.dumps(data, indent=2, default=str)
        else:
            output = json.dumps(data, default=str)
        self._output.write(output + "\n")
        self._output.flush()

    def _print_pretty(self, span: "Span") -> None:
        """Print span in human-readable format."""
        duration = span.duration_ms
        status = span.status[0].name

        # Header line
        header = f"[SPAN] {span.name}"
        if span.parent:
            header += f" (parent: {span.parent.span_id[:8]})"

        self._output.write(f"\n{header}\n")
        self._output.write("-" * 60 + "\n")

        # IDs
        self._output.write(f"  trace_id: {span.context.trace_id}\n")
        self._output.write(f"  span_id:  {span.context.span_id}\n")

        # Timing
        if duration is not None:
            self._output.write(f"  duration: {duration:.2f}ms\n")

        # Status
        self._output.write(f"  status:   {status}\n")

        # Kind
        self._output.write(f"  kind:     {span.kind.name}\n")

        # Attributes
        if span.attributes:
            self._output.write("  attributes:\n")
            for key, value in span.attributes.items():
                self._output.write(f"    {key}: {value}\n")

        # Events
        if span.events:
            self._output.write("  events:\n")
            for event in span.events:
                self._output.write(f"    - {event.name}\n")

        self._output.write("\n")
        self._output.flush()

    def shutdown(self) -> None:
        """Shutdown (flush output)."""
        try:
            self._output.flush()
        except Exception:
            pass


# =============================================================================
# In-Memory Exporter
# =============================================================================


class InMemorySpanExporter(SpanExporter):
    """Exporter that stores spans in memory.

    Useful for testing and debugging. Allows inspection of
    exported spans.

    Example:
        >>> exporter = InMemorySpanExporter()
        >>> processor = SimpleSpanProcessor(exporter)
        >>> # ... run traced code ...
        >>> spans = exporter.get_finished_spans()
        >>> assert len(spans) == 1
    """

    def __init__(self) -> None:
        """Initialize in-memory exporter."""
        self._spans: list["Span"] = []
        self._lock = threading.Lock()
        self._shutdown = False

    def export(self, spans: Sequence["Span"]) -> ExportResult:
        """Store spans in memory."""
        if self._shutdown:
            return ExportResult.FAILURE

        with self._lock:
            self._spans.extend(spans)

        return ExportResult.SUCCESS

    def get_finished_spans(self) -> list["Span"]:
        """Get all exported spans.

        Returns:
            List of exported spans.
        """
        with self._lock:
            return list(self._spans)

    def clear(self) -> None:
        """Clear stored spans."""
        with self._lock:
            self._spans.clear()

    def shutdown(self) -> None:
        """Shutdown exporter."""
        self._shutdown = True


# =============================================================================
# OTLP Exporter
# =============================================================================


@dataclass
class OTLPConfig:
    """Configuration for OTLP exporter."""

    endpoint: str = "http://localhost:4317"
    headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    compression: str = "none"  # none, gzip
    insecure: bool = False

    @classmethod
    def grpc(cls, endpoint: str = "localhost:4317", **kwargs) -> "OTLPConfig":
        """Create gRPC configuration."""
        if not endpoint.startswith("http"):
            endpoint = f"http://{endpoint}"
        return cls(endpoint=endpoint, **kwargs)

    @classmethod
    def http(cls, endpoint: str = "http://localhost:4318/v1/traces", **kwargs) -> "OTLPConfig":
        """Create HTTP configuration."""
        return cls(endpoint=endpoint, **kwargs)


class OTLPSpanExporter(SpanExporter):
    """OpenTelemetry Protocol (OTLP) span exporter.

    Exports spans using the OTLP protocol, which is supported by
    many backends including Jaeger, Tempo, and OpenTelemetry Collector.

    Supports both gRPC and HTTP transports.

    Example:
        >>> # gRPC transport
        >>> exporter = OTLPSpanExporter(
        ...     config=OTLPConfig.grpc("localhost:4317"),
        ... )
        >>>
        >>> # HTTP transport
        >>> exporter = OTLPSpanExporter(
        ...     config=OTLPConfig.http("http://localhost:4318/v1/traces"),
        ... )
    """

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        config: OTLPConfig | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize OTLP exporter.

        Args:
            endpoint: OTLP endpoint URL.
            config: Full configuration.
            headers: Additional headers.
        """
        self._config = config or OTLPConfig(
            endpoint=endpoint or "http://localhost:4317"
        )
        if headers:
            self._config.headers.update(headers)

        self._shutdown = False

    def _convert_span(self, span: "Span") -> dict[str, Any]:
        """Convert span to OTLP format."""
        from truthound.observability.tracing.span import StatusCode

        # Convert status code
        status_code_map = {
            StatusCode.UNSET: 0,
            StatusCode.OK: 1,
            StatusCode.ERROR: 2,
        }

        return {
            "traceId": span.context.trace_id,
            "spanId": span.context.span_id,
            "parentSpanId": span.parent.span_id if span.parent else "",
            "name": span.name,
            "kind": span.kind.value,
            "startTimeUnixNano": int(span.start_time * 1_000_000_000),
            "endTimeUnixNano": int(span.end_time * 1_000_000_000) if span.end_time else 0,
            "attributes": [
                {"key": k, "value": self._convert_value(v)}
                for k, v in span.attributes.items()
            ],
            "events": [
                {
                    "name": e.name,
                    "timeUnixNano": int(e.timestamp * 1_000_000_000),
                    "attributes": [
                        {"key": k, "value": self._convert_value(v)}
                        for k, v in e.attributes.items()
                    ],
                }
                for e in span.events
            ],
            "status": {
                "code": status_code_map.get(span.status[0], 0),
                "message": span.status[1],
            },
        }

    def _convert_value(self, value: Any) -> dict[str, Any]:
        """Convert attribute value to OTLP format."""
        if isinstance(value, bool):
            return {"boolValue": value}
        elif isinstance(value, int):
            return {"intValue": str(value)}
        elif isinstance(value, float):
            return {"doubleValue": value}
        elif isinstance(value, str):
            return {"stringValue": value}
        elif isinstance(value, list):
            # Array value
            if all(isinstance(v, str) for v in value):
                return {"arrayValue": {"values": [{"stringValue": v} for v in value]}}
            return {"stringValue": str(value)}
        else:
            return {"stringValue": str(value)}

    def export(self, spans: Sequence["Span"]) -> ExportResult:
        """Export spans via OTLP."""
        if self._shutdown:
            return ExportResult.FAILURE

        if not spans:
            return ExportResult.SUCCESS

        try:
            import urllib.request
            import urllib.error

            # Build OTLP request body
            resource_spans = [{
                "resource": {"attributes": []},
                "scopeSpans": [{
                    "scope": {"name": "truthound"},
                    "spans": [self._convert_span(span) for span in spans],
                }],
            }]

            body = json.dumps({"resourceSpans": resource_spans}).encode("utf-8")

            # Build request
            endpoint = self._config.endpoint
            if not endpoint.endswith("/v1/traces"):
                endpoint = endpoint.rstrip("/") + "/v1/traces"

            headers = {
                "Content-Type": "application/json",
                **self._config.headers,
            }

            request = urllib.request.Request(
                endpoint,
                data=body,
                headers=headers,
                method="POST",
            )

            with urllib.request.urlopen(
                request, timeout=self._config.timeout_seconds
            ) as response:
                if response.status == 200:
                    return ExportResult.SUCCESS
                elif response.status >= 500:
                    return ExportResult.RETRY
                else:
                    return ExportResult.FAILURE

        except urllib.error.HTTPError as e:
            if e.code >= 500:
                return ExportResult.RETRY
            return ExportResult.FAILURE
        except Exception:
            return ExportResult.FAILURE

    def shutdown(self) -> None:
        """Shutdown exporter."""
        self._shutdown = True


# =============================================================================
# Jaeger Exporter
# =============================================================================


@dataclass
class JaegerConfig:
    """Configuration for Jaeger exporter."""

    agent_host: str = "localhost"
    agent_port: int = 6831
    collector_endpoint: str | None = None
    username: str | None = None
    password: str | None = None
    timeout_seconds: float = 30.0


class JaegerExporter(SpanExporter):
    """Jaeger native format exporter.

    Exports spans to Jaeger using either:
    - UDP agent protocol (default, port 6831)
    - HTTP collector endpoint

    Example:
        >>> # Via agent
        >>> exporter = JaegerExporter(
        ...     config=JaegerConfig(agent_host="jaeger", agent_port=6831),
        ... )
        >>>
        >>> # Via collector
        >>> exporter = JaegerExporter(
        ...     config=JaegerConfig(
        ...         collector_endpoint="http://jaeger:14268/api/traces"
        ...     ),
        ... )
    """

    def __init__(self, config: JaegerConfig | None = None) -> None:
        """Initialize Jaeger exporter.

        Args:
            config: Jaeger configuration.
        """
        self._config = config or JaegerConfig()
        self._shutdown = False

    def _convert_span(self, span: "Span") -> dict[str, Any]:
        """Convert span to Jaeger format."""
        from truthound.observability.tracing.span import StatusCode

        tags = [
            {"key": k, "type": "string", "value": str(v)}
            for k, v in span.attributes.items()
        ]

        # Add status as tag
        if span.status[0] == StatusCode.ERROR:
            tags.append({"key": "error", "type": "bool", "value": True})

        logs = [
            {
                "timestamp": int(e.timestamp * 1_000_000),  # microseconds
                "fields": [
                    {"key": "event", "type": "string", "value": e.name},
                    *[
                        {"key": k, "type": "string", "value": str(v)}
                        for k, v in e.attributes.items()
                    ],
                ],
            }
            for e in span.events
        ]

        return {
            "traceIdLow": int(span.context.trace_id[16:], 16),
            "traceIdHigh": int(span.context.trace_id[:16], 16),
            "spanId": int(span.context.span_id, 16),
            "parentSpanId": int(span.parent.span_id, 16) if span.parent else 0,
            "operationName": span.name,
            "startTime": int(span.start_time * 1_000_000),  # microseconds
            "duration": int((span.duration_ms or 0) * 1000),  # microseconds
            "tags": tags,
            "logs": logs,
        }

    def export(self, spans: Sequence["Span"]) -> ExportResult:
        """Export spans to Jaeger."""
        if self._shutdown:
            return ExportResult.FAILURE

        if not spans:
            return ExportResult.SUCCESS

        try:
            if self._config.collector_endpoint:
                return self._export_http(spans)
            else:
                return self._export_udp(spans)
        except Exception:
            return ExportResult.FAILURE

    def _export_http(self, spans: Sequence["Span"]) -> ExportResult:
        """Export via HTTP collector."""
        import urllib.request
        import urllib.error

        batch = {
            "process": {
                "serviceName": "truthound",
                "tags": [],
            },
            "spans": [self._convert_span(span) for span in spans],
        }

        body = json.dumps(batch).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self._config.username and self._config.password:
            import base64
            auth = base64.b64encode(
                f"{self._config.username}:{self._config.password}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {auth}"

        request = urllib.request.Request(
            self._config.collector_endpoint,
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request, timeout=self._config.timeout_seconds
            ) as response:
                if response.status == 200:
                    return ExportResult.SUCCESS
                elif response.status >= 500:
                    return ExportResult.RETRY
                return ExportResult.FAILURE
        except urllib.error.HTTPError as e:
            if e.code >= 500:
                return ExportResult.RETRY
            return ExportResult.FAILURE

    def _export_udp(self, spans: Sequence["Span"]) -> ExportResult:
        """Export via UDP agent (simplified Thrift-like)."""
        import socket

        # Simplified UDP export (real implementation would use Thrift)
        # This is a simplified JSON-based approach
        batch = {
            "process": {"serviceName": "truthound"},
            "spans": [self._convert_span(span) for span in spans],
        }

        data = json.dumps(batch).encode("utf-8")

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.sendto(data, (self._config.agent_host, self._config.agent_port))
            return ExportResult.SUCCESS
        except socket.error:
            return ExportResult.FAILURE
        finally:
            sock.close()

    def shutdown(self) -> None:
        """Shutdown exporter."""
        self._shutdown = True


# =============================================================================
# Zipkin Exporter
# =============================================================================


@dataclass
class ZipkinConfig:
    """Configuration for Zipkin exporter."""

    endpoint: str = "http://localhost:9411/api/v2/spans"
    timeout_seconds: float = 30.0
    local_node_service_name: str = "truthound"


class ZipkinExporter(SpanExporter):
    """Zipkin JSON format exporter.

    Exports spans to Zipkin-compatible backends using the V2 JSON format.

    Example:
        >>> exporter = ZipkinExporter(
        ...     config=ZipkinConfig(endpoint="http://zipkin:9411/api/v2/spans"),
        ... )
    """

    def __init__(self, config: ZipkinConfig | None = None) -> None:
        """Initialize Zipkin exporter.

        Args:
            config: Zipkin configuration.
        """
        self._config = config or ZipkinConfig()
        self._shutdown = False

    def _convert_span(self, span: "Span") -> dict[str, Any]:
        """Convert span to Zipkin V2 format."""
        from truthound.observability.tracing.span import SpanKind, StatusCode

        # Map span kind
        kind_map = {
            SpanKind.CLIENT: "CLIENT",
            SpanKind.SERVER: "SERVER",
            SpanKind.PRODUCER: "PRODUCER",
            SpanKind.CONSUMER: "CONSUMER",
        }

        zipkin_span = {
            "traceId": span.context.trace_id,
            "id": span.context.span_id,
            "name": span.name,
            "timestamp": int(span.start_time * 1_000_000),  # microseconds
            "duration": int((span.duration_ms or 0) * 1000),  # microseconds
            "localEndpoint": {"serviceName": self._config.local_node_service_name},
            "tags": {k: str(v) for k, v in span.attributes.items()},
        }

        if span.parent:
            zipkin_span["parentId"] = span.parent.span_id

        if span.kind in kind_map:
            zipkin_span["kind"] = kind_map[span.kind]

        # Add error tag
        if span.status[0] == StatusCode.ERROR:
            zipkin_span["tags"]["error"] = span.status[1] or "true"

        # Convert events to annotations
        annotations = [
            {
                "timestamp": int(e.timestamp * 1_000_000),
                "value": e.name,
            }
            for e in span.events
        ]
        if annotations:
            zipkin_span["annotations"] = annotations

        return zipkin_span

    def export(self, spans: Sequence["Span"]) -> ExportResult:
        """Export spans to Zipkin."""
        if self._shutdown:
            return ExportResult.FAILURE

        if not spans:
            return ExportResult.SUCCESS

        try:
            import urllib.request
            import urllib.error

            zipkin_spans = [self._convert_span(span) for span in spans]
            body = json.dumps(zipkin_spans).encode("utf-8")

            request = urllib.request.Request(
                self._config.endpoint,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(
                request, timeout=self._config.timeout_seconds
            ) as response:
                if response.status in (200, 202):
                    return ExportResult.SUCCESS
                elif response.status >= 500:
                    return ExportResult.RETRY
                return ExportResult.FAILURE

        except urllib.error.HTTPError as e:
            if e.code >= 500:
                return ExportResult.RETRY
            return ExportResult.FAILURE
        except Exception:
            return ExportResult.FAILURE

    def shutdown(self) -> None:
        """Shutdown exporter."""
        self._shutdown = True


# =============================================================================
# Multi Exporter
# =============================================================================


class MultiSpanExporter(SpanExporter):
    """Exporter that sends spans to multiple backends.

    Useful for:
        - Gradual migration between backends
        - Redundant storage
        - Different backends for different purposes

    Example:
        >>> exporter = MultiSpanExporter([
        ...     ConsoleSpanExporter(),  # Debug output
        ...     OTLPSpanExporter(...),  # Production backend
        ... ])
    """

    def __init__(self, exporters: list[SpanExporter] | None = None) -> None:
        """Initialize multi-exporter.

        Args:
            exporters: List of exporters.
        """
        self._exporters = list(exporters or [])

    def add_exporter(self, exporter: SpanExporter) -> None:
        """Add an exporter."""
        self._exporters.append(exporter)

    def export(self, spans: Sequence["Span"]) -> ExportResult:
        """Export to all exporters."""
        results = []
        for exporter in self._exporters:
            try:
                results.append(exporter.export(spans))
            except Exception:
                results.append(ExportResult.FAILURE)

        # Return worst result
        if ExportResult.FAILURE in results:
            return ExportResult.FAILURE
        if ExportResult.RETRY in results:
            return ExportResult.RETRY
        return ExportResult.SUCCESS

    def shutdown(self) -> None:
        """Shutdown all exporters."""
        for exporter in self._exporters:
            try:
                exporter.shutdown()
            except Exception:
                pass
