"""Bridge components for cross-backend interoperability.

This module provides bridge classes that enable using components from
one backend with the other:

- Use OpenTelemetry SDK exporters with Truthound spans
- Use Truthound processors with OpenTelemetry spans
- Bridge samplers, propagators, and resources
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from truthound.observability.tracing.otel.detection import is_otel_sdk_available
from truthound.observability.tracing.otel.adapter import SpanContextAdapter

logger = logging.getLogger(__name__)


# =============================================================================
# Span Context Bridge
# =============================================================================


class SpanContextBridge:
    """Bridges span contexts between Truthound and OpenTelemetry formats.

    Handles conversion of trace/span IDs, trace flags, and trace state
    between the two format conventions.

    Example:
        >>> bridge = SpanContextBridge()
        >>> otel_ctx = bridge.truthound_to_otel(truthound_ctx)
        >>> truthound_ctx = bridge.otel_to_truthound(otel_ctx)
    """

    def truthound_to_otel(self, ctx: Any) -> Any:
        """Convert Truthound SpanContextData to OTEL SpanContext.

        Args:
            ctx: Truthound SpanContextData.

        Returns:
            OpenTelemetry SpanContext.

        Raises:
            ImportError: If OTEL SDK is not available.
        """
        if not is_otel_sdk_available():
            raise ImportError("OpenTelemetry SDK not available")

        from opentelemetry.trace import SpanContext, TraceState

        # Parse trace state
        trace_state = TraceState()
        if hasattr(ctx, "trace_state") and ctx.trace_state:
            state_str = ctx.trace_state if isinstance(ctx.trace_state, str) else str(ctx.trace_state)
            for pair in state_str.split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    trace_state = trace_state.add(key.strip(), value.strip())

        # Convert IDs
        trace_id = ctx.trace_id
        span_id = ctx.span_id

        if isinstance(trace_id, str):
            trace_id = int(trace_id, 16)
        if isinstance(span_id, str):
            span_id = int(span_id, 16)

        return SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            is_remote=getattr(ctx, "is_remote", False),
            trace_flags=ctx.trace_flags,
            trace_state=trace_state,
        )

    def otel_to_truthound(self, ctx: Any) -> Any:
        """Convert OTEL SpanContext to Truthound SpanContextData.

        Args:
            ctx: OpenTelemetry SpanContext.

        Returns:
            Truthound SpanContextData.
        """
        from truthound.observability.tracing.span import SpanContextData

        # Convert trace state to string
        trace_state = ""
        if hasattr(ctx, "trace_state") and ctx.trace_state:
            pairs = [f"{k}={v}" for k, v in ctx.trace_state.items()]
            trace_state = ",".join(pairs)

        # Format IDs as hex strings
        trace_id = ctx.trace_id
        span_id = ctx.span_id

        if isinstance(trace_id, int):
            trace_id = format(trace_id, "032x")
        if isinstance(span_id, int):
            span_id = format(span_id, "016x")

        return SpanContextData(
            trace_id=trace_id,
            span_id=span_id,
            trace_flags=ctx.trace_flags,
            trace_state=trace_state,
            is_remote=ctx.is_remote,
        )


# =============================================================================
# Span Bridge
# =============================================================================


class SpanBridge:
    """Bridges spans between Truthound and OpenTelemetry formats.

    Converts span data including attributes, events, links, and status.

    Example:
        >>> bridge = SpanBridge()
        >>> otel_data = bridge.truthound_to_otel_data(truthound_span)
        >>> # Use otel_data with OTEL exporter
    """

    def __init__(self) -> None:
        """Initialize span bridge."""
        self._context_bridge = SpanContextBridge()

    def truthound_to_otel_data(self, span: Any) -> dict[str, Any]:
        """Convert Truthound span to OTEL-compatible data dict.

        This is useful for bridging to OTEL exporters.

        Args:
            span: Truthound Span.

        Returns:
            Dictionary with OTEL-compatible span data.
        """
        from truthound.observability.tracing.span import StatusCode, SpanKind

        # Map status code
        status_map = {
            StatusCode.UNSET: 0,
            StatusCode.OK: 1,
            StatusCode.ERROR: 2,
        }

        # Map span kind
        kind_map = {
            SpanKind.INTERNAL: 0,
            SpanKind.SERVER: 1,
            SpanKind.CLIENT: 2,
            SpanKind.PRODUCER: 3,
            SpanKind.CONSUMER: 4,
        }

        status = span.status if hasattr(span, "status") else (StatusCode.UNSET, "")
        if isinstance(status, tuple):
            status_code, status_message = status
        else:
            status_code = status
            status_message = ""

        return {
            "trace_id": span.context.trace_id if hasattr(span, "context") else "",
            "span_id": span.context.span_id if hasattr(span, "context") else "",
            "parent_span_id": span.parent.span_id if hasattr(span, "parent") and span.parent else None,
            "name": span.name,
            "kind": kind_map.get(span.kind, 0) if hasattr(span, "kind") else 0,
            "start_time_ns": int(span.start_time * 1_000_000_000) if hasattr(span, "start_time") else 0,
            "end_time_ns": int(span.end_time * 1_000_000_000) if hasattr(span, "end_time") and span.end_time else 0,
            "attributes": dict(span.attributes) if hasattr(span, "attributes") else {},
            "events": [
                {
                    "name": e.name,
                    "timestamp_ns": int(e.timestamp * 1_000_000_000),
                    "attributes": dict(e.attributes),
                }
                for e in (span.events if hasattr(span, "events") else [])
            ],
            "links": [
                {
                    "trace_id": l.context.trace_id,
                    "span_id": l.context.span_id,
                    "attributes": dict(l.attributes),
                }
                for l in (span.links if hasattr(span, "links") else [])
            ],
            "status": {
                "code": status_map.get(status_code, 0),
                "message": status_message,
            },
        }

    def otel_to_truthound_data(self, span_data: dict[str, Any]) -> dict[str, Any]:
        """Convert OTEL span data to Truthound-compatible format.

        Args:
            span_data: OTEL span data dictionary.

        Returns:
            Truthound-compatible span data.
        """
        # Map status code back
        status_code_map = {0: "UNSET", 1: "OK", 2: "ERROR"}

        # Map kind back
        kind_map = {0: "INTERNAL", 1: "SERVER", 2: "CLIENT", 3: "PRODUCER", 4: "CONSUMER"}

        status = span_data.get("status", {})

        return {
            "trace_id": span_data.get("trace_id", ""),
            "span_id": span_data.get("span_id", ""),
            "parent_span_id": span_data.get("parent_span_id"),
            "name": span_data.get("name", ""),
            "kind": kind_map.get(span_data.get("kind", 0), "INTERNAL"),
            "start_time": span_data.get("start_time_ns", 0) / 1_000_000_000,
            "end_time": span_data.get("end_time_ns", 0) / 1_000_000_000 if span_data.get("end_time_ns") else None,
            "attributes": span_data.get("attributes", {}),
            "events": span_data.get("events", []),
            "links": span_data.get("links", []),
            "status_code": status_code_map.get(status.get("code", 0), "UNSET"),
            "status_message": status.get("message", ""),
        }


# =============================================================================
# Span Processor Bridge
# =============================================================================


class SpanProcessorBridge:
    """Bridges span processors between backends.

    Allows using OTEL span processors with Truthound spans and vice versa.

    Example:
        >>> # Use OTEL processor with Truthound
        >>> bridge = SpanProcessorBridge()
        >>> truthound_processor = bridge.wrap_otel_processor(otel_processor)
        >>> provider.add_processor(truthound_processor)
    """

    def __init__(self) -> None:
        """Initialize processor bridge."""
        self._span_bridge = SpanBridge()
        self._context_bridge = SpanContextBridge()

    def wrap_otel_processor(self, otel_processor: Any) -> "TruthoundProcessorWrapper":
        """Wrap an OTEL processor for use with Truthound.

        Args:
            otel_processor: OpenTelemetry SpanProcessor.

        Returns:
            Truthound-compatible processor wrapper.
        """
        return TruthoundProcessorWrapper(otel_processor, self._span_bridge, self._context_bridge)

    def wrap_truthound_processor(self, truthound_processor: Any) -> "OTELProcessorWrapper":
        """Wrap a Truthound processor for use with OTEL SDK.

        Args:
            truthound_processor: Truthound SpanProcessor.

        Returns:
            OTEL-compatible processor wrapper.
        """
        return OTELProcessorWrapper(truthound_processor, self._span_bridge, self._context_bridge)


class TruthoundProcessorWrapper:
    """Wraps an OTEL processor for use with Truthound's TracerProvider.

    Translates Truthound spans to OTEL format before passing to the
    wrapped processor.
    """

    def __init__(
        self,
        otel_processor: Any,
        span_bridge: SpanBridge,
        context_bridge: SpanContextBridge,
    ) -> None:
        """Initialize wrapper.

        Args:
            otel_processor: OTEL processor to wrap.
            span_bridge: Span bridge for conversions.
            context_bridge: Context bridge for conversions.
        """
        self._processor = otel_processor
        self._span_bridge = span_bridge
        self._context_bridge = context_bridge

    def on_start(self, span: Any, parent_context: Any | None = None) -> None:
        """Called when a span starts.

        Args:
            span: Truthound span.
            parent_context: Parent context.
        """
        if not is_otel_sdk_available():
            return

        try:
            # OTEL processor expects OTEL span, but we can try to pass data
            # Most processors don't heavily use on_start
            pass
        except Exception as e:
            logger.debug(f"Error in wrapped on_start: {e}")

    def on_end(self, span: Any) -> None:
        """Called when a span ends.

        Args:
            span: Truthound span.
        """
        if not is_otel_sdk_available():
            return

        try:
            from opentelemetry.sdk.trace import ReadableSpan

            # Create a readable span from Truthound span
            otel_span = _create_readable_span_from_truthound(
                span, self._span_bridge, self._context_bridge
            )
            self._processor.on_end(otel_span)
        except Exception as e:
            logger.debug(f"Error in wrapped on_end: {e}")

    def shutdown(self) -> bool:
        """Shutdown the processor."""
        try:
            self._processor.shutdown()
            return True
        except Exception:
            return False

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush the processor."""
        try:
            return self._processor.force_flush(timeout_millis)
        except Exception:
            return False


class OTELProcessorWrapper:
    """Wraps a Truthound processor for use with OTEL SDK's TracerProvider.

    Translates OTEL spans to Truthound format before passing to the
    wrapped processor.
    """

    def __init__(
        self,
        truthound_processor: Any,
        span_bridge: SpanBridge,
        context_bridge: SpanContextBridge,
    ) -> None:
        """Initialize wrapper.

        Args:
            truthound_processor: Truthound processor to wrap.
            span_bridge: Span bridge for conversions.
            context_bridge: Context bridge for conversions.
        """
        self._processor = truthound_processor
        self._span_bridge = span_bridge
        self._context_bridge = context_bridge

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        """Called when a span starts.

        Args:
            span: OTEL span.
            parent_context: OTEL context.
        """
        try:
            # Convert parent context if provided
            truthound_parent = None
            if parent_context:
                from opentelemetry.trace import get_current_span

                current = get_current_span(parent_context)
                if current and hasattr(current, "get_span_context"):
                    truthound_parent = self._context_bridge.otel_to_truthound(
                        current.get_span_context()
                    )

            # Create minimal Truthound span wrapper
            wrapper = _create_truthound_span_from_otel(span, self._context_bridge)
            self._processor.on_start(wrapper, truthound_parent)
        except Exception as e:
            logger.debug(f"Error in wrapped on_start: {e}")

    def on_end(self, span: Any) -> None:
        """Called when a span ends.

        Args:
            span: OTEL span (ReadableSpan).
        """
        try:
            wrapper = _create_truthound_span_from_otel(span, self._context_bridge)
            self._processor.on_end(wrapper)
        except Exception as e:
            logger.debug(f"Error in wrapped on_end: {e}")

    def shutdown(self) -> bool:
        """Shutdown the processor."""
        try:
            return self._processor.shutdown()
        except Exception:
            return False

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush the processor."""
        try:
            return self._processor.force_flush(timeout_millis)
        except Exception:
            return False


# =============================================================================
# Span Exporter Bridge
# =============================================================================


class SpanExporterBridge:
    """Bridges span exporters between backends.

    Allows using OTEL exporters with Truthound spans and vice versa.

    Example:
        >>> # Use OTEL OTLP exporter with Truthound
        >>> from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        >>>
        >>> bridge = SpanExporterBridge()
        >>> truthound_exporter = bridge.wrap_otel_exporter(OTLPSpanExporter())
        >>> provider.add_processor(SimpleSpanProcessor(truthound_exporter))
    """

    def __init__(self) -> None:
        """Initialize exporter bridge."""
        self._span_bridge = SpanBridge()
        self._context_bridge = SpanContextBridge()

    def wrap_otel_exporter(self, otel_exporter: Any) -> "TruthoundExporterWrapper":
        """Wrap an OTEL exporter for use with Truthound.

        Args:
            otel_exporter: OpenTelemetry SpanExporter.

        Returns:
            Truthound-compatible exporter wrapper.
        """
        return TruthoundExporterWrapper(otel_exporter, self._span_bridge, self._context_bridge)

    def wrap_truthound_exporter(self, truthound_exporter: Any) -> "OTELExporterWrapper":
        """Wrap a Truthound exporter for use with OTEL SDK.

        Args:
            truthound_exporter: Truthound SpanExporter.

        Returns:
            OTEL-compatible exporter wrapper.
        """
        return OTELExporterWrapper(truthound_exporter, self._span_bridge, self._context_bridge)


class TruthoundExporterWrapper:
    """Wraps an OTEL exporter for use with Truthound.

    Converts Truthound spans to OTEL ReadableSpan format before exporting.
    """

    def __init__(
        self,
        otel_exporter: Any,
        span_bridge: SpanBridge,
        context_bridge: SpanContextBridge,
    ) -> None:
        """Initialize wrapper.

        Args:
            otel_exporter: OTEL exporter to wrap.
            span_bridge: Span bridge for conversions.
            context_bridge: Context bridge for conversions.
        """
        self._exporter = otel_exporter
        self._span_bridge = span_bridge
        self._context_bridge = context_bridge
        self._shutdown = False

    def export(self, spans: Sequence[Any]) -> Any:
        """Export spans.

        Args:
            spans: Truthound spans to export.

        Returns:
            Export result.
        """
        if self._shutdown:
            from truthound.observability.tracing.exporter import ExportResult
            return ExportResult.FAILURE

        if not is_otel_sdk_available():
            from truthound.observability.tracing.exporter import ExportResult
            return ExportResult.FAILURE

        try:
            # Convert to OTEL readable spans
            otel_spans = [
                _create_readable_span_from_truthound(s, self._span_bridge, self._context_bridge)
                for s in spans
            ]

            # Export using OTEL exporter
            result = self._exporter.export(otel_spans)

            # Convert result
            from opentelemetry.sdk.trace.export import SpanExportResult
            from truthound.observability.tracing.exporter import ExportResult

            result_map = {
                SpanExportResult.SUCCESS: ExportResult.SUCCESS,
                SpanExportResult.FAILURE: ExportResult.FAILURE,
            }
            return result_map.get(result, ExportResult.FAILURE)

        except Exception as e:
            logger.error(f"Error exporting spans via OTEL: {e}")
            from truthound.observability.tracing.exporter import ExportResult
            return ExportResult.FAILURE

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        self._shutdown = True
        try:
            self._exporter.shutdown()
        except Exception:
            pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush the exporter."""
        try:
            if hasattr(self._exporter, "force_flush"):
                return self._exporter.force_flush(timeout_millis)
            return True
        except Exception:
            return False


class OTELExporterWrapper:
    """Wraps a Truthound exporter for use with OTEL SDK.

    Converts OTEL ReadableSpan to Truthound span format before exporting.
    """

    def __init__(
        self,
        truthound_exporter: Any,
        span_bridge: SpanBridge,
        context_bridge: SpanContextBridge,
    ) -> None:
        """Initialize wrapper.

        Args:
            truthound_exporter: Truthound exporter to wrap.
            span_bridge: Span bridge for conversions.
            context_bridge: Context bridge for conversions.
        """
        self._exporter = truthound_exporter
        self._span_bridge = span_bridge
        self._context_bridge = context_bridge
        self._shutdown = False

    def export(self, spans: Sequence[Any]) -> Any:
        """Export spans.

        Args:
            spans: OTEL spans (ReadableSpan) to export.

        Returns:
            OTEL SpanExportResult.
        """
        if not is_otel_sdk_available():
            from opentelemetry.sdk.trace.export import SpanExportResult
            return SpanExportResult.FAILURE

        try:
            # Convert to Truthound spans
            truthound_spans = [
                _create_truthound_span_from_otel(s, self._context_bridge)
                for s in spans
            ]

            # Export using Truthound exporter
            result = self._exporter.export(truthound_spans)

            # Convert result
            from opentelemetry.sdk.trace.export import SpanExportResult
            from truthound.observability.tracing.exporter import ExportResult

            result_map = {
                ExportResult.SUCCESS: SpanExportResult.SUCCESS,
                ExportResult.FAILURE: SpanExportResult.FAILURE,
                ExportResult.RETRY: SpanExportResult.FAILURE,
            }
            return result_map.get(result, SpanExportResult.FAILURE)

        except Exception as e:
            logger.error(f"Error exporting spans via Truthound: {e}")
            from opentelemetry.sdk.trace.export import SpanExportResult
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        self._shutdown = True
        try:
            self._exporter.shutdown()
        except Exception:
            pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush the exporter."""
        try:
            if hasattr(self._exporter, "force_flush"):
                return self._exporter.force_flush(timeout_millis)
            return True
        except Exception:
            return False


# =============================================================================
# Sampler Bridge
# =============================================================================


class SamplerBridge:
    """Bridges samplers between backends.

    Allows using OTEL samplers with Truthound and vice versa.
    """

    def __init__(self) -> None:
        """Initialize sampler bridge."""
        self._context_bridge = SpanContextBridge()

    def wrap_otel_sampler(self, otel_sampler: Any) -> "TruthoundSamplerWrapper":
        """Wrap an OTEL sampler for use with Truthound.

        Args:
            otel_sampler: OpenTelemetry Sampler.

        Returns:
            Truthound-compatible sampler wrapper.
        """
        return TruthoundSamplerWrapper(otel_sampler, self._context_bridge)

    def wrap_truthound_sampler(self, truthound_sampler: Any) -> "OTELSamplerWrapper":
        """Wrap a Truthound sampler for use with OTEL SDK.

        Args:
            truthound_sampler: Truthound Sampler.

        Returns:
            OTEL-compatible sampler wrapper.
        """
        return OTELSamplerWrapper(truthound_sampler, self._context_bridge)


class TruthoundSamplerWrapper:
    """Wraps an OTEL sampler for use with Truthound."""

    def __init__(self, otel_sampler: Any, context_bridge: SpanContextBridge) -> None:
        self._sampler = otel_sampler
        self._context_bridge = context_bridge

    def should_sample(
        self,
        parent_context: Any,
        trace_id: str | int,
        name: str,
        kind: Any = None,
        attributes: Mapping[str, Any] | None = None,
        links: Sequence[Any] | None = None,
    ) -> Any:
        """Make sampling decision."""
        from truthound.observability.tracing.sampler import SamplingResult, SamplingDecision

        if not is_otel_sdk_available():
            return SamplingResult(decision=SamplingDecision.RECORD_AND_SAMPLE)

        try:
            from opentelemetry.trace import SpanKind

            # Convert kind
            otel_kind = SpanKind.INTERNAL
            if kind and hasattr(kind, "name"):
                kind_map = {
                    "INTERNAL": SpanKind.INTERNAL,
                    "SERVER": SpanKind.SERVER,
                    "CLIENT": SpanKind.CLIENT,
                    "PRODUCER": SpanKind.PRODUCER,
                    "CONSUMER": SpanKind.CONSUMER,
                }
                otel_kind = kind_map.get(kind.name, SpanKind.INTERNAL)

            # Convert trace_id
            if isinstance(trace_id, str):
                trace_id = int(trace_id, 16)

            # Call OTEL sampler
            otel_result = self._sampler.should_sample(
                parent_context=None,  # OTEL uses Context, not SpanContext directly
                trace_id=trace_id,
                name=name,
                kind=otel_kind,
                attributes=dict(attributes) if attributes else None,
                links=links,
            )

            # Convert result
            from opentelemetry.sdk.trace.sampling import Decision

            decision_map = {
                Decision.DROP: SamplingDecision.DROP,
                Decision.RECORD_ONLY: SamplingDecision.RECORD_ONLY,
                Decision.RECORD_AND_SAMPLE: SamplingDecision.RECORD_AND_SAMPLE,
            }

            return SamplingResult(
                decision=decision_map.get(otel_result.decision, SamplingDecision.RECORD_AND_SAMPLE),
                attributes=dict(otel_result.attributes) if otel_result.attributes else {},
            )

        except Exception as e:
            logger.debug(f"Error in sampler wrapper: {e}")
            return SamplingResult(decision=SamplingDecision.RECORD_AND_SAMPLE)

    def get_description(self) -> str:
        """Get sampler description."""
        return f"OTELWrapper({self._sampler.get_description()})"


class OTELSamplerWrapper:
    """Wraps a Truthound sampler for use with OTEL SDK."""

    def __init__(self, truthound_sampler: Any, context_bridge: SpanContextBridge) -> None:
        self._sampler = truthound_sampler
        self._context_bridge = context_bridge

    def should_sample(
        self,
        parent_context: Any,
        trace_id: int,
        name: str,
        kind: Any = None,
        attributes: Mapping[str, Any] | None = None,
        links: Sequence[Any] | None = None,
    ) -> Any:
        """Make sampling decision."""
        if not is_otel_sdk_available():
            from opentelemetry.sdk.trace.sampling import Decision, SamplingResult
            return SamplingResult(decision=Decision.RECORD_AND_SAMPLE, attributes={})

        try:
            from opentelemetry.sdk.trace.sampling import Decision, SamplingResult
            from truthound.observability.tracing.sampler import SamplingDecision

            # Call Truthound sampler
            result = self._sampler.should_sample(
                parent_context=None,
                trace_id=format(trace_id, "032x"),
                name=name,
                kind=kind,
                attributes=attributes,
                links=links,
            )

            # Convert result
            decision_map = {
                SamplingDecision.DROP: Decision.DROP,
                SamplingDecision.RECORD_ONLY: Decision.RECORD_ONLY,
                SamplingDecision.RECORD_AND_SAMPLE: Decision.RECORD_AND_SAMPLE,
            }

            return SamplingResult(
                decision=decision_map.get(result.decision, Decision.RECORD_AND_SAMPLE),
                attributes=result.attributes,
            )

        except Exception as e:
            logger.debug(f"Error in sampler wrapper: {e}")
            from opentelemetry.sdk.trace.sampling import Decision, SamplingResult
            return SamplingResult(decision=Decision.RECORD_AND_SAMPLE, attributes={})

    def get_description(self) -> str:
        """Get sampler description."""
        return f"TruthoundWrapper({self._sampler.get_description()})"


# =============================================================================
# Propagator Bridge
# =============================================================================


class PropagatorBridge:
    """Bridges propagators between backends.

    Allows using OTEL propagators with Truthound and vice versa.
    """

    def __init__(self) -> None:
        """Initialize propagator bridge."""
        self._context_bridge = SpanContextBridge()

    def wrap_otel_propagator(self, otel_propagator: Any) -> "TruthoundPropagatorWrapper":
        """Wrap an OTEL propagator for use with Truthound."""
        return TruthoundPropagatorWrapper(otel_propagator, self._context_bridge)

    def wrap_truthound_propagator(self, truthound_propagator: Any) -> "OTELPropagatorWrapper":
        """Wrap a Truthound propagator for use with OTEL SDK."""
        return OTELPropagatorWrapper(truthound_propagator, self._context_bridge)


class TruthoundPropagatorWrapper:
    """Wraps an OTEL propagator for use with Truthound."""

    def __init__(self, otel_propagator: Any, context_bridge: SpanContextBridge) -> None:
        self._propagator = otel_propagator
        self._context_bridge = context_bridge

    def extract(
        self,
        carrier: Mapping[str, str],
        context: Any = None,
        getter: Any = None,
    ) -> Any:
        """Extract context from carrier."""
        if not is_otel_sdk_available():
            return None

        try:
            from opentelemetry.propagate import extract

            otel_context = extract(carrier)
            from opentelemetry.trace import get_current_span

            span = get_current_span(otel_context)
            if span and hasattr(span, "get_span_context"):
                return self._context_bridge.otel_to_truthound(span.get_span_context())
        except Exception as e:
            logger.debug(f"Error extracting context: {e}")

        return None

    def inject(
        self,
        carrier: dict[str, str],
        context: Any = None,
        setter: Any = None,
    ) -> None:
        """Inject context into carrier."""
        if not is_otel_sdk_available() or context is None:
            return

        try:
            from opentelemetry.propagate import inject
            from opentelemetry.trace import set_span_in_context, NonRecordingSpan

            otel_ctx = self._context_bridge.truthound_to_otel(context)
            span = NonRecordingSpan(otel_ctx)
            otel_context = set_span_in_context(span)
            inject(carrier, context=otel_context)
        except Exception as e:
            logger.debug(f"Error injecting context: {e}")

    @property
    def fields(self) -> set[str]:
        """Get propagator fields."""
        if hasattr(self._propagator, "fields"):
            return set(self._propagator.fields)
        return {"traceparent", "tracestate"}


class OTELPropagatorWrapper:
    """Wraps a Truthound propagator for use with OTEL SDK."""

    def __init__(self, truthound_propagator: Any, context_bridge: SpanContextBridge) -> None:
        self._propagator = truthound_propagator
        self._context_bridge = context_bridge

    def extract(
        self,
        carrier: Mapping[str, str],
        context: Any = None,
        getter: Any = None,
    ) -> Any:
        """Extract context from carrier."""
        try:
            truthound_ctx = self._propagator.extract(carrier)
            if truthound_ctx:
                from opentelemetry.trace import set_span_in_context, NonRecordingSpan

                otel_ctx = self._context_bridge.truthound_to_otel(truthound_ctx)
                span = NonRecordingSpan(otel_ctx)
                return set_span_in_context(span, context)
        except Exception as e:
            logger.debug(f"Error extracting context: {e}")

        return context

    def inject(
        self,
        carrier: dict[str, str],
        context: Any = None,
        setter: Any = None,
    ) -> None:
        """Inject context into carrier."""
        if not is_otel_sdk_available():
            return

        try:
            from opentelemetry.trace import get_current_span

            span = get_current_span(context)
            if span and hasattr(span, "get_span_context"):
                truthound_ctx = self._context_bridge.otel_to_truthound(span.get_span_context())
                self._propagator.inject(carrier, truthound_ctx)
        except Exception as e:
            logger.debug(f"Error injecting context: {e}")

    @property
    def fields(self) -> set[str]:
        """Get propagator fields."""
        if hasattr(self._propagator, "fields"):
            return self._propagator.fields
        return {"traceparent", "tracestate"}


# =============================================================================
# Helper Functions
# =============================================================================


def _create_readable_span_from_truthound(
    span: Any,
    span_bridge: SpanBridge,
    context_bridge: SpanContextBridge,
) -> Any:
    """Create an OTEL ReadableSpan from a Truthound span.

    This creates a minimal ReadableSpan-like object for exporting.
    """
    if not is_otel_sdk_available():
        raise ImportError("OpenTelemetry SDK not available")

    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.trace import SpanKind, StatusCode, Status
    from opentelemetry.sdk.trace.export import SpanExportResult

    # Get span data
    data = span_bridge.truthound_to_otel_data(span)

    # Create SpanContext
    otel_context = context_bridge.truthound_to_otel(span.context)

    # Create parent context if available
    parent_context = None
    if hasattr(span, "parent") and span.parent:
        parent_context = context_bridge.truthound_to_otel(span.parent)

    # Map kind
    kind_map = {
        0: SpanKind.INTERNAL,
        1: SpanKind.SERVER,
        2: SpanKind.CLIENT,
        3: SpanKind.PRODUCER,
        4: SpanKind.CONSUMER,
    }
    span_kind = kind_map.get(data["kind"], SpanKind.INTERNAL)

    # Map status
    status_code_map = {0: StatusCode.UNSET, 1: StatusCode.OK, 2: StatusCode.ERROR}
    status = Status(
        status_code=status_code_map.get(data["status"]["code"], StatusCode.UNSET),
        description=data["status"]["message"],
    )

    # Create a ReadableSpan-like object
    # Note: This is a simplified version; full implementation would need
    # to properly implement the ReadableSpan interface
    class _ReadableSpanAdapter:
        def __init__(self):
            self._context = otel_context
            self._parent = parent_context
            self._name = data["name"]
            self._kind = span_kind
            self._start_time = data["start_time_ns"]
            self._end_time = data["end_time_ns"]
            self._attributes = data["attributes"]
            self._events = tuple()
            self._links = tuple()
            self._status = status
            self._resource = None
            self._instrumentation_scope = None

        def get_span_context(self):
            return self._context

        @property
        def name(self):
            return self._name

        @property
        def context(self):
            return self._context

        @property
        def parent(self):
            return self._parent

        @property
        def kind(self):
            return self._kind

        @property
        def start_time(self):
            return self._start_time

        @property
        def end_time(self):
            return self._end_time

        @property
        def attributes(self):
            return self._attributes

        @property
        def events(self):
            return self._events

        @property
        def links(self):
            return self._links

        @property
        def status(self):
            return self._status

        @property
        def resource(self):
            return self._resource

        @property
        def instrumentation_scope(self):
            return self._instrumentation_scope

    return _ReadableSpanAdapter()


def _create_truthound_span_from_otel(
    otel_span: Any,
    context_bridge: SpanContextBridge,
) -> Any:
    """Create a Truthound Span-like object from an OTEL span.

    This creates a minimal span wrapper for Truthound processors.
    """
    from truthound.observability.tracing.span import SpanContextData, SpanKind, StatusCode

    # Get context
    otel_context = otel_span.get_span_context() if hasattr(otel_span, "get_span_context") else otel_span.context
    truthound_context = context_bridge.otel_to_truthound(otel_context)

    # Map kind
    kind_map = {
        "INTERNAL": SpanKind.INTERNAL,
        "SERVER": SpanKind.SERVER,
        "CLIENT": SpanKind.CLIENT,
        "PRODUCER": SpanKind.PRODUCER,
        "CONSUMER": SpanKind.CONSUMER,
    }

    span_kind = SpanKind.INTERNAL
    if hasattr(otel_span, "kind") and otel_span.kind:
        span_kind = kind_map.get(otel_span.kind.name, SpanKind.INTERNAL)

    # Create wrapper class
    class _TruthoundSpanAdapter:
        def __init__(self):
            self._context = truthound_context
            self._name = otel_span.name if hasattr(otel_span, "name") else ""
            self._kind = span_kind
            self._start_time = otel_span.start_time / 1_000_000_000 if hasattr(otel_span, "start_time") else time.time()
            self._end_time = otel_span.end_time / 1_000_000_000 if hasattr(otel_span, "end_time") and otel_span.end_time else None
            self._attributes = dict(otel_span.attributes) if hasattr(otel_span, "attributes") and otel_span.attributes else {}
            self._events = []
            self._links = []
            self._status = (StatusCode.UNSET, "")
            self._parent = None

            if hasattr(otel_span, "parent") and otel_span.parent:
                self._parent = context_bridge.otel_to_truthound(otel_span.parent)

            if hasattr(otel_span, "status") and otel_span.status:
                status_map = {"UNSET": StatusCode.UNSET, "OK": StatusCode.OK, "ERROR": StatusCode.ERROR}
                self._status = (
                    status_map.get(otel_span.status.status_code.name, StatusCode.UNSET),
                    otel_span.status.description or "",
                )

        @property
        def context(self):
            return self._context

        @property
        def name(self):
            return self._name

        @property
        def kind(self):
            return self._kind

        @property
        def start_time(self):
            return self._start_time

        @property
        def end_time(self):
            return self._end_time

        @property
        def attributes(self):
            return self._attributes

        @property
        def events(self):
            return self._events

        @property
        def links(self):
            return self._links

        @property
        def status(self):
            return self._status

        @property
        def parent(self):
            return self._parent

        @property
        def duration_ms(self):
            if self._end_time is None:
                return None
            return (self._end_time - self._start_time) * 1000

    return _TruthoundSpanAdapter()
