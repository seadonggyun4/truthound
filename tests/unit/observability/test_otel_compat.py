"""Tests for OpenTelemetry compatibility layer.

This module tests the OTEL adapter, bridge, and compatibility utilities
to ensure proper interoperability between Truthound and OpenTelemetry.
"""

from __future__ import annotations

import time
import threading
from unittest.mock import Mock, patch, MagicMock

import pytest


# =============================================================================
# Detection Tests
# =============================================================================


class TestOTELDetection:
    """Tests for OpenTelemetry package detection."""

    def test_detect_availability_returns_object(self):
        """Test that detection returns an OTELAvailability object."""
        from truthound.observability.tracing.otel.detection import (
            detect_otel_availability,
            OTELAvailability,
        )

        result = detect_otel_availability()
        assert isinstance(result, OTELAvailability)

    def test_availability_properties(self):
        """Test OTELAvailability properties."""
        from truthound.observability.tracing.otel.detection import OTELAvailability

        avail = OTELAvailability(
            api_available=True,
            sdk_available=True,
            api_version="1.0.0",
            sdk_version="1.0.0",
        )

        assert avail.fully_available is True
        assert avail.api_available is True
        assert avail.sdk_available is True

    def test_availability_partial(self):
        """Test partial availability."""
        from truthound.observability.tracing.otel.detection import OTELAvailability

        avail = OTELAvailability(
            api_available=True,
            sdk_available=False,
        )

        assert avail.fully_available is False
        assert avail.api_available is True
        assert avail.sdk_available is False

    def test_availability_features(self):
        """Test feature detection."""
        from truthound.observability.tracing.otel.detection import OTELAvailability

        avail = OTELAvailability(
            features={"otlp_export", "trace_api"},
        )

        assert avail.can_export_otlp is True
        assert avail.can_export_jaeger is False
        assert avail.can_export_zipkin is False

    def test_availability_to_dict(self):
        """Test conversion to dictionary."""
        from truthound.observability.tracing.otel.detection import OTELAvailability

        avail = OTELAvailability(
            api_available=True,
            sdk_available=True,
            api_version="1.0.0",
        )

        result = avail.to_dict()
        assert isinstance(result, dict)
        assert result["api_available"] is True
        assert result["sdk_available"] is True
        assert result["api_version"] == "1.0.0"

    def test_cache_clearing(self):
        """Test detection cache clearing."""
        from truthound.observability.tracing.otel.detection import (
            detect_otel_availability,
            clear_detection_cache,
        )

        # First detection
        result1 = detect_otel_availability()

        # Clear cache
        clear_detection_cache()

        # Second detection (should re-detect)
        result2 = detect_otel_availability(force_refresh=True)

        # Results should be equivalent
        assert result1.api_available == result2.api_available

    def test_convenience_functions(self):
        """Test convenience detection functions."""
        from truthound.observability.tracing.otel.detection import (
            is_otel_sdk_available,
            is_otel_api_available,
            get_otel_version,
        )

        # These should not raise
        sdk_available = is_otel_sdk_available()
        api_available = is_otel_api_available()
        version = get_otel_version()

        assert isinstance(sdk_available, bool)
        assert isinstance(api_available, bool)
        assert version is None or isinstance(version, str)


# =============================================================================
# Span Context Adapter Tests
# =============================================================================


class TestSpanContextAdapter:
    """Tests for SpanContextAdapter."""

    def test_create_from_strings(self):
        """Test creating adapter from string IDs."""
        from truthound.observability.tracing.otel.adapter import SpanContextAdapter

        ctx = SpanContextAdapter(
            trace_id="0123456789abcdef0123456789abcdef",
            span_id="0123456789abcdef",
            trace_flags=1,
        )

        assert ctx.trace_id == "0123456789abcdef0123456789abcdef"
        assert ctx.span_id == "0123456789abcdef"
        assert ctx.trace_flags == 1
        assert ctx.is_sampled is True
        assert ctx.is_valid is True

    def test_create_from_integers(self):
        """Test creating adapter from integer IDs."""
        from truthound.observability.tracing.otel.adapter import SpanContextAdapter

        ctx = SpanContextAdapter(
            trace_id=12345678901234567890,
            span_id=1234567890,
            trace_flags=0,
        )

        assert isinstance(ctx.trace_id, str)
        assert isinstance(ctx.span_id, str)
        assert ctx.is_sampled is False

    def test_w3c_traceparent_conversion(self):
        """Test W3C traceparent header conversion."""
        from truthound.observability.tracing.otel.adapter import SpanContextAdapter

        ctx = SpanContextAdapter(
            trace_id="0123456789abcdef0123456789abcdef",
            span_id="0123456789abcdef",
            trace_flags=1,
        )

        traceparent = ctx.to_w3c_traceparent()
        assert traceparent == "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"

    def test_from_w3c_traceparent(self):
        """Test parsing W3C traceparent header."""
        from truthound.observability.tracing.otel.adapter import SpanContextAdapter

        header = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
        ctx = SpanContextAdapter.from_w3c_traceparent(header)

        assert ctx is not None
        assert ctx.trace_id == "0123456789abcdef0123456789abcdef"
        assert ctx.span_id == "0123456789abcdef"
        assert ctx.trace_flags == 1
        assert ctx.is_remote is True

    def test_invalid_traceparent(self):
        """Test invalid traceparent returns None."""
        from truthound.observability.tracing.otel.adapter import SpanContextAdapter

        result = SpanContextAdapter.from_w3c_traceparent("invalid")
        assert result is None

    def test_from_truthound_context(self):
        """Test creating adapter from Truthound context."""
        from truthound.observability.tracing.otel.adapter import SpanContextAdapter
        from truthound.observability.tracing.span import SpanContextData

        truthound_ctx = SpanContextData(
            trace_id="0123456789abcdef0123456789abcdef",
            span_id="0123456789abcdef",
            trace_flags=1,
        )

        adapter = SpanContextAdapter.from_truthound(truthound_ctx)
        assert adapter.trace_id == truthound_ctx.trace_id
        assert adapter.span_id == truthound_ctx.span_id

    def test_to_truthound_context(self):
        """Test converting adapter to Truthound context."""
        from truthound.observability.tracing.otel.adapter import SpanContextAdapter

        adapter = SpanContextAdapter(
            trace_id="0123456789abcdef0123456789abcdef",
            span_id="0123456789abcdef",
            trace_flags=1,
        )

        truthound_ctx = adapter.to_truthound()
        assert truthound_ctx.trace_id == adapter.trace_id
        assert truthound_ctx.span_id == adapter.span_id


# =============================================================================
# Span Adapter Tests
# =============================================================================


class TestSpanAdapter:
    """Tests for SpanAdapter."""

    def test_wrap_truthound_span(self):
        """Test wrapping a Truthound span."""
        from truthound.observability.tracing.otel.adapter import (
            SpanAdapter,
            TracingBackend,
        )
        from truthound.observability.tracing.span import (
            Span,
            SpanContextData,
            SpanKind,
        )

        # Create a Truthound span
        context = SpanContextData(
            trace_id="0" * 32,
            span_id="0" * 16,
            trace_flags=1,
        )
        span = Span(name="test-span", context=context, kind=SpanKind.INTERNAL)

        # Wrap it
        adapter = SpanAdapter(span, TracingBackend.TRUTHOUND)

        assert adapter.name == "test-span"
        assert adapter.is_recording() is True

    def test_set_attribute(self):
        """Test setting attributes through adapter."""
        from truthound.observability.tracing.otel.adapter import (
            SpanAdapter,
            TracingBackend,
        )
        from truthound.observability.tracing.span import Span, SpanContextData

        context = SpanContextData(
            trace_id="0" * 32,
            span_id="0" * 16,
        )
        span = Span(name="test", context=context)
        adapter = SpanAdapter(span, TracingBackend.TRUTHOUND)

        adapter.set_attribute("key", "value")
        assert span.attributes.get("key") == "value"

    def test_set_attributes(self):
        """Test setting multiple attributes."""
        from truthound.observability.tracing.otel.adapter import (
            SpanAdapter,
            TracingBackend,
        )
        from truthound.observability.tracing.span import Span, SpanContextData

        context = SpanContextData(trace_id="0" * 32, span_id="0" * 16)
        span = Span(name="test", context=context)
        adapter = SpanAdapter(span, TracingBackend.TRUTHOUND)

        adapter.set_attributes({"key1": "val1", "key2": "val2"})
        assert span.attributes.get("key1") == "val1"
        assert span.attributes.get("key2") == "val2"

    def test_add_event(self):
        """Test adding events through adapter."""
        from truthound.observability.tracing.otel.adapter import (
            SpanAdapter,
            TracingBackend,
        )
        from truthound.observability.tracing.span import Span, SpanContextData

        context = SpanContextData(trace_id="0" * 32, span_id="0" * 16)
        span = Span(name="test", context=context)
        adapter = SpanAdapter(span, TracingBackend.TRUTHOUND)

        adapter.add_event("test-event", {"attr": "value"})
        assert len(span.events) == 1
        assert span.events[0].name == "test-event"

    def test_end_span(self):
        """Test ending span through adapter."""
        from truthound.observability.tracing.otel.adapter import (
            SpanAdapter,
            TracingBackend,
        )
        from truthound.observability.tracing.span import Span, SpanContextData

        context = SpanContextData(trace_id="0" * 32, span_id="0" * 16)
        span = Span(name="test", context=context)
        adapter = SpanAdapter(span, TracingBackend.TRUTHOUND)

        assert adapter.is_recording() is True
        adapter.end()
        assert adapter.is_recording() is False

    def test_context_manager(self):
        """Test adapter as context manager."""
        from truthound.observability.tracing.otel.adapter import (
            SpanAdapter,
            TracingBackend,
        )
        from truthound.observability.tracing.span import Span, SpanContextData

        context = SpanContextData(trace_id="0" * 32, span_id="0" * 16)
        span = Span(name="test", context=context)
        adapter = SpanAdapter(span, TracingBackend.TRUTHOUND)

        with adapter as s:
            s.set_attribute("in_context", True)

        assert span.end_time is not None
        assert span.attributes.get("in_context") is True


# =============================================================================
# Tracer Adapter Tests
# =============================================================================


class TestTracerAdapter:
    """Tests for TracerAdapter."""

    def test_start_span(self):
        """Test starting a span through adapter."""
        from truthound.observability.tracing.otel.adapter import (
            TracerAdapter,
            TracingBackend,
        )
        from truthound.observability.tracing.provider import TracerProvider

        provider = TracerProvider()
        tracer = provider.get_tracer("test")
        adapter = TracerAdapter(tracer, TracingBackend.TRUTHOUND)

        span = adapter.start_span("test-operation")
        assert span is not None
        assert span.name == "test-operation" or hasattr(span, "_span")
        span.end()

    def test_start_as_current_span(self):
        """Test starting span as current."""
        from truthound.observability.tracing.otel.adapter import (
            TracerAdapter,
            TracingBackend,
        )
        from truthound.observability.tracing.provider import TracerProvider

        provider = TracerProvider()
        tracer = provider.get_tracer("test")
        adapter = TracerAdapter(tracer, TracingBackend.TRUTHOUND)

        with adapter.start_as_current_span("test-operation") as span:
            span.set_attribute("key", "value")

        # Span should be ended
        # Note: we check the unwrapped span
        unwrapped = span.unwrap() if hasattr(span, "unwrap") else span._span
        assert unwrapped.end_time is not None


# =============================================================================
# TracerProvider Adapter Tests
# =============================================================================


class TestTracerProviderAdapter:
    """Tests for TracerProviderAdapter."""

    def test_create_with_truthound_backend(self):
        """Test creating adapter with Truthound backend."""
        from truthound.observability.tracing.otel.adapter import (
            TracerProviderAdapter,
            TracingBackend,
        )
        from truthound.observability.tracing.provider import TracerProvider

        provider = TracerProvider()
        adapter = TracerProviderAdapter(provider, TracingBackend.TRUTHOUND)

        assert adapter.backend == TracingBackend.TRUTHOUND

    def test_get_tracer(self):
        """Test getting tracer through adapter."""
        from truthound.observability.tracing.otel.adapter import (
            TracerProviderAdapter,
            TracingBackend,
        )
        from truthound.observability.tracing.provider import TracerProvider

        provider = TracerProvider()
        adapter = TracerProviderAdapter(provider, TracingBackend.TRUTHOUND)

        tracer = adapter.get_tracer("test-tracer", "1.0.0")
        assert tracer is not None
        assert tracer.name == "test-tracer"

    def test_tracer_caching(self):
        """Test that tracers are cached."""
        from truthound.observability.tracing.otel.adapter import (
            TracerProviderAdapter,
            TracingBackend,
        )
        from truthound.observability.tracing.provider import TracerProvider

        provider = TracerProvider()
        adapter = TracerProviderAdapter(provider, TracingBackend.TRUTHOUND)

        tracer1 = adapter.get_tracer("test-tracer")
        tracer2 = adapter.get_tracer("test-tracer")

        assert tracer1 is tracer2


# =============================================================================
# Global Configuration Tests
# =============================================================================


class TestGlobalConfiguration:
    """Tests for global configuration functions."""

    def test_configure_default(self):
        """Test default configuration."""
        from truthound.observability.tracing.otel.adapter import (
            configure,
            get_tracer_provider,
            reset_global_adapter,
        )

        reset_global_adapter()
        provider = configure()

        assert provider is not None
        assert get_tracer_provider() is provider

        reset_global_adapter()

    def test_configure_with_service_name(self):
        """Test configuration with service name."""
        from truthound.observability.tracing.otel.adapter import (
            configure,
            reset_global_adapter,
        )

        reset_global_adapter()
        provider = configure(service_name="test-service")

        assert provider is not None

        reset_global_adapter()

    def test_get_current_backend(self):
        """Test getting current backend."""
        from truthound.observability.tracing.otel.adapter import (
            configure,
            get_current_backend,
            TracingBackend,
            reset_global_adapter,
        )

        reset_global_adapter()
        configure(backend=TracingBackend.TRUTHOUND)

        backend = get_current_backend()
        assert backend == TracingBackend.TRUTHOUND

        reset_global_adapter()

    def test_get_tracer_convenience(self):
        """Test get_tracer convenience function."""
        from truthound.observability.tracing.otel.adapter import (
            get_tracer,
            reset_global_adapter,
        )

        reset_global_adapter()
        tracer = get_tracer("test-tracer")

        assert tracer is not None

        reset_global_adapter()


# =============================================================================
# Span Context Bridge Tests
# =============================================================================


class TestSpanContextBridge:
    """Tests for SpanContextBridge."""

    def test_truthound_to_truthound(self):
        """Test Truthound context conversion roundtrip."""
        from truthound.observability.tracing.otel.bridge import SpanContextBridge
        from truthound.observability.tracing.span import SpanContextData

        bridge = SpanContextBridge()

        original = SpanContextData(
            trace_id="0123456789abcdef0123456789abcdef",
            span_id="0123456789abcdef",
            trace_flags=1,
        )

        # Convert to OTEL format and back (if OTEL available)
        from truthound.observability.tracing.otel.detection import is_otel_sdk_available

        if is_otel_sdk_available():
            otel_ctx = bridge.truthound_to_otel(original)
            roundtrip = bridge.otel_to_truthound(otel_ctx)

            assert roundtrip.trace_id == original.trace_id
            assert roundtrip.span_id == original.span_id

    def test_otel_to_truthound(self):
        """Test OTEL to Truthound conversion."""
        from truthound.observability.tracing.otel.detection import is_otel_sdk_available

        if not is_otel_sdk_available():
            pytest.skip("OpenTelemetry SDK not available")

        from truthound.observability.tracing.otel.bridge import SpanContextBridge
        from opentelemetry.trace import SpanContext

        bridge = SpanContextBridge()

        otel_ctx = SpanContext(
            trace_id=0x0123456789ABCDEF0123456789ABCDEF,
            span_id=0x0123456789ABCDEF,
            is_remote=False,
            trace_flags=1,
        )

        truthound_ctx = bridge.otel_to_truthound(otel_ctx)
        assert truthound_ctx.trace_flags == 1


# =============================================================================
# Span Bridge Tests
# =============================================================================


class TestSpanBridge:
    """Tests for SpanBridge."""

    def test_truthound_to_otel_data(self):
        """Test converting Truthound span to OTEL data."""
        from truthound.observability.tracing.otel.bridge import SpanBridge
        from truthound.observability.tracing.span import (
            Span,
            SpanContextData,
            SpanKind,
        )

        bridge = SpanBridge()

        context = SpanContextData(
            trace_id="0" * 32,
            span_id="0" * 16,
            trace_flags=1,
        )
        span = Span(
            name="test-span",
            context=context,
            kind=SpanKind.SERVER,
            attributes={"key": "value"},
        )
        span.end()

        data = bridge.truthound_to_otel_data(span)

        assert data["name"] == "test-span"
        assert data["kind"] == 1  # SERVER
        assert data["attributes"]["key"] == "value"

    def test_otel_to_truthound_data(self):
        """Test converting OTEL data to Truthound format."""
        from truthound.observability.tracing.otel.bridge import SpanBridge

        bridge = SpanBridge()

        otel_data = {
            "name": "test-span",
            "kind": 1,  # SERVER
            "trace_id": "0" * 32,
            "span_id": "0" * 16,
            "start_time_ns": 1000000000,
            "end_time_ns": 2000000000,
            "attributes": {"key": "value"},
            "status": {"code": 1, "message": ""},
        }

        truthound_data = bridge.otel_to_truthound_data(otel_data)

        assert truthound_data["name"] == "test-span"
        assert truthound_data["kind"] == "SERVER"
        assert truthound_data["status_code"] == "OK"


# =============================================================================
# Compatibility Wrapper Tests
# =============================================================================


class TestOTELSpanWrapper:
    """Tests for OTELSpanWrapper."""

    def test_wrap_truthound_span(self):
        """Test wrapping Truthound span for OTEL compatibility."""
        from truthound.observability.tracing.otel.compat import OTELSpanWrapper
        from truthound.observability.tracing.span import Span, SpanContextData

        context = SpanContextData(
            trace_id="0" * 32,
            span_id="0" * 16,
            trace_flags=1,
        )
        span = Span(name="test", context=context)

        wrapper = OTELSpanWrapper(span)

        assert wrapper.name == "test"
        assert wrapper.is_recording() is True

    def test_get_span_context(self):
        """Test getting span context from wrapper."""
        from truthound.observability.tracing.otel.detection import is_otel_sdk_available

        if not is_otel_sdk_available():
            pytest.skip("OpenTelemetry SDK not available")

        from truthound.observability.tracing.otel.compat import OTELSpanWrapper
        from truthound.observability.tracing.span import Span, SpanContextData

        context = SpanContextData(
            trace_id="0123456789abcdef0123456789abcdef",
            span_id="0123456789abcdef",
            trace_flags=1,
        )
        span = Span(name="test", context=context)
        wrapper = OTELSpanWrapper(span)

        otel_ctx = wrapper.get_span_context()
        assert otel_ctx is not None
        assert otel_ctx.is_valid

    def test_set_attribute(self):
        """Test setting attribute through wrapper."""
        from truthound.observability.tracing.otel.compat import OTELSpanWrapper
        from truthound.observability.tracing.span import Span, SpanContextData

        context = SpanContextData(trace_id="0" * 32, span_id="0" * 16)
        span = Span(name="test", context=context)
        wrapper = OTELSpanWrapper(span)

        wrapper.set_attribute("key", "value")
        assert span.attributes.get("key") == "value"

    def test_context_manager(self):
        """Test wrapper as context manager."""
        from truthound.observability.tracing.otel.compat import OTELSpanWrapper
        from truthound.observability.tracing.span import Span, SpanContextData

        context = SpanContextData(trace_id="0" * 32, span_id="0" * 16)
        span = Span(name="test", context=context)
        wrapper = OTELSpanWrapper(span)

        with wrapper as w:
            w.set_attribute("in_context", True)

        assert span.end_time is not None

    def test_unwrap(self):
        """Test unwrapping to get original span."""
        from truthound.observability.tracing.otel.compat import OTELSpanWrapper
        from truthound.observability.tracing.span import Span, SpanContextData

        context = SpanContextData(trace_id="0" * 32, span_id="0" * 16)
        span = Span(name="test", context=context)
        wrapper = OTELSpanWrapper(span)

        unwrapped = wrapper.unwrap()
        assert unwrapped is span


class TestTruthoundSpanWrapper:
    """Tests for TruthoundSpanWrapper."""

    def test_wrap_mock_otel_span(self):
        """Test wrapping mock OTEL span."""
        from truthound.observability.tracing.otel.compat import TruthoundSpanWrapper

        # Create mock OTEL span
        mock_span = Mock()
        mock_span.name = "test-span"
        mock_span.get_span_context.return_value = Mock(
            trace_id=0x0123456789ABCDEF0123456789ABCDEF,
            span_id=0x0123456789ABCDEF,
            trace_flags=1,
            is_remote=False,
            trace_state=None,
        )
        mock_span.is_recording.return_value = True
        mock_span.kind = Mock(name="INTERNAL")
        mock_span.start_time = 1000000000
        mock_span.end_time = None
        mock_span.attributes = {}
        mock_span.events = []
        mock_span.links = []
        mock_span.parent = None
        mock_span.status = None

        wrapper = TruthoundSpanWrapper(mock_span)

        assert wrapper.name == "test-span"
        assert wrapper.context is not None
        assert wrapper.is_recording() is True


# =============================================================================
# Configuration Tests
# =============================================================================


class TestUnifiedTracingConfig:
    """Tests for UnifiedTracingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from truthound.observability.tracing.otel.config import UnifiedTracingConfig

        config = UnifiedTracingConfig()

        assert config.service_name == "truthound"
        assert config.exporter == "console"
        assert config.sampler == "always_on"
        assert config.sampling_ratio == 1.0

    def test_development_config(self):
        """Test development configuration preset."""
        from truthound.observability.tracing.otel.config import UnifiedTracingConfig

        config = UnifiedTracingConfig.development("my-dev-service")

        assert config.service_name == "my-dev-service"
        assert config.environment == "development"
        assert config.exporter == "console"
        assert config.batch_export is False
        assert config.console_debug is True

    def test_production_config(self):
        """Test production configuration preset."""
        from truthound.observability.tracing.otel.config import UnifiedTracingConfig

        config = UnifiedTracingConfig.production(
            service_name="my-prod-service",
            endpoint="http://collector:4317",
            sampling_ratio=0.1,
        )

        assert config.service_name == "my-prod-service"
        assert config.environment == "production"
        assert config.exporter == "otlp"
        assert config.endpoint == "http://collector:4317"
        assert config.sampling_ratio == 0.1
        assert config.batch_export is True

    def test_to_adapter_config(self):
        """Test conversion to AdapterConfig."""
        from truthound.observability.tracing.otel.config import UnifiedTracingConfig

        config = UnifiedTracingConfig(
            service_name="test-service",
            exporter="otlp",
            endpoint="http://localhost:4317",
        )

        adapter_config = config.to_adapter_config()

        assert adapter_config.service_name == "test-service"
        assert adapter_config.exporter_type == "otlp"
        assert adapter_config.exporter_endpoint == "http://localhost:4317"

    def test_to_truthound_config(self):
        """Test conversion to Truthound TracingConfig."""
        from truthound.observability.tracing.otel.config import UnifiedTracingConfig

        config = UnifiedTracingConfig(
            service_name="test-service",
            exporter="console",
        )

        truthound_config = config.to_truthound_config()

        assert truthound_config.service_name == "test-service"
        assert truthound_config.exporter == "console"


class TestSetupTracing:
    """Tests for setup_tracing function."""

    def test_setup_default(self):
        """Test default setup."""
        from truthound.observability.tracing.otel.config import setup_tracing
        from truthound.observability.tracing.otel.adapter import reset_global_adapter

        reset_global_adapter()
        provider = setup_tracing(service_name="test-service")

        assert provider is not None

        reset_global_adapter()

    def test_setup_with_config(self):
        """Test setup with configuration object."""
        from truthound.observability.tracing.otel.config import (
            setup_tracing,
            UnifiedTracingConfig,
        )
        from truthound.observability.tracing.otel.adapter import reset_global_adapter

        reset_global_adapter()

        config = UnifiedTracingConfig.testing("test-service")
        provider = setup_tracing(config)

        assert provider is not None

        reset_global_adapter()


class TestTracingStatus:
    """Tests for tracing status and diagnostics."""

    def test_get_tracing_status(self):
        """Test getting tracing status."""
        from truthound.observability.tracing.otel.config import get_tracing_status
        from truthound.observability.tracing.otel.adapter import reset_global_adapter

        reset_global_adapter()

        status = get_tracing_status()

        assert isinstance(status, dict)
        assert "otel_api_available" in status
        assert "otel_sdk_available" in status
        assert "current_backend" in status

    def test_diagnose_tracing(self):
        """Test tracing diagnostics."""
        from truthound.observability.tracing.otel.config import diagnose_tracing
        from truthound.observability.tracing.otel.adapter import reset_global_adapter

        reset_global_adapter()

        report = diagnose_tracing()

        assert isinstance(report, str)
        assert "Truthound Tracing Diagnostics" in report
        assert "OpenTelemetry Availability" in report


# =============================================================================
# Integration Tests
# =============================================================================


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_full_tracing_workflow(self):
        """Test complete tracing workflow with adapter."""
        from truthound.observability.tracing.otel import (
            setup_tracing,
            UnifiedTracingConfig,
            get_tracer,
        )
        from truthound.observability.tracing.otel.adapter import reset_global_adapter

        reset_global_adapter()

        # Setup tracing
        config = UnifiedTracingConfig.testing("integration-test")
        provider = setup_tracing(config)

        # Get tracer
        tracer = provider.get_tracer("test-component")

        # Create spans
        with tracer.start_as_current_span("parent-operation") as parent:
            parent.set_attribute("test.type", "integration")

            with tracer.start_as_current_span("child-operation") as child:
                child.set_attribute("child.key", "child.value")
                child.add_event("processing", {"step": 1})

        # Verify spans were created properly
        unwrapped_parent = parent.unwrap() if hasattr(parent, "unwrap") else parent._span
        assert unwrapped_parent.end_time is not None

        reset_global_adapter()

    def test_exception_handling(self):
        """Test exception recording in spans."""
        from truthound.observability.tracing.otel import setup_tracing
        from truthound.observability.tracing.otel.adapter import reset_global_adapter

        reset_global_adapter()

        provider = setup_tracing(service_name="exception-test")
        tracer = provider.get_tracer("test")

        with pytest.raises(ValueError):
            with tracer.start_as_current_span("failing-operation") as span:
                raise ValueError("Test error")

        # Span should have recorded the exception
        unwrapped = span.unwrap() if hasattr(span, "unwrap") else span._span
        assert len(unwrapped.events) > 0

        reset_global_adapter()

    def test_attribute_types(self):
        """Test various attribute types."""
        from truthound.observability.tracing.otel import setup_tracing
        from truthound.observability.tracing.otel.adapter import reset_global_adapter

        reset_global_adapter()

        provider = setup_tracing(service_name="attr-test")
        tracer = provider.get_tracer("test")

        with tracer.start_as_current_span("attr-operation") as span:
            span.set_attribute("string_attr", "value")
            span.set_attribute("int_attr", 42)
            span.set_attribute("float_attr", 3.14)
            span.set_attribute("bool_attr", True)

        unwrapped = span.unwrap() if hasattr(span, "unwrap") else span._span
        assert unwrapped.attributes.get("string_attr") == "value"
        assert unwrapped.attributes.get("int_attr") == 42
        assert unwrapped.attributes.get("float_attr") == 3.14
        assert unwrapped.attributes.get("bool_attr") is True

        reset_global_adapter()


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safety of the adapter layer."""

    def test_concurrent_span_creation(self):
        """Test creating spans from multiple threads."""
        from truthound.observability.tracing.otel import setup_tracing
        from truthound.observability.tracing.otel.adapter import reset_global_adapter

        reset_global_adapter()

        provider = setup_tracing(service_name="thread-test")
        tracer = provider.get_tracer("test")

        results = []
        errors = []

        def create_span(thread_id: int):
            try:
                with tracer.start_as_current_span(f"thread-{thread_id}") as span:
                    span.set_attribute("thread_id", thread_id)
                    time.sleep(0.01)
                results.append(thread_id)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=create_span, args=(i,))
            for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10

        reset_global_adapter()

    def test_concurrent_tracer_access(self):
        """Test accessing tracers from multiple threads."""
        from truthound.observability.tracing.otel import setup_tracing
        from truthound.observability.tracing.otel.adapter import reset_global_adapter

        reset_global_adapter()

        provider = setup_tracing(service_name="thread-test")

        results = []
        errors = []

        def get_tracer(thread_id: int):
            try:
                tracer = provider.get_tracer(f"tracer-{thread_id}")
                results.append(tracer)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=get_tracer, args=(i,))
            for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10

        reset_global_adapter()
