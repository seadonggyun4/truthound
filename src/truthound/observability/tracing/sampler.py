"""Sampling strategies for distributed tracing.

Samplers decide whether a span should be recorded and exported.
This helps control the volume of trace data while still providing
meaningful visibility into system behavior.

Sampling Strategies:
    - AlwaysOnSampler: Record all spans (development/testing)
    - AlwaysOffSampler: Record no spans (disable tracing)
    - TraceIdRatioSampler: Sample based on trace ID hash
    - ParentBasedSampler: Respect parent's sampling decision
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Mapping, Sequence

from truthound.observability.tracing.span import SpanContextData, SpanKind, Link


# =============================================================================
# Sampling Decision
# =============================================================================


class SamplingDecision(Enum):
    """Sampling decision for a span."""

    DROP = auto()  # Don't record or export
    RECORD_ONLY = auto()  # Record but don't export
    RECORD_AND_SAMPLE = auto()  # Record and export


@dataclass
class SamplingResult:
    """Result of a sampling decision.

    Contains the decision and any additional attributes to add to the span.
    """

    decision: SamplingDecision
    attributes: dict[str, Any] = field(default_factory=dict)
    trace_state: str = ""

    @property
    def is_sampled(self) -> bool:
        """Check if span should be sampled (exported)."""
        return self.decision == SamplingDecision.RECORD_AND_SAMPLE

    @property
    def is_recording(self) -> bool:
        """Check if span should be recorded."""
        return self.decision in (
            SamplingDecision.RECORD_ONLY,
            SamplingDecision.RECORD_AND_SAMPLE,
        )


# =============================================================================
# Sampler Interface
# =============================================================================


class Sampler(ABC):
    """Abstract base class for samplers.

    Samplers make the decision whether to record and export a span.
    """

    @abstractmethod
    def should_sample(
        self,
        parent_context: SpanContextData | None,
        trace_id: str,
        name: str,
        kind: SpanKind,
        attributes: Mapping[str, Any] | None = None,
        links: Sequence[Link] | None = None,
    ) -> SamplingResult:
        """Make a sampling decision.

        Args:
            parent_context: Parent span context (None for root spans).
            trace_id: Trace ID being sampled.
            name: Span name.
            kind: Span kind.
            attributes: Initial span attributes.
            links: Span links.

        Returns:
            SamplingResult with the decision.
        """
        pass

    @abstractmethod
    def description(self) -> str:
        """Get human-readable sampler description."""
        pass


# =============================================================================
# Always On Sampler
# =============================================================================


class AlwaysOnSampler(Sampler):
    """Sampler that always records and exports spans.

    Use for:
        - Development and testing
        - Low-volume services
        - When you need complete visibility

    Example:
        >>> sampler = AlwaysOnSampler()
        >>> provider = TracerProvider(sampler=sampler)
    """

    def should_sample(
        self,
        parent_context: SpanContextData | None,
        trace_id: str,
        name: str,
        kind: SpanKind,
        attributes: Mapping[str, Any] | None = None,
        links: Sequence[Link] | None = None,
    ) -> SamplingResult:
        """Always return RECORD_AND_SAMPLE."""
        return SamplingResult(decision=SamplingDecision.RECORD_AND_SAMPLE)

    def description(self) -> str:
        return "AlwaysOnSampler"


# =============================================================================
# Always Off Sampler
# =============================================================================


class AlwaysOffSampler(Sampler):
    """Sampler that never records spans.

    Use for:
        - Completely disabling tracing
        - Production emergency (too much overhead)

    Example:
        >>> sampler = AlwaysOffSampler()
        >>> provider = TracerProvider(sampler=sampler)
    """

    def should_sample(
        self,
        parent_context: SpanContextData | None,
        trace_id: str,
        name: str,
        kind: SpanKind,
        attributes: Mapping[str, Any] | None = None,
        links: Sequence[Link] | None = None,
    ) -> SamplingResult:
        """Always return DROP."""
        return SamplingResult(decision=SamplingDecision.DROP)

    def description(self) -> str:
        return "AlwaysOffSampler"


# =============================================================================
# Trace ID Ratio Sampler
# =============================================================================


class TraceIdRatioSampler(Sampler):
    """Sampler that samples based on trace ID.

    Samples a configurable percentage of traces based on the trace ID.
    This ensures consistent sampling across all spans in a trace.

    Args:
        ratio: Probability of sampling (0.0 to 1.0).
               0.0 = never sample, 1.0 = always sample.

    Example:
        >>> # Sample 10% of traces
        >>> sampler = TraceIdRatioSampler(0.1)
        >>> provider = TracerProvider(sampler=sampler)
    """

    def __init__(self, ratio: float = 1.0) -> None:
        """Initialize with sampling ratio.

        Args:
            ratio: Sampling ratio between 0.0 and 1.0.

        Raises:
            ValueError: If ratio is not between 0.0 and 1.0.
        """
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"Ratio must be between 0.0 and 1.0, got {ratio}")

        self._ratio = ratio
        # Convert ratio to threshold for comparison
        # trace_id is 128-bit hex, we use upper 64 bits for comparison
        self._threshold = int(ratio * (2**64 - 1))

    @property
    def ratio(self) -> float:
        """Get sampling ratio."""
        return self._ratio

    def _should_sample_trace_id(self, trace_id: str) -> bool:
        """Check if trace ID should be sampled.

        Uses deterministic hash of trace ID for consistent sampling.
        """
        try:
            # Use first 16 hex chars (64 bits) for comparison
            trace_id_upper = int(trace_id[:16], 16)
            return trace_id_upper < self._threshold
        except (ValueError, IndexError):
            # Invalid trace ID, sample it
            return True

    def should_sample(
        self,
        parent_context: SpanContextData | None,
        trace_id: str,
        name: str,
        kind: SpanKind,
        attributes: Mapping[str, Any] | None = None,
        links: Sequence[Link] | None = None,
    ) -> SamplingResult:
        """Sample based on trace ID ratio."""
        if self._should_sample_trace_id(trace_id):
            return SamplingResult(decision=SamplingDecision.RECORD_AND_SAMPLE)
        return SamplingResult(decision=SamplingDecision.DROP)

    def description(self) -> str:
        return f"TraceIdRatioSampler({self._ratio})"


# =============================================================================
# Parent-Based Sampler
# =============================================================================


class ParentBasedSampler(Sampler):
    """Sampler that respects parent's sampling decision.

    This sampler uses different delegated samplers based on the parent context:
    - Root spans (no parent): Use root sampler
    - Remote parent sampled: Use remote_parent_sampled sampler
    - Remote parent not sampled: Use remote_parent_not_sampled sampler
    - Local parent sampled: Use local_parent_sampled sampler
    - Local parent not sampled: Use local_parent_not_sampled sampler

    Example:
        >>> # Sample 10% of root spans, respect parent decisions
        >>> sampler = ParentBasedSampler(
        ...     root=TraceIdRatioSampler(0.1),
        ... )
        >>> provider = TracerProvider(sampler=sampler)
    """

    def __init__(
        self,
        root: Sampler | None = None,
        *,
        remote_parent_sampled: Sampler | None = None,
        remote_parent_not_sampled: Sampler | None = None,
        local_parent_sampled: Sampler | None = None,
        local_parent_not_sampled: Sampler | None = None,
    ) -> None:
        """Initialize parent-based sampler.

        Args:
            root: Sampler for root spans (default: AlwaysOnSampler).
            remote_parent_sampled: Sampler when remote parent is sampled
                                   (default: AlwaysOnSampler).
            remote_parent_not_sampled: Sampler when remote parent is not sampled
                                       (default: AlwaysOffSampler).
            local_parent_sampled: Sampler when local parent is sampled
                                  (default: AlwaysOnSampler).
            local_parent_not_sampled: Sampler when local parent is not sampled
                                      (default: AlwaysOffSampler).
        """
        self._root = root or AlwaysOnSampler()
        self._remote_parent_sampled = remote_parent_sampled or AlwaysOnSampler()
        self._remote_parent_not_sampled = remote_parent_not_sampled or AlwaysOffSampler()
        self._local_parent_sampled = local_parent_sampled or AlwaysOnSampler()
        self._local_parent_not_sampled = local_parent_not_sampled or AlwaysOffSampler()

    def should_sample(
        self,
        parent_context: SpanContextData | None,
        trace_id: str,
        name: str,
        kind: SpanKind,
        attributes: Mapping[str, Any] | None = None,
        links: Sequence[Link] | None = None,
    ) -> SamplingResult:
        """Sample based on parent context."""
        # No parent: use root sampler
        if parent_context is None:
            return self._root.should_sample(
                parent_context, trace_id, name, kind, attributes, links
            )

        # Determine which sampler to use based on parent
        if parent_context.is_remote:
            if parent_context.is_sampled:
                sampler = self._remote_parent_sampled
            else:
                sampler = self._remote_parent_not_sampled
        else:
            if parent_context.is_sampled:
                sampler = self._local_parent_sampled
            else:
                sampler = self._local_parent_not_sampled

        return sampler.should_sample(
            parent_context, trace_id, name, kind, attributes, links
        )

    def description(self) -> str:
        return f"ParentBasedSampler(root={self._root.description()})"


# =============================================================================
# Attribute-Based Sampler
# =============================================================================


class AttributeBasedSampler(Sampler):
    """Sampler that makes decisions based on span attributes.

    Useful for:
        - Always sampling errors
        - Sampling high-priority operations
        - Filtering out noisy endpoints

    Example:
        >>> # Always sample errors, 10% for everything else
        >>> sampler = AttributeBasedSampler(
        ...     rules=[
        ...         ({"error": True}, AlwaysOnSampler()),
        ...         ({"http.target": "/health"}, AlwaysOffSampler()),
        ...     ],
        ...     default=TraceIdRatioSampler(0.1),
        ... )
    """

    def __init__(
        self,
        rules: list[tuple[dict[str, Any], Sampler]] | None = None,
        default: Sampler | None = None,
    ) -> None:
        """Initialize attribute-based sampler.

        Args:
            rules: List of (attribute_match, sampler) tuples.
            default: Default sampler if no rules match.
        """
        self._rules = rules or []
        self._default = default or AlwaysOnSampler()

    def _matches_attributes(
        self,
        pattern: dict[str, Any],
        attributes: Mapping[str, Any] | None,
    ) -> bool:
        """Check if attributes match pattern."""
        if not attributes:
            return not pattern

        for key, value in pattern.items():
            if key not in attributes:
                return False
            if attributes[key] != value:
                return False

        return True

    def should_sample(
        self,
        parent_context: SpanContextData | None,
        trace_id: str,
        name: str,
        kind: SpanKind,
        attributes: Mapping[str, Any] | None = None,
        links: Sequence[Link] | None = None,
    ) -> SamplingResult:
        """Sample based on attributes."""
        for pattern, sampler in self._rules:
            if self._matches_attributes(pattern, attributes):
                return sampler.should_sample(
                    parent_context, trace_id, name, kind, attributes, links
                )

        return self._default.should_sample(
            parent_context, trace_id, name, kind, attributes, links
        )

    def description(self) -> str:
        return f"AttributeBasedSampler(rules={len(self._rules)})"


# =============================================================================
# Rate Limiting Sampler
# =============================================================================


class RateLimitingSampler(Sampler):
    """Sampler that limits the rate of sampled spans.

    Uses a token bucket algorithm to limit sampling rate.
    Useful for protecting backends from traffic spikes.

    Example:
        >>> # Max 100 spans per second
        >>> sampler = RateLimitingSampler(rate_limit=100)
    """

    def __init__(
        self,
        rate_limit: float = 100.0,
        *,
        burst_size: int | None = None,
        fallback: Sampler | None = None,
    ) -> None:
        """Initialize rate limiting sampler.

        Args:
            rate_limit: Maximum spans per second.
            burst_size: Maximum burst size (default: rate_limit).
            fallback: Sampler to use when rate limit exceeded.
        """
        import threading
        import time

        self._rate_limit = rate_limit
        self._burst_size = burst_size or int(rate_limit)
        self._fallback = fallback or AlwaysOffSampler()

        self._tokens = float(self._burst_size)
        self._last_update = time.time()
        self._lock = threading.Lock()

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        import time

        now = time.time()
        elapsed = now - self._last_update
        self._tokens = min(
            self._burst_size,
            self._tokens + elapsed * self._rate_limit,
        )
        self._last_update = now

    def should_sample(
        self,
        parent_context: SpanContextData | None,
        trace_id: str,
        name: str,
        kind: SpanKind,
        attributes: Mapping[str, Any] | None = None,
        links: Sequence[Link] | None = None,
    ) -> SamplingResult:
        """Sample with rate limiting."""
        with self._lock:
            self._refill_tokens()

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return SamplingResult(decision=SamplingDecision.RECORD_AND_SAMPLE)

        # Rate limit exceeded, use fallback
        return self._fallback.should_sample(
            parent_context, trace_id, name, kind, attributes, links
        )

    def description(self) -> str:
        return f"RateLimitingSampler(rate={self._rate_limit}/s)"
