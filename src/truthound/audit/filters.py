"""Filters and processors for audit events.

This module provides filtering and processing capabilities:
- Filters: Determine which events should be logged
- Processors: Transform/enrich events before storage
- Privacy: Mask sensitive data
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Pattern

from truthound.audit.core import (
    AuditEvent,
    AuditEventType,
    AuditSeverity,
    AuditCategory,
    AuditOutcome,
    AuditFilter,
    AuditProcessor,
    AuditConfig,
    mask_sensitive_value,
    anonymize_ip_address,
)


# =============================================================================
# Filters
# =============================================================================


class SeverityFilter(AuditFilter):
    """Filter events by minimum severity level.

    Example:
        >>> filter = SeverityFilter(min_severity=AuditSeverity.WARNING)
        >>> filter.should_log(event)  # Only WARNING and above
    """

    SEVERITY_ORDER = {
        AuditSeverity.DEBUG: 0,
        AuditSeverity.INFO: 1,
        AuditSeverity.WARNING: 2,
        AuditSeverity.ERROR: 3,
        AuditSeverity.CRITICAL: 4,
    }

    def __init__(self, min_severity: AuditSeverity = AuditSeverity.INFO) -> None:
        self._min_severity = min_severity
        self._min_level = self.SEVERITY_ORDER.get(min_severity, 1)

    def should_log(self, event: AuditEvent) -> bool:
        event_level = self.SEVERITY_ORDER.get(event.severity, 1)
        return event_level >= self._min_level


class EventTypeFilter(AuditFilter):
    """Filter events by event type.

    Example:
        >>> filter = EventTypeFilter(
        ...     include=[AuditEventType.CREATE, AuditEventType.UPDATE],
        ... )
    """

    def __init__(
        self,
        include: list[AuditEventType] | None = None,
        exclude: list[AuditEventType] | None = None,
    ) -> None:
        self._include = set(include) if include else None
        self._exclude = set(exclude) if exclude else set()

    def should_log(self, event: AuditEvent) -> bool:
        if event.event_type in self._exclude:
            return False
        if self._include is not None:
            return event.event_type in self._include
        return True


class CategoryFilter(AuditFilter):
    """Filter events by category.

    Example:
        >>> filter = CategoryFilter(
        ...     include=[AuditCategory.SECURITY, AuditCategory.AUTHENTICATION],
        ... )
    """

    def __init__(
        self,
        include: list[AuditCategory] | None = None,
        exclude: list[AuditCategory] | None = None,
    ) -> None:
        self._include = set(include) if include else None
        self._exclude = set(exclude) if exclude else set()

    def should_log(self, event: AuditEvent) -> bool:
        if event.category in self._exclude:
            return False
        if self._include is not None:
            return event.category in self._include
        return True


class ActorFilter(AuditFilter):
    """Filter events by actor.

    Example:
        >>> filter = ActorFilter(
        ...     exclude_ids=["system", "healthcheck"],
        ...     include_types=["user"],
        ... )
    """

    def __init__(
        self,
        include_ids: list[str] | None = None,
        exclude_ids: list[str] | None = None,
        include_types: list[str] | None = None,
        exclude_types: list[str] | None = None,
    ) -> None:
        self._include_ids = set(include_ids) if include_ids else None
        self._exclude_ids = set(exclude_ids) if exclude_ids else set()
        self._include_types = set(include_types) if include_types else None
        self._exclude_types = set(exclude_types) if exclude_types else set()

    def should_log(self, event: AuditEvent) -> bool:
        if not event.actor:
            return True

        # Check ID filters
        if event.actor.id in self._exclude_ids:
            return False
        if self._include_ids is not None and event.actor.id not in self._include_ids:
            return False

        # Check type filters
        if event.actor.type in self._exclude_types:
            return False
        if self._include_types is not None and event.actor.type not in self._include_types:
            return False

        return True


class ActionFilter(AuditFilter):
    """Filter events by action name.

    Supports glob-style patterns.

    Example:
        >>> filter = ActionFilter(
        ...     exclude_patterns=["healthcheck*", "internal_*"],
        ... )
    """

    def __init__(
        self,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> None:
        self._include_patterns = [
            self._compile_pattern(p) for p in (include_patterns or [])
        ]
        self._exclude_patterns = [
            self._compile_pattern(p) for p in (exclude_patterns or [])
        ]

    def should_log(self, event: AuditEvent) -> bool:
        action = event.action or ""

        # Check excludes
        for pattern in self._exclude_patterns:
            if pattern.match(action):
                return False

        # Check includes
        if self._include_patterns:
            for pattern in self._include_patterns:
                if pattern.match(action):
                    return True
            return False

        return True

    def _compile_pattern(self, pattern: str) -> Pattern[str]:
        """Convert glob pattern to regex."""
        regex = pattern.replace("*", ".*").replace("?", ".")
        return re.compile(f"^{regex}$")


class OutcomeFilter(AuditFilter):
    """Filter events by outcome.

    Example:
        >>> filter = OutcomeFilter(include=[AuditOutcome.FAILURE])
    """

    def __init__(
        self,
        include: list[AuditOutcome] | None = None,
        exclude: list[AuditOutcome] | None = None,
    ) -> None:
        self._include = set(include) if include else None
        self._exclude = set(exclude) if exclude else set()

    def should_log(self, event: AuditEvent) -> bool:
        if event.outcome in self._exclude:
            return False
        if self._include is not None:
            return event.outcome in self._include
        return True


class SamplingFilter(AuditFilter):
    """Sample events at a specified rate.

    Example:
        >>> filter = SamplingFilter(rate=0.1)  # Log 10% of events
    """

    def __init__(self, rate: float = 1.0) -> None:
        """Initialize sampling filter.

        Args:
            rate: Sampling rate (0.0 to 1.0).
        """
        import random

        self._rate = max(0.0, min(1.0, rate))
        self._random = random.Random()

    def should_log(self, event: AuditEvent) -> bool:
        if self._rate >= 1.0:
            return True
        if self._rate <= 0.0:
            return False
        return self._random.random() < self._rate


class RateLimitFilter(AuditFilter):
    """Rate limit events per key.

    Example:
        >>> filter = RateLimitFilter(
        ...     events_per_second=10,
        ...     key_func=lambda e: e.actor.id if e.actor else "global",
        ... )
    """

    def __init__(
        self,
        events_per_second: float = 100.0,
        key_func: Callable[[AuditEvent], str] | None = None,
    ) -> None:
        import time

        self._rate = events_per_second
        self._key_func = key_func or (lambda e: "global")
        self._last_event: dict[str, float] = {}
        self._min_interval = 1.0 / events_per_second if events_per_second > 0 else 0

    def should_log(self, event: AuditEvent) -> bool:
        import time

        if self._rate <= 0:
            return False

        key = self._key_func(event)
        now = time.time()
        last = self._last_event.get(key, 0)

        if now - last >= self._min_interval:
            self._last_event[key] = now
            return True
        return False


class CompositeFilter(AuditFilter):
    """Combine multiple filters with AND/OR logic.

    Example:
        >>> filter = CompositeFilter(
        ...     filters=[severity_filter, type_filter],
        ...     mode="all",  # AND logic
        ... )
    """

    def __init__(
        self,
        filters: list[AuditFilter],
        mode: str = "all",  # "all" (AND) or "any" (OR)
    ) -> None:
        self._filters = filters
        self._mode = mode

    def should_log(self, event: AuditEvent) -> bool:
        if self._mode == "all":
            return all(f.should_log(event) for f in self._filters)
        else:  # "any"
            return any(f.should_log(event) for f in self._filters)


class CallableFilter(AuditFilter):
    """Filter using a custom callable.

    Example:
        >>> filter = CallableFilter(
        ...     lambda e: e.actor and e.actor.type == "user"
        ... )
    """

    def __init__(self, func: Callable[[AuditEvent], bool]) -> None:
        self._func = func

    def should_log(self, event: AuditEvent) -> bool:
        return self._func(event)


# =============================================================================
# Processors
# =============================================================================


class PrivacyProcessor(AuditProcessor):
    """Process events to protect sensitive data.

    Example:
        >>> processor = PrivacyProcessor(
        ...     mask_fields=["password", "token", "api_key"],
        ...     anonymize_ip=True,
        ... )
    """

    DEFAULT_SENSITIVE_FIELDS = {
        "password",
        "passwd",
        "secret",
        "token",
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "credit_card",
        "card_number",
        "cvv",
        "ssn",
        "social_security",
    }

    def __init__(
        self,
        mask_fields: list[str] | None = None,
        anonymize_ip: bool = False,
        mask_char: str = "*",
    ) -> None:
        self._mask_fields = set(mask_fields) if mask_fields else self.DEFAULT_SENSITIVE_FIELDS
        self._anonymize_ip = anonymize_ip
        self._mask_char = mask_char

    def process(self, event: AuditEvent) -> AuditEvent:
        # Mask sensitive data
        if event.data:
            event.data = self._mask_dict(event.data)

        if event.old_value:
            event.old_value = self._mask_value(event.old_value)

        if event.new_value:
            event.new_value = self._mask_value(event.new_value)

        if event.changes:
            event.changes = self._mask_dict(event.changes)

        # Anonymize IP
        if self._anonymize_ip and event.actor and event.actor.ip_address:
            event.actor.ip_address = anonymize_ip_address(event.actor.ip_address)

        return event

    def _mask_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Mask sensitive fields in dictionary."""
        result = {}
        for key, value in data.items():
            lower_key = key.lower()
            if any(field in lower_key for field in self._mask_fields):
                result[key] = mask_sensitive_value(value, self._mask_char)
            elif isinstance(value, dict):
                result[key] = self._mask_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self._mask_dict(v) if isinstance(v, dict) else v
                    for v in value
                ]
            else:
                result[key] = value
        return result

    def _mask_value(self, value: Any) -> Any:
        """Mask a value if it's a dict or contains sensitive data."""
        if isinstance(value, dict):
            return self._mask_dict(value)
        return value


class EnrichmentProcessor(AuditProcessor):
    """Enrich events with additional context.

    Example:
        >>> processor = EnrichmentProcessor(
        ...     add_hostname=True,
        ...     add_environment=True,
        ...     custom_enrichers={"region": lambda e: "us-west-2"},
        ... )
    """

    def __init__(
        self,
        add_hostname: bool = False,
        add_environment: bool = False,
        add_service_info: bool = False,
        service_name: str = "",
        service_version: str = "",
        environment: str = "",
        custom_enrichers: dict[str, Callable[[AuditEvent], Any]] | None = None,
    ) -> None:
        self._add_hostname = add_hostname
        self._add_environment = add_environment
        self._add_service_info = add_service_info
        self._service_name = service_name
        self._service_version = service_version
        self._environment = environment
        self._custom_enrichers = custom_enrichers or {}

        # Cache hostname
        self._hostname: str | None = None

    def process(self, event: AuditEvent) -> AuditEvent:
        # Add hostname
        if self._add_hostname:
            if self._hostname is None:
                import socket
                try:
                    self._hostname = socket.gethostname()
                except Exception:
                    self._hostname = "unknown"
            event.context.host = self._hostname

        # Add environment
        if self._add_environment and self._environment:
            event.context.environment = self._environment

        # Add service info
        if self._add_service_info:
            if self._service_name:
                event.context.service_name = self._service_name
            if self._service_version:
                event.context.service_version = self._service_version

        # Apply custom enrichers
        for key, enricher in self._custom_enrichers.items():
            try:
                value = enricher(event)
                event.data[key] = value
            except Exception:
                pass

        return event


class ChecksumProcessor(AuditProcessor):
    """Add integrity checksums to events.

    Example:
        >>> processor = ChecksumProcessor()
    """

    def process(self, event: AuditEvent) -> AuditEvent:
        event.checksum = event.compute_checksum()
        return event


class TaggingProcessor(AuditProcessor):
    """Add tags to events based on rules.

    Example:
        >>> processor = TaggingProcessor(
        ...     rules=[
        ...         (lambda e: e.severity == AuditSeverity.CRITICAL, "critical"),
        ...         (lambda e: e.category == AuditCategory.SECURITY, "security"),
        ...     ]
        ... )
    """

    def __init__(
        self,
        rules: list[tuple[Callable[[AuditEvent], bool], str]] | None = None,
        static_tags: list[str] | None = None,
    ) -> None:
        self._rules = rules or []
        self._static_tags = static_tags or []

    def process(self, event: AuditEvent) -> AuditEvent:
        # Add static tags
        for tag in self._static_tags:
            if tag not in event.tags:
                event.tags.append(tag)

        # Apply rules
        for condition, tag in self._rules:
            try:
                if condition(event) and tag not in event.tags:
                    event.tags.append(tag)
            except Exception:
                pass

        return event


class CompositeProcessor(AuditProcessor):
    """Chain multiple processors.

    Example:
        >>> processor = CompositeProcessor([
        ...     privacy_processor,
        ...     enrichment_processor,
        ...     checksum_processor,
        ... ])
    """

    def __init__(self, processors: list[AuditProcessor]) -> None:
        self._processors = processors

    def process(self, event: AuditEvent) -> AuditEvent:
        for processor in self._processors:
            event = processor.process(event)
        return event


# =============================================================================
# Filter Chain from Config
# =============================================================================


def create_filter_from_config(config: AuditConfig) -> AuditFilter:
    """Create filter chain from audit configuration.

    Args:
        config: Audit configuration.

    Returns:
        Composite filter based on config.
    """
    filters: list[AuditFilter] = []

    # Severity filter
    filters.append(SeverityFilter(min_severity=config.min_severity))

    # Exclude read events if configured
    if not config.include_read_events:
        filters.append(EventTypeFilter(exclude=[AuditEventType.READ]))

    # Exclude debug if configured
    if not config.include_debug_events:
        filters.append(SeverityFilter(min_severity=AuditSeverity.INFO))

    # Excluded actions
    if config.excluded_actions:
        filters.append(ActionFilter(exclude_patterns=config.excluded_actions))

    # Excluded actors
    if config.excluded_actors:
        filters.append(ActorFilter(exclude_ids=config.excluded_actors))

    return CompositeFilter(filters, mode="all")


def create_processor_from_config(config: AuditConfig) -> AuditProcessor:
    """Create processor chain from audit configuration.

    Args:
        config: Audit configuration.

    Returns:
        Composite processor based on config.
    """
    processors: list[AuditProcessor] = []

    # Privacy processor
    if config.mask_sensitive_data:
        processors.append(
            PrivacyProcessor(
                mask_fields=config.sensitive_fields,
                anonymize_ip=config.anonymize_ip,
            )
        )

    # Enrichment processor
    if config.service_name or config.environment:
        processors.append(
            EnrichmentProcessor(
                add_hostname=True,
                add_environment=True,
                add_service_info=True,
                service_name=config.service_name,
                service_version=config.service_version,
                environment=config.environment,
            )
        )

    # Checksum processor
    if config.compute_checksums:
        processors.append(ChecksumProcessor())

    return CompositeProcessor(processors)
