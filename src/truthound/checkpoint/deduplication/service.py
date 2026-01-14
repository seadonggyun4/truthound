"""Notification Deduplication Service.

This module provides the main deduplication service that
coordinates stores, processors, and policies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from truthound.checkpoint.deduplication.protocols import (
    DeduplicationRecord,
    DeduplicationResult,
    DeduplicationStats,
    DeduplicationStore,
    NotificationFingerprint,
    TimeWindow,
)
from truthound.checkpoint.deduplication.processor import (
    SlidingWindowStrategy,
    TimeWindowProcessor,
    WindowStrategy,
)
from truthound.checkpoint.deduplication.stores import InMemoryDeduplicationStore

if TYPE_CHECKING:
    from truthound.checkpoint.checkpoint import CheckpointResult

logger = logging.getLogger(__name__)


class DeduplicationPolicy(str, Enum):
    """Deduplication policies."""

    # No deduplication
    NONE = "none"

    # Deduplicate based on checkpoint + action type
    BASIC = "basic"

    # Deduplicate based on checkpoint + action + severity
    SEVERITY = "severity"

    # Deduplicate based on checkpoint + action + issues
    ISSUE_BASED = "issue_based"

    # Full deduplication including all components
    STRICT = "strict"

    # Custom fingerprint function
    CUSTOM = "custom"


@dataclass
class DeduplicationConfig:
    """Configuration for the deduplication service.

    Attributes:
        enabled: Whether deduplication is enabled.
        policy: Deduplication policy to use.
        default_window: Default time window for deduplication.
        action_windows: Per-action-type window overrides.
        severity_windows: Per-severity window overrides.
        max_suppressed_count: Maximum times to suppress before alerting.
        on_suppression_exceeded: Callback when max suppression exceeded.
        include_data_asset: Include data asset in fingerprint.
        include_metadata: Include metadata in fingerprint.
    """

    enabled: bool = True
    policy: DeduplicationPolicy = DeduplicationPolicy.SEVERITY
    default_window: TimeWindow = field(default_factory=lambda: TimeWindow(minutes=5))

    # Per-action overrides
    action_windows: dict[str, TimeWindow] = field(default_factory=dict)

    # Per-severity overrides (for SEVERITY policy)
    severity_windows: dict[str, TimeWindow] = field(
        default_factory=lambda: {
            "critical": TimeWindow(minutes=1),
            "high": TimeWindow(minutes=5),
            "medium": TimeWindow(minutes=15),
            "low": TimeWindow(hours=1),
            "info": TimeWindow(hours=1),
        }
    )

    # Suppression limits
    max_suppressed_count: int = 100
    on_suppression_exceeded: Callable[[DeduplicationRecord], None] | None = None

    # Fingerprint options
    include_data_asset: bool = True
    include_metadata: bool = False

    # Custom fingerprint function
    custom_fingerprint_fn: (
        Callable[[Any, str], NotificationFingerprint] | None
    ) = None

    def get_window_for_action(self, action_type: str) -> TimeWindow:
        """Get the time window for an action type."""
        return self.action_windows.get(action_type, self.default_window)

    def get_window_for_severity(self, severity: str) -> TimeWindow:
        """Get the time window for a severity level."""
        return self.severity_windows.get(severity.lower(), self.default_window)


@dataclass
class NotificationDeduplicator:
    """Main service for notification deduplication.

    Coordinates all deduplication components and provides a simple
    interface for checking and recording notifications.

    Attributes:
        store: Storage backend for deduplication records.
        config: Deduplication configuration.
        processor: Time window processor.

    Example:
        >>> deduplicator = NotificationDeduplicator(
        ...     store=InMemoryDeduplicationStore(),
        ...     config=DeduplicationConfig(
        ...         policy=DeduplicationPolicy.SEVERITY,
        ...         default_window=TimeWindow(minutes=5),
        ...     ),
        ... )
        >>>
        >>> # Check before sending
        >>> result = deduplicator.check(checkpoint_result, "slack")
        >>> if result.should_send:
        ...     send_notification()
        ...     deduplicator.mark_sent(result.fingerprint)
    """

    store: DeduplicationStore = field(default_factory=InMemoryDeduplicationStore)
    config: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    processor: TimeWindowProcessor = field(
        default_factory=lambda: TimeWindowProcessor(
            strategy=SlidingWindowStrategy(),
            default_window=TimeWindow(minutes=5),
        )
    )

    def __post_init__(self) -> None:
        """Update processor with config default window."""
        self.processor.default_window = self.config.default_window

    def check(
        self,
        checkpoint_result: CheckpointResult,
        action_type: str,
        *,
        severity: str | None = None,
        custom_components: dict[str, Any] | None = None,
    ) -> DeduplicationResult:
        """Check if a notification should be sent.

        Args:
            checkpoint_result: The checkpoint result.
            action_type: Type of notification action.
            severity: Override severity level.
            custom_components: Additional fingerprint components.

        Returns:
            DeduplicationResult indicating if notification is a duplicate.
        """
        if not self.config.enabled:
            # Deduplication disabled, always allow
            fingerprint = self._generate_fingerprint(
                checkpoint_result, action_type, severity, custom_components
            )
            return DeduplicationResult(
                is_duplicate=False,
                fingerprint=fingerprint,
                message="Deduplication disabled",
            )

        # Generate fingerprint
        fingerprint = self._generate_fingerprint(
            checkpoint_result, action_type, severity, custom_components
        )

        # Record the check for stats
        if hasattr(self.store, "record_check"):
            self.store.record_check()

        # Get effective window
        effective_window = self._get_effective_window(
            action_type, severity or self._extract_severity(checkpoint_result)
        )

        # Get deduplication key
        dedup_key = self.processor.get_dedup_key(fingerprint, effective_window)

        # Check for existing record
        existing = self.store.get(dedup_key)

        if existing is None:
            # No existing record, not a duplicate
            return DeduplicationResult(
                is_duplicate=False,
                fingerprint=fingerprint,
                message="No duplicate found",
            )

        # Check if within window
        if self.processor.is_duplicate(fingerprint, existing, effective_window):
            # Is a duplicate - increment count
            updated = self.store.increment(dedup_key)

            if updated:
                # Check if we've exceeded max suppression
                if updated.count > self.config.max_suppressed_count:
                    self._handle_suppression_exceeded(updated)

                return DeduplicationResult(
                    is_duplicate=True,
                    fingerprint=fingerprint,
                    original_record=updated,
                    suppressed_count=updated.count,
                    message=f"Duplicate suppressed (count: {updated.count})",
                )

        # Record expired or window passed
        return DeduplicationResult(
            is_duplicate=False,
            fingerprint=fingerprint,
            message="Previous record expired",
        )

    def mark_sent(
        self,
        fingerprint: NotificationFingerprint,
        window: TimeWindow | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DeduplicationRecord:
        """Mark a notification as sent.

        Args:
            fingerprint: The notification fingerprint.
            window: Time window for deduplication.
            metadata: Additional metadata to store.

        Returns:
            Created deduplication record.
        """
        effective_window = window or self._get_effective_window(
            fingerprint.action_type,
            fingerprint.components.get("severity", "medium"),
        )

        record = self.store.put(fingerprint, effective_window, metadata)

        logger.debug(
            f"Marked notification sent: {fingerprint.checkpoint_name}/"
            f"{fingerprint.action_type} (expires: {record.expires_at})"
        )

        return record

    def check_and_mark(
        self,
        checkpoint_result: CheckpointResult,
        action_type: str,
        *,
        severity: str | None = None,
        custom_components: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DeduplicationResult:
        """Check for duplicate and mark as sent if not duplicate.

        Convenience method that combines check() and mark_sent().

        Args:
            checkpoint_result: The checkpoint result.
            action_type: Type of notification action.
            severity: Override severity level.
            custom_components: Additional fingerprint components.
            metadata: Metadata to store with record.

        Returns:
            DeduplicationResult.
        """
        result = self.check(
            checkpoint_result,
            action_type,
            severity=severity,
            custom_components=custom_components,
        )

        if result.should_send:
            self.mark_sent(result.fingerprint, metadata=metadata)

        return result

    def is_duplicate(
        self,
        checkpoint_result: CheckpointResult,
        action_type: str,
        *,
        severity: str | None = None,
    ) -> bool:
        """Simple check if notification is a duplicate.

        Args:
            checkpoint_result: The checkpoint result.
            action_type: Type of notification action.
            severity: Override severity level.

        Returns:
            True if notification is a duplicate.
        """
        result = self.check(checkpoint_result, action_type, severity=severity)
        return result.is_duplicate

    def get_stats(self) -> DeduplicationStats:
        """Get deduplication statistics."""
        return self.store.get_stats()

    def cleanup(self) -> int:
        """Clean up expired records.

        Returns:
            Number of records removed.
        """
        return self.store.cleanup_expired()

    def clear(self) -> None:
        """Clear all deduplication records."""
        self.store.clear()

    def _generate_fingerprint(
        self,
        checkpoint_result: CheckpointResult,
        action_type: str,
        severity: str | None = None,
        custom_components: dict[str, Any] | None = None,
    ) -> NotificationFingerprint:
        """Generate fingerprint based on policy."""
        policy = self.config.policy

        if policy == DeduplicationPolicy.NONE:
            # Unique fingerprint for each notification
            return NotificationFingerprint.generate(
                checkpoint_name=checkpoint_result.checkpoint_name,
                action_type=action_type,
                custom_key=f"{checkpoint_result.run_id}_{datetime.now().timestamp()}",
            )

        if policy == DeduplicationPolicy.CUSTOM and self.config.custom_fingerprint_fn:
            return self.config.custom_fingerprint_fn(checkpoint_result, action_type)

        # Build extra components based on policy
        extra_components: dict[str, Any] = {}

        # Include status
        if policy in (
            DeduplicationPolicy.BASIC,
            DeduplicationPolicy.SEVERITY,
            DeduplicationPolicy.ISSUE_BASED,
            DeduplicationPolicy.STRICT,
        ):
            extra_components["status"] = str(checkpoint_result.status)

        # Include severity
        effective_severity: str | None = None
        if policy in (
            DeduplicationPolicy.SEVERITY,
            DeduplicationPolicy.ISSUE_BASED,
            DeduplicationPolicy.STRICT,
        ):
            effective_severity = severity or self._extract_severity(checkpoint_result)
            extra_components["severity"] = effective_severity

        # Include data asset
        if self.config.include_data_asset and checkpoint_result.data_asset:
            extra_components["data_asset"] = checkpoint_result.data_asset

        # Include issues
        if policy in (DeduplicationPolicy.ISSUE_BASED, DeduplicationPolicy.STRICT):
            if checkpoint_result.validation_result:
                issue_types = set()
                for issue in checkpoint_result.validation_result.issues:
                    issue_types.add(issue.validator_name)
                if issue_types:
                    extra_components["issue_types"] = sorted(issue_types)

        # Include metadata
        if self.config.include_metadata and policy == DeduplicationPolicy.STRICT:
            extra_components["metadata"] = checkpoint_result.metadata

        # Add custom components
        if custom_components:
            extra_components.update(custom_components)

        return NotificationFingerprint.generate(
            checkpoint_name=checkpoint_result.checkpoint_name,
            action_type=action_type,
            severity=effective_severity,
            data_asset=checkpoint_result.data_asset if self.config.include_data_asset else None,
            **{k: v for k, v in extra_components.items() if k not in ("severity", "data_asset")},
        )

    def _get_effective_window(
        self,
        action_type: str,
        severity: str,
    ) -> TimeWindow:
        """Get effective window based on config and policy."""
        # Check action-specific override first
        if action_type in self.config.action_windows:
            return self.config.action_windows[action_type]

        # For severity policy, use severity-based windows
        if self.config.policy == DeduplicationPolicy.SEVERITY:
            return self.config.get_window_for_severity(severity)

        return self.config.default_window

    def _extract_severity(self, checkpoint_result: CheckpointResult) -> str:
        """Extract highest severity from checkpoint result."""
        if not checkpoint_result.validation_result:
            return "medium"

        severity_order = ["critical", "high", "medium", "low", "info"]
        highest = "info"

        for issue in checkpoint_result.validation_result.issues:
            issue_severity = getattr(issue, "severity", "medium")
            if isinstance(issue_severity, str):
                sev = issue_severity.lower()
            else:
                sev = str(issue_severity).lower()

            try:
                if severity_order.index(sev) < severity_order.index(highest):
                    highest = sev
            except ValueError:
                continue

        return highest

    def _handle_suppression_exceeded(self, record: DeduplicationRecord) -> None:
        """Handle when max suppression count is exceeded."""
        logger.warning(
            f"Max suppression count exceeded for {record.fingerprint.checkpoint_name}/"
            f"{record.fingerprint.action_type}: {record.count} duplicates"
        )

        if self.config.on_suppression_exceeded:
            try:
                self.config.on_suppression_exceeded(record)
            except Exception as e:
                logger.error(f"Error in suppression exceeded callback: {e}")


@dataclass
class DeduplicatorBuilder:
    """Builder for creating NotificationDeduplicator instances.

    Example:
        >>> deduplicator = (
        ...     DeduplicatorBuilder()
        ...     .with_policy(DeduplicationPolicy.SEVERITY)
        ...     .with_default_window(TimeWindow(minutes=5))
        ...     .with_severity_window("critical", TimeWindow(minutes=1))
        ...     .with_redis_store("redis://localhost:6379")
        ...     .build()
        ... )
    """

    _config: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    _store: DeduplicationStore | None = None
    _strategy: WindowStrategy | None = None

    def with_policy(self, policy: DeduplicationPolicy) -> DeduplicatorBuilder:
        """Set deduplication policy."""
        self._config.policy = policy
        return self

    def with_default_window(self, window: TimeWindow) -> DeduplicatorBuilder:
        """Set default time window."""
        self._config.default_window = window
        return self

    def with_action_window(
        self, action_type: str, window: TimeWindow
    ) -> DeduplicatorBuilder:
        """Set window for specific action type."""
        self._config.action_windows[action_type] = window
        return self

    def with_severity_window(
        self, severity: str, window: TimeWindow
    ) -> DeduplicatorBuilder:
        """Set window for specific severity level."""
        self._config.severity_windows[severity.lower()] = window
        return self

    def with_max_suppression(self, count: int) -> DeduplicatorBuilder:
        """Set maximum suppression count."""
        self._config.max_suppressed_count = count
        return self

    def with_suppression_callback(
        self, callback: Callable[[DeduplicationRecord], None]
    ) -> DeduplicatorBuilder:
        """Set callback for suppression exceeded."""
        self._config.on_suppression_exceeded = callback
        return self

    def with_memory_store(self, max_size: int = 100000) -> DeduplicatorBuilder:
        """Use in-memory store."""
        self._store = InMemoryDeduplicationStore(max_size=max_size)
        return self

    def with_redis_store(
        self,
        redis_url: str,
        stream_key: str = "truthound:dedup:stream",
    ) -> DeduplicatorBuilder:
        """Use Redis Streams store."""
        from truthound.checkpoint.deduplication.stores import (
            RedisStreamsDeduplicationStore,
        )

        self._store = RedisStreamsDeduplicationStore(
            redis_url=redis_url,
            stream_key=stream_key,
        )
        return self

    def with_store(self, store: DeduplicationStore) -> DeduplicatorBuilder:
        """Use custom store."""
        self._store = store
        return self

    def with_strategy(self, strategy: WindowStrategy) -> DeduplicatorBuilder:
        """Set window strategy."""
        self._strategy = strategy
        return self

    def with_sliding_window(self) -> DeduplicatorBuilder:
        """Use sliding window strategy."""
        self._strategy = SlidingWindowStrategy()
        return self

    def with_tumbling_window(self) -> DeduplicatorBuilder:
        """Use tumbling window strategy."""
        from truthound.checkpoint.deduplication.processor import TumblingWindowStrategy

        self._strategy = TumblingWindowStrategy()
        return self

    def with_session_window(
        self,
        gap_duration: TimeWindow | None = None,
        max_duration: TimeWindow | None = None,
    ) -> DeduplicatorBuilder:
        """Use session window strategy."""
        from truthound.checkpoint.deduplication.processor import SessionWindowStrategy

        self._strategy = SessionWindowStrategy(
            gap_duration=gap_duration or TimeWindow(minutes=5),
            max_session_duration=max_duration or TimeWindow(hours=1),
        )
        return self

    def disabled(self) -> DeduplicatorBuilder:
        """Disable deduplication."""
        self._config.enabled = False
        return self

    def enabled(self) -> DeduplicatorBuilder:
        """Enable deduplication."""
        self._config.enabled = True
        return self

    def build(self) -> NotificationDeduplicator:
        """Build the deduplicator."""
        store = self._store or InMemoryDeduplicationStore()
        strategy = self._strategy or SlidingWindowStrategy()

        processor = TimeWindowProcessor(
            strategy=strategy,
            default_window=self._config.default_window,
        )

        return NotificationDeduplicator(
            store=store,
            config=self._config,
            processor=processor,
        )


# Convenience factory functions
def create_deduplicator(
    policy: DeduplicationPolicy = DeduplicationPolicy.SEVERITY,
    window: TimeWindow | None = None,
    redis_url: str | None = None,
) -> NotificationDeduplicator:
    """Create a deduplicator with common defaults.

    Args:
        policy: Deduplication policy.
        window: Default time window.
        redis_url: Redis URL (if using Redis backend).

    Returns:
        Configured NotificationDeduplicator.
    """
    builder = DeduplicatorBuilder().with_policy(policy)

    if window:
        builder.with_default_window(window)

    if redis_url:
        builder.with_redis_store(redis_url)
    else:
        builder.with_memory_store()

    return builder.build()
