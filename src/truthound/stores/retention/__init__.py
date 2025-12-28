"""TTL and Retention policy system for validation stores.

This module provides a flexible retention policy layer that can be applied
to any store backend, enabling automatic cleanup based on time, count,
size, or custom rules.

Example:
    >>> from truthound.stores import get_store
    >>> from truthound.stores.retention import (
    ...     RetentionStore,
    ...     RetentionConfig,
    ...     TimeBasedPolicy,
    ...     CountBasedPolicy,
    ...     SizeBasedPolicy,
    ...     CompositePolicy,
    ... )
    >>>
    >>> # Create base store
    >>> base_store = get_store("filesystem", base_path=".truthound/results")
    >>>
    >>> # Define retention policies
    >>> policies = [
    ...     TimeBasedPolicy(max_age_days=30),  # Keep 30 days
    ...     CountBasedPolicy(max_count=1000),  # Max 1000 items
    ...     SizeBasedPolicy(max_size_mb=500),  # Max 500 MB
    ... ]
    >>>
    >>> # Wrap with retention
    >>> store = RetentionStore(
    ...     base_store,
    ...     RetentionConfig(
    ...         policies=policies,
    ...         auto_cleanup=True,
    ...         cleanup_interval_hours=24,
    ...     ),
    ... )
    >>>
    >>> # Manual cleanup
    >>> deleted = store.run_cleanup()
    >>> print(f"Deleted {deleted} items")
"""

from truthound.stores.retention.base import (
    RetentionConfig,
    RetentionPolicy,
    RetentionResult,
    RetentionSchedule,
)
from truthound.stores.retention.policies import (
    TimeBasedPolicy,
    CountBasedPolicy,
    SizeBasedPolicy,
    StatusBasedPolicy,
    TagBasedPolicy,
    CompositePolicy,
)
from truthound.stores.retention.store import RetentionStore
from truthound.stores.retention.scheduler import RetentionScheduler

__all__ = [
    # Base types
    "RetentionConfig",
    "RetentionPolicy",
    "RetentionResult",
    "RetentionSchedule",
    # Policies
    "TimeBasedPolicy",
    "CountBasedPolicy",
    "SizeBasedPolicy",
    "StatusBasedPolicy",
    "TagBasedPolicy",
    "CompositePolicy",
    # Store wrapper
    "RetentionStore",
    # Scheduler
    "RetentionScheduler",
]
