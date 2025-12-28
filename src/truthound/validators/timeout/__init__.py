"""Distributed timeout management for validators.

This module provides enterprise-grade timeout mechanisms for distributed
validation environments. It extends the core process timeout with:
- Distributed coordination via Redis/DynamoDB
- Deadline propagation across services
- Cascading timeout handling
- Graceful degradation policies

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    Distributed Timeout Manager                       │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
    ┌───────────────┬───────────────┼───────────────┬─────────────────────┐
    │               │               │               │                     │
    ▼               ▼               ▼               ▼                     ▼
┌─────────┐   ┌─────────┐    ┌──────────┐   ┌──────────┐    ┌────────────┐
│Deadline │   │ Timeout │    │ Cascading│   │ Graceful │    │ Distributed│
│Propagation│  │ Budgeting│   │ Handler  │   │Degradation│   │ Coordinator│
└─────────┘   └─────────┘    └──────────┘   └──────────┘    └────────────┘

Usage:
    from truthound.validators.timeout import (
        DistributedTimeoutManager,
        DeadlineContext,
        TimeoutBudget,
        with_deadline,
    )

    # Create deadline context
    deadline = DeadlineContext.from_seconds(60)

    # Use with timeout budget
    budget = TimeoutBudget(total_seconds=120)

    async with DistributedTimeoutManager(budget) as manager:
        result = await manager.execute_with_deadline(
            validate_fn,
            deadline,
        )
"""

from truthound.validators.timeout.distributed import (
    DistributedTimeoutManager,
    DistributedTimeoutConfig,
    CoordinatorBackend,
)

from truthound.validators.timeout.deadline import (
    DeadlineContext,
    DeadlinePropagator,
    TimeoutBudget,
    BudgetAllocation,
    with_deadline,
    propagate_deadline,
)

from truthound.validators.timeout.cascade import (
    CascadeTimeoutHandler,
    CascadePolicy,
    CascadeLevel,
    TimeoutCascadeResult,
)

from truthound.validators.timeout.degradation import (
    GracefulDegradation,
    DegradationPolicy,
    DegradationLevel,
    DegradationAction,
    DegradationResult,
)

# Advanced timeout features
from truthound.validators.timeout import advanced

__all__ = [
    # Distributed
    "DistributedTimeoutManager",
    "DistributedTimeoutConfig",
    "CoordinatorBackend",
    # Deadline
    "DeadlineContext",
    "DeadlinePropagator",
    "TimeoutBudget",
    "BudgetAllocation",
    "with_deadline",
    "propagate_deadline",
    # Cascade
    "CascadeTimeoutHandler",
    "CascadePolicy",
    "CascadeLevel",
    "TimeoutCascadeResult",
    # Degradation
    "GracefulDegradation",
    "DegradationPolicy",
    "DegradationLevel",
    "DegradationAction",
    "DegradationResult",
    # Advanced module
    "advanced",
]
