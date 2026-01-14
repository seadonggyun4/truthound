"""Escalation Policy System.

This module provides a comprehensive escalation policy system for
multi-level alerting with APScheduler-based scheduling.

Key Components:
    - protocols: Core abstractions and data types
    - states: State machine for escalation lifecycle
    - levels: Escalation level management
    - scheduler: APScheduler-based scheduling
    - stores: Persistence backends (InMemory, Redis, SQLite)
    - engine: Escalation policy orchestration
    - integration: Routing system integration

Example:
    >>> from truthound.checkpoint.escalation import (
    ...     EscalationPolicy,
    ...     EscalationLevel,
    ...     EscalationEngine,
    ... )
    >>> policy = EscalationPolicy(
    ...     name="critical_alerts",
    ...     levels=[
    ...         EscalationLevel(level=1, delay_minutes=0, targets=["team-lead"]),
    ...         EscalationLevel(level=2, delay_minutes=15, targets=["manager"]),
    ...         EscalationLevel(level=3, delay_minutes=30, targets=["director"]),
    ...     ],
    ... )
    >>> engine = EscalationEngine(policy)
    >>> await engine.trigger("incident-123", context)
"""

from truthound.checkpoint.escalation.protocols import (
    BaseEscalationStore,
    EscalationLevel,
    EscalationPolicy,
    EscalationPolicyConfig,
    EscalationRecord,
    EscalationResult,
    EscalationStats,
    EscalationStoreProtocol,
    EscalationTarget,
    EscalationTrigger,
    TargetType,
)
from truthound.checkpoint.escalation.states import (
    EscalationEvent,
    EscalationState,
    EscalationStateMachine,
    EscalationStateManager,
    StateTransition,
)
from truthound.checkpoint.escalation.scheduler import (
    AsyncioScheduler,
    BaseEscalationScheduler,
    EscalationSchedulerProtocol,
    InMemoryScheduler,
    JobStatus,
    ScheduledJob,
    SchedulerConfig,
    SchedulerType,
    create_scheduler,
)
from truthound.checkpoint.escalation.stores import (
    InMemoryEscalationStore,
    RedisEscalationStore,
    SQLiteEscalationStore,
    create_store,
)
from truthound.checkpoint.escalation.engine import (
    EscalationEngine,
    EscalationEngineConfig,
    EscalationPolicyManager,
)
from truthound.checkpoint.escalation.integration import (
    ActionNotificationAdapter,
    EscalationAction,
    EscalationRule,
    EscalationRuleConfig,
    create_escalation_route,
    setup_escalation_with_existing_actions,
)

__all__ = [
    # Protocols
    "BaseEscalationStore",
    "EscalationLevel",
    "EscalationPolicy",
    "EscalationPolicyConfig",
    "EscalationRecord",
    "EscalationResult",
    "EscalationStats",
    "EscalationStoreProtocol",
    "EscalationTarget",
    "EscalationTrigger",
    "TargetType",
    # States
    "EscalationEvent",
    "EscalationState",
    "EscalationStateMachine",
    "EscalationStateManager",
    "StateTransition",
    # Scheduler
    "AsyncioScheduler",
    "BaseEscalationScheduler",
    "EscalationSchedulerProtocol",
    "InMemoryScheduler",
    "JobStatus",
    "ScheduledJob",
    "SchedulerConfig",
    "SchedulerType",
    "create_scheduler",
    # Stores
    "InMemoryEscalationStore",
    "RedisEscalationStore",
    "SQLiteEscalationStore",
    "create_store",
    # Engine
    "EscalationEngine",
    "EscalationEngineConfig",
    "EscalationPolicyManager",
    # Integration
    "ActionNotificationAdapter",
    "EscalationAction",
    "EscalationRule",
    "EscalationRuleConfig",
    "create_escalation_route",
    "setup_escalation_with_existing_actions",
]
