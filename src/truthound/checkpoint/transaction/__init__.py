"""Transaction and Compensation Framework for Checkpoint Actions.

This module provides a robust transaction management system with support for:
- Saga pattern for distributed transactions
- Compensating transactions (rollback/undo)
- Transaction boundaries and isolation
- Idempotency guarantees

Architecture:
    The framework uses the Saga pattern to manage transactions across multiple
    actions. Each action can optionally implement compensation logic that will
    be executed if a subsequent action fails.

Example:
    >>> from truthound.checkpoint.transaction import (
    ...     TransactionCoordinator,
    ...     TransactionalCheckpoint,
    ...     CompensatingAction,
    ... )
    >>>
    >>> # Create transactional checkpoint
    >>> checkpoint = TransactionalCheckpoint(
    ...     name="transactional_validation",
    ...     actions=[
    ...         StoreResultAction(path="/data/results"),  # Has compensation
    ...         SlackNotification(webhook_url="..."),      # No compensation needed
    ...         UpdateDocsAction(site="s3_docs"),          # Has compensation
    ...     ],
    ...     transaction_config=TransactionConfig(
    ...         rollback_on_failure=True,
    ...         savepoint_enabled=True,
    ...     ),
    ... )
    >>>
    >>> # Run with automatic rollback on failure
    >>> result = checkpoint.run()

Compensation Strategies:
    1. BACKWARD: Roll back in reverse order (default, Saga pattern)
    2. FORWARD: Attempt to complete remaining actions (forward recovery)
    3. CUSTOM: Use custom compensation orchestration
"""

from truthound.checkpoint.transaction.base import (
    # Core types
    TransactionState,
    TransactionPhase,
    CompensationStrategy,
    IsolationLevel,
    # Configuration
    TransactionConfig,
    # Results
    TransactionResult,
    CompensationResult,
    # Context
    TransactionContext,
    Savepoint,
)

from truthound.checkpoint.transaction.compensatable import (
    # Interfaces
    Compensatable,
    CompensatableAction,
    # Wrapper
    CompensationWrapper,
    # Decorators
    compensatable,
    with_compensation,
)

from truthound.checkpoint.transaction.coordinator import (
    TransactionCoordinator,
    SagaOrchestrator,
)

from truthound.checkpoint.transaction.executor import (
    TransactionalExecutor,
    TransactionBoundary,
)

from truthound.checkpoint.transaction.idempotency import (
    IdempotencyKey,
    IdempotencyStore,
    InMemoryIdempotencyStore,
    FileIdempotencyStore,
    IdempotencyManager,
    IdempotencyConflictError,
    idempotent,
)

from truthound.checkpoint.transaction.executor import (
    TransactionManager,
)

__all__ = [
    # Core types
    "TransactionState",
    "TransactionPhase",
    "CompensationStrategy",
    "IsolationLevel",
    # Configuration
    "TransactionConfig",
    # Results
    "TransactionResult",
    "CompensationResult",
    # Context
    "TransactionContext",
    "Savepoint",
    # Interfaces
    "Compensatable",
    "CompensatableAction",
    # Wrapper
    "CompensationWrapper",
    # Decorators
    "compensatable",
    "with_compensation",
    # Coordinator
    "TransactionCoordinator",
    "SagaOrchestrator",
    # Executor
    "TransactionalExecutor",
    "TransactionBoundary",
    # Idempotency
    "IdempotencyKey",
    "IdempotencyStore",
    "InMemoryIdempotencyStore",
    "FileIdempotencyStore",
    "IdempotencyManager",
    "IdempotencyConflictError",
    "idempotent",
    # Manager
    "TransactionManager",
]
