"""Enterprise Saga Pattern Framework.

This module provides advanced Saga pattern implementations for complex,
distributed transaction scenarios in enterprise environments.

Key Components:
    - SagaDefinition: Declarative saga definition with fluent API
    - SagaStateMachine: State machine-based saga execution
    - SagaEventStore: Event sourcing for saga history
    - Advanced compensation strategies (Semantic, Pivot, Countermeasure)
    - Complex scenario builders (Chained, Nested, Parallel sagas)

Example:
    >>> from truthound.checkpoint.transaction.saga import (
    ...     SagaBuilder,
    ...     SagaRunner,
    ...     CompensationPolicy,
    ... )
    >>>
    >>> saga = (
    ...     SagaBuilder("order_processing")
    ...     .step("validate_order", ValidateAction())
    ...         .compensate_with(RejectOrderAction())
    ...     .step("reserve_inventory", ReserveAction())
    ...         .compensate_with(ReleaseInventoryAction())
    ...         .with_timeout(30)
    ...     .step("process_payment", PaymentAction())
    ...         .compensate_with(RefundAction())
    ...         .with_retry(max_attempts=3, backoff="exponential")
    ...     .step("ship_order", ShipAction())
    ...     .with_policy(CompensationPolicy.SEMANTIC)
    ...     .build()
    ... )
    >>>
    >>> runner = SagaRunner()
    >>> result = runner.execute(saga, context)
"""

from truthound.checkpoint.transaction.saga.definition import (
    SagaDefinition,
    SagaStepDefinition,
    StepDependency,
    DependencyType,
)

from truthound.checkpoint.transaction.saga.builder import (
    SagaBuilder,
    StepBuilder,
)

from truthound.checkpoint.transaction.saga.state_machine import (
    SagaState,
    SagaEvent,
    SagaEventType,
    SagaStateMachine,
    SagaTransition,
)

from truthound.checkpoint.transaction.saga.event_store import (
    SagaEventStore,
    InMemorySagaEventStore,
    FileSagaEventStore,
    SagaSnapshot,
)

from truthound.checkpoint.transaction.saga.strategies import (
    CompensationPolicy,
    SemanticCompensation,
    PivotTransaction,
    CountermeasureStrategy,
    CompensationPlan,
    CompensationPlanner,
)

from truthound.checkpoint.transaction.saga.runner import (
    SagaRunner,
    SagaExecutionContext,
    SagaExecutionResult,
    SagaMetrics,
)

from truthound.checkpoint.transaction.saga.patterns import (
    ChainedSagaPattern,
    NestedSagaPattern,
    ParallelSagaPattern,
    ChoreographySagaPattern,
    OrchestratorSagaPattern,
)

from truthound.checkpoint.transaction.saga.testing import (
    SagaTestHarness,
    SagaScenario,
    FailureInjector,
    SagaAssertion,
    ScenarioBuilder,
)


__all__ = [
    # Definition
    "SagaDefinition",
    "SagaStepDefinition",
    "StepDependency",
    "DependencyType",
    # Builder
    "SagaBuilder",
    "StepBuilder",
    # State Machine
    "SagaState",
    "SagaEvent",
    "SagaEventType",
    "SagaStateMachine",
    "SagaTransition",
    # Event Store
    "SagaEventStore",
    "InMemorySagaEventStore",
    "FileSagaEventStore",
    "SagaSnapshot",
    # Strategies
    "CompensationPolicy",
    "SemanticCompensation",
    "PivotTransaction",
    "CountermeasureStrategy",
    "CompensationPlan",
    "CompensationPlanner",
    # Runner
    "SagaRunner",
    "SagaExecutionContext",
    "SagaExecutionResult",
    "SagaMetrics",
    # Patterns
    "ChainedSagaPattern",
    "NestedSagaPattern",
    "ParallelSagaPattern",
    "ChoreographySagaPattern",
    "OrchestratorSagaPattern",
    # Testing
    "SagaTestHarness",
    "SagaScenario",
    "FailureInjector",
    "SagaAssertion",
    "ScenarioBuilder",
]
