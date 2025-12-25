"""Checkpoint & CI/CD integration for automated validation pipelines.

This module provides a comprehensive checkpoint system for orchestrating
data quality validations with actions, triggers, and CI/CD integrations.

Supports both synchronous and asynchronous execution:

Sync Example:
    >>> from truthound.checkpoint import Checkpoint
    >>> from truthound.checkpoint.actions import (
    ...     StoreValidationResult,
    ...     SlackNotification,
    ... )
    >>>
    >>> checkpoint = Checkpoint(
    ...     name="daily_user_validation",
    ...     data_source="users.csv",
    ...     validators=["null", "duplicate"],
    ...     actions=[
    ...         StoreValidationResult(store_path="./results"),
    ...         SlackNotification(
    ...             webhook_url="https://hooks.slack.com/...",
    ...             notify_on="failure",
    ...         ),
    ...     ],
    ... )
    >>> result = checkpoint.run()

Async Example:
    >>> from truthound.checkpoint import AsyncCheckpoint
    >>> from truthound.checkpoint.async_actions import (
    ...     AsyncSlackNotification,
    ...     AsyncWebhookAction,
    ... )
    >>>
    >>> async def main():
    ...     checkpoint = AsyncCheckpoint(
    ...         name="async_validation",
    ...         data_source="large_dataset.parquet",
    ...         actions=[
    ...             AsyncSlackNotification(webhook_url="..."),
    ...             AsyncWebhookAction(url="..."),
    ...         ],
    ...         max_concurrent_actions=5,
    ...     )
    ...     result = await checkpoint.run_async()
    ...
    ...     # Or run multiple checkpoints concurrently
    ...     results = await run_checkpoints_async([cp1, cp2, cp3])
"""

from truthound.checkpoint.checkpoint import (
    Checkpoint,
    CheckpointConfig,
    CheckpointResult,
    CheckpointStatus,
)
from truthound.checkpoint.runner import CheckpointRunner
from truthound.checkpoint.registry import (
    CheckpointRegistry,
    get_checkpoint,
    register_checkpoint,
    list_checkpoints,
    load_checkpoints,
)

# Async exports
from truthound.checkpoint.async_checkpoint import (
    AsyncCheckpoint,
    AsyncCheckpointConfig,
    to_async_checkpoint,
    run_checkpoints_async,
)
from truthound.checkpoint.async_runner import (
    AsyncCheckpointRunner,
    AsyncRunnerConfig,
    run_checkpoint_async,
    run_checkpoints_parallel,
    CheckpointPool,
)
from truthound.checkpoint.async_base import (
    AsyncBaseAction,
    AsyncBaseTrigger,
    AsyncExecutionContext,
    SyncActionAdapter,
    adapt_to_async,
    ExecutionStrategy,
    SequentialStrategy,
    ConcurrentStrategy,
    PipelineStrategy,
    with_retry,
    with_timeout,
    with_semaphore,
)

__all__ = [
    # Core (Sync)
    "Checkpoint",
    "CheckpointConfig",
    "CheckpointResult",
    "CheckpointStatus",
    # Runner (Sync)
    "CheckpointRunner",
    # Registry
    "CheckpointRegistry",
    "get_checkpoint",
    "register_checkpoint",
    "list_checkpoints",
    "load_checkpoints",
    # Async Checkpoint
    "AsyncCheckpoint",
    "AsyncCheckpointConfig",
    "to_async_checkpoint",
    "run_checkpoints_async",
    # Async Runner
    "AsyncCheckpointRunner",
    "AsyncRunnerConfig",
    "run_checkpoint_async",
    "run_checkpoints_parallel",
    "CheckpointPool",
    # Async Base Classes
    "AsyncBaseAction",
    "AsyncBaseTrigger",
    "AsyncExecutionContext",
    "SyncActionAdapter",
    "adapt_to_async",
    # Execution Strategies
    "ExecutionStrategy",
    "SequentialStrategy",
    "ConcurrentStrategy",
    "PipelineStrategy",
    # Decorators
    "with_retry",
    "with_timeout",
    "with_semaphore",
]
