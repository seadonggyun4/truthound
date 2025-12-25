"""Base action class and common interfaces.

This module defines the abstract base class for all checkpoint actions
and common types used across action implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from truthound.checkpoint.checkpoint import CheckpointResult


class ActionStatus(str, Enum):
    """Status of an action execution."""

    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"
    ERROR = "error"

    def __str__(self) -> str:
        return self.value


class NotifyCondition(str, Enum):
    """Conditions for when to execute notification actions."""

    ALWAYS = "always"
    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"
    WARNING = "warning"
    FAILURE_OR_ERROR = "failure_or_error"
    NOT_SUCCESS = "not_success"

    def __str__(self) -> str:
        return self.value

    def should_notify(self, result_status: str) -> bool:
        """Check if notification should be sent based on result status.

        Args:
            result_status: The status of the validation result.

        Returns:
            True if notification should be sent.
        """
        if self == NotifyCondition.ALWAYS:
            return True
        if self == NotifyCondition.SUCCESS:
            return result_status == "success"
        if self == NotifyCondition.FAILURE:
            return result_status == "failure"
        if self == NotifyCondition.ERROR:
            return result_status == "error"
        if self == NotifyCondition.WARNING:
            return result_status == "warning"
        if self == NotifyCondition.FAILURE_OR_ERROR:
            return result_status in ("failure", "error")
        if self == NotifyCondition.NOT_SUCCESS:
            return result_status != "success"
        return False


@dataclass
class ActionConfig:
    """Base configuration for all actions.

    Subclasses should extend this with action-specific options.

    Attributes:
        name: Optional name for this action instance.
        enabled: Whether this action is enabled.
        notify_on: Condition for when to execute (for notification actions).
        timeout_seconds: Maximum time to wait for action completion.
        retry_count: Number of retries on failure.
        retry_delay_seconds: Delay between retries.
        fail_checkpoint_on_error: Whether action failure should fail the checkpoint.
        metadata: Additional metadata for the action.
    """

    name: str | None = None
    enabled: bool = True
    notify_on: NotifyCondition | str = NotifyCondition.ALWAYS
    timeout_seconds: int = 30
    retry_count: int = 0
    retry_delay_seconds: float = 1.0
    fail_checkpoint_on_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Convert string notify_on to enum."""
        if isinstance(self.notify_on, str):
            self.notify_on = NotifyCondition(self.notify_on.lower())


@dataclass
class ActionResult:
    """Result of an action execution.

    Attributes:
        action_name: Name of the action that was executed.
        action_type: Type of the action.
        status: Execution status.
        message: Human-readable result message.
        started_at: When the action started.
        completed_at: When the action completed.
        duration_ms: Execution duration in milliseconds.
        details: Additional details about the execution.
        error: Error message if action failed.
    """

    action_name: str
    action_type: str
    status: ActionStatus
    message: str = ""
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def success(self) -> bool:
        """Check if action was successful."""
        return self.status == ActionStatus.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "action_name": self.action_name,
            "action_type": self.action_type,
            "status": self.status.value,
            "message": self.message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "details": self.details,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionResult":
        """Create from dictionary."""
        return cls(
            action_name=data["action_name"],
            action_type=data["action_type"],
            status=ActionStatus(data["status"]),
            message=data.get("message", ""),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            duration_ms=data.get("duration_ms", 0.0),
            details=data.get("details", {}),
            error=data.get("error"),
        )


ConfigT = TypeVar("ConfigT", bound=ActionConfig)


class BaseAction(ABC, Generic[ConfigT]):
    """Abstract base class for all checkpoint actions.

    Actions are executed after validation completes. Each action can
    perform specific tasks like storing results, sending notifications,
    or updating external systems.

    Type Parameters:
        ConfigT: The configuration type for this action.

    Example:
        >>> class MyAction(BaseAction[MyActionConfig]):
        ...     action_type = "my_action"
        ...
        ...     def _execute(self, checkpoint_result: CheckpointResult) -> ActionResult:
        ...         # Implementation
        ...         return ActionResult(
        ...             action_name=self.name,
        ...             action_type=self.action_type,
        ...             status=ActionStatus.SUCCESS,
        ...             message="Action completed",
        ...         )
    """

    action_type: str = "base"

    def __init__(self, config: ConfigT | None = None, **kwargs: Any) -> None:
        """Initialize the action.

        Args:
            config: Action configuration. If None, uses default configuration.
            **kwargs: Additional configuration options to override.
        """
        self._config = config or self._default_config()

        # Apply kwargs to config
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

    @classmethod
    @abstractmethod
    def _default_config(cls) -> ConfigT:
        """Create default configuration for this action type."""
        pass

    @property
    def config(self) -> ConfigT:
        """Get the action configuration."""
        return self._config

    @property
    def name(self) -> str:
        """Get the action name."""
        return self._config.name or f"{self.action_type}_{id(self)}"

    @property
    def enabled(self) -> bool:
        """Check if this action is enabled."""
        return self._config.enabled

    def should_run(self, result_status: str) -> bool:
        """Check if this action should run based on result status.

        Args:
            result_status: The status of the validation result.

        Returns:
            True if this action should run.
        """
        if not self.enabled:
            return False

        notify_condition = self._config.notify_on
        if isinstance(notify_condition, str):
            notify_condition = NotifyCondition(notify_condition.lower())

        return notify_condition.should_notify(result_status)

    def execute(self, checkpoint_result: "CheckpointResult") -> ActionResult:
        """Execute the action with timing and error handling.

        This method wraps the actual implementation with timing,
        retry logic, and error handling.

        Args:
            checkpoint_result: The result of the checkpoint run.

        Returns:
            ActionResult describing the execution outcome.
        """
        import time

        started_at = datetime.now()
        result_status = checkpoint_result.status.value if hasattr(checkpoint_result.status, 'value') else str(checkpoint_result.status)

        # Check if action should run
        if not self.should_run(result_status):
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.SKIPPED,
                message=f"Skipped: notify_on={self._config.notify_on}, result_status={result_status}",
                started_at=started_at,
                completed_at=datetime.now(),
            )

        # Execute with retries
        last_error: Exception | None = None
        for attempt in range(self._config.retry_count + 1):
            try:
                result = self._execute(checkpoint_result)
                result.started_at = started_at
                result.completed_at = datetime.now()
                result.duration_ms = (result.completed_at - started_at).total_seconds() * 1000
                return result
            except Exception as e:
                last_error = e
                if attempt < self._config.retry_count:
                    time.sleep(self._config.retry_delay_seconds)

        # All retries failed
        completed_at = datetime.now()
        return ActionResult(
            action_name=self.name,
            action_type=self.action_type,
            status=ActionStatus.ERROR,
            message=f"Action failed after {self._config.retry_count + 1} attempts",
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=(completed_at - started_at).total_seconds() * 1000,
            error=str(last_error),
        )

    @abstractmethod
    def _execute(self, checkpoint_result: "CheckpointResult") -> ActionResult:
        """Execute the action implementation.

        Subclasses must implement this method to define the action's behavior.

        Args:
            checkpoint_result: The result of the checkpoint run.

        Returns:
            ActionResult describing the execution outcome.
        """
        pass

    def validate_config(self) -> list[str]:
        """Validate the action configuration.

        Returns:
            List of validation error messages (empty if valid).
        """
        return []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
