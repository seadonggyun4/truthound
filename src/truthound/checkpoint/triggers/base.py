"""Base trigger class and common interfaces.

This module defines the abstract base class for all checkpoint triggers
and common types used across trigger implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Iterator

if TYPE_CHECKING:
    from truthound.checkpoint.checkpoint import Checkpoint


class TriggerStatus(str, Enum):
    """Status of a trigger."""

    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"

    def __str__(self) -> str:
        return self.value


@dataclass
class TriggerConfig:
    """Base configuration for all triggers.

    Subclasses should extend this with trigger-specific options.

    Attributes:
        name: Optional name for this trigger instance.
        enabled: Whether this trigger is enabled.
        max_runs: Maximum number of runs (0 = unlimited).
        run_immediately: Run once immediately on start.
        catch_up: Run missed executions on startup.
        max_concurrent: Maximum concurrent checkpoint runs.
        metadata: Additional metadata for the trigger.
    """

    name: str | None = None
    enabled: bool = True
    max_runs: int = 0
    run_immediately: bool = False
    catch_up: bool = False
    max_concurrent: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TriggerResult:
    """Result of a trigger evaluation.

    Attributes:
        should_run: Whether the checkpoint should run now.
        reason: Reason for the decision.
        next_run: When the next run is scheduled (if applicable).
        context: Additional context for the run.
    """

    should_run: bool
    reason: str = ""
    next_run: datetime | None = None
    context: dict[str, Any] = field(default_factory=dict)


ConfigT = TypeVar("ConfigT", bound=TriggerConfig)


class BaseTrigger(ABC, Generic[ConfigT]):
    """Abstract base class for all checkpoint triggers.

    Triggers determine when checkpoints should be executed.
    They can be time-based, event-based, or custom.

    Type Parameters:
        ConfigT: The configuration type for this trigger.

    Example:
        >>> class MyTrigger(BaseTrigger[MyTriggerConfig]):
        ...     trigger_type = "my_trigger"
        ...
        ...     def should_trigger(self) -> TriggerResult:
        ...         # Check if trigger condition is met
        ...         return TriggerResult(should_run=True, reason="Condition met")
    """

    trigger_type: str = "base"

    def __init__(self, config: ConfigT | None = None, **kwargs: Any) -> None:
        """Initialize the trigger.

        Args:
            config: Trigger configuration. If None, uses default configuration.
            **kwargs: Additional configuration options to override.
        """
        self._config = config or self._default_config()

        # Apply kwargs to config
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

        self._status = TriggerStatus.STOPPED
        self._run_count = 0
        self._last_run: datetime | None = None
        self._checkpoint: Checkpoint | None = None

    @classmethod
    @abstractmethod
    def _default_config(cls) -> ConfigT:
        """Create default configuration for this trigger type."""
        pass

    @property
    def config(self) -> ConfigT:
        """Get the trigger configuration."""
        return self._config

    @property
    def name(self) -> str:
        """Get the trigger name."""
        return self._config.name or f"{self.trigger_type}_{id(self)}"

    @property
    def status(self) -> TriggerStatus:
        """Get the trigger status."""
        return self._status

    @property
    def enabled(self) -> bool:
        """Check if this trigger is enabled."""
        return self._config.enabled

    @property
    def run_count(self) -> int:
        """Get the number of times this trigger has fired."""
        return self._run_count

    @property
    def last_run(self) -> datetime | None:
        """Get the last run time."""
        return self._last_run

    def attach(self, checkpoint: "Checkpoint") -> None:
        """Attach this trigger to a checkpoint.

        Args:
            checkpoint: The checkpoint to trigger.
        """
        self._checkpoint = checkpoint

    @abstractmethod
    def should_trigger(self) -> TriggerResult:
        """Check if the trigger condition is met.

        Returns:
            TriggerResult indicating whether to run and why.
        """
        pass

    def start(self) -> None:
        """Start the trigger."""
        if not self.enabled:
            return

        self._status = TriggerStatus.ACTIVE
        self._on_start()

        if self._config.run_immediately:
            self._run_count += 1
            self._last_run = datetime.now()

    def stop(self) -> None:
        """Stop the trigger."""
        self._status = TriggerStatus.STOPPED
        self._on_stop()

    def pause(self) -> None:
        """Pause the trigger."""
        self._status = TriggerStatus.PAUSED

    def resume(self) -> None:
        """Resume a paused trigger."""
        if self._status == TriggerStatus.PAUSED:
            self._status = TriggerStatus.ACTIVE

    def record_run(self) -> None:
        """Record that a run occurred."""
        self._run_count += 1
        self._last_run = datetime.now()

    def should_continue(self) -> bool:
        """Check if trigger should continue running.

        Returns:
            True if trigger should continue, False if max_runs reached.
        """
        if self._config.max_runs == 0:
            return True
        return self._run_count < self._config.max_runs

    def _on_start(self) -> None:
        """Called when trigger starts. Override for custom behavior."""
        pass

    def _on_stop(self) -> None:
        """Called when trigger stops. Override for custom behavior."""
        pass

    def validate_config(self) -> list[str]:
        """Validate the trigger configuration.

        Returns:
            List of validation error messages (empty if valid).
        """
        return []

    def to_dict(self) -> dict[str, Any]:
        """Serialize trigger to dictionary."""
        return {
            "trigger_type": self.trigger_type,
            "name": self.name,
            "status": self._status.value,
            "run_count": self._run_count,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "enabled": self.enabled,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, status={self._status.value!r})"
