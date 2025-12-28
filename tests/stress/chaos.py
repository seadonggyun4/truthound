"""Chaos engineering for stress testing.

Provides failure injection and chaos testing capabilities
for distributed saga stress tests.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Awaitable, TypeVar

import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


class FailureType(Enum):
    """Types of failures that can be injected."""

    # Network failures
    NETWORK_DELAY = auto()
    NETWORK_TIMEOUT = auto()
    NETWORK_DISCONNECT = auto()
    NETWORK_PARTITION = auto()

    # Application failures
    EXCEPTION = auto()
    SLOW_RESPONSE = auto()
    CORRUPT_RESPONSE = auto()
    EMPTY_RESPONSE = auto()

    # Resource failures
    MEMORY_PRESSURE = auto()
    CPU_PRESSURE = auto()
    DISK_FULL = auto()

    # Service failures
    SERVICE_UNAVAILABLE = auto()
    RATE_LIMITED = auto()
    AUTHENTICATION_FAILURE = auto()

    # Transaction failures
    PARTIAL_COMMIT = auto()
    ROLLBACK = auto()
    DEADLOCK = auto()


@dataclass
class FailureConfig:
    """Configuration for failure injection.

    Attributes:
        failure_type: Type of failure to inject.
        probability: Probability of failure (0.0-1.0).
        duration_ms: Duration of failure effect in milliseconds.
        targets: Optional list of target identifiers.
        metadata: Additional failure parameters.
    """

    failure_type: FailureType
    probability: float = 0.1
    duration_ms: float = 1000.0
    targets: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("Probability must be between 0.0 and 1.0")
        if self.duration_ms < 0:
            raise ValueError("Duration must be non-negative")


@dataclass
class FailureEvent:
    """Record of an injected failure.

    Attributes:
        failure_type: Type of failure.
        target: Target identifier.
        timestamp: When failure was injected.
        duration_ms: Duration of failure.
        metadata: Additional details.
    """

    failure_type: FailureType
    target: str
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class FailureInjector:
    """Injects failures into operations.

    Wraps operations and injects configured failures
    based on probability and rules.

    Example:
        >>> injector = FailureInjector()
        >>> injector.add_failure(FailureConfig(
        ...     failure_type=FailureType.NETWORK_DELAY,
        ...     probability=0.1,
        ...     duration_ms=500,
        ... ))
        >>>
        >>> @injector.wrap
        >>> async def my_operation():
        ...     return await fetch_data()
    """

    def __init__(
        self,
        enabled: bool = True,
        seed: int | None = None,
    ) -> None:
        """Initialize failure injector.

        Args:
            enabled: Whether injection is enabled.
            seed: Random seed for reproducibility.
        """
        self._enabled = enabled
        self._rng = random.Random(seed)
        self._failures: list[FailureConfig] = []
        self._events: list[FailureEvent] = []
        self._paused = False

    def add_failure(self, config: FailureConfig) -> "FailureInjector":
        """Add a failure configuration.

        Args:
            config: Failure configuration.

        Returns:
            Self for chaining.
        """
        self._failures.append(config)
        return self

    def remove_failure(self, failure_type: FailureType) -> bool:
        """Remove failures of a specific type.

        Args:
            failure_type: Type to remove.

        Returns:
            True if any were removed.
        """
        original_count = len(self._failures)
        self._failures = [f for f in self._failures if f.failure_type != failure_type]
        return len(self._failures) < original_count

    def clear_failures(self) -> None:
        """Clear all failure configurations."""
        self._failures.clear()

    def pause(self) -> None:
        """Pause failure injection."""
        self._paused = True

    def resume(self) -> None:
        """Resume failure injection."""
        self._paused = False

    @property
    def events(self) -> list[FailureEvent]:
        """Get list of injected failure events."""
        return self._events.copy()

    def clear_events(self) -> None:
        """Clear recorded events."""
        self._events.clear()

    async def maybe_inject(
        self,
        target: str = "default",
    ) -> FailureEvent | None:
        """Maybe inject a failure based on configuration.

        Args:
            target: Target identifier.

        Returns:
            FailureEvent if failure was injected, None otherwise.
        """
        if not self._enabled or self._paused:
            return None

        for config in self._failures:
            # Check if target matches
            if config.targets and target not in config.targets:
                continue

            # Check probability
            if self._rng.random() > config.probability:
                continue

            # Inject failure
            event = await self._inject_failure(config, target)
            self._events.append(event)
            return event

        return None

    async def _inject_failure(
        self,
        config: FailureConfig,
        target: str,
    ) -> FailureEvent:
        """Inject a specific failure.

        Args:
            config: Failure configuration.
            target: Target identifier.

        Returns:
            FailureEvent describing what was injected.
        """
        failure_type = config.failure_type
        duration_ms = config.duration_ms

        logger.debug(f"Injecting {failure_type.name} failure on {target}")

        if failure_type == FailureType.NETWORK_DELAY:
            await asyncio.sleep(duration_ms / 1000)

        elif failure_type == FailureType.NETWORK_TIMEOUT:
            await asyncio.sleep(duration_ms / 1000)
            raise asyncio.TimeoutError(f"Simulated timeout on {target}")

        elif failure_type == FailureType.NETWORK_DISCONNECT:
            raise ConnectionError(f"Simulated disconnect on {target}")

        elif failure_type == FailureType.EXCEPTION:
            error_msg = config.metadata.get("message", f"Simulated error on {target}")
            raise RuntimeError(error_msg)

        elif failure_type == FailureType.SLOW_RESPONSE:
            await asyncio.sleep(duration_ms / 1000)

        elif failure_type == FailureType.SERVICE_UNAVAILABLE:
            raise ConnectionRefusedError(f"Service unavailable: {target}")

        elif failure_type == FailureType.RATE_LIMITED:
            raise RuntimeError(f"Rate limited on {target}")

        elif failure_type == FailureType.AUTHENTICATION_FAILURE:
            raise PermissionError(f"Authentication failed for {target}")

        elif failure_type == FailureType.ROLLBACK:
            raise RuntimeError(f"Transaction rollback on {target}")

        elif failure_type == FailureType.DEADLOCK:
            # Simulate a deadlock by waiting indefinitely (up to duration)
            await asyncio.sleep(duration_ms / 1000)
            raise RuntimeError(f"Deadlock detected on {target}")

        return FailureEvent(
            failure_type=failure_type,
            target=target,
            duration_ms=duration_ms,
            metadata=config.metadata,
        )

    def wrap(
        self,
        func: Callable[..., Awaitable[T]],
    ) -> Callable[..., Awaitable[T]]:
        """Wrap an async function with failure injection.

        Args:
            func: Async function to wrap.

        Returns:
            Wrapped function.
        """
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            # Get target from kwargs or use function name
            target = kwargs.pop("_failure_target", func.__name__)

            # Maybe inject failure
            await self.maybe_inject(target)

            # Call original function
            return await func(*args, **kwargs)

        return wrapper


class ChaosEngine:
    """Comprehensive chaos engineering engine.

    Coordinates multiple failure injectors and provides
    high-level chaos scenarios.

    Example:
        >>> chaos = ChaosEngine()
        >>> chaos.enable_scenario("network_chaos", probability=0.1)
        >>>
        >>> async with chaos.chaos_context("my_operation"):
        ...     await perform_operation()
    """

    # Pre-defined chaos scenarios
    SCENARIOS = {
        "network_chaos": [
            FailureConfig(FailureType.NETWORK_DELAY, probability=0.1, duration_ms=500),
            FailureConfig(FailureType.NETWORK_TIMEOUT, probability=0.05, duration_ms=5000),
            FailureConfig(FailureType.NETWORK_DISCONNECT, probability=0.02),
        ],
        "service_chaos": [
            FailureConfig(FailureType.SERVICE_UNAVAILABLE, probability=0.05),
            FailureConfig(FailureType.RATE_LIMITED, probability=0.1),
            FailureConfig(FailureType.SLOW_RESPONSE, probability=0.1, duration_ms=2000),
        ],
        "transaction_chaos": [
            FailureConfig(FailureType.ROLLBACK, probability=0.05),
            FailureConfig(FailureType.PARTIAL_COMMIT, probability=0.02),
            FailureConfig(FailureType.DEADLOCK, probability=0.01, duration_ms=10000),
        ],
        "full_chaos": [
            FailureConfig(FailureType.NETWORK_DELAY, probability=0.05, duration_ms=500),
            FailureConfig(FailureType.NETWORK_TIMEOUT, probability=0.02, duration_ms=5000),
            FailureConfig(FailureType.SERVICE_UNAVAILABLE, probability=0.02),
            FailureConfig(FailureType.EXCEPTION, probability=0.05),
            FailureConfig(FailureType.ROLLBACK, probability=0.02),
        ],
    }

    def __init__(
        self,
        enabled: bool = True,
        seed: int | None = None,
    ) -> None:
        """Initialize chaos engine.

        Args:
            enabled: Whether chaos is enabled globally.
            seed: Random seed for reproducibility.
        """
        self._enabled = enabled
        self._seed = seed
        self._injector = FailureInjector(enabled=enabled, seed=seed)
        self._active_scenarios: set[str] = set()
        self._stats: dict[str, int] = {
            "total_operations": 0,
            "failures_injected": 0,
        }

    def enable_scenario(
        self,
        scenario_name: str,
        probability_multiplier: float = 1.0,
    ) -> bool:
        """Enable a pre-defined chaos scenario.

        Args:
            scenario_name: Name of scenario to enable.
            probability_multiplier: Multiplier for failure probabilities.

        Returns:
            True if scenario was enabled.
        """
        if scenario_name not in self.SCENARIOS:
            logger.warning(f"Unknown chaos scenario: {scenario_name}")
            return False

        # Add failures with adjusted probabilities
        for base_config in self.SCENARIOS[scenario_name]:
            adjusted_config = FailureConfig(
                failure_type=base_config.failure_type,
                probability=min(1.0, base_config.probability * probability_multiplier),
                duration_ms=base_config.duration_ms,
                targets=base_config.targets,
                metadata=base_config.metadata,
            )
            self._injector.add_failure(adjusted_config)

        self._active_scenarios.add(scenario_name)
        logger.info(f"Enabled chaos scenario: {scenario_name}")
        return True

    def disable_scenario(self, scenario_name: str) -> bool:
        """Disable a chaos scenario.

        Args:
            scenario_name: Name of scenario to disable.

        Returns:
            True if scenario was disabled.
        """
        if scenario_name not in self._active_scenarios:
            return False

        # Remove failures associated with this scenario
        for config in self.SCENARIOS.get(scenario_name, []):
            self._injector.remove_failure(config.failure_type)

        self._active_scenarios.discard(scenario_name)
        logger.info(f"Disabled chaos scenario: {scenario_name}")
        return True

    def disable_all(self) -> None:
        """Disable all chaos scenarios."""
        self._injector.clear_failures()
        self._active_scenarios.clear()
        logger.info("Disabled all chaos scenarios")

    @property
    def active_scenarios(self) -> set[str]:
        """Get active scenario names."""
        return self._active_scenarios.copy()

    @property
    def injector(self) -> FailureInjector:
        """Get the underlying failure injector."""
        return self._injector

    @property
    def stats(self) -> dict[str, int]:
        """Get chaos statistics."""
        return self._stats.copy()

    async def maybe_inject_chaos(self, target: str = "default") -> bool:
        """Maybe inject chaos at this point.

        Args:
            target: Target identifier.

        Returns:
            True if chaos was injected.
        """
        self._stats["total_operations"] += 1

        event = await self._injector.maybe_inject(target)
        if event:
            self._stats["failures_injected"] += 1
            return True
        return False

    class ChaosContext:
        """Context manager for chaos injection."""

        def __init__(
            self,
            engine: "ChaosEngine",
            target: str,
        ) -> None:
            self._engine = engine
            self._target = target
            self._start_time: float | None = None

        async def __aenter__(self) -> "ChaosEngine.ChaosContext":
            self._start_time = time.monotonic()
            await self._engine.maybe_inject_chaos(self._target)
            return self

        async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
            # Could inject more chaos on exit if desired
            return False

    def chaos_context(self, target: str = "default") -> ChaosContext:
        """Create a chaos context manager.

        Args:
            target: Target identifier.

        Returns:
            Context manager for chaos injection.
        """
        return self.ChaosContext(self, target)

    def get_failure_summary(self) -> dict[str, Any]:
        """Get summary of injected failures.

        Returns:
            Summary dictionary.
        """
        events = self._injector.events

        # Count by type
        by_type: dict[str, int] = {}
        for event in events:
            type_name = event.failure_type.name
            by_type[type_name] = by_type.get(type_name, 0) + 1

        return {
            "total_failures": len(events),
            "total_operations": self._stats["total_operations"],
            "failure_rate": (
                len(events) / self._stats["total_operations"]
                if self._stats["total_operations"] > 0
                else 0.0
            ),
            "by_type": by_type,
            "active_scenarios": list(self._active_scenarios),
        }
