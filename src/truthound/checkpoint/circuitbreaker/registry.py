"""Circuit Breaker Registry for centralized management.

This module provides a singleton registry for managing multiple
circuit breakers, enabling centralized monitoring and configuration.
"""

from __future__ import annotations

from threading import RLock
from typing import Callable, Iterator

from truthound.checkpoint.circuitbreaker.core import (
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    CircuitState,
    StateChangeEvent,
)
from truthound.checkpoint.circuitbreaker.breaker import CircuitBreaker
from truthound.checkpoint.circuitbreaker.detection import FailureDetector


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers.

    Provides centralized access to circuit breakers with features like:
    - Singleton or instance-based usage
    - Global state change listeners
    - Bulk operations (reset all, get all metrics)
    - Health checking

    Example:
        >>> registry = CircuitBreakerRegistry()
        >>>
        >>> # Register breakers
        >>> registry.register("api", CircuitBreakerConfig(failure_threshold=5))
        >>> registry.register("db", CircuitBreakerConfig(failure_threshold=3))
        >>>
        >>> # Get breaker
        >>> api_breaker = registry.get("api")
        >>>
        >>> # Get all metrics
        >>> for name, metrics in registry.get_all_metrics().items():
        ...     print(f"{name}: {metrics.state}")
    """

    _instance: CircuitBreakerRegistry | None = None
    _instance_lock = RLock()

    def __init__(self):
        """Initialize registry."""
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = RLock()
        self._global_listeners: list[Callable[[StateChangeEvent], None]] = []

    @classmethod
    def get_instance(cls) -> CircuitBreakerRegistry:
        """Get singleton instance.

        Returns:
            Global registry instance
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = CircuitBreakerRegistry()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (mainly for testing)."""
        with cls._instance_lock:
            cls._instance = None

    def register(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        detector: FailureDetector | None = None,
        replace: bool = False,
    ) -> CircuitBreaker:
        """Register a new circuit breaker.

        Args:
            name: Unique name for the breaker
            config: Configuration (uses defaults if not provided)
            detector: Custom failure detector
            replace: If True, replace existing breaker with same name

        Returns:
            The registered CircuitBreaker

        Raises:
            ValueError: If name already exists and replace=False
        """
        with self._lock:
            if name in self._breakers and not replace:
                raise ValueError(f"Circuit breaker '{name}' already registered")

            breaker = CircuitBreaker(name, config, detector)

            # Add global listeners
            for listener in self._global_listeners:
                breaker.add_listener(listener)

            self._breakers[name] = breaker
            return breaker

    def register_breaker(self, breaker: CircuitBreaker, replace: bool = False) -> None:
        """Register an existing circuit breaker instance.

        Args:
            breaker: Circuit breaker to register
            replace: If True, replace existing breaker with same name

        Raises:
            ValueError: If name already exists and replace=False
        """
        with self._lock:
            if breaker.name in self._breakers and not replace:
                raise ValueError(f"Circuit breaker '{breaker.name}' already registered")

            # Add global listeners
            for listener in self._global_listeners:
                breaker.add_listener(listener)

            self._breakers[breaker.name] = breaker

    def get(self, name: str) -> CircuitBreaker | None:
        """Get circuit breaker by name.

        Args:
            name: Breaker name

        Returns:
            CircuitBreaker or None if not found
        """
        with self._lock:
            return self._breakers.get(name)

    def get_or_create(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        detector: FailureDetector | None = None,
    ) -> CircuitBreaker:
        """Get existing breaker or create new one.

        Args:
            name: Breaker name
            config: Configuration for new breaker
            detector: Detector for new breaker

        Returns:
            Existing or new CircuitBreaker
        """
        with self._lock:
            if name in self._breakers:
                return self._breakers[name]
            return self.register(name, config, detector)

    def unregister(self, name: str) -> CircuitBreaker | None:
        """Unregister a circuit breaker.

        Args:
            name: Breaker name to remove

        Returns:
            Removed breaker or None if not found
        """
        with self._lock:
            return self._breakers.pop(name, None)

    def __contains__(self, name: str) -> bool:
        """Check if breaker exists."""
        return name in self._breakers

    def __getitem__(self, name: str) -> CircuitBreaker:
        """Get breaker by name (raises KeyError if not found)."""
        with self._lock:
            return self._breakers[name]

    def __iter__(self) -> Iterator[str]:
        """Iterate over breaker names."""
        with self._lock:
            return iter(list(self._breakers.keys()))

    def __len__(self) -> int:
        """Get number of registered breakers."""
        return len(self._breakers)

    def get_all(self) -> dict[str, CircuitBreaker]:
        """Get all registered breakers.

        Returns:
            Dict mapping names to breakers
        """
        with self._lock:
            return dict(self._breakers)

    def get_all_metrics(self) -> dict[str, CircuitBreakerMetrics]:
        """Get metrics for all breakers.

        Returns:
            Dict mapping names to metrics
        """
        with self._lock:
            return {
                name: breaker.get_metrics()
                for name, breaker in self._breakers.items()
            }

    def get_health_status(self) -> dict[str, bool]:
        """Get health status for all breakers.

        Returns:
            Dict mapping names to health status (True = healthy/closed)
        """
        with self._lock:
            return {
                name: breaker.is_closed
                for name, breaker in self._breakers.items()
            }

    def get_open_breakers(self) -> list[str]:
        """Get names of all open breakers.

        Returns:
            List of breaker names in OPEN state
        """
        with self._lock:
            return [
                name for name, breaker in self._breakers.items()
                if breaker.is_open
            ]

    def get_unhealthy_breakers(self) -> list[str]:
        """Get names of all unhealthy (open or half-open) breakers.

        Returns:
            List of breaker names not in CLOSED state
        """
        with self._lock:
            return [
                name for name, breaker in self._breakers.items()
                if not breaker.is_closed
            ]

    def reset_all(self) -> None:
        """Reset all breakers to closed state."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()

    def reset(self, name: str) -> bool:
        """Reset specific breaker.

        Args:
            name: Breaker name to reset

        Returns:
            True if breaker was found and reset
        """
        with self._lock:
            if name in self._breakers:
                self._breakers[name].reset()
                return True
            return False

    def add_global_listener(self, listener: Callable[[StateChangeEvent], None]) -> None:
        """Add listener to all current and future breakers.

        Args:
            listener: Callback for state changes
        """
        with self._lock:
            self._global_listeners.append(listener)
            # Add to existing breakers
            for breaker in self._breakers.values():
                breaker.add_listener(listener)

    def remove_global_listener(self, listener: Callable[[StateChangeEvent], None]) -> None:
        """Remove global listener.

        Args:
            listener: Callback to remove
        """
        with self._lock:
            if listener in self._global_listeners:
                self._global_listeners.remove(listener)
            # Remove from existing breakers
            for breaker in self._breakers.values():
                breaker.remove_listener(listener)

    def clear(self) -> None:
        """Remove all registered breakers."""
        with self._lock:
            self._breakers.clear()

    def to_dict(self) -> dict:
        """Export registry state as dictionary.

        Returns:
            Dict with all breaker metrics
        """
        with self._lock:
            return {
                "breakers": {
                    name: breaker.get_metrics().to_dict()
                    for name, breaker in self._breakers.items()
                },
                "total_count": len(self._breakers),
                "healthy_count": sum(1 for b in self._breakers.values() if b.is_closed),
                "open_count": sum(1 for b in self._breakers.values() if b.is_open),
                "half_open_count": sum(1 for b in self._breakers.values() if b.is_half_open),
            }


# Module-level convenience functions using singleton
def get_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry.

    Returns:
        Global CircuitBreakerRegistry instance
    """
    return CircuitBreakerRegistry.get_instance()


def get_breaker(name: str) -> CircuitBreaker | None:
    """Get a circuit breaker from the global registry.

    Args:
        name: Breaker name

    Returns:
        CircuitBreaker or None
    """
    return get_registry().get(name)


def register_breaker(
    name: str,
    config: CircuitBreakerConfig | None = None,
    detector: FailureDetector | None = None,
    replace: bool = False,
) -> CircuitBreaker:
    """Register a circuit breaker in the global registry.

    Args:
        name: Unique name for the breaker
        config: Configuration
        detector: Custom failure detector
        replace: Replace existing if True

    Returns:
        Registered CircuitBreaker
    """
    return get_registry().register(name, config, detector, replace)
