"""Factory for creating sandbox engines.

This module provides the SandboxFactory which creates the appropriate
sandbox engine based on the requested isolation level.

The factory follows the Strategy pattern, allowing different isolation
mechanisms to be selected at runtime.
"""

from __future__ import annotations

import logging

from truthound.plugins.security.protocols import IsolationLevel, SandboxEngine

logger = logging.getLogger(__name__)


class SandboxFactory:
    """Factory for creating sandbox engines.

    Uses the Strategy pattern to select appropriate sandbox engine
    based on isolation level.

    Example:
        >>> engine = SandboxFactory.create(IsolationLevel.PROCESS)
        >>> context = engine.create_sandbox("my-plugin", policy)
    """

    # Registry of engine classes by isolation level
    _engines: dict[IsolationLevel, type[SandboxEngine]] = {}

    # Lazy-loaded engine instances (singletons per level)
    _instances: dict[IsolationLevel, SandboxEngine] = {}
    _retired: list[SandboxEngine] = []
    _defaults_loaded = False

    @classmethod
    def register(
        cls,
        level: IsolationLevel,
        engine_class: type[SandboxEngine],
    ) -> None:
        """Register a sandbox engine for an isolation level.

        Args:
            level: Isolation level this engine provides
            engine_class: Engine class to register

        Example:
            >>> SandboxFactory.register(IsolationLevel.WASM, WasmSandboxEngine)
        """
        cls._engines[level] = engine_class
        # Clear cached instance if exists
        previous = cls._instances.pop(level, None)
        if previous is not None:
            cls._retired.append(previous)
        logger.debug(f"Registered sandbox engine {engine_class.__name__} for {level.name}")

    @classmethod
    def unregister(cls, level: IsolationLevel) -> bool:
        """Unregister a sandbox engine.

        Args:
            level: Isolation level to unregister

        Returns:
            True if engine was registered, False otherwise
        """
        previous = cls._instances.pop(level, None)
        if previous is not None:
            cls._retired.append(previous)
        return cls._engines.pop(level, None) is not None

    @classmethod
    def create(
        cls,
        level: IsolationLevel,
        singleton: bool = True,
    ) -> SandboxEngine:
        """Create or get a sandbox engine for the specified isolation level.

        Args:
            level: Desired isolation level
            singleton: If True, return cached instance; if False, create new

        Returns:
            SandboxEngine instance

        Raises:
            ValueError: If no engine registered for the level
        """
        # Check for singleton instance
        if singleton and level in cls._instances:
            return cls._instances[level]

        # Get engine class
        engine_class = cls._engines.get(level)
        if engine_class is None:
            # Try to lazily load default engines
            cls._load_default_engines()
            engine_class = cls._engines.get(level)

        if engine_class is None:
            available = [level.name for level in cls._engines]
            raise ValueError(
                f"No sandbox engine registered for isolation level {level.name}. "
                f"Available: {available}"
            )

        # Create instance
        instance = engine_class()

        # Cache if singleton
        if singleton:
            cls._instances[level] = instance

        logger.debug(f"Created sandbox engine {engine_class.__name__} for {level.name}")
        return instance

    @classmethod
    def _load_default_engines(cls) -> None:
        """Lazily load default sandbox engines."""
        if cls._defaults_loaded:
            return

        # Import engines here to avoid circular imports
        from truthound.plugins.security.sandbox.engines.container import ContainerSandboxEngine
        from truthound.plugins.security.sandbox.engines.noop import NoopSandboxEngine
        from truthound.plugins.security.sandbox.engines.process import ProcessSandboxEngine

        defaults = {
            IsolationLevel.NONE: NoopSandboxEngine,
            IsolationLevel.PROCESS: ProcessSandboxEngine,
            IsolationLevel.CONTAINER: ContainerSandboxEngine,
            # WASM not implemented yet
        }
        for level, engine in defaults.items():
            cls._engines.setdefault(level, engine)
        cls._defaults_loaded = True

    @classmethod
    def is_available(cls, level: IsolationLevel) -> bool:
        """Check if a sandbox engine is available for the level.

        Args:
            level: Isolation level to check

        Returns:
            True if engine is registered
        """
        cls._load_default_engines()
        return level in cls._engines

    @classmethod
    def list_available(cls) -> list[IsolationLevel]:
        """List all available isolation levels.

        Returns:
            List of registered isolation levels
        """
        cls._load_default_engines()
        return list(cls._engines.keys())

    @classmethod
    def get_best_available(cls, preferred: IsolationLevel) -> SandboxEngine:
        """Get the best available sandbox engine.

        Falls back to less isolated options if preferred is not available.

        Args:
            preferred: Preferred isolation level

        Returns:
            Best available SandboxEngine

        Raises:
            ValueError: If no engines are available
        """
        cls._load_default_engines()

        # Priority order from most to least isolated
        fallback_order = [
            IsolationLevel.CONTAINER,
            IsolationLevel.PROCESS,
            IsolationLevel.NONE,
        ]

        # Try preferred first
        if cls.is_available(preferred):
            return cls.create(preferred)

        # Find best available fallback
        preferred_index = fallback_order.index(preferred) if preferred in fallback_order else 0
        for level in fallback_order[preferred_index:]:
            if cls.is_available(level):
                logger.warning(
                    f"Preferred isolation {preferred.name} not available, "
                    f"falling back to {level.name}"
                )
                return cls.create(level)

        raise ValueError("No sandbox engines available")

    @classmethod
    async def cleanup_all(cls) -> None:
        """Clean up all cached engine instances."""
        failures = []
        for level, engine in list(cls._instances.items()):
            try:
                await engine.cleanup()
            except Exception as error:
                failures.append(error)
            else:
                if cls._instances.get(level) is engine:
                    cls._instances.pop(level)
        for engine in list(cls._retired):
            try:
                await engine.cleanup()
            except Exception as error:
                failures.append(error)
            else:
                cls._retired.remove(engine)
        if failures:
            raise ExceptionGroup("Sandbox owner cleanup failed", failures)

    @classmethod
    def reset(cls) -> None:
        """Reset factory to initial state (mainly for testing)."""
        cls._retired.extend(cls._instances.values())
        cls._engines.clear()
        cls._instances.clear()
        cls._defaults_loaded = False
