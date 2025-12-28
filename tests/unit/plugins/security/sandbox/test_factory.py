"""Tests for sandbox factory."""

import pytest

from truthound.plugins.security.protocols import IsolationLevel, SandboxEngine
from truthound.plugins.security.sandbox.factory import SandboxFactory


class TestSandboxFactory:
    """Tests for SandboxFactory."""

    def setup_method(self):
        """Reset factory before each test."""
        SandboxFactory.reset()

    def teardown_method(self):
        """Clean up after each test."""
        SandboxFactory.reset()

    def test_create_noop_engine(self):
        """Test creating no-op sandbox engine."""
        engine = SandboxFactory.create(IsolationLevel.NONE)

        assert engine is not None
        assert engine.isolation_level == IsolationLevel.NONE

    def test_create_process_engine(self):
        """Test creating process sandbox engine."""
        engine = SandboxFactory.create(IsolationLevel.PROCESS)

        assert engine is not None
        assert engine.isolation_level == IsolationLevel.PROCESS

    def test_create_container_engine(self):
        """Test creating container sandbox engine."""
        engine = SandboxFactory.create(IsolationLevel.CONTAINER)

        assert engine is not None
        assert engine.isolation_level == IsolationLevel.CONTAINER

    def test_singleton_behavior(self):
        """Test factory returns singleton instances by default."""
        engine1 = SandboxFactory.create(IsolationLevel.NONE)
        engine2 = SandboxFactory.create(IsolationLevel.NONE)

        assert engine1 is engine2

    def test_non_singleton_behavior(self):
        """Test factory can create new instances."""
        engine1 = SandboxFactory.create(IsolationLevel.NONE, singleton=False)
        engine2 = SandboxFactory.create(IsolationLevel.NONE, singleton=False)

        assert engine1 is not engine2

    def test_is_available(self):
        """Test checking engine availability."""
        assert SandboxFactory.is_available(IsolationLevel.NONE) is True
        assert SandboxFactory.is_available(IsolationLevel.PROCESS) is True
        assert SandboxFactory.is_available(IsolationLevel.CONTAINER) is True
        # WASM is not implemented
        assert SandboxFactory.is_available(IsolationLevel.WASM) is False

    def test_list_available(self):
        """Test listing available isolation levels."""
        available = SandboxFactory.list_available()

        assert IsolationLevel.NONE in available
        assert IsolationLevel.PROCESS in available
        assert IsolationLevel.CONTAINER in available

    def test_create_invalid_level_raises(self):
        """Test creating engine for unavailable level raises error."""
        # WASM is not implemented
        with pytest.raises(ValueError, match="No sandbox engine registered"):
            SandboxFactory.create(IsolationLevel.WASM)

    def test_get_best_available(self):
        """Test getting best available engine."""
        # Should return the preferred if available
        engine = SandboxFactory.get_best_available(IsolationLevel.PROCESS)
        assert engine.isolation_level == IsolationLevel.PROCESS

    def test_get_best_available_fallback(self):
        """Test fallback to less isolated engine."""
        # Remove container engine to test fallback
        SandboxFactory.reset()
        SandboxFactory._load_default_engines()
        SandboxFactory.unregister(IsolationLevel.CONTAINER)

        # Should fallback to process
        engine = SandboxFactory.get_best_available(IsolationLevel.CONTAINER)
        assert engine.isolation_level == IsolationLevel.PROCESS


class TestSandboxFactoryRegistration:
    """Tests for engine registration."""

    def setup_method(self):
        """Reset factory before each test."""
        SandboxFactory.reset()

    def teardown_method(self):
        """Clean up after each test."""
        SandboxFactory.reset()

    def test_register_custom_engine(self):
        """Test registering a custom engine."""
        from truthound.plugins.security.sandbox.engines.noop import NoopSandboxEngine

        # Create a mock custom engine
        class CustomEngine(NoopSandboxEngine):
            @property
            def isolation_level(self):
                return IsolationLevel.WASM

        # Register for WASM level
        SandboxFactory.register(IsolationLevel.WASM, CustomEngine)

        # Should now be available
        assert SandboxFactory.is_available(IsolationLevel.WASM)
        engine = SandboxFactory.create(IsolationLevel.WASM)
        assert engine.isolation_level == IsolationLevel.WASM

    def test_unregister_engine(self):
        """Test unregistering an engine."""
        SandboxFactory._load_default_engines()

        result = SandboxFactory.unregister(IsolationLevel.CONTAINER)
        assert result is True

        assert SandboxFactory.is_available(IsolationLevel.CONTAINER) is False

    def test_unregister_nonexistent_returns_false(self):
        """Test unregistering nonexistent engine returns False."""
        SandboxFactory.reset()

        result = SandboxFactory.unregister(IsolationLevel.WASM)
        assert result is False

    def test_register_clears_cached_instance(self):
        """Test registering clears cached singleton."""
        from truthound.plugins.security.sandbox.engines.noop import NoopSandboxEngine

        # Get cached instance
        engine1 = SandboxFactory.create(IsolationLevel.NONE)

        # Re-register should clear cache
        SandboxFactory.register(IsolationLevel.NONE, NoopSandboxEngine)

        engine2 = SandboxFactory.create(IsolationLevel.NONE)

        # Should be different instances
        assert engine1 is not engine2


class TestSandboxFactoryCleanup:
    """Tests for factory cleanup."""

    def setup_method(self):
        """Reset factory before each test."""
        SandboxFactory.reset()

    def teardown_method(self):
        """Clean up after each test."""
        SandboxFactory.reset()

    @pytest.mark.asyncio
    async def test_cleanup_all(self):
        """Test cleaning up all engine instances."""
        # Create some engines
        SandboxFactory.create(IsolationLevel.NONE)
        SandboxFactory.create(IsolationLevel.PROCESS)

        # Cleanup
        await SandboxFactory.cleanup_all()

        # Instances should be cleared
        assert len(SandboxFactory._instances) == 0

    def test_reset(self):
        """Test resetting factory."""
        # Create some engines
        SandboxFactory.create(IsolationLevel.NONE)
        SandboxFactory.create(IsolationLevel.PROCESS)

        SandboxFactory.reset()

        assert len(SandboxFactory._engines) == 0
        assert len(SandboxFactory._instances) == 0
