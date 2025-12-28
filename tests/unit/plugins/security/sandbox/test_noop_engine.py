"""Tests for no-op sandbox engine."""

import asyncio
import pytest

from truthound.plugins.security.protocols import IsolationLevel, SecurityPolicy, ResourceLimits
from truthound.plugins.security.sandbox.engines.noop import NoopSandboxEngine
from truthound.plugins.security.exceptions import SandboxTimeoutError


class TestNoopSandboxEngine:
    """Tests for NoopSandboxEngine."""

    def test_isolation_level(self):
        """Test engine reports correct isolation level."""
        engine = NoopSandboxEngine()
        assert engine.isolation_level == IsolationLevel.NONE

    def test_create_sandbox(self):
        """Test creating a sandbox context."""
        engine = NoopSandboxEngine()
        policy = SecurityPolicy.standard()

        context = engine.create_sandbox("test-plugin", policy)

        assert context.plugin_id == "test-plugin"
        assert context.policy == policy
        assert context.is_alive() is True

    def test_create_multiple_sandboxes(self):
        """Test creating multiple sandbox contexts."""
        engine = NoopSandboxEngine()
        policy = SecurityPolicy.standard()

        ctx1 = engine.create_sandbox("plugin-1", policy)
        ctx2 = engine.create_sandbox("plugin-2", policy)

        assert ctx1.sandbox_id != ctx2.sandbox_id
        assert ctx1.plugin_id == "plugin-1"
        assert ctx2.plugin_id == "plugin-2"

    @pytest.mark.asyncio
    async def test_execute_sync_function(self):
        """Test executing a sync function."""
        engine = NoopSandboxEngine()
        policy = SecurityPolicy.standard()
        context = engine.create_sandbox("test", policy)

        def add(a, b):
            return a + b

        result = await engine.execute(context, add, 2, 3)

        assert result == 5
        assert context.is_alive() is False

    @pytest.mark.asyncio
    async def test_execute_async_function(self):
        """Test executing an async function."""
        engine = NoopSandboxEngine()
        policy = SecurityPolicy.standard()
        context = engine.create_sandbox("test", policy)

        async def async_multiply(a, b):
            await asyncio.sleep(0.01)
            return a * b

        result = await engine.execute(context, async_multiply, 4, 5)

        assert result == 20
        assert context.is_alive() is False

    @pytest.mark.asyncio
    async def test_execute_with_kwargs(self):
        """Test executing function with keyword arguments."""
        engine = NoopSandboxEngine()
        policy = SecurityPolicy.standard()
        context = engine.create_sandbox("test", policy)

        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result = await engine.execute(context, greet, "World", greeting="Hi")

        assert result == "Hi, World!"

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        """Test execution timeout."""
        engine = NoopSandboxEngine()
        # Create policy with very short timeout
        limits = ResourceLimits(max_execution_time_sec=0.1)
        policy = SecurityPolicy(resource_limits=limits)
        context = engine.create_sandbox("test", policy)

        async def slow_function():
            await asyncio.sleep(10)  # Way longer than timeout
            return "done"

        with pytest.raises(SandboxTimeoutError) as exc_info:
            await engine.execute(context, slow_function)

        assert exc_info.value.plugin_id == "test"
        assert exc_info.value.timeout_seconds == 0.1
        assert context.is_alive() is False

    def test_terminate(self):
        """Test terminating a sandbox."""
        engine = NoopSandboxEngine()
        policy = SecurityPolicy.standard()
        context = engine.create_sandbox("test", policy)

        assert context.is_alive() is True

        engine.terminate(context)

        assert context.is_alive() is False

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test cleaning up all sandboxes."""
        engine = NoopSandboxEngine()
        policy = SecurityPolicy.standard()

        ctx1 = engine.create_sandbox("plugin-1", policy)
        ctx2 = engine.create_sandbox("plugin-2", policy)

        await engine.cleanup()

        assert ctx1.is_alive() is False
        assert ctx2.is_alive() is False


class TestNoopSandboxEngineEdgeCases:
    """Edge case tests for NoopSandboxEngine."""

    @pytest.mark.asyncio
    async def test_execute_function_that_raises(self):
        """Test executing a function that raises an exception."""
        engine = NoopSandboxEngine()
        policy = SecurityPolicy.standard()
        context = engine.create_sandbox("test", policy)

        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await engine.execute(context, failing_function)

    @pytest.mark.asyncio
    async def test_execute_returns_none(self):
        """Test executing a function that returns None."""
        engine = NoopSandboxEngine()
        policy = SecurityPolicy.standard()
        context = engine.create_sandbox("test", policy)

        def void_function():
            pass

        result = await engine.execute(context, void_function)

        assert result is None

    def test_terminate_context_not_tracked(self):
        """Test terminating context not tracked by this engine.

        Since the context is a SandboxContextImpl, terminate() can still
        mark it as terminated without looking it up in _contexts.
        """
        engine = NoopSandboxEngine()
        policy = SecurityPolicy.standard()

        # Create context directly (not through engine)
        from truthound.plugins.security.sandbox.context import SandboxContextImpl
        context = SandboxContextImpl(plugin_id="external", policy=policy)

        # Should be alive initially
        assert context.is_alive() is True

        # Terminating works - it's a SandboxContextImpl so it can be marked
        engine.terminate(context)

        # Context should be terminated
        assert context.is_alive() is False
