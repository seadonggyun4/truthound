"""Tests for sandbox context implementation."""

import pytest
from datetime import datetime, timezone

from truthound.plugins.security.protocols import SecurityPolicy
from truthound.plugins.security.sandbox.context import SandboxContextImpl


class TestSandboxContextImpl:
    """Tests for SandboxContextImpl."""

    def test_create_context(self):
        """Test creating a sandbox context."""
        policy = SecurityPolicy.standard()
        context = SandboxContextImpl(
            plugin_id="test-plugin",
            policy=policy,
        )

        assert context.plugin_id == "test-plugin"
        assert context.policy == policy
        assert context.sandbox_id != ""
        assert context.created_at is not None
        assert context.started_at is None
        assert context.finished_at is None

    def test_auto_generate_sandbox_id(self):
        """Test sandbox ID is auto-generated."""
        policy = SecurityPolicy.standard()
        context1 = SandboxContextImpl(plugin_id="test", policy=policy)
        context2 = SandboxContextImpl(plugin_id="test", policy=policy)

        # Different contexts should have different IDs
        assert context1.sandbox_id != context2.sandbox_id
        # IDs should be 16 characters (hex)
        assert len(context1.sandbox_id) == 16

    def test_custom_sandbox_id(self):
        """Test providing custom sandbox ID."""
        policy = SecurityPolicy.standard()
        context = SandboxContextImpl(
            plugin_id="test",
            policy=policy,
            sandbox_id="custom-id-12345",
        )

        assert context.sandbox_id == "custom-id-12345"

    def test_is_alive_initially_true(self):
        """Test context is alive when created."""
        policy = SecurityPolicy.standard()
        context = SandboxContextImpl(plugin_id="test", policy=policy)

        assert context.is_alive() is True

    def test_mark_started(self):
        """Test marking context as started."""
        policy = SecurityPolicy.standard()
        context = SandboxContextImpl(plugin_id="test", policy=policy)

        assert context.started_at is None
        context.mark_started()

        assert context.started_at is not None
        assert context.is_alive() is True

    def test_mark_finished(self):
        """Test marking context as finished."""
        policy = SecurityPolicy.standard()
        context = SandboxContextImpl(plugin_id="test", policy=policy)

        context.mark_started()
        context.mark_finished()

        assert context.finished_at is not None
        assert context.is_alive() is False

    def test_mark_terminated(self):
        """Test marking context as terminated."""
        policy = SecurityPolicy.standard()
        context = SandboxContextImpl(plugin_id="test", policy=policy)

        context.mark_terminated()

        assert context.is_alive() is False
        assert context.finished_at is not None

    def test_get_resource_usage_initial(self):
        """Test getting initial resource usage."""
        policy = SecurityPolicy.standard()
        context = SandboxContextImpl(plugin_id="test", policy=policy)

        usage = context.get_resource_usage()

        assert usage["memory_mb"] == 0.0
        assert usage["cpu_percent"] == 0.0
        assert usage["execution_time_sec"] == 0.0

    def test_update_resource_usage(self):
        """Test updating resource usage."""
        policy = SecurityPolicy.standard()
        context = SandboxContextImpl(plugin_id="test", policy=policy)

        context.update_resource_usage(memory_mb=256.5, cpu_percent=45.2)
        usage = context.get_resource_usage()

        assert usage["memory_mb"] == 256.5
        assert usage["cpu_percent"] == 45.2

    def test_set_process_id(self):
        """Test setting process ID."""
        policy = SecurityPolicy.standard()
        context = SandboxContextImpl(plugin_id="test", policy=policy)

        assert context.process_id is None
        context.set_process_id(12345)
        assert context.process_id == 12345

    def test_set_container_id(self):
        """Test setting container ID."""
        policy = SecurityPolicy.standard()
        context = SandboxContextImpl(plugin_id="test", policy=policy)

        assert context.container_id is None
        context.set_container_id("abc123def456")
        assert context.container_id == "abc123def456"

    def test_to_dict(self):
        """Test converting context to dictionary."""
        policy = SecurityPolicy.standard()
        context = SandboxContextImpl(plugin_id="test-plugin", policy=policy)
        context.set_process_id(9999)

        data = context.to_dict()

        assert data["plugin_id"] == "test-plugin"
        assert data["sandbox_id"] == context.sandbox_id
        assert "created_at" in data
        assert data["is_alive"] is True
        assert data["process_id"] == 9999
        assert "resource_usage" in data


class TestSandboxContextExecution:
    """Tests for sandbox context execution tracking."""

    def test_execution_time_tracking(self):
        """Test execution time is tracked correctly."""
        import time

        policy = SecurityPolicy.standard()
        context = SandboxContextImpl(plugin_id="test", policy=policy)

        context.mark_started()
        time.sleep(0.1)  # 100ms
        context.mark_finished()

        # Execution time should be approximately 0.1 seconds
        exec_time = context.execution_time_sec
        assert 0.05 < exec_time < 0.5  # Allow some tolerance

    def test_execution_time_while_running(self):
        """Test execution time updates while running."""
        import time

        policy = SecurityPolicy.standard()
        context = SandboxContextImpl(plugin_id="test", policy=policy)

        context.mark_started()
        time.sleep(0.05)

        # Should report current execution time
        time1 = context.execution_time_sec
        time.sleep(0.05)
        time2 = context.execution_time_sec

        assert time2 > time1

        context.mark_finished()
