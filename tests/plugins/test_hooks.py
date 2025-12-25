"""Tests for hook system."""

from __future__ import annotations

import pytest
import asyncio
from typing import Any

from truthound.plugins.hooks import (
    HookManager,
    Hook,
    HookType,
    hook,
    before_validation,
    after_validation,
    before_profile,
    after_profile,
    on_report_generate,
    on_issue_found,
    on_error,
    get_hook_manager,
    reset_hook_manager,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def hook_manager():
    """Create a fresh hook manager for each test."""
    return HookManager()


@pytest.fixture(autouse=True)
def reset_global_hooks():
    """Reset global hook manager between tests."""
    reset_hook_manager()
    yield
    reset_hook_manager()


# =============================================================================
# Hook Tests
# =============================================================================


class TestHook:
    """Tests for Hook dataclass."""

    def test_hook_creation(self):
        """Test creating a hook."""
        def handler():
            pass

        hook = Hook(
            hook_type=HookType.BEFORE_VALIDATION,
            handler=handler,
            priority=50,
            source="test",
        )

        assert hook.hook_type == HookType.BEFORE_VALIDATION
        assert hook.priority == 50
        assert hook.source == "test"
        assert hook.enabled is True

    def test_hook_callable(self):
        """Test that hooks are callable."""
        result = []

        def handler(x):
            result.append(x)

        hook = Hook(
            hook_type=HookType.CUSTOM,
            handler=handler,
        )

        hook("test")
        assert result == ["test"]

    def test_disabled_hook(self):
        """Test that disabled hooks don't execute."""
        result = []

        def handler():
            result.append("called")

        hook = Hook(
            hook_type=HookType.CUSTOM,
            handler=handler,
            enabled=False,
        )

        hook()
        assert result == []

    def test_async_hook_detection(self):
        """Test async handler detection."""
        async def async_handler():
            pass

        def sync_handler():
            pass

        async_hook = Hook(hook_type=HookType.CUSTOM, handler=async_handler)
        sync_hook = Hook(hook_type=HookType.CUSTOM, handler=sync_handler)

        assert async_hook.async_handler is True
        assert sync_hook.async_handler is False


# =============================================================================
# HookManager Tests
# =============================================================================


class TestHookManager:
    """Tests for HookManager."""

    def test_register_hook(self, hook_manager):
        """Test registering a hook."""
        def handler():
            pass

        hook = hook_manager.register(HookType.BEFORE_VALIDATION, handler)

        assert isinstance(hook, Hook)
        assert len(hook_manager) == 1

    def test_register_with_priority(self, hook_manager):
        """Test hooks are sorted by priority."""
        results = []

        def handler1():
            results.append(1)

        def handler2():
            results.append(2)

        def handler3():
            results.append(3)

        hook_manager.register(HookType.CUSTOM, handler2, priority=100)
        hook_manager.register(HookType.CUSTOM, handler1, priority=50)
        hook_manager.register(HookType.CUSTOM, handler3, priority=150)

        hook_manager.trigger(HookType.CUSTOM)

        # Should execute in priority order
        assert results == [1, 2, 3]

    def test_trigger_hooks(self, hook_manager):
        """Test triggering hooks."""
        results = []

        def handler1(x):
            results.append(f"handler1:{x}")

        def handler2(x):
            results.append(f"handler2:{x}")

        hook_manager.register(HookType.BEFORE_VALIDATION, handler1)
        hook_manager.register(HookType.BEFORE_VALIDATION, handler2)

        hook_manager.trigger(HookType.BEFORE_VALIDATION, "test")

        assert "handler1:test" in results
        assert "handler2:test" in results

    def test_trigger_with_kwargs(self, hook_manager):
        """Test triggering hooks with keyword arguments."""
        received = {}

        def handler(datasource=None, validators=None):
            received["datasource"] = datasource
            received["validators"] = validators

        hook_manager.register(HookType.BEFORE_VALIDATION, handler)
        hook_manager.trigger(
            HookType.BEFORE_VALIDATION,
            datasource="test.csv",
            validators=["null", "range"],
        )

        assert received["datasource"] == "test.csv"
        assert received["validators"] == ["null", "range"]

    def test_trigger_returns_results(self, hook_manager):
        """Test that trigger returns results from handlers."""
        def handler1():
            return 1

        def handler2():
            return 2

        hook_manager.register(HookType.CUSTOM, handler1)
        hook_manager.register(HookType.CUSTOM, handler2)

        results = hook_manager.trigger(HookType.CUSTOM)
        assert 1 in results
        assert 2 in results

    def test_trigger_handles_errors(self, hook_manager):
        """Test error handling in trigger."""
        def good_handler():
            return "ok"

        def bad_handler():
            raise RuntimeError("Error!")

        hook_manager.register(HookType.CUSTOM, good_handler, priority=50)
        hook_manager.register(HookType.CUSTOM, bad_handler, priority=100)

        # Should not raise, should continue
        results = hook_manager.trigger(HookType.CUSTOM)
        assert "ok" in results

    def test_unregister_hook(self, hook_manager):
        """Test unregistering hooks."""
        def handler():
            pass

        hook_manager.register(HookType.CUSTOM, handler)
        assert len(hook_manager.get_hooks(HookType.CUSTOM)) == 1

        removed = hook_manager.unregister(HookType.CUSTOM, handler=handler)
        assert removed == 1
        assert len(hook_manager.get_hooks(HookType.CUSTOM)) == 0

    def test_unregister_by_source(self, hook_manager):
        """Test unregistering hooks by source."""
        def handler1():
            pass

        def handler2():
            pass

        hook_manager.register(HookType.CUSTOM, handler1, source="plugin-a")
        hook_manager.register(HookType.CUSTOM, handler2, source="plugin-b")

        removed = hook_manager.unregister(HookType.CUSTOM, source="plugin-a")
        assert removed == 1

        hooks = hook_manager.get_hooks(HookType.CUSTOM)
        assert len(hooks) == 1
        assert hooks[0].source == "plugin-b"

    def test_get_hooks(self, hook_manager):
        """Test getting registered hooks."""
        def handler1():
            pass

        def handler2():
            pass

        hook_manager.register(HookType.BEFORE_VALIDATION, handler1)
        hook_manager.register(HookType.AFTER_VALIDATION, handler2)

        before_hooks = hook_manager.get_hooks(HookType.BEFORE_VALIDATION)
        all_hooks = hook_manager.get_hooks()

        assert len(before_hooks) == 1
        assert len(all_hooks) == 2

    def test_clear_hooks(self, hook_manager):
        """Test clearing hooks."""
        hook_manager.register(HookType.BEFORE_VALIDATION, lambda: None)
        hook_manager.register(HookType.AFTER_VALIDATION, lambda: None)

        hook_manager.clear(HookType.BEFORE_VALIDATION)
        assert len(hook_manager.get_hooks(HookType.BEFORE_VALIDATION)) == 0
        assert len(hook_manager.get_hooks(HookType.AFTER_VALIDATION)) == 1

        hook_manager.clear()
        assert len(hook_manager) == 0

    def test_enable_disable_hooks(self, hook_manager):
        """Test enabling and disabling hooks."""
        results = []

        def handler():
            results.append("called")

        hook_manager.register(HookType.CUSTOM, handler, source="test")

        hook_manager.disable(source="test")
        hook_manager.trigger(HookType.CUSTOM)
        assert results == []

        hook_manager.enable(source="test")
        hook_manager.trigger(HookType.CUSTOM)
        assert results == ["called"]

    @pytest.mark.asyncio
    async def test_trigger_async(self, hook_manager):
        """Test async hook triggering."""
        results = []

        async def async_handler():
            await asyncio.sleep(0.01)
            results.append("async")
            return "async_result"

        def sync_handler():
            results.append("sync")
            return "sync_result"

        hook_manager.register(HookType.CUSTOM, sync_handler, priority=50)
        hook_manager.register(HookType.CUSTOM, async_handler, priority=100)

        trigger_results = await hook_manager.trigger_async(HookType.CUSTOM)

        assert "sync" in results
        assert "async" in results
        assert "sync_result" in trigger_results
        assert "async_result" in trigger_results


# =============================================================================
# Decorator Tests
# =============================================================================


class TestHookDecorators:
    """Tests for hook decorators."""

    def test_hook_decorator(self):
        """Test generic hook decorator."""
        @hook(HookType.BEFORE_VALIDATION, priority=50)
        def my_handler():
            pass

        assert hasattr(my_handler, "_truthound_hook")
        info = my_handler._truthound_hook
        assert info["hook_type"] == HookType.BEFORE_VALIDATION
        assert info["priority"] == 50

    def test_before_validation_decorator(self):
        """Test before_validation decorator."""
        @before_validation(priority=25)
        def handler(datasource, validators):
            pass

        assert handler._truthound_hook["hook_type"] == HookType.BEFORE_VALIDATION
        assert handler._truthound_hook["priority"] == 25

    def test_after_validation_decorator(self):
        """Test after_validation decorator."""
        @after_validation()
        def handler(datasource, result, issues):
            pass

        assert handler._truthound_hook["hook_type"] == HookType.AFTER_VALIDATION

    def test_before_profile_decorator(self):
        """Test before_profile decorator."""
        @before_profile()
        def handler(datasource, config):
            pass

        assert handler._truthound_hook["hook_type"] == HookType.BEFORE_PROFILE

    def test_after_profile_decorator(self):
        """Test after_profile decorator."""
        @after_profile()
        def handler(datasource, profile):
            pass

        assert handler._truthound_hook["hook_type"] == HookType.AFTER_PROFILE

    def test_on_report_generate_decorator(self):
        """Test on_report_generate decorator."""
        @on_report_generate()
        def handler(report, format):
            return report

        assert handler._truthound_hook["hook_type"] == HookType.ON_REPORT_GENERATE

    def test_on_issue_found_decorator(self):
        """Test on_issue_found decorator."""
        @on_issue_found()
        def handler(issue, validator):
            pass

        assert handler._truthound_hook["hook_type"] == HookType.ON_ISSUE_FOUND

    def test_on_error_decorator(self):
        """Test on_error decorator."""
        @on_error()
        def handler(error, context):
            pass

        assert handler._truthound_hook["hook_type"] == HookType.ON_ERROR

    def test_register_decorated_function(self, hook_manager):
        """Test registering a decorated function."""
        @before_validation(priority=10)
        def my_handler(datasource, validators):
            return "handled"

        hook = hook_manager.register_decorated(my_handler)

        assert hook.priority == 10
        assert hook.hook_type == HookType.BEFORE_VALIDATION

        results = hook_manager.trigger(
            HookType.BEFORE_VALIDATION,
            datasource="test",
            validators=[],
        )
        assert "handled" in results

    def test_register_non_decorated_raises(self, hook_manager):
        """Test that registering non-decorated function raises."""
        def plain_handler():
            pass

        with pytest.raises(ValueError, match="not decorated"):
            hook_manager.register_decorated(plain_handler)


# =============================================================================
# HookType Tests
# =============================================================================


class TestHookType:
    """Tests for HookType enumeration."""

    def test_all_hook_types(self):
        """Test all hook types are defined."""
        expected = [
            "before_validation",
            "after_validation",
            "on_issue_found",
            "before_profile",
            "after_profile",
            "on_report_generate",
            "on_report_write",
            "on_datasource_connect",
            "on_datasource_disconnect",
            "on_error",
            "on_plugin_load",
            "on_plugin_unload",
            "custom",
        ]
        actual = [h.value for h in HookType]
        assert set(expected) == set(actual)

    def test_hook_type_as_string(self, hook_manager):
        """Test using string hook type."""
        def handler():
            return "ok"

        hook_manager.register("custom_hook", handler)
        results = hook_manager.trigger("custom_hook")
        assert "ok" in results


# =============================================================================
# Global Hook Manager Tests
# =============================================================================


class TestGlobalHookManager:
    """Tests for global hook manager."""

    def test_get_hook_manager(self):
        """Test getting global hook manager."""
        manager1 = get_hook_manager()
        manager2 = get_hook_manager()
        assert manager1 is manager2

    def test_reset_hook_manager(self):
        """Test resetting global hook manager."""
        manager1 = get_hook_manager()
        manager1.register(HookType.CUSTOM, lambda: None)

        reset_hook_manager()
        manager2 = get_hook_manager()

        assert manager2 is not manager1
        assert len(manager2) == 0
