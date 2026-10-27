"""Native hooks distinguish None success from failure without leaking messages."""

import asyncio

import pytest

from truthound.plugins.hooks import HookManager, HookType, before_validation

pytestmark = pytest.mark.contract


def test_native_detailed_hook_outcomes_keep_none_success_and_failure_distinct(caplog):
    hooks = HookManager()

    def nothing(**kwargs):
        return None

    def failing(**kwargs):
        raise ValueError("synthetic-private-hook-detail")

    hooks.register(HookType.CUSTOM, nothing, source="nothing")
    hooks.register(HookType.CUSTOM, failing, source="failing")
    assert hooks.trigger(HookType.CUSTOM) == [None, None]
    assert "synthetic-private-hook-detail" not in caplog.text
    result = hooks.trigger_detailed(HookType.CUSTOM)
    assert [(row.success, row.result, row.error_type) for row in result] == [
        (True, None, None),
        (False, None, "ValueError"),
    ]
    assert hooks.trigger(HookType.CUSTOM) == [None, None]
    assert "synthetic-private-hook-detail" not in caplog.text


def test_decorated_async_native_hook_is_awaited_and_exceptions_recorded():
    async def run():
        calls = []
        hooks = HookManager()

        @before_validation()
        async def before(**kwargs):
            await asyncio.sleep(0)
            calls.append("awaited")
            return "native"

        hooks.register_decorated(before)
        import inspect

        legacy = await hooks.trigger_async(HookType.BEFORE_VALIDATION)
        for value in legacy:
            if inspect.iscoroutine(value):
                value.close()
        assert calls == ["awaited"] and legacy == ["native"]
        results = await hooks.trigger_detailed_async(HookType.BEFORE_VALIDATION)
        assert (
            calls == ["awaited", "awaited"] and results[0].success and results[0].result == "native"
        )
        assert await hooks.trigger_async(HookType.BEFORE_VALIDATION) == ["native"]
        assert calls == ["awaited", "awaited", "awaited"]

    asyncio.run(run())


def test_native_hook_guard_rechecks_between_handlers_and_preserves_revocation():
    async def run():
        hooks = HookManager()
        calls = []
        authorized = True

        def guard():
            if not authorized:
                raise PermissionError("revoked")

        async def first():
            nonlocal authorized
            calls.append("first")
            authorized = False

        async def second():
            calls.append("second")

        hooks.register(HookType.CUSTOM, first, priority=1)
        hooks.register(HookType.CUSTOM, second, priority=2)
        hooks.set_execution_guard(guard)
        with pytest.raises(PermissionError, match="revoked"):
            await hooks.trigger_detailed_async(HookType.CUSTOM)
        assert calls == ["first"]

    asyncio.run(run())
