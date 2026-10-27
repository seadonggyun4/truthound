"""Factory registration must retain defaults and cleanup ownership."""

import pytest

from truthound.plugins.security.protocols import IsolationLevel
from truthound.plugins.security.sandbox.engines.noop import NoopSandboxEngine
from truthound.plugins.security.sandbox.factory import SandboxFactory


def test_custom_registration_does_not_hide_unrelated_defaults():
    SandboxFactory.reset()
    try:
        SandboxFactory.register(IsolationLevel.WASM, NoopSandboxEngine)
        assert (
            SandboxFactory.create(IsolationLevel.PROCESS).isolation_level == IsolationLevel.PROCESS
        )
    finally:
        SandboxFactory.reset()


@pytest.mark.asyncio
async def test_replaced_owner_cleanup_failure_is_retained_for_retry():
    class FailingOwner(NoopSandboxEngine):
        failed = True
        calls = 0

        async def cleanup(self):
            self.calls += 1
            if self.failed:
                raise RuntimeError("owned cleanup failed")
            await super().cleanup()

    SandboxFactory.reset()
    try:
        SandboxFactory.register(IsolationLevel.NONE, FailingOwner)
        owner = SandboxFactory.create(IsolationLevel.NONE)
        SandboxFactory.register(IsolationLevel.NONE, NoopSandboxEngine)
        SandboxFactory.create(IsolationLevel.NONE)
        with pytest.raises(ExceptionGroup):
            await SandboxFactory.cleanup_all()
        assert owner.calls == 1
        owner.failed = False
        await SandboxFactory.cleanup_all()
        assert owner.calls == 2
    finally:
        SandboxFactory.reset()
