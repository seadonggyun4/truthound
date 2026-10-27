"""Trusted thread work must be retained until actual completion."""

import asyncio
import threading
from dataclasses import replace

import pytest

from truthound.plugins.security.exceptions import SandboxTimeoutError
from truthound.plugins.security.protocols import SecurityPolicy
from truthound.plugins.security.sandbox.engines.noop import NoopSandboxEngine


@pytest.mark.asyncio
async def test_cleanup_waits_for_timed_out_native_thread():
    engine = NoopSandboxEngine()
    policy = SecurityPolicy.development()
    policy = replace(
        policy, resource_limits=replace(policy.resource_limits, max_execution_time_sec=0.05)
    )
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def work():
        started.set()
        release.wait(5)
        finished.set()

    cleanup = None
    try:
        with pytest.raises(SandboxTimeoutError):
            await engine.execute(engine.create_sandbox("thread", policy), work)
        assert started.is_set() and not finished.is_set()
        cleanup = asyncio.create_task(engine.cleanup())
        await asyncio.sleep(0.05)
        assert not cleanup.done(), "Cleanup claimed completion while its worker was running"
        release.set()
        await cleanup
        assert finished.is_set()
    finally:
        release.set()
        if cleanup is not None:
            await cleanup
        await engine.cleanup()


@pytest.mark.asyncio
async def test_operation_failure_finishes_owned_context_and_rejects_foreign_execution():
    engine, other = NoopSandboxEngine(), NoopSandboxEngine()
    context = engine.create_sandbox("failure", SecurityPolicy.development())

    def fail():
        raise ValueError("selected failure")

    with pytest.raises(ValueError, match="selected failure"):
        await engine.execute(context, fail)
    assert not context.is_alive()
    foreign = engine.create_sandbox("foreign", SecurityPolicy.development())
    with pytest.raises(ValueError):
        await other.execute(foreign, abs, -1)
    await engine.cleanup()
    await other.cleanup()
