"""Actual process-worker IPC and cancellation boundaries."""

import asyncio
import os
import time

import psutil
import pytest

from truthound.plugins.security.exceptions import SandboxError
from truthound.plugins.security.protocols import SecurityPolicy
from truthound.plugins.security.sandbox.engines.process import ProcessSandboxEngine


@pytest.mark.asyncio
async def test_child_return_value_cannot_execute_pickle_reducers_in_parent():
    engine = ProcessSandboxEngine()
    context = engine.create_sandbox("data-only-result", SecurityPolicy.development())
    # A reducer returning the process ID demonstrates parent-side execution
    # without writing files, launching commands, or changing external state.
    expression = "type('Payload', (), {'__reduce__': lambda self: (__import__('os').getpid, ())})()"
    with pytest.raises(SandboxError):
        result = await engine.execute(context, eval, expression)
        assert result != os.getpid(), "Untrusted child reducer executed in the parent"


@pytest.mark.asyncio
async def test_cancelled_execution_reaps_its_actual_owned_process():
    engine = ProcessSandboxEngine()
    context = engine.create_sandbox("cancel-owned", SecurityPolicy.development())
    task = asyncio.create_task(engine.execute(context, time.sleep, 10))
    try:
        async with asyncio.timeout(5):
            while context.process_id is None:
                await asyncio.sleep(0.01)
        process = psutil.Process(context.process_id)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not process.is_running()
        assert not context.is_alive()
    finally:
        await engine.cleanup()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_actual_worker_preserves_closed_scalar_container_values_and_bounds_output():
    from datetime import date

    engine = ProcessSandboxEngine(max_result_bytes=1024)
    policy = SecurityPolicy.development()
    context = engine.create_sandbox("values", policy)
    result = await engine.execute(
        context, eval, "(b'abc', {'date': __import__('datetime').date(2024, 1, 2)})"
    )
    assert result == (b"abc", {"date": date(2024, 1, 2)})
    assert not context.is_alive()
    assert not engine._processes and not engine._temp_dirs
    with pytest.raises(SandboxError):
        await engine.execute(engine.create_sandbox("bounded", policy), str, "x" * 2048)
    await engine.cleanup()


@pytest.mark.asyncio
async def test_foreign_context_cannot_select_another_engines_process():
    owner, other = ProcessSandboxEngine(), ProcessSandboxEngine()
    context = owner.create_sandbox("owned", SecurityPolicy.development())
    with pytest.raises(ValueError):
        await other.execute(context, abs, -1)
    with pytest.raises(ValueError):
        other.terminate(context)
    await owner.cleanup()
    await other.cleanup()


@pytest.mark.asyncio
async def test_actual_memory_exhaustion_is_a_resource_failure():
    from dataclasses import replace

    from truthound.plugins.security.exceptions import SandboxResourceError
    from truthound.plugins.security.protocols import ResourceLimits

    policy = replace(
        SecurityPolicy.development(),
        resource_limits=ResourceLimits(
            max_memory_mb=64,
            max_execution_time_sec=3,
            max_file_descriptors=100,
        ),
    )
    engine = ProcessSandboxEngine()
    try:
        with pytest.raises(SandboxResourceError) as failure:
            await engine.execute(
                engine.create_sandbox("memory", policy),
                eval,
                "(bytearray(128 * 1024 * 1024), __import__('time').sleep(1))[1]",
            )
        assert failure.value.resource_type == "memory"
        assert not engine._processes
    finally:
        await engine.cleanup()
