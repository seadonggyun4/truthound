"""Owned process execution with bounded data-only results and native limits.

Import filtering is advisory Python behavior, not an OS security boundary.
"""

from __future__ import annotations

import asyncio
import math
import os
import pickle
import shutil
import sys
import tempfile
from contextlib import suppress
from pathlib import Path

import psutil

from truthound.plugins.security.exceptions import (
    SandboxResourceError,
    SandboxTimeoutError,
)
from truthound.plugins.security.protocols import IsolationLevel
from truthound.plugins.security.sandbox._ipc import read_result
from truthound.plugins.security.sandbox.context import SandboxContextImpl


def _create_executor_script(
    func_module,
    func_name,
    data_path,
    result_path,
    max_memory_mb,
    max_cpu_time,
    blocked_modules,
    max_file_descriptors=100,
    max_result_bytes=16 * 1024 * 1024,
):
    codec = Path(__file__).resolve().parents[4] / "ml" / "serialization.py"
    return f"""
import asyncio
import importlib.util
import json
import pickle
import sys

spec = importlib.util.spec_from_file_location("_truthound_values", {str(codec)!r})
codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)

def save(value):
    content = json.dumps(value, allow_nan=False).encode("utf-8")
    if len(content) > {max_result_bytes}:
        raise ValueError("Result exceeds IPC budget")
    with open({str(result_path)!r}, "wb") as stream:
        stream.write(content)

stage = "limits"
try:
    try:
        import resource
    except ImportError:
        if {max_file_descriptors}:
            raise RuntimeError("Native resource limits unavailable")
    else:
        if {max_memory_mb}:
            try:
                resource.setrlimit(resource.RLIMIT_AS, ({max_memory_mb} * 1024 * 1024,) * 2)
            except (ValueError, OSError):
                pass  # Parent RSS monitor remains active on platforms without RLIMIT_AS.
        resource.setrlimit(resource.RLIMIT_CPU, ({max_cpu_time},) * 2)
        if {max_file_descriptors}:
            resource.setrlimit(resource.RLIMIT_NOFILE, ({max_file_descriptors},) * 2)
        resource.setrlimit(resource.RLIMIT_FSIZE, ({max_result_bytes},) * 2)
    blocked = {tuple(blocked_modules)!r}
    class ImportBlocker:
        def find_spec(self, fullname, path=None, target=None):
            if any(fullname == item or fullname.startswith(item + ".") for item in blocked):
                raise ImportError("Import blocked by selected policy")
            return None
    sys.meta_path.insert(0, ImportBlocker())
    with open({str(data_path)!r}, "rb") as stream:
        data = pickle.load(stream)
    stage = "operation"
    result = data["func"](*data["args"], **data["kwargs"])
    if hasattr(result, "__await__"):
        async def finish():
            return await result
        result = asyncio.run(finish())
    stage = "result"
    save({{"success": True, "result": codec.encode(result)}})
except BaseException as error:
    save({{"success": False, "error": type(error).__name__, "stage": stage}})
"""


class ProcessSandboxEngine:
    """Run trusted selected callables in owned workers, never unpickle results.

    Arguments and callable selection are trusted local Python inputs. Returned
    values use Core's closed scalar/container codec. This is not hostile-code
    filesystem/network isolation; use an independently configured OS sandbox.
    """

    isolation_level = IsolationLevel.PROCESS

    def __init__(self, *, max_result_bytes=16 * 1024 * 1024):
        if type(max_result_bytes) is not int or max_result_bytes < 256:
            raise ValueError("max_result_bytes must be an integer of at least 256")
        self.max_result_bytes = max_result_bytes
        self._contexts = {}
        self._temp_dirs = []
        self._processes = {}
        self._tasks = {}
        self._spawns = {}
        self._monitors = {}

    def create_sandbox(self, plugin_id, policy):
        limits = policy.resource_limits
        duration = limits.max_execution_time_sec
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise ValueError("Execution time must be finite and positive")
        for value in (limits.max_memory_mb, limits.max_file_descriptors):
            if type(value) is not int or value < 0:
                raise ValueError("Native resource limits must be nonnegative integers")
        if any(not isinstance(name, str) or not name for name in policy.blocked_modules):
            raise ValueError("Blocked modules must have nonempty names")
        context = SandboxContextImpl(plugin_id=plugin_id, policy=policy)
        self._contexts[context.sandbox_id] = context
        return context

    async def execute(self, context, func, *args, **kwargs):
        impl = self._get_impl(context)
        if impl.started_at is not None or not impl.is_alive():
            raise ValueError("Create a fresh context for each execution")
        if not callable(func):
            raise TypeError("Selected operation must be callable")
        impl.mark_started()
        identity = impl.sandbox_id
        self._tasks[identity] = asyncio.current_task()
        directory = Path(tempfile.mkdtemp(prefix="truthound_sandbox_"))
        self._temp_dirs.append(directory)
        process = None
        completed = False
        monitoring = None
        memory_exceeded = []
        limits = context.policy.resource_limits
        try:
            data_path, result_path = directory / "input.pkl", directory / "result.json"
            with data_path.open("wb") as stream:
                pickle.dump({"func": func, "args": args, "kwargs": kwargs}, stream)
            script = directory / "executor.py"
            script.write_text(
                _create_executor_script(
                    getattr(func, "__module__", ""),
                    getattr(func, "__name__", ""),
                    data_path,
                    result_path,
                    limits.max_memory_mb,
                    max(1, math.ceil(limits.max_execution_time_sec)),
                    context.policy.blocked_modules,
                    limits.max_file_descriptors,
                    self.max_result_bytes,
                )
            )
            environment = {
                key: value
                for key, value in os.environ.items()
                if key
                in {
                    "PATH",
                    "SYSTEMROOT",
                    "WINDIR",
                    "TEMP",
                    "TMP",
                    "PROCESSOR_ARCHITECTURE",
                    "PROCESSOR_ARCHITEW6432",
                }
            }
            environment.update(
                POLARS_MAX_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
            )
            spawning = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    sys.executable,
                    "-I",
                    str(script),
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                    cwd=str(directory),
                    env=environment,
                )
            )
            self._spawns[identity] = spawning
            try:
                process = await asyncio.shield(spawning)
            except asyncio.CancelledError:
                process = await asyncio.shield(spawning)
                raise
            self._processes[identity] = process
            impl.set_process_id(process.pid)

            async def observe_memory():
                if not limits.max_memory_mb:
                    return
                try:
                    owned = psutil.Process(process.pid)
                    created = owned.create_time()
                    while process.returncode is None:
                        if owned.create_time() != created:
                            return
                        used = owned.memory_info().rss / (1024 * 1024)
                        impl.update_resource_usage(memory_mb=used)
                        if used > limits.max_memory_mb:
                            memory_exceeded.append(used)
                            process.kill()
                            return
                        await asyncio.sleep(0.01)
                except psutil.NoSuchProcess:
                    return

            monitoring = asyncio.create_task(observe_memory())
            self._monitors[identity] = monitoring
            try:
                await asyncio.wait_for(process.wait(), timeout=limits.max_execution_time_sec)
            except TimeoutError:
                raise SandboxTimeoutError(
                    "Execution deadline exceeded",
                    plugin_id=context.plugin_id,
                    sandbox_id=identity,
                    timeout_seconds=limits.max_execution_time_sec,
                ) from None
            await monitoring
            if memory_exceeded:
                raise SandboxResourceError(
                    "Worker exceeded memory budget",
                    plugin_id=context.plugin_id,
                    sandbox_id=identity,
                    resource_type="memory",
                    limit=limits.max_memory_mb,
                    actual=memory_exceeded[0],
                )
            if process.returncode != 0 or not result_path.exists():
                raise SandboxResourceError(
                    "Worker exited without a complete result",
                    plugin_id=context.plugin_id,
                    sandbox_id=identity,
                    resource_type="worker",
                )
            value = read_result(
                result_path,
                self.max_result_bytes,
                plugin_id=context.plugin_id,
                sandbox_id=identity,
                memory_limit=limits.max_memory_mb,
            )
            completed = True
            return value
        finally:
            if process is None and identity in self._spawns:
                with suppress(Exception):
                    process = await asyncio.shield(self._spawns[identity])
            if process is not None:
                if process.returncode is None:
                    process.kill()
                await process.wait()
            if monitoring is not None:
                if not monitoring.done():
                    monitoring.cancel()
                await asyncio.gather(monitoring, return_exceptions=True)
            self._monitors.pop(identity, None)
            self._spawns.pop(identity, None)
            self._processes.pop(identity, None)
            self._tasks.pop(identity, None)
            if completed:
                impl.mark_finished()
            else:
                impl.mark_terminated()
            self._cleanup_temp_dir(directory)

    def terminate(self, context):
        impl = self._get_impl(context)
        process = self._processes.get(impl.sandbox_id)
        if process is not None and process.returncode is None:
            process.kill()
        elif impl.sandbox_id in self._tasks:
            self._tasks[impl.sandbox_id].cancel()
        else:
            impl.mark_terminated()

    async def cleanup(self):
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        errors = [error for error in outcomes if isinstance(error, Exception)]
        spawned = await asyncio.gather(*self._spawns.values(), return_exceptions=True)
        errors.extend(error for error in spawned if isinstance(error, Exception))
        processes = {
            *self._processes.values(),
            *(item for item in spawned if isinstance(item, asyncio.subprocess.Process)),
        }
        for process in processes:
            try:
                if process.returncode is None:
                    process.kill()
                await process.wait()
            except Exception as error:
                errors.append(error)
        for monitor in self._monitors.values():
            monitor.cancel()
        await asyncio.gather(*self._monitors.values(), return_exceptions=True)
        self._monitors.clear()
        self._spawns.clear()
        self._processes.clear()
        self._tasks.clear()
        for context in self._contexts.values():
            context.mark_terminated()
        for directory in list(self._temp_dirs):
            try:
                self._cleanup_temp_dir(directory)
            except Exception as error:
                errors.append(error)
        self._contexts.clear()
        if errors:
            raise ExceptionGroup("Process worker cleanup failed", errors)

    def _cleanup_temp_dir(self, directory):
        if directory.exists():
            shutil.rmtree(directory)
        if directory in self._temp_dirs:
            self._temp_dirs.remove(directory)

    def _get_impl(self, context):
        impl = self._contexts.get(context.sandbox_id)
        if impl is not context:
            raise ValueError("Context is not owned by this engine")
        return impl
