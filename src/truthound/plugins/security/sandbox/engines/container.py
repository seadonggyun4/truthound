"""Container-based sandbox engine using Docker/Podman.

Selected container restrictions and owned lifecycle with data-only results.
Selected syscall denials apply inside Linux workers. Explicit host path grants
are mounted separately from the image filesystem and owned IPC workspace.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import pickle
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from collections.abc import Callable

from truthound.plugins.security.exceptions import (
    SandboxError,
    SandboxResourceError,
    SandboxTimeoutError,
)
from truthound.plugins.security.protocols import (
    IsolationLevel,
    SandboxContext,
    SecurityPolicy,
)
from truthound.plugins.security.sandbox._ipc import read_result
from truthound.plugins.security.sandbox._linux_syscalls import SUPPORTED
from truthound.plugins.security.sandbox.context import SandboxContextImpl

logger = logging.getLogger(__name__)


class ContainerSandboxEngine:
    """Container-based sandbox engine using Docker or Podman.

    Runs selected trusted callables under explicit container restrictions.
    Automatically detects and uses Docker or Podman.

    Features:
        - Separate root filesystem with an owned writable IPC mount
        - Network isolation (configurable)
        - Memory and CPU limits via cgroups
        - Read-only filesystem option
        - Configurable image
    """

    DEFAULT_IMAGE = "python:3.11-slim"

    @property
    def isolation_level(self) -> IsolationLevel:
        """Return the isolation level provided by this engine."""
        return IsolationLevel.CONTAINER

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        runtime: str | None = None,
        *,
        max_result_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        """Initialize the container sandbox engine.

        Args:
            image: Default Docker image to use
            runtime: Container runtime ("docker" or "podman"), auto-detected if None
        """
        if type(max_result_bytes) is not int or max_result_bytes < 256:
            raise ValueError("max_result_bytes must be an integer of at least 256")
        self.max_result_bytes = max_result_bytes
        self._image = image
        self._runtime = runtime
        self._contexts: dict[str, SandboxContextImpl] = {}
        self._container_ids: list[str] = []
        self._tasks = {}
        self._cleanup_tasks = set()
        self._temp_dirs = set()
        self._pending = {}

    async def _detect_runtime(self) -> str:
        """Detect available container runtime."""
        if self._runtime:
            return self._runtime

        # Try Docker first, then Podman
        for runtime in ["docker", "podman"]:
            try:
                process = await asyncio.create_subprocess_exec(
                    runtime,
                    "version",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await process.communicate()
                if process.returncode == 0:
                    self._runtime = runtime
                    return runtime
            except FileNotFoundError:
                continue

        raise SandboxError("No container runtime available. Install Docker or Podman.")

    @staticmethod
    def _host_mounts(policy):
        limits = policy.resource_limits
        if limits.writable_paths and not policy.allow_file_write:
            raise ValueError("Writable host paths require allow_file_write")
        mounts = {}
        for paths, writable in ((limits.allowed_paths, False), (limits.writable_paths, True)):
            if not isinstance(paths, (tuple, list)):
                raise ValueError("Host paths must be a sequence of absolute paths")
            for value in paths:
                if not isinstance(value, str) or any(char in value for char in (",", "\0", "\n")):
                    raise ValueError("Invalid host mount path")
                path = Path(value)
                if not path.is_absolute() or str(path.resolve()) != str(path):
                    raise ValueError("Host grants require canonical absolute paths")
                if path == Path("/") or any(
                    path == root or root in path.parents
                    for root in (Path("/workspace"), Path("/proc"), Path("/sys"), Path("/dev"))
                ):
                    raise ValueError("Reserved host mount destination")
                if not path.is_file() and not path.is_dir():
                    raise ValueError("Host grants require an existing regular file or directory")
                mounts[str(path)] = mounts.get(str(path), False) or writable
        return sorted(mounts.items())

    def create_sandbox(
        self,
        plugin_id: str,
        policy: SecurityPolicy,
    ) -> SandboxContext:
        """Create a sandbox context.

        Args:
            plugin_id: Plugin identifier
            policy: Security policy to apply

        Returns:
            SandboxContext for execution
        """
        for name in ("allow_network", "allow_subprocess", "allow_file_write"):
            if type(getattr(policy, name)) is not bool:
                raise ValueError("Container policy flags must be boolean")
        self._host_mounts(policy)
        limits = policy.resource_limits
        if set(limits.denied_syscalls) - SUPPORTED:
            raise ValueError("Unsupported container syscall policy")
        for value in (limits.max_memory_mb, limits.max_file_descriptors):
            if type(value) is not int or value < 0:
                raise ValueError("Memory and descriptor limits must be nonnegative integers")
        for value in (limits.max_execution_time_sec, limits.max_cpu_percent):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("Time and CPU limits must be finite and positive")
        context = SandboxContextImpl(
            plugin_id=plugin_id,
            policy=deepcopy(policy),
        )
        self._contexts[context.sandbox_id] = context
        logger.debug(f"Created container sandbox context for {plugin_id}")
        return context

    async def execute(
        self,
        context: SandboxContext,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute function in container.

        Args:
            context: Sandbox context
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            SandboxTimeoutError: If execution times out
            SandboxResourceError: If resource limits exceeded
            SandboxError: If container execution fails
        """
        impl = self._get_impl(context)
        if impl.started_at is not None or not impl.is_alive():
            raise ValueError("Sandbox context is single-use")
        policy = context.policy
        limits = policy.resource_limits

        # Detect runtime
        runtime = await self._detect_runtime()

        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp(prefix="truthound_container_"))
        self._temp_dirs.add(temp_dir)
        impl.mark_started()
        self._tasks[context.sandbox_id] = asyncio.current_task()
        data_path = temp_dir / "data.pkl"
        result_path = temp_dir / "result.json"
        script_path = temp_dir / "executor.py"

        process = None
        creation = None
        startup = None
        container_id = None
        token = uuid4().hex
        owned_name = "truthound-worker-" + token
        try:
            # Serialize function and data
            with open(data_path, "wb") as f:
                pickle.dump(
                    {
                        "func": func,
                        "args": args,
                        "kwargs": kwargs,
                    },
                    f,
                )

            shutil.copyfile(
                Path(__file__).resolve().parents[4] / "ml" / "serialization.py",
                temp_dir / "values.py",
            )

            shutil.copyfile(
                Path(__file__).resolve().parent.parent / "_linux_syscalls.py",
                temp_dir / "syscalls.py",
            )

            # Create executor script
            script = self._create_executor_script(policy)
            with open(script_path, "w") as f:
                f.write(script)

            # Build container command
            container_cmd = self._build_container_command(
                runtime=runtime,
                policy=policy,
                temp_dir=temp_dir,
            )

            container_cmd[2:2] = ["--name", owned_name, "--label", "org.truthound.worker=" + token]
            self._pending[owned_name] = token
            creation = asyncio.create_task(self._create_container(container_cmd))
            container_id = await asyncio.shield(creation)
            impl.set_container_id(container_id)
            startup = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    runtime,
                    "start",
                    "--attach",
                    container_id,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
            )
            process = await asyncio.shield(startup)

            try:
                # Wait with timeout
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=limits.max_execution_time_sec,
                )
            except TimeoutError:
                process.kill()
                await process.wait()
                impl.mark_terminated()
                raise SandboxTimeoutError(
                    f"Container timed out after {limits.max_execution_time_sec}s",
                    plugin_id=context.plugin_id,
                    sandbox_id=context.sandbox_id,
                    timeout_seconds=limits.max_execution_time_sec,
                    execution_time=impl.execution_time_sec,
                ) from None

            impl.mark_finished()

            if process.returncode == 137:
                raise SandboxResourceError(
                    "Container was killed; memory exhaustion is possible",
                    resource_type="container",
                    limit=limits.max_memory_mb,
                )
            if process.returncode != 0:
                raise SandboxError("Container worker failed", plugin_id=context.plugin_id)
            return read_result(
                result_path,
                self.max_result_bytes,
                plugin_id=context.plugin_id,
                sandbox_id=context.sandbox_id,
                memory_limit=limits.max_memory_mb,
            )

        finally:

            async def release():
                owned_id = container_id
                if creation is not None and owned_id is None:
                    try:
                        owned_id = await creation
                    except Exception:
                        owned_id = await self._reconcile_creation(owned_name, token)
                if owned_id is not None:
                    await self._kill_container(owned_id)
                self._pending.pop(owned_name, None)
                attached = process
                if attached is None and startup is not None:
                    attached = await startup
                if attached is not None and attached.returncode is None:
                    attached.kill()
                    await attached.wait()
                shutil.rmtree(temp_dir)
                self._temp_dirs.discard(temp_dir)
                impl.mark_finished()
                self._tasks.pop(context.sandbox_id, None)

            cleanup = asyncio.create_task(release())
            self._cleanup_tasks.add(cleanup)
            await asyncio.shield(cleanup)
            self._cleanup_tasks.discard(cleanup)

    async def _reconcile_creation(self, name, token):
        process = await asyncio.create_subprocess_exec(
            self._runtime,
            "container",
            "ls",
            "--all",
            "--no-trunc",
            "--filter",
            "name=^/" + name + "$",
            "--filter",
            "label=org.truthound.worker=" + token,
            "--format",
            "{{.ID}}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        output, _ = await process.communicate()
        if process.returncode:
            raise SandboxError("Container creation outcome remains unknown")
        identity = output.decode("ascii", errors="replace").strip()
        if not identity:
            return None
        if len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
            raise SandboxError("Invalid owned container identity")
        if identity not in self._container_ids:
            self._container_ids.append(identity)
        return identity

    async def _create_container(self, command):
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        output, _ = await process.communicate()
        identity = output.decode("ascii", errors="replace").strip()
        if (
            process.returncode
            or len(identity) != 64
            or any(c not in "0123456789abcdef" for c in identity)
        ):
            raise SandboxError("Container creation failed")
        self._container_ids.append(identity)
        return identity

    def _build_container_command(
        self,
        runtime: str,
        policy: SecurityPolicy,
        temp_dir: Path,
    ) -> list[str]:
        """Build container run command."""
        limits = policy.resource_limits

        cmd = [
            runtime,
            "create",
            "--pull=never",
            f"--cpus={limits.max_cpu_percent / 100}",
            f"--network={'bridge' if policy.allow_network else 'none'}",
            "-v",
            f"{temp_dir}:/workspace:rw",
            "-w",
            "/workspace",
        ]

        # With all capabilities dropped, container root cannot bypass the
        # host owner's private IPC mode on native Linux bind mounts.
        if hasattr(os, "getuid") and hasattr(os, "getgid"):
            cmd.extend(["--user", f"{os.getuid()}:{os.getgid()}"])

        if limits.max_memory_mb:
            cmd.append(f"--memory={limits.max_memory_mb}m")
        if limits.max_file_descriptors:
            cmd.append(
                f"--ulimit=nofile={limits.max_file_descriptors}:{limits.max_file_descriptors}"
            )

        for path, writable in self._host_mounts(policy):
            mount = f"type=bind,source={path},target={path}"
            if not writable:
                mount += ",readonly"
            cmd.extend(["--mount", mount])

        # Add read-only root filesystem if not allowing file writes
        if not policy.allow_file_write:
            cmd.extend(
                [
                    "--read-only",
                    "--tmpfs",
                    "/tmp:rw,noexec,nosuid",
                ]
            )

        # Add security options
        cmd.extend(
            [
                "--security-opt",
                "no-new-privileges",
                "--cap-drop",
                "ALL",
            ]
        )

        # Add image and command
        cmd.extend(
            [
                self._image,
                "python",
                "/workspace/executor.py",
            ]
        )

        return cmd

    def _create_executor_script(self, policy) -> str:
        """Create Python script for container execution."""
        return f"""
import asyncio
import importlib.util
import json
import pickle
import resource

spec = importlib.util.spec_from_file_location("_truthound_values", "/workspace/values.py")
codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)

def save(value):
    content = json.dumps(value, allow_nan=False).encode("utf-8")
    if len(content) > {self.max_result_bytes}:
        raise ValueError("Result exceeds IPC budget")
    with open("/workspace/result.json", "wb") as stream:
        stream.write(content)

stage = "limits"
try:
    resource.setrlimit(resource.RLIMIT_FSIZE, ({self.max_result_bytes},) * 2)
    syscall_spec = importlib.util.spec_from_file_location("_truthound_syscalls", "/workspace/syscalls.py")
    syscalls = importlib.util.module_from_spec(syscall_spec)
    syscall_spec.loader.exec_module(syscalls)
    syscalls.install({tuple(policy.resource_limits.denied_syscalls)!r}, {policy.allow_subprocess!r})
    with open("/workspace/data.pkl", "rb") as stream:
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

    def terminate(self, context: SandboxContext) -> None:
        impl = self._get_impl(context)
        impl.mark_terminated()
        task = self._tasks.get(context.sandbox_id)
        if task is not None:
            task.cancel()

    async def _kill_container(self, container_id: str) -> None:
        if container_id not in self._container_ids:
            raise ValueError("Container is not owned by this engine")
        process = await asyncio.create_subprocess_exec(
            self._runtime,
            "rm",
            "--force",
            container_id,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await process.wait()
        if process.returncode:
            raise SandboxError("Owned container removal failed")
        self._container_ids.remove(container_id)

    async def cleanup(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        results = await asyncio.gather(*self._cleanup_tasks, return_exceptions=True)
        self._cleanup_tasks.clear()
        failures = [result for result in results if isinstance(result, Exception)]
        for name, token in list(self._pending.items()):
            try:
                await self._reconcile_creation(name, token)
            except Exception as error:
                failures.append(error)
            else:
                self._pending.pop(name, None)
        for identity in list(self._container_ids):
            try:
                await self._kill_container(identity)
            except Exception as error:
                failures.append(error)
        if failures:
            raise ExceptionGroup("Container cleanup failed", failures)
        for directory in list(self._temp_dirs):
            shutil.rmtree(directory)
            self._temp_dirs.remove(directory)
        self._tasks.clear()
        self._contexts.clear()

    def _get_impl(self, context: SandboxContext) -> SandboxContextImpl:
        impl = self._contexts.get(context.sandbox_id)
        if impl is not context:
            raise ValueError("Sandbox context is not owned by this engine")
        return impl

    async def check_available(self) -> bool:
        """Check if container runtime is available.

        Returns:
            True if Docker or Podman is available
        """
        try:
            await self._detect_runtime()
            return True
        except SandboxError:
            return False
