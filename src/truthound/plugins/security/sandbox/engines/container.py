"""Container-based sandbox engine using Docker/Podman.

This engine provides maximum isolation by running code in containers:
- Complete filesystem isolation
- Network isolation
- Resource limits via cgroups
- Seccomp profiles for syscall filtering
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from truthound.plugins.security.protocols import (
    IsolationLevel,
    SecurityPolicy,
    SandboxContext,
)
from truthound.plugins.security.sandbox.context import SandboxContextImpl
from truthound.plugins.security.exceptions import (
    SandboxTimeoutError,
    SandboxResourceError,
    SandboxError,
)

logger = logging.getLogger(__name__)


class ContainerSandboxEngine:
    """Container-based sandbox engine using Docker or Podman.

    Provides the strongest isolation by running code in containers.
    Automatically detects and uses Docker or Podman.

    Features:
        - Complete filesystem isolation
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
    ) -> None:
        """Initialize the container sandbox engine.

        Args:
            image: Default Docker image to use
            runtime: Container runtime ("docker" or "podman"), auto-detected if None
        """
        self._image = image
        self._runtime = runtime
        self._contexts: dict[str, SandboxContextImpl] = {}
        self._container_ids: list[str] = []

    async def _detect_runtime(self) -> str:
        """Detect available container runtime."""
        if self._runtime:
            return self._runtime

        # Try Docker first, then Podman
        for runtime in ["docker", "podman"]:
            try:
                process = await asyncio.create_subprocess_exec(
                    runtime, "version",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await process.communicate()
                if process.returncode == 0:
                    self._runtime = runtime
                    return runtime
            except FileNotFoundError:
                continue

        raise SandboxError(
            "No container runtime available. Install Docker or Podman."
        )

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
        context = SandboxContextImpl(
            plugin_id=plugin_id,
            policy=policy,
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
        impl.mark_started()
        policy = context.policy
        limits = policy.resource_limits

        # Detect runtime
        runtime = await self._detect_runtime()

        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp(prefix="truthound_container_"))
        data_path = temp_dir / "data.pkl"
        result_path = temp_dir / "result.pkl"
        script_path = temp_dir / "executor.py"

        try:
            # Serialize function and data
            with open(data_path, "wb") as f:
                pickle.dump({
                    "func": func,
                    "args": args,
                    "kwargs": kwargs,
                }, f)

            # Create executor script
            script = self._create_executor_script()
            with open(script_path, "w") as f:
                f.write(script)

            # Build container command
            container_cmd = self._build_container_command(
                runtime=runtime,
                policy=policy,
                temp_dir=temp_dir,
            )

            # Execute container
            process = await asyncio.create_subprocess_exec(
                *container_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                # Wait with timeout
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=limits.max_execution_time_sec + 10,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                impl.mark_terminated()
                raise SandboxTimeoutError(
                    f"Container timed out after {limits.max_execution_time_sec}s",
                    plugin_id=context.plugin_id,
                    sandbox_id=context.sandbox_id,
                    timeout_seconds=limits.max_execution_time_sec,
                    execution_time=impl.execution_time_sec,
                )

            impl.mark_finished()

            # Check for result
            if not result_path.exists():
                stderr_text = stderr.decode() if stderr else "No output"
                raise SandboxError(
                    f"Container produced no result. stderr: {stderr_text}",
                    plugin_id=context.plugin_id,
                    sandbox_id=context.sandbox_id,
                )

            # Load result
            with open(result_path, "rb") as f:
                result_data = pickle.load(f)

            if result_data.get("success"):
                return result_data.get("result")

            # Handle errors
            error_type = result_data.get("error", "UnknownError")
            error_msg = result_data.get("message", "Unknown error")

            if error_type == "MemoryError":
                raise SandboxResourceError(
                    f"Container memory limit exceeded: {error_msg}",
                    plugin_id=context.plugin_id,
                    sandbox_id=context.sandbox_id,
                    resource_type="memory",
                    limit=limits.max_memory_mb,
                )
            else:
                raise SandboxError(
                    f"Container error ({error_type}): {error_msg}",
                    plugin_id=context.plugin_id,
                    sandbox_id=context.sandbox_id,
                )

        finally:
            # Cleanup temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _build_container_command(
        self,
        runtime: str,
        policy: SecurityPolicy,
        temp_dir: Path,
    ) -> list[str]:
        """Build container run command."""
        limits = policy.resource_limits

        cmd = [
            runtime, "run",
            "--rm",  # Auto-remove container
            f"--memory={limits.max_memory_mb}m",
            f"--cpus={limits.max_cpu_percent / 100}",
            f"--network={'bridge' if policy.allow_network else 'none'}",
            "-v", f"{temp_dir}:/workspace:rw",
            "-w", "/workspace",
        ]

        # Add read-only root filesystem if not allowing file writes
        if not policy.allow_file_write:
            cmd.extend([
                "--read-only",
                "--tmpfs", "/tmp:rw,noexec,nosuid",
            ])

        # Add security options
        cmd.extend([
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",
        ])

        # Add image and command
        cmd.extend([
            self._image,
            "python", "/workspace/executor.py",
        ])

        return cmd

    def _create_executor_script(self) -> str:
        """Create Python script for container execution."""
        return '''
import sys
import pickle

try:
    # Load data
    with open("/workspace/data.pkl", "rb") as f:
        data = pickle.load(f)

    func = data["func"]
    args = data["args"]
    kwargs = data["kwargs"]

    # Execute
    result = func(*args, **kwargs)

    # Save result
    with open("/workspace/result.pkl", "wb") as f:
        pickle.dump({"success": True, "result": result}, f)

except MemoryError as e:
    with open("/workspace/result.pkl", "wb") as f:
        pickle.dump({"success": False, "error": "MemoryError", "message": str(e)}, f)
except Exception as e:
    with open("/workspace/result.pkl", "wb") as f:
        pickle.dump({"success": False, "error": type(e).__name__, "message": str(e)}, f)
'''

    def terminate(self, context: SandboxContext) -> None:
        """Terminate sandbox container.

        Args:
            context: Sandbox to terminate
        """
        impl = self._get_impl(context)
        impl.mark_terminated()

        # Kill container if running
        container_id = impl.container_id
        if container_id and self._runtime:
            try:
                asyncio.create_task(self._kill_container(container_id))
            except Exception:
                pass

        self._contexts.pop(context.sandbox_id, None)

    async def _kill_container(self, container_id: str) -> None:
        """Kill a container by ID."""
        if not self._runtime:
            return
        try:
            process = await asyncio.create_subprocess_exec(
                self._runtime, "rm", "-f", container_id,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await process.wait()
        except Exception as e:
            logger.warning(f"Failed to kill container {container_id}: {e}")

    async def cleanup(self) -> None:
        """Clean up all sandbox resources."""
        # Terminate all contexts
        for context in list(self._contexts.values()):
            self.terminate(context)
        self._contexts.clear()

        # Kill any remaining containers
        for container_id in self._container_ids:
            await self._kill_container(container_id)
        self._container_ids.clear()

    def _get_impl(self, context: SandboxContext) -> SandboxContextImpl:
        """Get implementation from context."""
        if isinstance(context, SandboxContextImpl):
            return context
        impl = self._contexts.get(context.sandbox_id)
        if impl is None:
            raise ValueError(f"Unknown sandbox context: {context.sandbox_id}")
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
