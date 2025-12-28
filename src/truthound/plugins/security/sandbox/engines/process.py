"""Process-based sandbox engine with resource limits.

This engine executes code in a separate subprocess with:
- Memory limits (via resource module on Linux/macOS)
- CPU time limits
- Timeout enforcement
- Module import restrictions
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import pickle
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
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
    SandboxSecurityViolation,
    SandboxError,
)

logger = logging.getLogger(__name__)


def _create_executor_script(
    func_module: str,
    func_name: str,
    data_path: Path,
    result_path: Path,
    max_memory_mb: int,
    max_cpu_time: int,
    blocked_modules: tuple[str, ...],
) -> str:
    """Create Python script for subprocess execution."""
    blocked_set = set(blocked_modules)
    return f'''
import sys
import pickle
import json

# Set resource limits (Linux/macOS only)
try:
    import resource
    # Memory limit
    memory_bytes = {max_memory_mb} * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    # CPU time limit
    resource.setrlimit(resource.RLIMIT_CPU, ({max_cpu_time}, {max_cpu_time}))
except (ImportError, ValueError):
    pass  # Windows or limit already exceeded

# Block dangerous modules
blocked = {blocked_set}

class ImportBlocker:
    def find_module(self, name, path=None):
        if name in blocked or any(name.startswith(m + ".") for m in blocked):
            return self
        return None

    def load_module(self, name):
        raise ImportError(f"Module '{{name}}' is blocked in sandbox")

sys.meta_path.insert(0, ImportBlocker())

try:
    # Load function and data
    with open("{data_path}", "rb") as f:
        data = pickle.load(f)

    func = data["func"]
    args = data["args"]
    kwargs = data["kwargs"]

    # Execute
    result = func(*args, **kwargs)

    # Save result
    with open("{result_path}", "wb") as f:
        pickle.dump({{"success": True, "result": result}}, f)

except MemoryError as e:
    with open("{result_path}", "wb") as f:
        pickle.dump({{"success": False, "error": "memory_limit", "message": str(e)}}, f)
except Exception as e:
    with open("{result_path}", "wb") as f:
        pickle.dump({{"success": False, "error": type(e).__name__, "message": str(e)}}, f)
'''


class ProcessSandboxEngine:
    """Process-based sandbox engine with resource limits.

    Executes code in a separate subprocess with memory and CPU limits
    enforced via the resource module (Linux/macOS).

    Features:
        - Memory limiting
        - CPU time limiting
        - Module import blocking
        - Timeout enforcement
        - Process isolation
    """

    @property
    def isolation_level(self) -> IsolationLevel:
        """Return the isolation level provided by this engine."""
        return IsolationLevel.PROCESS

    def __init__(self) -> None:
        """Initialize the process sandbox engine."""
        self._contexts: dict[str, SandboxContextImpl] = {}
        self._temp_dirs: list[Path] = []

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
        logger.debug(f"Created process sandbox context for {plugin_id}")
        return context

    async def execute(
        self,
        context: SandboxContext,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute function in isolated subprocess.

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
            SandboxSecurityViolation: If blocked module imported
        """
        impl = self._get_impl(context)
        impl.mark_started()
        policy = context.policy
        limits = policy.resource_limits

        # Create temporary directory for IPC
        temp_dir = Path(tempfile.mkdtemp(prefix="truthound_sandbox_"))
        self._temp_dirs.append(temp_dir)

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
            script = _create_executor_script(
                func_module=getattr(func, "__module__", "__main__"),
                func_name=getattr(func, "__name__", "func"),
                data_path=data_path,
                result_path=result_path,
                max_memory_mb=limits.max_memory_mb,
                max_cpu_time=int(limits.max_execution_time_sec),
                blocked_modules=policy.blocked_modules,
            )
            with open(script_path, "w") as f:
                f.write(script)

            # Execute in subprocess
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(temp_dir),
            )
            impl.set_process_id(process.pid or 0)

            try:
                # Wait with timeout (add grace period)
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=limits.max_execution_time_sec + 5,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                impl.mark_terminated()
                raise SandboxTimeoutError(
                    f"Execution timed out after {limits.max_execution_time_sec}s",
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
                    f"Subprocess produced no result. stderr: {stderr_text}",
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

            if error_type == "memory_limit":
                raise SandboxResourceError(
                    f"Memory limit exceeded: {error_msg}",
                    plugin_id=context.plugin_id,
                    sandbox_id=context.sandbox_id,
                    resource_type="memory",
                    limit=limits.max_memory_mb,
                )
            elif error_type == "ImportError":
                raise SandboxSecurityViolation(
                    f"Blocked import: {error_msg}",
                    plugin_id=context.plugin_id,
                    sandbox_id=context.sandbox_id,
                    violation_type="import",
                    attempted_action=error_msg,
                )
            else:
                raise SandboxError(
                    f"Execution error ({error_type}): {error_msg}",
                    plugin_id=context.plugin_id,
                    sandbox_id=context.sandbox_id,
                )

        finally:
            # Cleanup temp directory
            self._cleanup_temp_dir(temp_dir)

    def terminate(self, context: SandboxContext) -> None:
        """Terminate sandbox subprocess.

        Args:
            context: Sandbox to terminate
        """
        impl = self._get_impl(context)
        impl.mark_terminated()

        # Kill process if running
        pid = impl.process_id
        if pid:
            try:
                os.kill(pid, 9)  # SIGKILL
            except (OSError, ProcessLookupError):
                pass  # Process already dead

        self._contexts.pop(context.sandbox_id, None)

    async def cleanup(self) -> None:
        """Clean up all sandbox resources."""
        # Terminate all contexts
        for context in list(self._contexts.values()):
            self.terminate(context)
        self._contexts.clear()

        # Clean up temp directories
        import shutil
        for temp_dir in self._temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        self._temp_dirs.clear()

    def _cleanup_temp_dir(self, temp_dir: Path) -> None:
        """Clean up a single temp directory."""
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        if temp_dir in self._temp_dirs:
            self._temp_dirs.remove(temp_dir)

    def _get_impl(self, context: SandboxContext) -> SandboxContextImpl:
        """Get implementation from context."""
        if isinstance(context, SandboxContextImpl):
            return context
        impl = self._contexts.get(context.sandbox_id)
        if impl is None:
            raise ValueError(f"Unknown sandbox context: {context.sandbox_id}")
        return impl
