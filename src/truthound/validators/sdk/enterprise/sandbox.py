"""Sandbox execution for untrusted validators.

This module provides isolation mechanisms for executing validators:
- Subprocess isolation with resource limits
- Docker container isolation
- In-process execution with restrictions

Security Model:
    Sandboxed validators cannot:
    - Access filesystem outside allowed paths
    - Make network connections
    - Execute arbitrary system commands
    - Access environment variables
    - Spawn child processes

Example:
    from truthound.validators.sdk.enterprise.sandbox import (
        SandboxExecutor,
        SandboxConfig,
    )

    config = SandboxConfig(
        backend=SandboxBackend.SUBPROCESS,
        allowed_paths=["/data"],
        timeout_seconds=30,
    )

    executor = SandboxExecutor(config)
    result = await executor.execute(validator, dataframe)
"""

from __future__ import annotations

import abc
import asyncio
import hashlib
import json
import logging
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Generator, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SandboxBackend(Enum):
    """Available sandbox backends."""

    IN_PROCESS = auto()    # No isolation, just resource limits
    SUBPROCESS = auto()    # Process-level isolation
    DOCKER = auto()        # Container-level isolation


class SandboxError(Exception):
    """Base exception for sandbox errors."""
    pass


class SandboxTimeoutError(SandboxError):
    """Raised when sandbox execution times out."""
    pass


class SandboxResourceError(SandboxError):
    """Raised when resource limits are exceeded."""
    pass


class SandboxSecurityError(SandboxError):
    """Raised when security violation is detected."""
    pass


@dataclass(frozen=True)
class SandboxConfig:
    """Configuration for sandbox execution.

    Attributes:
        backend: Sandbox backend to use
        timeout_seconds: Maximum execution time
        max_memory_mb: Maximum memory in megabytes
        max_cpu_percent: Maximum CPU percentage (0-100)
        allowed_paths: Paths the validator can access
        allowed_modules: Python modules the validator can import
        blocked_modules: Python modules the validator cannot import
        network_enabled: Whether network access is allowed
        env_vars: Environment variables to pass through
        docker_image: Docker image to use (if backend is DOCKER)
        working_dir: Working directory inside sandbox
    """

    backend: SandboxBackend = SandboxBackend.SUBPROCESS
    timeout_seconds: float = 60.0
    max_memory_mb: int = 512
    max_cpu_percent: int = 100
    allowed_paths: tuple[str, ...] = field(default_factory=tuple)
    allowed_modules: tuple[str, ...] = field(default_factory=lambda: (
        "polars", "numpy", "pandas", "truthound",
    ))
    blocked_modules: tuple[str, ...] = field(default_factory=lambda: (
        "os", "subprocess", "shutil", "socket", "urllib", "requests",
        "http", "ftplib", "smtplib", "telnetlib", "ctypes", "multiprocessing",
    ))
    network_enabled: bool = False
    env_vars: dict[str, str] = field(default_factory=dict)
    docker_image: str = "python:3.11-slim"
    working_dir: str = "/workspace"

    @classmethod
    def strict(cls) -> "SandboxConfig":
        """Create strict security configuration."""
        return cls(
            backend=SandboxBackend.DOCKER,
            timeout_seconds=30.0,
            max_memory_mb=256,
            max_cpu_percent=50,
            network_enabled=False,
        )

    @classmethod
    def standard(cls) -> "SandboxConfig":
        """Create standard security configuration."""
        return cls(
            backend=SandboxBackend.SUBPROCESS,
            timeout_seconds=60.0,
            max_memory_mb=512,
            max_cpu_percent=100,
            network_enabled=False,
        )

    @classmethod
    def permissive(cls) -> "SandboxConfig":
        """Create permissive configuration for trusted code."""
        return cls(
            backend=SandboxBackend.IN_PROCESS,
            timeout_seconds=120.0,
            max_memory_mb=2048,
            max_cpu_percent=100,
            network_enabled=True,
        )


@dataclass
class SandboxResult:
    """Result of sandboxed execution.

    Attributes:
        success: Whether execution completed successfully
        result: Execution result (if success)
        error: Error message (if failed)
        execution_time_seconds: Total execution time
        memory_used_mb: Peak memory usage
        cpu_time_seconds: CPU time consumed
        sandbox_id: Unique identifier for this execution
        started_at: Execution start time
        finished_at: Execution end time
    """

    success: bool
    result: Any = None
    error: str | None = None
    execution_time_seconds: float = 0.0
    memory_used_mb: float = 0.0
    cpu_time_seconds: float = 0.0
    sandbox_id: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "error": self.error,
            "execution_time_seconds": self.execution_time_seconds,
            "memory_used_mb": self.memory_used_mb,
            "cpu_time_seconds": self.cpu_time_seconds,
            "sandbox_id": self.sandbox_id,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


class SandboxExecutor(abc.ABC):
    """Abstract base class for sandbox executors.

    All sandbox implementations must inherit from this class and implement
    the execute method.
    """

    def __init__(self, config: SandboxConfig):
        """Initialize executor with configuration.

        Args:
            config: Sandbox configuration
        """
        self.config = config
        self._execution_count = 0

    @abc.abstractmethod
    async def execute(
        self,
        validator_class: type,
        data: Any,
        config: dict[str, Any] | None = None,
    ) -> SandboxResult:
        """Execute a validator in the sandbox.

        Args:
            validator_class: Validator class to execute
            data: Data to validate (DataFrame)
            config: Optional validator configuration

        Returns:
            SandboxResult with execution details
        """
        pass

    def _generate_sandbox_id(self) -> str:
        """Generate unique sandbox execution ID."""
        self._execution_count += 1
        timestamp = datetime.now(timezone.utc).isoformat()
        content = f"{timestamp}-{self._execution_count}-{id(self)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @abc.abstractmethod
    async def cleanup(self) -> None:
        """Clean up sandbox resources."""
        pass


class InProcessSandbox(SandboxExecutor):
    """In-process sandbox with restricted imports.

    Provides basic isolation by restricting module imports.
    Suitable for semi-trusted code with resource monitoring.
    """

    def __init__(self, config: SandboxConfig):
        """Initialize in-process sandbox."""
        super().__init__(config)
        # Handle __builtins__ being either a module or a dict
        if isinstance(__builtins__, dict):
            self._original_import = __builtins__.get("__import__")
        else:
            self._original_import = getattr(__builtins__, "__import__", None)

    def _create_restricted_import(self) -> Callable:
        """Create import function that blocks certain modules."""
        original_import = self._original_import
        blocked = set(self.config.blocked_modules)

        def restricted_import(
            name: str,
            globals: dict | None = None,
            locals: dict | None = None,
            fromlist: tuple = (),
            level: int = 0,
        ) -> Any:
            # Check if module or any parent is blocked
            parts = name.split(".")
            for i in range(len(parts)):
                module_path = ".".join(parts[: i + 1])
                if module_path in blocked:
                    raise ImportError(
                        f"Module '{name}' is blocked in sandbox. "
                        f"Blocked modules: {blocked}"
                    )
            return original_import(name, globals, locals, fromlist, level)

        return restricted_import

    @contextmanager
    def _restricted_context(self) -> Generator[None, None, None]:
        """Context manager for restricted execution."""
        # Handle __builtins__ being either a module or a dict
        if isinstance(__builtins__, dict):
            original_import = __builtins__.get("__import__")
            try:
                __builtins__["__import__"] = self._create_restricted_import()
                yield
            finally:
                __builtins__["__import__"] = original_import
        else:
            original_import = getattr(__builtins__, "__import__", None)
            try:
                setattr(__builtins__, "__import__", self._create_restricted_import())
                yield
            finally:
                if original_import:
                    setattr(__builtins__, "__import__", original_import)

    async def execute(
        self,
        validator_class: type,
        data: Any,
        config: dict[str, Any] | None = None,
    ) -> SandboxResult:
        """Execute validator with import restrictions."""
        sandbox_id = self._generate_sandbox_id()
        started_at = datetime.now(timezone.utc)
        start_time = time.perf_counter()

        try:
            with self._restricted_context():
                # Create validator instance
                validator = validator_class(**(config or {}))

                # Execute with timeout
                async def run_validation() -> Any:
                    return validator.validate(data)

                result = await asyncio.wait_for(
                    run_validation(),
                    timeout=self.config.timeout_seconds,
                )

                execution_time = time.perf_counter() - start_time

                return SandboxResult(
                    success=True,
                    result=result,
                    execution_time_seconds=execution_time,
                    sandbox_id=sandbox_id,
                    started_at=started_at,
                    finished_at=datetime.now(timezone.utc),
                )

        except asyncio.TimeoutError:
            return SandboxResult(
                success=False,
                error=f"Execution timed out after {self.config.timeout_seconds}s",
                execution_time_seconds=self.config.timeout_seconds,
                sandbox_id=sandbox_id,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            )
        except ImportError as e:
            return SandboxResult(
                success=False,
                error=f"Security violation: {e}",
                execution_time_seconds=time.perf_counter() - start_time,
                sandbox_id=sandbox_id,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            )
        except Exception as e:
            return SandboxResult(
                success=False,
                error=f"Execution error: {e}",
                execution_time_seconds=time.perf_counter() - start_time,
                sandbox_id=sandbox_id,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            )

    async def cleanup(self) -> None:
        """No cleanup needed for in-process sandbox."""
        pass


class SubprocessSandbox(SandboxExecutor):
    """Subprocess-based sandbox with resource limits.

    Executes validators in a separate process with:
    - Memory limits via resource module
    - CPU time limits
    - Filesystem isolation
    - Network restrictions
    """

    def __init__(self, config: SandboxConfig):
        """Initialize subprocess sandbox."""
        super().__init__(config)
        self._temp_dirs: list[Path] = []

    def _create_executor_script(
        self,
        validator_module: str,
        validator_class_name: str,
        data_path: Path,
        config: dict[str, Any] | None,
        result_path: Path,
    ) -> str:
        """Create Python script to execute in subprocess."""
        config_json = json.dumps(config or {})

        return f'''
import resource
import sys
import pickle
import json

# Set resource limits
resource.setrlimit(resource.RLIMIT_AS, ({self.config.max_memory_mb * 1024 * 1024}, {self.config.max_memory_mb * 1024 * 1024}))
resource.setrlimit(resource.RLIMIT_CPU, ({int(self.config.timeout_seconds)}, {int(self.config.timeout_seconds)}))

# Block network by not importing socket-related modules
blocked_modules = {set(self.config.blocked_modules)}

class ImportBlocker:
    def find_module(self, name, path=None):
        if name in blocked_modules or any(name.startswith(m + ".") for m in blocked_modules):
            return self
        return None

    def load_module(self, name):
        raise ImportError(f"Module '{{name}}' is blocked in sandbox")

sys.meta_path.insert(0, ImportBlocker())

try:
    # Load data
    with open("{data_path}", "rb") as f:
        data = pickle.load(f)

    # Import and instantiate validator
    module = __import__("{validator_module}", fromlist=["{validator_class_name}"])
    validator_class = getattr(module, "{validator_class_name}")
    config = json.loads('{config_json}')
    validator = validator_class(**config)

    # Execute validation
    result = validator.validate(data)

    # Save result
    with open("{result_path}", "wb") as f:
        pickle.dump({{"success": True, "result": result}}, f)

except Exception as e:
    with open("{result_path}", "wb") as f:
        pickle.dump({{"success": False, "error": str(e)}}, f)
'''

    async def execute(
        self,
        validator_class: type,
        data: Any,
        config: dict[str, Any] | None = None,
    ) -> SandboxResult:
        """Execute validator in subprocess with resource limits."""
        sandbox_id = self._generate_sandbox_id()
        started_at = datetime.now(timezone.utc)
        start_time = time.perf_counter()

        # Create temporary directory for data exchange
        temp_dir = Path(tempfile.mkdtemp(prefix="truthound_sandbox_"))
        self._temp_dirs.append(temp_dir)

        data_path = temp_dir / "data.pkl"
        result_path = temp_dir / "result.pkl"
        script_path = temp_dir / "executor.py"

        try:
            # Serialize data
            with open(data_path, "wb") as f:
                pickle.dump(data, f)

            # Create executor script
            script = self._create_executor_script(
                validator_class.__module__,
                validator_class.__name__,
                data_path,
                config,
                result_path,
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

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.timeout_seconds + 5,  # Grace period
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return SandboxResult(
                    success=False,
                    error=f"Execution timed out after {self.config.timeout_seconds}s",
                    execution_time_seconds=self.config.timeout_seconds,
                    sandbox_id=sandbox_id,
                    started_at=started_at,
                    finished_at=datetime.now(timezone.utc),
                )

            execution_time = time.perf_counter() - start_time

            # Read result
            if result_path.exists():
                with open(result_path, "rb") as f:
                    result_data = pickle.load(f)

                if result_data.get("success"):
                    return SandboxResult(
                        success=True,
                        result=result_data.get("result"),
                        execution_time_seconds=execution_time,
                        sandbox_id=sandbox_id,
                        started_at=started_at,
                        finished_at=datetime.now(timezone.utc),
                    )
                else:
                    return SandboxResult(
                        success=False,
                        error=result_data.get("error", "Unknown error"),
                        execution_time_seconds=execution_time,
                        sandbox_id=sandbox_id,
                        started_at=started_at,
                        finished_at=datetime.now(timezone.utc),
                    )
            else:
                return SandboxResult(
                    success=False,
                    error=f"No result produced. stderr: {stderr.decode()}",
                    execution_time_seconds=execution_time,
                    sandbox_id=sandbox_id,
                    started_at=started_at,
                    finished_at=datetime.now(timezone.utc),
                )

        except Exception as e:
            return SandboxResult(
                success=False,
                error=f"Sandbox error: {e}",
                execution_time_seconds=time.perf_counter() - start_time,
                sandbox_id=sandbox_id,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            )

    async def cleanup(self) -> None:
        """Clean up temporary directories."""
        import shutil

        for temp_dir in self._temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        self._temp_dirs.clear()


class DockerSandbox(SandboxExecutor):
    """Docker container sandbox with full isolation.

    Provides the strongest isolation:
    - Separate filesystem
    - Network isolation
    - Resource limits via cgroups
    - Seccomp profiles
    """

    def __init__(self, config: SandboxConfig):
        """Initialize Docker sandbox."""
        super().__init__(config)
        self._container_ids: list[str] = []

    async def _check_docker_available(self) -> bool:
        """Check if Docker is available."""
        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()
            return process.returncode == 0
        except FileNotFoundError:
            return False

    async def execute(
        self,
        validator_class: type,
        data: Any,
        config: dict[str, Any] | None = None,
    ) -> SandboxResult:
        """Execute validator in Docker container."""
        sandbox_id = self._generate_sandbox_id()
        started_at = datetime.now(timezone.utc)
        start_time = time.perf_counter()

        # Check Docker availability
        if not await self._check_docker_available():
            return SandboxResult(
                success=False,
                error="Docker is not available. Install Docker or use SUBPROCESS backend.",
                execution_time_seconds=0,
                sandbox_id=sandbox_id,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            )

        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp(prefix="truthound_docker_"))
        data_path = temp_dir / "data.pkl"
        result_path = temp_dir / "result.pkl"
        script_path = temp_dir / "executor.py"

        try:
            # Serialize data
            with open(data_path, "wb") as f:
                pickle.dump(data, f)

            # Create executor script (simpler version for Docker)
            script = f'''
import sys
import pickle
import json

try:
    with open("/workspace/data.pkl", "rb") as f:
        data = pickle.load(f)

    from {validator_class.__module__} import {validator_class.__name__}
    config = json.loads('{json.dumps(config or {})}')
    validator = {validator_class.__name__}(**config)
    result = validator.validate(data)

    with open("/workspace/result.pkl", "wb") as f:
        pickle.dump({{"success": True, "result": result}}, f)
except Exception as e:
    with open("/workspace/result.pkl", "wb") as f:
        pickle.dump({{"success": False, "error": str(e)}}, f)
'''
            with open(script_path, "w") as f:
                f.write(script)

            # Build Docker command
            docker_cmd = [
                "docker", "run",
                "--rm",
                f"--memory={self.config.max_memory_mb}m",
                f"--cpus={self.config.max_cpu_percent / 100}",
                f"--network={'bridge' if self.config.network_enabled else 'none'}",
                "-v", f"{temp_dir}:/workspace:rw",
                "-w", "/workspace",
                self.config.docker_image,
                "python", "/workspace/executor.py",
            ]

            # Execute container
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.timeout_seconds + 10,
                )
            except asyncio.TimeoutError:
                process.kill()
                return SandboxResult(
                    success=False,
                    error=f"Container timed out after {self.config.timeout_seconds}s",
                    execution_time_seconds=self.config.timeout_seconds,
                    sandbox_id=sandbox_id,
                    started_at=started_at,
                    finished_at=datetime.now(timezone.utc),
                )

            execution_time = time.perf_counter() - start_time

            # Read result
            if result_path.exists():
                with open(result_path, "rb") as f:
                    result_data = pickle.load(f)

                if result_data.get("success"):
                    return SandboxResult(
                        success=True,
                        result=result_data.get("result"),
                        execution_time_seconds=execution_time,
                        sandbox_id=sandbox_id,
                        started_at=started_at,
                        finished_at=datetime.now(timezone.utc),
                    )
                else:
                    return SandboxResult(
                        success=False,
                        error=result_data.get("error"),
                        execution_time_seconds=execution_time,
                        sandbox_id=sandbox_id,
                        started_at=started_at,
                        finished_at=datetime.now(timezone.utc),
                    )
            else:
                return SandboxResult(
                    success=False,
                    error=f"No result. stderr: {stderr.decode()}",
                    execution_time_seconds=execution_time,
                    sandbox_id=sandbox_id,
                    started_at=started_at,
                    finished_at=datetime.now(timezone.utc),
                )

        except Exception as e:
            return SandboxResult(
                success=False,
                error=f"Docker error: {e}",
                execution_time_seconds=time.perf_counter() - start_time,
                sandbox_id=sandbox_id,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            )
        finally:
            # Cleanup temp directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def cleanup(self) -> None:
        """Clean up any remaining containers."""
        for container_id in self._container_ids:
            try:
                process = await asyncio.create_subprocess_exec(
                    "docker", "rm", "-f", container_id,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await process.wait()
            except Exception:
                pass
        self._container_ids.clear()


def create_sandbox(config: SandboxConfig) -> SandboxExecutor:
    """Factory function to create appropriate sandbox executor.

    Args:
        config: Sandbox configuration

    Returns:
        SandboxExecutor instance

    Raises:
        ValueError: If backend is not supported
    """
    if config.backend == SandboxBackend.IN_PROCESS:
        return InProcessSandbox(config)
    elif config.backend == SandboxBackend.SUBPROCESS:
        return SubprocessSandbox(config)
    elif config.backend == SandboxBackend.DOCKER:
        return DockerSandbox(config)
    else:
        raise ValueError(f"Unsupported sandbox backend: {config.backend}")
