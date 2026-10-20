"""Bounded local Core requests; no dependency on an MCP transport."""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import threading
from pathlib import Path

from truthound.mcp.policy import MCPError, MCPPolicy
from truthound.mcp.projection import response


class MCPService:
    def __init__(self, policy: MCPPolicy):
        self.policy = policy
        self._busy = threading.Lock()

    def call(self, operation: str, path: str) -> dict:
        """Synchronous facade for non-transport callers; SDK remains optional."""
        import anyio

        return anyio.run(self.call_async, operation, path)

    def _worker_argv(self) -> list[str]:
        return [sys.executable, "-I", "-m", "truthound.mcp.worker"]

    async def call_async(self, operation: str, path: str) -> dict:
        """Own the child in the request's cancellation scope through reaping."""
        import anyio

        await anyio.lowlevel.checkpoint()
        if operation not in {"check", "profile", "scan"}:
            return response("unknown", error="unknown_operation")
        if not self._busy.acquire(blocking=False):
            return response(operation, error="busy")
        try:
            source = self.policy.resolve(path)
            with tempfile.TemporaryDirectory(prefix="truthound-mcp-") as directory:
                # Core receives an immutable private input copy, not a user project.
                target = Path(directory) / ("input" + source.suffix.lower())
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
                fd = os.open(source, flags)
                with os.fdopen(fd, "rb") as stream, target.open("xb") as output:
                    info = os.fstat(stream.fileno())
                    if not stat.S_ISREG(info.st_mode):
                        raise MCPError("access_denied")
                    copied = 0
                    while chunk := stream.read(1024 * 1024):
                        copied += len(chunk)
                        if copied > self.policy.max_file_bytes:
                            raise MCPError("file_too_large")
                        output.write(chunk)
                        await anyio.lowlevel.checkpoint()
                request = {
                    "operation": operation,
                    "path": str(target),
                    "max_items": self.policy.max_items,
                    "max_response_bytes": self.policy.max_response_bytes,
                }
                env = {
                    k: v
                    for k, v in os.environ.items()
                    if k
                    in {
                        "PATH",
                        "SYSTEMROOT",
                        "WINDIR",
                        "TEMP",
                        "TMP",
                        # CPython 3.11/3.12 platform.machine() on Windows needs
                        # these public OS identifiers for Polars CPU detection.
                        "PROCESSOR_ARCHITECTURE",
                        "PROCESSOR_ARCHITEW6432",
                    }
                }
                env.update(POLARS_MAX_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                # AnyIO owns and shield-finalizes the process: cancellation and
                # timeout kill/reap it before deleting its input and releasing busy.
                with anyio.fail_after(self.policy.timeout_seconds):
                    worker = await anyio.run_process(
                        self._worker_argv(),
                        input=json.dumps(request).encode(),
                        check=False,
                        cwd=directory,
                        env=env,
                    )
                if worker.returncode or len(worker.stdout) > self.policy.max_response_bytes:
                    raise MCPError("worker_error")
                return json.loads(worker.stdout)
        except MCPError as exc:
            return response(operation, error=exc.code)
        except TimeoutError:
            return response(operation, error="timeout")
        except Exception:
            return response(operation, error="worker_error")
        finally:
            self._busy.release()
