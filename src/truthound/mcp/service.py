"""Bounded local Core requests; no dependency on an MCP transport."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
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
        self._identity_key = secrets.token_bytes(32)
        self._session = None
        self._closed = False

    def close(self):
        if self._busy.locked():
            raise MCPError("busy")
        self._closed = True
        if self._session is not None:
            self._session.close()

    def call(self, operation: str, path: str) -> dict:
        """Synchronous facade for non-transport callers; SDK remains optional."""
        import anyio

        return anyio.run(self.call_async, operation, path)

    def _worker_argv(self) -> list[str]:
        return [sys.executable, "-I", "-m", "truthound.mcp.worker"]

    async def call_async(self, operation: str, path: str) -> dict:
        """Own the child in the request's cancellation scope through reaping."""
        return await self._call_async(operation, path)

    async def call_operation_async(self, name: str, arguments: dict) -> dict:
        """Validate and authorize explicit typed operations before touching input."""
        from truthound.mcp.operations import envelope

        if self.policy.profile != "core":
            return envelope(name, error="unknown_operation")
        from pydantic import ValidationError

        from truthound.mcp.catalog import CATALOG, discover, needs_disclosure

        operation = CATALOG.get(name)
        if operation is None:
            return envelope(name, error="unknown_operation")
        try:
            if self._closed:
                raise MCPError("session_closed")
            if not isinstance(arguments, dict):
                raise MCPError("invalid_arguments")
            try:
                size = len(json.dumps(arguments, allow_nan=False).encode())
            except (ValueError, TypeError, RecursionError):
                raise MCPError("invalid_arguments") from None
            if size > 65536:
                raise MCPError("resource_exhausted")
            parsed = operation.input_type.model_validate(arguments)
            if name in {"capabilities_list", "capability_describe"}:
                data = discover(
                    name,
                    parsed,
                    sensitive=self.policy.allow_sensitive_results,
                    workspaces=bool(self.policy.workspace_directories),
                    control=self.policy.control_directory is not None,
                )
                result = envelope(name, data=data)
                operation.output_type().model_validate(result)
                if (
                    len(json.dumps(result, allow_nan=False).encode())
                    > self.policy.max_response_bytes
                ):
                    raise MCPError("response_too_large")
                return result
            if needs_disclosure(operation, parsed) and not self.policy.allow_sensitive_results:
                raise MCPError("approval_required")
            if operation.route == "session":
                if self._closed:
                    raise MCPError("session_closed")
                if not self._busy.acquire(blocking=False):
                    raise MCPError("busy")
                try:
                    from truthound.mcp.session import MCPSession

                    if self._session is None:
                        self._session = MCPSession(self)
                    result = await self._session.call(name, parsed)
                    operation.output_type().model_validate(result)
                    if (
                        len(json.dumps(result, allow_nan=False).encode())
                        > self.policy.max_response_bytes
                    ):
                        raise MCPError("response_too_large")
                    return result
                except TimeoutError:
                    return envelope(name, error="timeout")
                finally:
                    self._busy.release()
            return await self._call_async(name, parsed.path, arguments=parsed.model_dump())
        except ValidationError:
            return envelope(name, error="invalid_arguments")
        except MCPError as exc:
            return envelope(name, error=exc.code)
        except Exception:
            return envelope(name, error="worker_error")

    async def _execute_request(self, request, directory):
        import anyio

        env = {
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
        env.update(POLARS_MAX_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
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

    async def _call_async(
        self, operation: str, path: str, *, arguments: dict | None = None
    ) -> dict:
        import anyio

        from truthound.mcp.operations import envelope

        def failure(code):
            return (
                envelope(operation, error=code)
                if arguments is not None
                else response(operation, error=code)
            )

        await anyio.lowlevel.checkpoint()
        if self._closed:
            return failure("session_closed")
        if arguments is None and operation not in {"check", "profile", "scan"}:
            return response("unknown", error="unknown_operation")
        if not self._busy.acquire(blocking=False):
            return failure("busy")
        try:
            source = self.policy.resolve(path)
            with tempfile.TemporaryDirectory(prefix="truthound-mcp-") as directory:
                # Core receives an immutable private input copy, not a user project.
                target = Path(directory) / ("input" + source.suffix.lower())
                from truthound.mcp.ingestion import capture

                digest = await capture(self.policy, source, target)
                request = {
                    "operation": operation,
                    "path": str(target),
                    "max_items": self.policy.max_items,
                    "max_response_bytes": self.policy.max_response_bytes,
                }
                if arguments is not None:
                    from truthound import __version__

                    request.update(
                        contract_version=1,
                        arguments=arguments,
                        provenance={
                            "core_version": __version__,
                            "input_identity": hmac.new(
                                self._identity_key, digest.digest(), hashlib.sha256
                            ).hexdigest(),
                            **(
                                {"input_sha256": digest.hexdigest()}
                                if self.policy.allow_sensitive_results
                                else {}
                            ),
                            "context_mode": "transient",
                        },
                    )
                # AnyIO owns and shield-finalizes the process: cancellation and
                # timeout kill/reap it before deleting its input and releasing busy.
                result = await self._execute_request(request, directory)
                if arguments is not None:
                    from truthound.mcp.catalog import CATALOG

                    CATALOG[operation].output_type().model_validate(result)
                return result
        except MCPError as exc:
            return failure(exc.code)
        except TimeoutError:
            return failure("timeout")
        except Exception:
            return failure("worker_error")
        finally:
            self._busy.release()
