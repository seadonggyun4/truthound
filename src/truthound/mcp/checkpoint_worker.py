"""Isolated installed factory; no wire-selected imports or connection settings."""

import asyncio
import contextlib
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from truthound.mcp.approval import private_read
from truthound.mcp.encoding import encode
from truthound.mcp.parent_pipe import watch_parent
from truthound.mcp.plugin_bindings import approved_capabilities, fingerprint
from truthound.mcp.policy import MCPError
from truthound.mcp.resources import BoundedWriter


@dataclass(frozen=True)
class BindingGuard:
    """A private serializable authority check for approved native child execution."""

    control: str
    digest: str
    plugins: dict

    def __call__(self):
        raw = private_read(Path(self.control) / "bindings.json", 1024**2)
        if hashlib.sha256(raw).hexdigest() != self.digest:
            raise MCPError("access_denied")
        for spec in self.plugins.values():
            package, digest, _ = fingerprint(spec["distribution"])
            if package.version != spec["version"] or digest != spec["sha256"]:
                raise MCPError("plugin_identity_mismatch")


def binding_guard(request):
    return BindingGuard(request["control"], request["digest"], request["plugins"])


async def execute_async(request):
    from truthound.checkpoint.async_checkpoint import AsyncCheckpoint

    guard = binding_guard(request)
    guard()
    with approved_capabilities(request["plugins"]) as capabilities:
        async with capabilities[request["capability"]]() as checkpoint:
            if not isinstance(checkpoint, AsyncCheckpoint):
                raise MCPError("invalid_binding")
            guard()
            result = await checkpoint.run_async(run_id=request["run_id"])
            guard()
            packed = encode(result.to_dict())
    guard()
    return packed


def execute(request):
    from truthound.checkpoint import Checkpoint

    if request.get("mode", "sync") == "async":
        return asyncio.run(execute_async(request))
    guard = binding_guard(request)
    guard()
    # A bare Checkpoint has no general resource-close contract. Its factory
    # must explicitly own datasource and action cleanup in a context manager.
    with (
        approved_capabilities(request["plugins"]) as capabilities,
        capabilities[request["capability"]]() as checkpoint,
    ):
        if not isinstance(checkpoint, Checkpoint):
            raise MCPError("invalid_binding")
        guard()
        result = checkpoint.run(run_id=request["run_id"], use_router=request["use_router"])
        guard()
        packed = encode(result.to_dict())
    guard()
    return packed


def main():
    wire = sys.stdout
    try:
        line = sys.stdin.buffer.readline(2 * 1024**2 + 1)
        if len(line) > 2 * 1024**2 or not line.endswith(b"\n"):
            raise MCPError("resource_exhausted")
        request = json.loads(line)

        watch_parent(sys.stdin.fileno())
        with (
            open(os.devnull, "w") as sink,
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            result = execute(request)
        with BoundedWriter(Path(request["output_path"]), request["budget"]) as output:
            output.write(json.dumps(result, allow_nan=False).encode())
        wire.write('{"ok":true}\n')
    except Exception:
        wire.write('{"ok":false,"code":"checkpoint_execution_failed"}\n')
    wire.flush()


if __name__ == "__main__":
    main()
