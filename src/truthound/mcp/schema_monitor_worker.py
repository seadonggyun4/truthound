"""Owned process hosting approved native watchers and notifier objects."""

import contextlib
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

from truthound.mcp.approval import private_read
from truthound.mcp.encoding import encode
from truthound.mcp.native_stdio import native_worker_wire
from truthound.mcp.plugin_bindings import approved_capabilities, fingerprint
from truthound.mcp.policy import MCPError
from truthound.mcp.resources import BoundedWriter
from truthound.mcp.schema_history_worker import FailClosed
from truthound.mcp.schema_monitor_runtime import MonitorRuntime


def main():
    wire = native_worker_wire()
    runtime = None
    for name in ("history", "watcher", "breaking_alerts"):
        handler = FailClosed()
        handler.setLevel(logging.WARNING)
        logging.getLogger("truthound.profiler.evolution." + name).addHandler(handler)
    with contextlib.ExitStack() as stack:
        plugins = stack.enter_context(contextlib.ExitStack())
        sink = stack.enter_context(open(os.devnull, "w", encoding="utf-8"))
        # Keep background plugin prints off the protocol for the whole lifetime.
        stack.enter_context(contextlib.redirect_stdout(sink))
        stack.enter_context(contextlib.redirect_stderr(sink))
        try:
            while line := sys.stdin.buffer.readline(131073):
                try:
                    if len(line) > 131072 or not line.endswith(b"\n"):
                        raise MCPError("resource_exhausted")
                    request = json.loads(line)
                    if "open" in request:
                        if runtime is not None:
                            raise MCPError("invalid_arguments")
                        opening = request

                        def guard(opening=opening):
                            raw = private_read(Path(opening["control"]) / "bindings.json", 1024**2)
                            if hashlib.sha256(raw).hexdigest() != opening["digest"]:
                                raise MCPError("access_denied")
                            for spec in opening["plugins"].values():
                                package, digest, _ = fingerprint(spec["distribution"])
                                if digest != spec["sha256"] or package.version != spec["version"]:
                                    raise MCPError("plugin_identity_mismatch")

                        guard()
                        capabilities = plugins.enter_context(
                            approved_capabilities(request["plugins"])
                        )
                        binding = request["binding"]
                        source = capabilities[binding["source"]]()
                        notifier = (
                            capabilities[binding["notifier"]]() if binding["notifier"] else None
                        )
                        args = request["open"]
                        runtime = MonitorRuntime(
                            source,
                            notifier,
                            mode=args["mode"],
                            interval=args["poll_interval"],
                            max_events=args["max_events"],
                            budget=request["budget"],
                            guard=guard,
                        )
                        runtime.start()
                    else:
                        if runtime is None:
                            raise MCPError("invalid_handle")
                        if request["operation"] == "schema_monitor_stop":
                            runtime.stop()
                            plugins.close()
                        elif request["operation"] != "schema_monitor_status":
                            raise MCPError("unsupported_operation")
                        with BoundedWriter(
                            Path(request["output_path"]), request["budget"]
                        ) as output:
                            output.write(
                                json.dumps(encode(runtime.snapshot()), allow_nan=False).encode()
                            )
                    wire.write('{"ok":true}\n')
                    wire.flush()
                except Exception:
                    wire.write('{"ok":false,"code":"monitor_execution_failed"}\n')
                    wire.flush()
                    break
        finally:
            if runtime is not None:
                runtime.stop()


if __name__ == "__main__":
    main()
