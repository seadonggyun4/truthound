"""One isolated native poll or delivery; persistent authority stays in the parent."""

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

from truthound.mcp.approval import private_read
from truthound.mcp.plugin_bindings import approved_capabilities, fingerprint
from truthound.mcp.policy import MCPError
from truthound.mcp.resources import BoundedWriter
from truthound.mcp.schema_history_worker import FailClosed
from truthound.mcp.schema_monitor_runtime import MonitorRuntime


def execute(request):
    def guard():
        raw = private_read(Path(request["control"]) / "bindings.json", 1024**2)
        if hashlib.sha256(raw).hexdigest() != request["digest"]:
            raise MCPError("access_denied")
        for spec in request["plugins"].values():
            package, digest, _ = fingerprint(spec["distribution"])
            if package.version != spec["version"] or digest != spec["sha256"]:
                raise MCPError("plugin_identity_mismatch")

    guard()
    with approved_capabilities(request["plugins"]) as capabilities:
        binding = request["binding"]
        if request["task"] == "deliver":
            from truthound.profiler.evolution.breaking_alerts import (
                AlertNotifier,
                BreakingChangeAlert,
            )

            notifier = capabilities[binding["notifier"]]()
            if not isinstance(notifier, AlertNotifier):
                raise MCPError("invalid_binding")
            guard()
            if notifier.send(BreakingChangeAlert.from_dict(request["alert"])) is not True:
                raise MCPError("notification_delivery_unknown")
            return {"delivered": True}
        source = capabilities[binding["source"]]()
        options = request["options"]
        runtime = MonitorRuntime(
            source,
            None,
            mode=options["mode"],
            interval=options["poll_interval"],
            max_events=options["max_events"],
            budget=request["budget"],
            guard=guard,
        )
        previous = request["checkpoint"]
        if previous is not None:
            restore(runtime, previous)
        if options["mode"] == "sync":
            runtime.watcher.check_now()
        else:
            asyncio.run(runtime.watcher.check_now())
        if runtime.error:
            raise MCPError("monitor_execution_failed")
        snapshot = runtime.snapshot()
        return {
            **snapshot,
            "schemas": {
                name: {key: str(value) for key, value in schema.items()}
                for name, schema in runtime.watcher._last_schemas.items()
            },
            "checksums": runtime.watcher._last_checksums,
        }


def restore(runtime, previous):
    """Pinned native state adapter, preserving counters and native serialized objects."""
    from truthound.profiler.evolution.breaking_alerts import BreakingChangeAlert
    from truthound.profiler.evolution.history import SchemaVersion

    watcher = runtime.watcher
    if set(previous["schemas"]) != set(watcher._sources):
        raise MCPError("invalid_control_state")
    watcher._last_schemas = previous["schemas"]
    watcher._last_checksums = previous["checksums"]
    watcher._event_counter = runtime.events = previous["events"]
    runtime.polls = previous["polls"]
    versions = [SchemaVersion.from_dict(item) for item in reversed(previous["versions"])]
    runtime.history._versions = {item.version_id: item for item in versions}
    runtime.history._order = [item.version_id for item in versions]
    runtime.alerts._alerts = [BreakingChangeAlert.from_dict(item) for item in previous["alerts"]]
    runtime.alerts._alert_counter = len(runtime.alerts._alerts)


def main():
    wire = sys.stdout
    for name in ("history", "watcher", "breaking_alerts"):
        handler = FailClosed()
        handler.setLevel(logging.WARNING)
        logging.getLogger("truthound.profiler.evolution." + name).addHandler(handler)
    try:
        line = sys.stdin.buffer.readline(2 * 1024**2 + 1)
        if len(line) > 2 * 1024**2 or not line.endswith(b"\n"):
            raise MCPError("resource_exhausted")
        request = json.loads(line)
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
        wire.write('{"ok":false,"code":"monitor_execution_failed"}\n')
    wire.flush()


if __name__ == "__main__":
    main()
