"""One isolated native poll or delivery; persistent authority stays in the parent."""

import asyncio
import base64
import contextlib
import hashlib
import io
import json
import logging
import os
import sys
from pathlib import Path

from truthound.mcp.approval import private_read
from truthound.mcp.encoding import encode
from truthound.mcp.native_stdio import native_worker_wire
from truthound.mcp.parent_pipe import watch_parent
from truthound.mcp.plugin_bindings import approved_capabilities, fingerprint
from truthound.mcp.policy import MCPError
from truthound.mcp.resources import BoundedWriter
from truthound.mcp.schema_history_worker import FailClosed
from truthound.mcp.schema_monitor_runtime import MonitorRuntime
from truthound.mcp.tables import decode


def pack_schema(schema):
    """Lossless closed values and native Arrow dtype metadata, never pickle/eval."""
    import polars as pl

    packed = {}
    for name, value in schema.items():
        if type(name) is not str:
            raise MCPError("unsupported_value")
        if isinstance(value, pl.DataType) or (
            isinstance(value, type) and issubclass(value, pl.DataType)
        ):
            buffer = io.BytesIO()
            pl.DataFrame(schema={"value": value}).write_ipc(buffer)
            raw = buffer.getvalue()
            if pl.read_ipc(io.BytesIO(raw)).schema["value"] != value:
                raise MCPError("unsupported_value")
            packed[name] = {"kind": "polars", "ipc": base64.b64encode(raw).decode("ascii")}
        else:
            encoded = encode(value)
            restored = decode(encoded)
            if type(restored) is not type(value) or restored != value:
                raise MCPError("unsupported_value")
            packed[name] = {"kind": "value", "value": encoded}
    return packed


def unpack_schema(schema):
    import polars as pl

    restored = {}
    for name, value in schema.items():
        if value["kind"] == "value":
            restored[name] = decode(value["value"])
        elif value["kind"] == "polars":
            frame = pl.read_ipc(io.BytesIO(base64.b64decode(value["ipc"], validate=True)))
            if frame.shape != (0, 1) or frame.columns != ["value"]:
                raise MCPError("invalid_control_state")
            restored[name] = frame.schema["value"]
        else:
            raise MCPError("invalid_control_state")
    return restored


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
        runtime.alerts._alert_counter = max(
            runtime.alerts._alert_counter, request.get("alert_counter", 0)
        )
        if options["mode"] == "sync":
            runtime.watcher.check_now()
        else:
            asyncio.run(runtime.watcher.check_now())
        if runtime.error:
            raise MCPError("monitor_execution_failed")
        return checkpoint(runtime)


def checkpoint(runtime):
    return {
        **runtime.snapshot(),
        "schema_format": 2,
        "alert_counter": runtime.alerts._alert_counter,
        "schemas": {
            name: pack_schema(schema) for name, schema in runtime.watcher._last_schemas.items()
        },
        "version_schemas": {
            item.version_id: pack_schema(item.schema) for item in runtime.history.list_versions()
        },
        "checksums": runtime.watcher._last_checksums,
    }


def restore(runtime, previous):
    """Pinned native state adapter, preserving counters and native serialized objects."""
    from truthound.profiler.evolution.breaking_alerts import BreakingChangeAlert
    from truthound.profiler.evolution.history import SchemaVersion

    watcher = runtime.watcher
    if previous.get("schema_format") != 2 or set(previous["schemas"]) != set(watcher._sources):
        raise MCPError("invalid_control_state")
    watcher._last_schemas = {
        name: unpack_schema(schema) for name, schema in previous["schemas"].items()
    }
    watcher._last_checksums = previous["checksums"]
    watcher._event_counter = runtime.events = previous["events"]
    runtime.polls = previous["polls"]
    versions = [SchemaVersion.from_dict(item) for item in reversed(previous["versions"])]
    for item in versions:
        item.schema = unpack_schema(previous["version_schemas"][item.version_id])
    runtime.history._versions = {item.version_id: item for item in versions}
    runtime.history._order = [item.version_id for item in versions]
    runtime.alerts._alerts = [BreakingChangeAlert.from_dict(item) for item in previous["alerts"]]
    runtime.alerts._alert_counter = max(
        previous.get("alert_counter", 0),
        *(int(item.alert_id.removeprefix("ALERT-")) for item in runtime.alerts._alerts),
        0,
    )


def main():
    wire = native_worker_wire()
    for name in ("history", "watcher", "breaking_alerts"):
        handler = FailClosed()
        handler.setLevel(logging.WARNING)
        logging.getLogger("truthound.profiler.evolution." + name).addHandler(handler)
    try:
        line = sys.stdin.buffer.readline(2 * 1024**2 + 1)
        if len(line) > 2 * 1024**2 or not line.endswith(b"\n"):
            raise MCPError("resource_exhausted")
        request = json.loads(line)

        watch_parent(sys.stdin.fileno())
        with (
            open(os.devnull, "w", encoding="utf-8") as sink,
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
