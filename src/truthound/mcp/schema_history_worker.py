"""Resident native history with bounded artifact output and closed operations."""

import contextlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from truthound.mcp.encoding import encode
from truthound.mcp.native_stdio import native_worker_wire
from truthound.mcp.policy import MCPError
from truthound.mcp.resources import BoundedWriter


class FailClosed(logging.Handler):
    """Native history/watchers log some failures instead of raising them."""

    def emit(self, record):
        raise MCPError("history_execution_failed")


def main():
    from truthound.mcp.schema_history import HistoryAction, HistoryCreate
    from truthound.mcp.validator_wire import _known_type

    history = None
    manager = None
    revision = 0
    configuration = None
    root = Path.cwd() / "history"
    sources = {}
    for module in ("history", "watcher", "breaking_alerts"):
        handler = FailClosed()
        handler.setLevel(logging.WARNING)
        logging.getLogger("truthound.profiler.evolution." + module).addHandler(handler)
    wire = native_worker_wire()
    with open(os.devnull, "w", encoding="utf-8") as sink:
        while line := sys.stdin.buffer.readline(131073):
            try:
                if len(line) > 131072 or not line.endswith(b"\n"):
                    raise MCPError("resource_exhausted")
                request = json.loads(line)
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    if "create" in request:
                        if history is not None:
                            raise MCPError("invalid_arguments")
                        args = HistoryCreate.model_validate(request["create"])
                        configuration = args
                        storage_budget = request["storage_budget"]
                        prefix = "truthound.profiler.evolution.history."
                        for name in ("SchemaVersion", "SchemaDiff"):
                            _known_type(prefix + name)
                        strategies = {
                            "semantic": "SemanticVersionStrategy",
                            "incremental": "IncrementalVersionStrategy",
                            "timestamp": "TimestampVersionStrategy",
                            "git_like": "GitLikeVersionStrategy",
                        }
                        if request.get("restore_path"):
                            from truthound.mcp.history_files import restore

                            restore(request["restore_path"], root, args)
                        cls = (
                            "FileSchemaHistory"
                            if args.backend == "file"
                            else "InMemorySchemaHistory"
                        )
                        history = _known_type(prefix + cls)(
                            **({"base_path": root} if args.backend == "file" else {}),
                            max_versions=args.max_versions,
                            version_strategy=_known_type(prefix + strategies[args.strategy])(),
                        )
                        manager = _known_type(prefix + "SchemaHistory")(history)
                        watch_prefix = "truthound.profiler.evolution.watcher."
                        alert_prefix = "truthound.profiler.evolution.breaking_alerts."
                        alerts = _known_type(alert_prefix + "BreakingChangeAlertManager")(
                            alert_storage_path=root / "alerts.json"
                            if args.backend == "file"
                            else None
                        )
                        watcher = _known_type(watch_prefix + "SchemaWatcher")(history=manager)
                        watcher.add_handler(
                            _known_type(watch_prefix + "HistoryEventHandler")(manager)
                        )
                        watcher.add_handler(
                            _known_type(watch_prefix + "AlertingEventHandler")(alerts)
                        )
                    else:
                        args = HistoryAction.model_validate(request["arguments"])
                        if history is None or args.expected_revision != revision:
                            raise MCPError("revision_conflict")
                        command = args.command
                        action = command.action
                        options = command.model_dump(exclude={"action"})
                        if configuration.backend == "file":
                            for key in ("version_id", "source_id", "target_id"):
                                value = options.get(key)
                                if value and any(
                                    character in value for character in ("/", "\\", "\0", ":")
                                ):
                                    raise MCPError("invalid_arguments")
                        if action == "list":
                            for key in ("since", "until"):
                                if options[key] is not None:
                                    options[key] = datetime.fromisoformat(options[key])
                        if action in {"save", "watch_add", "watch_check"}:
                            import polars as pl

                            schema = dict(
                                pl.scan_ipc(
                                    request["asset_path"], memory_map=False
                                ).collect_schema()
                            )
                        if action == "save":
                            result = history.save_version(
                                schema,
                                version=command.version,
                                metadata=command.metadata,
                            )
                        elif action == "snapshot":
                            from truthound.mcp.history_files import snapshot

                            if configuration.backend != "file":
                                raise MCPError("unsupported_operation")
                            result = snapshot(root, configuration)
                        elif action == "watch_add":
                            if (
                                command.name in sources
                                or len(sources) >= configuration.max_versions
                            ):
                                raise MCPError("resource_exhausted")
                            source = _known_type(watch_prefix + "DictSchemaSource")(
                                {key: str(value) for key, value in schema.items()},
                                name=command.name,
                            )
                            watcher.add_source(source)
                            sources[command.name] = source
                            result = watcher.get_source_status()
                        elif action == "watch_check":
                            if command.name not in sources:
                                raise MCPError("invalid_arguments")
                            if alerts.get_stats()["total"] >= configuration.max_versions:
                                raise MCPError("resource_exhausted")
                            sources[command.name].update(
                                {key: str(value) for key, value in schema.items()}
                            )
                            result = watcher.check_now()
                        elif action == "watch_remove":
                            result = watcher.remove_source(command.name)
                            sources.pop(command.name, None)
                        elif action == "watch_status":
                            result = watcher.get_source_status()
                        elif action == "alerts":
                            result = alerts.get_alert_history()
                        elif action == "alert_stats":
                            result = alerts.get_stats()
                        elif action.startswith("alert_"):
                            method = {
                                "alert_get": alerts.get_alert,
                                "alert_acknowledge": alerts.acknowledge_alert,
                                "alert_resolve": alerts.resolve_alert,
                            }[action]
                            result = method(command.alert_id)
                        elif action == "diff":
                            result = manager.diff(command.source_id, command.target_id)
                        elif action == "rollback":
                            result = manager.rollback(command.version_id, reason=command.reason)
                        elif action == "count":
                            result = manager.version_count
                        elif action == "breaking_since":
                            result = manager.has_breaking_changes_since(command.version_id)
                        else:
                            methods = {
                                "get": history.get_version,
                                "delete": history.delete_version,
                                "latest": history.get_latest,
                                "baseline": history.get_baseline,
                                "list": history.list_versions,
                                "by_version": history.get_version_by_string,
                            }
                            result = methods[action](**options)
                        if action in {
                            "save",
                            "delete",
                            "rollback",
                            "watch_add",
                            "watch_check",
                            "watch_remove",
                            "alert_acknowledge",
                            "alert_resolve",
                        }:
                            revision += 1
                        if configuration.backend == "file" and sum(
                            path.stat().st_size for path in root.rglob("*") if path.is_file()
                        ) > min(storage_budget, request["output_budget"]):
                            raise MCPError("resource_exhausted")
                        if isinstance(result, list):
                            payload = [item.to_dict() for item in result]
                        else:
                            payload = result.to_dict() if hasattr(result, "to_dict") else result
                        with BoundedWriter(
                            Path(request["output_path"]), request["output_budget"]
                        ) as output:
                            output.write(json.dumps(encode(payload), allow_nan=False).encode())
                wire.write(json.dumps({"ok": True, "revision": revision}) + "\n")
                wire.flush()
            except Exception:
                wire.write(json.dumps({"ok": False, "code": "history_execution_failed"}) + "\n")
                wire.flush()
                return


if __name__ == "__main__":
    main()
