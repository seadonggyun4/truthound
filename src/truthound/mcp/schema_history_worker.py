"""Resident native history with bounded artifact output and closed operations."""

import contextlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from truthound.mcp.encoding import encode
from truthound.mcp.policy import MCPError
from truthound.mcp.resources import BoundedWriter


def main():
    from truthound.mcp.schema_history import HistoryAction, HistoryCreate
    from truthound.mcp.validator_wire import _known_type

    history = None
    manager = None
    revision = 0
    wire = sys.stdout
    with open(os.devnull, "w") as sink:
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
                        prefix = "truthound.profiler.evolution.history."
                        for name in ("SchemaVersion", "SchemaDiff"):
                            _known_type(prefix + name)
                        strategies = {
                            "semantic": "SemanticVersionStrategy",
                            "incremental": "IncrementalVersionStrategy",
                            "timestamp": "TimestampVersionStrategy",
                            "git_like": "GitLikeVersionStrategy",
                        }
                        history = _known_type(prefix + "InMemorySchemaHistory")(
                            max_versions=args.max_versions,
                            version_strategy=_known_type(prefix + strategies[args.strategy])(),
                        )
                        manager = _known_type(prefix + "SchemaHistory")(history)
                    else:
                        args = HistoryAction.model_validate(request["arguments"])
                        if history is None or args.expected_revision != revision:
                            raise MCPError("revision_conflict")
                        command = args.command
                        action = command.action
                        options = command.model_dump(exclude={"action"})
                        if action == "list":
                            for key in ("since", "until"):
                                if options[key] is not None:
                                    options[key] = datetime.fromisoformat(options[key])
                        if action == "save":
                            import polars as pl

                            result = history.save_version(
                                pl.scan_ipc(
                                    request["asset_path"], memory_map=False
                                ).collect_schema(),
                                version=command.version,
                                metadata=command.metadata,
                            )
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
                        if action in {"save", "delete", "rollback"}:
                            revision += 1
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
