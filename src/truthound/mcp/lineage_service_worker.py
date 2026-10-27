"""One native tracker per approved factory; a context job never spans wire calls."""

import asyncio
import inspect
import json
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

from truthound.lineage import LineageTracker, OperationType
from truthound.mcp.broker_worker import NativeServiceBytes, NativeServiceResult
from truthound.mcp.encoding import encode
from truthound.mcp.lineage_contracts import RenderSettings, TrackingEvent
from truthound.mcp.lineage_services import LineageJob, LineageRuntime
from truthound.mcp.lineage_worker import apply_tracking_event, render_artifact
from truthound.mcp.policy import MCPError


class LineageOwner:
    def __init__(self, runtime, request=None):
        if not isinstance(runtime, LineageRuntime) or not isinstance(
            runtime.tracker, LineageTracker
        ):
            raise MCPError("invalid_binding")
        if (
            not isinstance(runtime.jobs, dict)
            or not isinstance(runtime.files, dict)
            or any(
                not isinstance(alias, str) or not alias for alias in [*runtime.jobs, *runtime.files]
            )
        ):
            raise MCPError("invalid_binding")
        self.tracker = runtime.tracker
        if self.tracker._active_contexts:
            raise MCPError("invalid_binding")
        self.jobs = {}
        for alias, job in runtime.jobs.items():
            if (
                not isinstance(job, LineageJob)
                or not callable(job.operation)
                or not isinstance(job.name, str)
                or not job.name
                or inspect.iscoroutinefunction(job.operation)
                or inspect.isasyncgenfunction(job.operation)
                or inspect.iscoroutinefunction(type(job.operation).__call__)
                or inspect.isasyncgenfunction(type(job.operation).__call__)
            ):
                raise MCPError("invalid_binding")
            try:
                self.jobs[alias] = replace(job, operation_type=OperationType(job.operation_type))
            except (ValueError, TypeError):
                raise MCPError("invalid_binding") from None
        self.files = {}
        for alias, value in runtime.files.items():
            try:
                path = Path(value)
            except (ValueError, TypeError):
                raise MCPError("invalid_binding") from None
            if not path.is_absolute() or path.is_symlink():
                raise MCPError("invalid_binding")
            self.files[alias] = path
        self._pending = None
        self._closed = False
        self.shutdown_requested = False

    @staticmethod
    def selected(mapping, key):
        try:
            return mapping[key]
        except KeyError:
            raise MCPError("invalid_handle") from None

    def state(self):
        tracker = self.tracker
        return {
            "node_count": tracker.graph.node_count,
            "edge_count": tracker.graph.edge_count,
            "history_count": len(tracker._history),
            "active_contexts": len(tracker._active_contexts),
            "persistence_configured": tracker._config.persist_path is not None,
            "persistence_unknown": tracker._persistence_failed,
        }

    def incomplete(self, reason="native_failure", *, native_completed=False):
        return NativeServiceResult(
            {
                "native_completed": native_completed,
                "effects_may_be_partial": True,
                "reason": reason,
                "state": self.state(),
            },
            "native_lineage_incomplete",
        )

    def bounded(self, value, budget):
        payload = value.content if isinstance(value, NativeServiceBytes) else None
        try:
            size = (
                len(payload)
                if payload is not None
                else len(json.dumps(encode(value), allow_nan=False).encode())
            )
        except (MCPError, TypeError, ValueError, RecursionError):
            return self.incomplete("unsupported_value", native_completed=True)
        return (
            self.incomplete("resource_exhausted", native_completed=True) if size > budget else value
        )

    def execute_native(self, args, budget):
        command, tracker = args["command"], self.tracker
        if command == "inspect":
            return self.bounded(
                {**self.state(), "jobs": sorted(self.jobs), "files": sorted(self.files)}, budget
            )
        if command == "shutdown":
            self.shutdown_requested = True
            return {"stopped": True}
        if tracker._persistence_failed and command in {"track", "run", "save", "clear"}:
            return self.incomplete("unconfirmed_persistence")
        job = self.selected(self.jobs, args["job_id"]) if command == "run" else None
        path = self.selected(self.files, args["file_id"]) if command in {"save", "load"} else None
        try:
            if command == "history":
                history = tracker.history
                offset, limit = args.get("offset", 0), args.get("limit", 100)
                rows = [
                    {
                        "operation_id": item.operation_id,
                        "operation_type": item.operation_type.value,
                        "started_at": item.started_at,
                        "sources": item.sources,
                        "targets": item.targets,
                        "metadata": item.metadata,
                        "completed": item.completed,
                        "error": "native_tracking_failed" if item.error is not None else None,
                    }
                    for item in history[offset : offset + limit]
                ]
                end = min(len(history), offset + limit)
                value = {
                    "history": rows,
                    "total": len(history),
                    "next_offset": end if limit and end < len(history) else None,
                    "state": self.state(),
                }
            elif command == "snapshot":
                value = {"graph": tracker.graph.to_dict(), "state": self.state()}
            elif command == "track":
                from pydantic import TypeAdapter

                apply_tracking_event(
                    tracker, TypeAdapter(TrackingEvent).validate_python(args["event"])
                )
                value = {
                    "native_completed": True,
                    "graph": tracker.graph.to_dict(),
                    "state": self.state(),
                }
            elif command == "run":
                with tracker.track(job.name, job.operation_type) as context:
                    result = job.operation(tracker, context)
                    if result is not None:
                        if inspect.iscoroutine(result):
                            result.close()
                        raise TypeError("Tracking jobs must return None")
                value = {
                    "native_completed": context.completed,
                    "operation_id": context.operation_id,
                    "graph": tracker.graph.to_dict(),
                    "state": self.state(),
                }
            elif command == "lineage":
                value = {
                    "lineage": tracker.get_lineage(args["node_id"], args.get("direction", "both")),
                    "state": self.state(),
                }
            elif command == "path":
                found = tracker.get_path(args["source_id"], args["target_id"])
                value = {
                    "path": [node.to_dict() for node in found] if found is not None else None,
                    "state": self.state(),
                }
            elif command == "save":
                tracker.save(str(path))
                value = {"saved": True, "state": self.state()}
            elif command == "load":
                tracker.load(str(path))
                value = {"loaded": True, "graph": tracker.graph.to_dict(), "state": self.state()}
            elif command == "clear":
                tracker.clear()
                value = {"cleared": True, "state": self.state()}
            elif command == "export":
                value = NativeServiceBytes(tracker.export_to_json().encode())
            elif command == "render":
                content, _ = render_artifact(
                    tracker.graph, RenderSettings.model_validate(args["rendering"])
                )
                value = NativeServiceBytes(content)
            else:
                raise MCPError("invalid_arguments")
        except MCPError:
            raise
        except Exception:
            return self.incomplete()
        return self.bounded(value, budget)

    async def execute(self, args, budget):
        if self._closed or self._pending is not None:
            raise MCPError("invalid_handle")
        task = asyncio.create_task(asyncio.to_thread(self.execute_native, args, budget))
        self._pending = task
        task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            with suppress(Exception):
                await asyncio.shield(task)
            raise
        finally:
            if task.done():
                self._pending = None

    async def close(self):
        self._closed = True
        if self._pending is not None:
            with suppress(Exception):
                await asyncio.shield(self._pending)
            self._pending = None
        self.jobs.clear()
        self.files.clear()
        self.tracker = None


if __name__ == "__main__":
    from truthound.mcp.broker_worker import main

    main(LineageOwner)
