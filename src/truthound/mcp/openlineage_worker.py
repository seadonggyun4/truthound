"""Own one native emitter; retain uncertain deliveries without closing the owner."""

import asyncio
import json
from copy import deepcopy
from pathlib import Path

from truthound.mcp.broker_worker import NativeServiceResult
from truthound.mcp.openlineage_services import OpenLineageRuntime
from truthound.mcp.policy import MCPError


class OpenLineageOwner:
    def __init__(self, runtime, request):
        from truthound.lineage.integrations.openlineage import OpenLineageEmitter

        if not isinstance(runtime, OpenLineageRuntime) or not isinstance(
            runtime.emitter, OpenLineageEmitter
        ):
            raise MCPError("invalid_binding")
        self.emitter = runtime.emitter
        self.datasets = deepcopy(runtime.datasets)
        self.shutdown_requested = False

    def status(self):
        return {
            "active": [
                {"run_id": key, "event_type": value.event_type.value}
                for key, value in self.emitter.active_runs.items()
            ],
            "unknown": [
                {"run_id": key, "event_type": value["event_type"]}
                for key, value in self.emitter.unknown_runs.items()
            ],
            "datasets": sorted(self.datasets),
        }

    async def execute(self, args, budget, inputs=None):
        from truthound.lineage import LineageGraph
        from truthound.lineage.integrations.openlineage import OpenLineageEmissionError

        emitter, command = self.emitter, args["command"]
        try:
            if command in {"start", "graph"}:
                status = await asyncio.to_thread(self.status)
                if len(status["active"]) + len(status["unknown"]) >= 1000:
                    raise MCPError("resource_exhausted")
            try:
                datasets = [deepcopy(self.datasets[key]) for key in args.get("datasets", [])]
            except KeyError:
                raise MCPError("invalid_binding") from None
            run = None
            if command == "start":
                run = await emitter.start_run_async(
                    args["job_name"], inputs=datasets, parent_run_id=args.get("parent_run_id")
                )
            elif command in {"running", "complete", "fail", "abort"}:
                state = await asyncio.to_thread(self.status)
                unknown = next(
                    (item for item in state["unknown"] if item["run_id"] == args["run_id"]), None
                )
                if unknown is not None:
                    return NativeServiceResult(
                        {**unknown, "outcome_unknown": True}, "outcome_unknown"
                    )
                try:
                    run = await asyncio.to_thread(emitter.get_run, args["run_id"])
                except KeyError:
                    raise MCPError("invalid_handle") from None
                if command == "running":
                    await emitter.emit_running_async(run)
                elif command == "complete":
                    await emitter.emit_complete_async(run, outputs=datasets)
                elif command == "fail":
                    await emitter.emit_fail_async(
                        run, RuntimeError("Operator reported run failure")
                    )
                else:
                    await emitter.emit_abort_async(run)
            elif command == "graph":
                document = json.loads(Path(inputs["graph_path"]).read_text(encoding="utf-8"))
                graph = LineageGraph.from_dict(document["graph"])
                runs = await emitter.emit_from_graph_async(graph, job_name=args["job_name"])
                return {
                    "runs": [
                        {"run_id": run.run_id, "event_type": run.event_type.value} for run in runs
                    ]
                }
            elif command == "resolve":
                try:
                    await emitter.resolve_unknown_async(args["run_id"], delivered=args["delivered"])
                except KeyError:
                    raise MCPError("invalid_handle") from None
            elif command == "shutdown":
                self.shutdown_requested = True
            elif command != "status":
                raise MCPError("invalid_arguments")
            result = await asyncio.to_thread(self.status)
            if run is not None:
                result["run"] = {"run_id": run.run_id, "event_type": run.event_type.value}
            return result
        except OpenLineageEmissionError as error:
            return NativeServiceResult(
                {"run_id": error.run_id, "event_type": error.event_type, "outcome_unknown": True},
                "outcome_unknown",
            )

    async def close(self):
        await self.emitter.aclose()


if __name__ == "__main__":
    from truthound.mcp.broker_worker import main

    main(OpenLineageOwner)
