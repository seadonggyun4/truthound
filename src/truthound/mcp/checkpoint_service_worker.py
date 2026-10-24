"""Resident native services; operator factories own connections and cleanup."""

import asyncio
import contextlib
import json
import os
import sys
from pathlib import Path

from truthound.mcp.checkpoint_worker import binding_guard
from truthound.mcp.encoding import encode
from truthound.mcp.parent_pipe import watch_resident_parent
from truthound.mcp.plugin_bindings import approved_capabilities
from truthound.mcp.policy import MCPError
from truthound.mcp.resources import BoundedWriter


async def requests():
    # Do not leave a Windows synchronous ReadFile pending while native scheduled
    # work starts subprocesses. Read only available bytes, on either platform.
    fd, pending = sys.stdin.fileno(), bytearray()
    if os.name == "nt":
        import _winapi
        import msvcrt

        handle = msvcrt.get_osfhandle(fd)

        def available():
            return min(_winapi.PeekNamedPipe(handle)[0], 65536)
    else:
        import select

        def available():
            return 65536 if select.select([fd], [], [], 0)[0] else 0

    while True:
        try:
            count = available()
        except OSError:
            return
        if count:
            data = os.read(fd, count)
            if not data:
                return
            pending.extend(data)
            if len(pending) > 131072:
                raise MCPError("resource_exhausted")
            if b"\n" in pending:
                line, _, remainder = pending.partition(b"\n")
                if remainder:
                    raise MCPError("invalid_arguments")
                pending.clear()
                yield json.loads(line)
        await asyncio.sleep(0.05)


async def execute(service, kind, args):
    command = args["command"]
    if kind == "runner":
        if command == "schedule_update":
            from dataclasses import asdict

            from truthound.mcp.checkpoint_services import ScheduleUpdate

            changes = ScheduleUpdate.model_validate(args["schedule_update"])
            trigger = service.edit_schedule(
                args["checkpoint_name"], args["trigger_name"], changes.model_dump(exclude_unset=True)
            )
            return {"trigger": trigger.to_dict(), "configuration": asdict(trigger.config)}
        elif command == "run_once":
            result = await asyncio.to_thread(service.run_once, args["checkpoint_name"])
            return {"result": result.to_dict()}
        elif command == "run_all":
            results = await asyncio.to_thread(service.run_all)
            return {"results": [result.to_dict() for result in results]}
        elif command == "remove_checkpoint":
            return {"removed": service.remove_checkpoint(args["checkpoint_name"])}
        elif command == "start":
            service.start()
        elif command == "stop":
            await asyncio.to_thread(service.stop)
        elif command == "results":
            results = await asyncio.to_thread(
                service.get_results, timeout=0.01, max_results=args.get("max_results", 100)
            )
            return {"results": [result.to_dict() for result in results]}
        elif command != "status":
            raise MCPError("invalid_arguments")
        return {"running": service.running, "checkpoints": list(service.checkpoints)}
    if command == "persist_configuration":
        if service.is_running:
            raise MCPError("invalid_arguments")
        return {
            "policies": {
                name: policy.to_dict() if (policy := service.get_policy(name)) is not None else None
                for name in args["policy_names"]
            }
        }
    if command in {"policy", "policy_update", "policy_create"}:
        policy = service.get_policy(args["policy_name"])
        if command in {"policy_update", "policy_create"}:
            if (command == "policy_update") != (policy is not None):
                raise MCPError("invalid_arguments")
            from dataclasses import replace

            from truthound.checkpoint.escalation.protocols import (
                EscalationLevel,
                EscalationPolicy,
                EscalationTrigger,
            )
            from truthound.mcp.checkpoint_services import EscalationPolicyUpdate

            update = EscalationPolicyUpdate.model_validate(
                {key: value for key, value in args["policy_update"].items() if value is not None}
            )
            patch = update.model_dump(exclude_none=True)
            if policy is None:
                policy = EscalationPolicy.from_dict({"name": args["policy_name"], **patch})
            else:
                if "levels" in patch:
                    patch["levels"] = [
                        EscalationLevel.from_dict(level) for level in patch["levels"]
                    ]
                if "triggers" in patch:
                    patch["triggers"] = [EscalationTrigger(value) for value in patch["triggers"]]
                # Preserve opaque operator-owned values in fields not edited.
                policy = replace(policy, **patch)
            service.register_policy(policy)
        return {"policy": policy.to_dict() if policy is not None else None}
    elif command == "policy_remove":
        return {"removed": service.unregister_policy(args["policy_name"])}
    elif command == "active_records":
        return {
            "records": [r.to_dict() for r in service.get_active_escalations(args.get("policy_name"))]
        }
    elif command == "incident_records":
        return {
            "records": [r.to_dict() for r in service.get_records_for_incident(args["incident_id"])]
        }
    elif command == "start":
        await service.start()
    elif command == "stop":
        await service.stop()
    elif command == "trigger":
        from truthound.checkpoint.escalation.protocols import EscalationTrigger

        return (
            await service.trigger(
                args["incident_id"],
                context=args.get("context", {}),
                trigger_type=EscalationTrigger(args.get("trigger_type", "unacknowledged")),
                policy_name=args.get("policy_name"),
            )
        ).to_dict()
    elif command == "record":
        record = service.get_record(args["record_id"])
        return {"record": record.to_dict() if record else None}
    elif command == "acknowledge":
        return (await service.acknowledge(args["record_id"], "mcp-approved-operator")).to_dict()
    elif command == "resolve":
        return (await service.resolve(args["record_id"], "mcp-approved-operator")).to_dict()
    elif command == "cancel":
        return (await service.cancel(args["record_id"], "mcp-approved-operator")).to_dict()
    elif command == "escalate":
        return (await service.escalate(args["record_id"], force=False)).to_dict()
    elif command != "status":
        raise MCPError("invalid_arguments")
    stats = service.get_stats().to_dict()
    # Core's level counts have integer keys. Preserve their types explicitly;
    # the shared rich-value codec deliberately rejects non-string dict keys.
    stats["escalations_by_level"] = [
        {"level": level, "count": count} for level, count in stats["escalations_by_level"].items()
    ]
    return {"running": service.is_running, "stats": stats}


async def serve(wire):
    from truthound.checkpoint.escalation.engine import EscalationEngine
    from truthound.checkpoint.runner import CheckpointRunner

    service = kind = guard = None
    async with contextlib.AsyncExitStack() as stack:
        try:
            async for request in requests():
                if "open" in request:
                    if service is not None:
                        raise MCPError("invalid_arguments")
                    guard = binding_guard(request)
                    guard()
                    capabilities = stack.enter_context(approved_capabilities(request["plugins"]))
                    kind = request["open"]
                    owner = capabilities[request["capability"]]()
                    if kind == "runner":
                        service = stack.enter_context(owner)
                        if not isinstance(service, CheckpointRunner) or service.running:
                            raise MCPError("invalid_binding")
                    elif kind == "escalation":
                        service = await stack.enter_async_context(owner)
                        if not isinstance(service, EscalationEngine) or service.is_running:
                            raise MCPError("invalid_binding")
                    else:
                        raise MCPError("invalid_binding")
                    if request.get("configuration_path") is not None:
                        if kind != "escalation":
                            raise MCPError("invalid_arguments")
                        from truthound.checkpoint.escalation.protocols import EscalationPolicy

                        # Restore only declarative policies, never actions, incidents,
                        # scheduler jobs or notifications. The factory owns its store.
                        snapshot = json.loads(
                            Path(request["configuration_path"]).read_text(encoding="utf-8")
                        )
                        for name, configuration in snapshot["policies"].items():
                            if configuration is None:
                                service.unregister_policy(name)
                            else:
                                service.register_policy(EscalationPolicy.from_dict(configuration))
                else:
                    if service is None:
                        raise MCPError("invalid_handle")
                    guard()
                    result = await execute(service, kind, request["arguments"])
                    guard()
                    with BoundedWriter(Path(request["output_path"]), request["budget"]) as output:
                        if request["arguments"]["command"] == "persist_configuration":
                            from truthound.mcp.durable_checkpoint_services import json_configuration

                            value = json_configuration(result)
                        else:
                            value = encode(result)
                        output.write(json.dumps(value, allow_nan=False).encode())
                guard()
                wire.write('{"ok":true}\n')
                wire.flush()
        finally:
            # Owner exit runs after native stop. Forced revocation/timeout cannot
            # promise third-party graceful cleanup or retract already sent work.
            if isinstance(service, CheckpointRunner):
                await asyncio.to_thread(service.stop)
            elif isinstance(service, EscalationEngine):
                await service.stop()


def main():
    wire = sys.stdout
    try:
        watch_resident_parent(sys.stdin.fileno())
        with (
            open(os.devnull, "w") as sink,
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            asyncio.run(serve(wire))
    except Exception:
        wire.write('{"ok":false,"code":"checkpoint_service_failed"}\n')
        wire.flush()


if __name__ == "__main__":
    main()
