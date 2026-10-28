"""Resident native services; operator factories own connections and cleanup."""

import asyncio
import contextlib
import json
import os
import sys
from pathlib import Path

from truthound.mcp.checkpoint_worker import binding_guard
from truthound.mcp.encoding import encode
from truthound.mcp.native_stdio import native_worker_wire
from truthound.mcp.parent_pipe import watch_resident_parent
from truthound.mcp.plugin_bindings import approved_capabilities
from truthound.mcp.policy import MCPError
from truthound.mcp.resources import BoundedWriter


def failure_code(error):
    """Closed diagnostic categories only: never serialize plugin exception text."""
    if isinstance(error, MCPError) and error.code in {
        "invalid_arguments",
        "invalid_binding",
        "access_denied",
        "plugin_identity_mismatch",
        "resource_exhausted",
        "unsupported_value",
        "invalid_handle",
    }:
        return "checkpoint_service_" + error.code
    for kind, label in (
        (OSError, "os_error"),
        (TypeError, "type_error"),
        (ValueError, "value_error"),
        (RuntimeError, "runtime_error"),
    ):
        if isinstance(error, kind):
            return "checkpoint_service_" + label
    return "checkpoint_service_failed"


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


def runner_configuration(service, names):
    from dataclasses import asdict

    from truthound.checkpoint.triggers.base import TriggerConfig
    from truthound.checkpoint.triggers.schedule import CronTrigger, ScheduleTrigger
    from truthound.mcp.checkpoint_services import CheckpointDefinitionUpdate, ScheduleUpdate

    if service.running or service._active_runs or (service._thread and service._thread.is_alive()):
        raise MCPError("invalid_arguments")
    output = {}
    for name in names:
        checkpoint = service.checkpoints.get(name)
        if checkpoint is None:
            output[name] = None
            continue
        settings = {
            key: getattr(checkpoint.config, key)
            for key in CheckpointDefinitionUpdate.model_fields
            if key != "checks"
        }
        checks = getattr(service, "_definition_checks", {}).get(name)
        if checks is not None:
            settings["checks"] = checks
        # Only common settings are portable for provider-defined triggers.
        triggers = {}
        if any(not trigger.config.name for trigger in checkpoint.triggers):
            raise MCPError("unsupported_value")
        if len({trigger.name for trigger in checkpoint.triggers}) != len(checkpoint.triggers):
            raise MCPError("unsupported_value")
        for trigger in checkpoint.triggers:
            fields = (
                ScheduleUpdate.model_fields
                if type(trigger) in {ScheduleTrigger, CronTrigger}
                else TriggerConfig.__dataclass_fields__
            )
            config = asdict(trigger.config)
            patch = {key: value for key, value in config.items() if key in fields and key != "name"}
            triggers[trigger.name] = ScheduleUpdate.model_validate(patch).model_dump(
                mode="json", exclude_unset=True
            )
        output[name] = {
            "settings": CheckpointDefinitionUpdate.model_validate(settings).model_dump(
                mode="json", exclude_unset=True
            ),
            "triggers": triggers,
        }
    return {"checkpoints": output}


async def execute(service, kind, args):
    command = args["command"]
    if command == "recovery_status":
        return service.recovery_status(
            limit=args.get("recovery_limit", 100), offset=args.get("recovery_offset", 0)
        )
    if command in {"recovery_resolve", "recovery_forget"}:
        args_values = (
            (args["attempt_id"], args["resolution"])
            if command == "recovery_resolve"
            else (args["attempt_ids"],)
        )
        method = getattr(service, command)
        if kind == "runner":
            return await asyncio.to_thread(method, *args_values)
        return await method(*args_values)
    if kind == "runner":
        if command == "persist_configuration":
            return runner_configuration(service, args["checkpoint_names"])
        if command == "checkpoint_update":
            from truthound.mcp.checkpoint_services import CheckpointDefinitionUpdate
            from truthound.mcp.validator_wire import constructor, settings

            update = CheckpointDefinitionUpdate.model_validate(args["checkpoint_update"])
            patch = update.model_dump(exclude_unset=True, exclude={"checks"})
            if update.checks is not None:
                patch["validators"] = [
                    constructor(check.validator)(
                        **settings(check.validator, check.settings, build=True)
                    )
                    for check in update.checks
                ]
                patch["validator_config"] = {}
            checkpoint = service.edit_checkpoint(args["checkpoint_name"], patch)
            if update.checks is not None:
                if not hasattr(service, "_definition_checks"):
                    service._definition_checks = {}
                service._definition_checks[checkpoint.name] = [
                    check.model_dump() for check in update.checks
                ]
            # Project only edited declarative fields, never the opaque source/transport.
            return {
                "checkpoint": checkpoint.name,
                "settings": update.model_dump(exclude_unset=True),
            }
        if command == "schedule_update":
            from dataclasses import asdict

            from truthound.mcp.checkpoint_services import ScheduleUpdate

            changes = ScheduleUpdate.model_validate(args["schedule_update"])
            trigger = service.edit_schedule(
                args["checkpoint_name"],
                args["trigger_name"],
                changes.model_dump(exclude_unset=True),
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
            "records": [
                r.to_dict() for r in service.get_active_escalations(args.get("policy_name"))
            ]
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
                if request == {"prepare": True}:
                    if service is not None:
                        raise MCPError("invalid_arguments")
                    # Core imports and transport are ready. No operator binding
                    # or factory has been loaded and no native effect started.
                    wire.write('{"ok":true}\n')
                    wire.flush()
                    continue
                if request == {"close": True}:
                    break
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
                        service.set_execution_guard(guard)
                        service._ensure_owner()
                    elif kind == "escalation":
                        service = await stack.enter_async_context(owner)
                        if not isinstance(service, EscalationEngine) or service.is_running:
                            raise MCPError("invalid_binding")
                        service.set_execution_guard(guard)
                        service._ensure_owner()
                    else:
                        raise MCPError("invalid_binding")
                    service.bind_recovery_scope(request["recovery_scope"])
                    if request.get("configuration_path") is not None:
                        from truthound.checkpoint.escalation.protocols import EscalationPolicy

                        snapshot = json.loads(
                            Path(request["configuration_path"]).read_text(encoding="utf-8")
                        )
                        if kind == "escalation":
                            for name, configuration in snapshot["policies"].items():
                                if configuration is None:
                                    service.unregister_policy(name)
                                else:
                                    service.register_policy(
                                        EscalationPolicy.from_dict(configuration)
                                    )
                        else:
                            for name, configuration in snapshot["checkpoints"].items():
                                if configuration is None:
                                    service.remove_checkpoint(name)
                                    continue
                                await execute(
                                    service,
                                    kind,
                                    {
                                        "command": "checkpoint_update",
                                        "checkpoint_name": name,
                                        "checkpoint_update": configuration["settings"],
                                    },
                                )
                                for trigger_name, changes in configuration["triggers"].items():
                                    await execute(
                                        service,
                                        kind,
                                        {
                                            "command": "schedule_update",
                                            "checkpoint_name": name,
                                            "trigger_name": trigger_name,
                                            "schedule_update": changes,
                                        },
                                    )
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
                await asyncio.to_thread(service.close)
            elif isinstance(service, EscalationEngine):
                await service.close()


def main():
    wire = native_worker_wire()
    try:
        watch_resident_parent(sys.stdin.fileno())
        asyncio.run(serve(wire))
    except Exception as error:
        wire.write(json.dumps({"ok": False, "code": failure_code(error)}) + "\n")
        wire.flush()
    finally:
        wire.close()


if __name__ == "__main__":
    main()
