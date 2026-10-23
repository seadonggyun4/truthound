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
        if command == "start":
            service.start()
        elif command == "stop":
            await asyncio.to_thread(service.stop)
        elif command == "results":
            results = await asyncio.to_thread(
                service.get_results, timeout=0.01, max_results=args["max_results"]
            )
            return {"results": [result.to_dict() for result in results]}
        elif command != "status":
            raise MCPError("invalid_arguments")
        return {"running": service.running, "checkpoints": list(service.checkpoints)}
    if command == "start":
        await service.start()
    elif command == "stop":
        await service.stop()
    elif command == "trigger":
        return (
            await service.trigger(args["incident_id"], policy_name=args["policy_name"])
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
                else:
                    if service is None:
                        raise MCPError("invalid_handle")
                    guard()
                    result = await execute(service, kind, request["arguments"])
                    guard()
                    with BoundedWriter(Path(request["output_path"]), request["budget"]) as output:
                        output.write(json.dumps(encode(result), allow_nan=False).encode())
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
