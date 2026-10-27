"""Resident broker owner; no wire-selected transport, credentials or topic."""

import asyncio
import contextlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from truthound.mcp.brokers import BrokerRuntime
from truthound.mcp.checkpoint_service_worker import failure_code, requests
from truthound.mcp.checkpoint_worker import binding_guard
from truthound.mcp.encoding import encode
from truthound.mcp.parent_pipe import watch_resident_parent
from truthound.mcp.plugin_bindings import approved_capabilities
from truthound.mcp.policy import MCPError
from truthound.mcp.resources import BoundedWriter


@dataclass(frozen=True)
class NativeServiceResult:
    """Private reply status is distinct from a retained native payload."""

    payload: object
    failure_code: str


@dataclass(frozen=True)
class NativeServiceBytes:
    """Native rendered bytes; the parent selects the fixed command media type."""

    content: bytes


class BrokerOwner:
    def __init__(self, runtime, request=None):
        from truthound.realtime.adapters.base import BaseStreamAdapter
        from truthound.realtime.processing.exactly_once import ExactlyOnceProcessor
        from truthound.realtime.processing.windows import WindowProcessor

        if not isinstance(runtime, BrokerRuntime) or not isinstance(
            runtime.adapter, BaseStreamAdapter
        ):
            raise MCPError("invalid_binding")
        if runtime.adapter.is_connected:
            raise MCPError("invalid_binding")
        if runtime.window is not None and not isinstance(runtime.window, WindowProcessor):
            raise MCPError("invalid_binding")
        if runtime.processor is not None and not isinstance(
            runtime.processor, ExactlyOnceProcessor
        ):
            raise MCPError("invalid_binding")
        self.runtime = runtime
        self.batches = {}
        self.windowed = set()
        self.processed = set()

    async def execute(self, args, budget):
        from truthound.realtime.protocols import StreamMessage

        runtime, command = self.runtime, args["command"]
        adapter = runtime.adapter
        if command == "connect":
            await adapter.connect()
        elif command == "disconnect":
            await adapter.disconnect()
            self.batches.clear()
            self.windowed.clear()
            self.processed.clear()
        elif command == "consume":
            if len(self.batches) >= 16:
                raise MCPError("resource_exhausted")
            batch = await asyncio.wait_for(
                adapter.consume_batch(args.get("limit", 100), args.get("timeout_ms", 1000)),
                args.get("timeout_ms", 1000) / 1000,
            )
            value = [asdict(message) for message in batch.messages]
            size = len(json.dumps(encode(value), allow_nan=False).encode())
            if size + sum(row[1] for row in self.batches.values()) > budget:
                raise MCPError("resource_exhausted")
            identity = "batch_" + uuid4().hex
            self.batches[identity] = (batch, size)
            return {"batch_id": identity, "messages": value, "size": batch.size}
        elif command in {"ack", "window_add", "process"}:
            identity = args["batch_id"]
            if identity not in self.batches:
                raise MCPError("invalid_handle")
            batch = self.batches[identity][0]
            if command == "ack":
                await adapter.commit_batch(batch.messages)
                del self.batches[identity]
                self.windowed.discard(identity)
                self.processed.discard(identity)
                return {"acknowledged": batch.size}
            if command == "window_add":
                if runtime.window is None or identity in self.windowed:
                    raise MCPError("invalid_arguments")
                output = []
                for message in batch.messages:
                    output.extend(await runtime.window.add(message))
                self.windowed.add(identity)
                return {"windows": [result.to_dict() for result in output]}
            if runtime.processor is None or identity in self.processed:
                raise MCPError("invalid_arguments")
            results = [
                await runtime.processor.process_message(message) for message in batch.messages
            ]
            self.processed.add(identity)
            return {
                "results": [
                    {
                        "success": result.success,
                        "result": result.result,
                        "deduplicated": result.deduplicated,
                        "error": "native_processing_failed" if result.error else None,
                    }
                    for result in results
                ]
            }
        elif command == "seek":
            if self.batches:
                raise MCPError("invalid_arguments")
            await adapter.seek(args["partition"], args["offset"])
        elif command == "produce":
            if not runtime.output_topic:
                raise MCPError("invalid_binding")
            from truthound.mcp.brokers import BrokerMessage

            message = BrokerMessage.model_validate(args["message"])
            await adapter.produce(
                StreamMessage(
                    key=message.key,
                    value=message.value,
                    partition=message.partition,
                    offset=0,
                    timestamp=message.timestamp,
                    topic=runtime.output_topic,
                )
            )
        elif command == "flush":
            await adapter.flush()
        elif command == "window_flush":
            if runtime.window is None:
                raise MCPError("invalid_binding")
            return {"windows": [result.to_dict() for result in await runtime.window.force_emit()]}
        elif command == "recover":
            if runtime.processor is None:
                raise MCPError("invalid_binding")
            return {"rolled_back": await runtime.processor.recover()}
        elif command != "status":
            raise MCPError("invalid_arguments")
        return {
            "state": adapter.state.value,
            "connected": adapter.is_connected,
            "metrics": asdict(adapter.metrics),
            "retained_batches": len(self.batches),
            "processing": runtime.processor.stats if runtime.processor else None,
            "watermark": runtime.window.watermark if runtime.window else None,
        }

    async def close(self):
        await self.runtime.adapter.disconnect()


async def serve(wire, owner_type=BrokerOwner):
    owner = None
    async with contextlib.AsyncExitStack() as stack:
        try:
            async for request in requests():
                if request == {"close": True}:
                    break
                reply = {"ok": True}
                shutdown = False
                if request.get("open"):
                    if owner is not None:
                        raise MCPError("invalid_arguments")
                    guard = binding_guard(request)
                    guard()
                    capabilities = stack.enter_context(approved_capabilities(request["plugins"]))
                    startup = contextlib.nullcontext()
                    if request.get("native_group") == "execution_services":
                        from truthound.execution.distributed._spark_startup import (
                            resident_gateway_launcher,
                        )

                        startup = resident_gateway_launcher()
                    with startup:
                        runtime = await stack.enter_async_context(capabilities[request["capability"]]())
                    owner = owner_type(runtime, request)
                else:
                    if owner is None:
                        raise MCPError("invalid_handle")
                    guard()
                    if request.get("inputs"):
                        value = await owner.execute(
                            request["arguments"], request["budget"], inputs=request["inputs"]
                        )
                    else:
                        value = await owner.execute(request["arguments"], request["budget"])
                    guard()
                    shutdown = getattr(owner, "shutdown_requested", False)
                    if shutdown:
                        # Acknowledge native teardown only after children and
                        # the approved factory have actually finished. This
                        # uses the existing command deadline, not close's
                        # short forced-cleanup fallback.
                        closing_owner, owner = owner, None
                        await closing_owner.close()
                        await stack.aclose()
                        reply["shutdown"] = True
                    if isinstance(value, NativeServiceResult):
                        reply["native_failed"] = value.failure_code
                        value = value.payload
                    with BoundedWriter(Path(request["output_path"]), request["budget"]) as output:
                        content = value.content if isinstance(value, NativeServiceBytes) else json.dumps(encode(value), allow_nan=False).encode()
                        output.write(content)
                guard()
                wire.write(json.dumps(reply) + "\n")
                wire.flush()
        finally:
            if owner is not None:
                await owner.close()


def main(owner_type=BrokerOwner):
    wire = sys.stdout
    try:
        watch_resident_parent(sys.stdin.fileno())
        with (
            open(os.devnull, "w", encoding="utf-8") as sink,
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            asyncio.run(serve(wire, owner_type))
    except Exception as error:
        wire.write(json.dumps({"ok": False, "code": failure_code(error)}) + "\n")
        wire.flush()


if __name__ == "__main__":
    main()
