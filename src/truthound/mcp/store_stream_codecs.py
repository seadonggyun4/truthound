"""Owned native codec streams with explicit completion and failed-state fencing."""

import io
import math
from uuid import uuid4

from truthound.mcp.broker_worker import NativeServiceResult
from truthound.mcp.governance_worker import selected
from truthound.mcp.policy import MCPError


class BoundedOutput(io.BytesIO):
    def __init__(self, owner, budget):
        super().__init__()
        self.owner, self.budget = owner, budget

    def write(self, data):
        if len(data) > remaining(self.owner, self.budget):
            raise MCPError("resource_exhausted")
        return super().write(data)


def buffered_bytes(row):
    native = row["native"]
    if row["kind"] == "compression":
        return native.buffered_bytes
    buffer = row["buffer"]
    return (0 if buffer.closed else buffer.getbuffer().nbytes) + (
        len(native._buffer) if row["direction"] == "encode" else 0
    )


def view(identity, row):
    metrics = row["native"].metrics.to_dict()
    return {
        "stream_id": identity,
        "kind": row["kind"],
        "direction": row["direction"],
        "state": row["state"],
        "buffered_bytes": buffered_bytes(row),
        "authenticated_complete": getattr(row["native"], "authenticated_complete", False) is True,
        "metrics": {
            key: value
            for key, value in metrics.items()
            if type(value) in {int, float} and math.isfinite(value)
        },
        "output": row.get("output"),
        "cleanup_uncertain": row.get("cleanup_uncertain", False),
    }


def close_native(row, *, abort):
    if row.get("closed"):
        return
    if row.get("cleanup_uncertain"):
        raise RuntimeError("Native stream cleanup outcome remains unknown")
    row["cleanup_uncertain"] = True
    try:
        # Honor the selected class's context exit exactly once. Exceptional
        # exit aborts unfinished data rather than implicitly committing it.
        error = RuntimeError("Native stream aborted") if abort else None
        row["native"].__exit__(type(error) if error is not None else None, error, None)
    finally:
        if row["buffer"] is not None:
            row["buffer"].close()
    row["closed"] = True
    row["cleanup_uncertain"] = False


def release_all(owner):
    errors = []
    for row in reversed(list(owner.streams.values())):
        try:
            close_native(row, abort=row["state"] != "finalized")
        except Exception as error:
            errors.append(error)
    owner.streams.clear()
    if errors:
        raise ExceptionGroup("Native codec stream cleanup failed", errors)


def remaining(owner, budget):
    value = (
        min(budget, 16 * 1024 * 1024)
        - sum(map(len, owner.payloads.values()))
        - owner.stream_buffer_bytes()
    )
    if value < 0:
        raise MCPError("resource_exhausted")
    return value


def execute_stream(owner, args, budget):
    from truthound.stores.compression.streaming import StreamingCompressor, StreamingDecompressor
    from truthound.stores.encryption.streaming import StreamingDecryptor, StreamingEncryptor

    owner.guard()
    command = args["command"]
    if command == "stream_open":
        if len(owner.streams) >= 8:
            raise MCPError("resource_exhausted")
        kind, direction = args["codec_kind"], args["direction"]
        binding = selected(getattr(owner.runtime, kind + "_streams"), args["codec_id"])
        buffer = None
        native = None
        try:
            if kind == "compression":
                native = (
                    binding.compressor_factory
                    if direction == "encode"
                    else binding.decompressor_factory
                )()
                expected = StreamingCompressor if direction == "encode" else StreamingDecompressor
            else:
                data = (
                    selected(owner.payloads, args["payload_id"]) if direction == "decode" else b""
                )
                if len(data) > remaining(owner, budget):
                    raise MCPError("resource_exhausted")
                buffer = io.BytesIO(data) if direction == "decode" else BoundedOutput(owner, budget)
                native = (
                    binding.encryptor_factory
                    if direction == "encode"
                    else binding.decryptor_factory
                )(buffer)
                expected = StreamingEncryptor if direction == "encode" else StreamingDecryptor
            if not isinstance(native, expected):
                raise MCPError("invalid_binding")
            native.set_execution_guard(owner.guard)
            native.__enter__()
            owner.guard()
            identity = "stream_" + uuid4().hex
            row = {
                "kind": kind,
                "direction": direction,
                "native": native,
                "buffer": buffer,
                "state": "open",
            }
            owner.streams[identity] = row
            return view(identity, row)
        except BaseException:
            if native is not None and isinstance(
                native,
                (
                    StreamingCompressor,
                    StreamingDecompressor,
                    StreamingEncryptor,
                    StreamingDecryptor,
                ),
            ):
                close_native(
                    {"native": native, "kind": kind, "direction": direction, "buffer": buffer},
                    abort=True,
                )
            elif buffer is not None:
                buffer.close()
            raise
    identity = args["stream_id"]
    row = selected(owner.streams, identity)
    if command == "stream_status":
        return view(identity, row)
    if command in {"stream_abort", "stream_release"}:
        try:
            close_native(row, abort=row["state"] != "finalized")
        except Exception:
            row["state"] = "failed"
            return NativeServiceResult(view(identity, row), "native_codec_incomplete")
        row["state"] = "aborted" if row["state"] != "finalized" else "finalized"
        result = view(identity, row)
        if command == "stream_release":
            del owner.streams[identity]
            result["released"] = True
        return result
    if row["state"] != "open":
        raise MCPError("invalid_handle")
    native, kind, direction = row["native"], row["kind"], row["direction"]
    if command == "stream_read" and (kind != "encryption" or direction != "decode"):
        raise MCPError("invalid_arguments")
    if command == "stream_write" and kind == "encryption" and direction == "decode":
        raise MCPError("invalid_arguments")
    if command == "stream_flush" and direction != "encode":
        raise MCPError("invalid_arguments")
    result = None
    row["output"] = None
    try:
        if command == "stream_write":
            data = selected(owner.payloads, args["payload_id"])
            if len(data) > remaining(owner, budget):
                raise MCPError("resource_exhausted")
            value = native.write(data)
            owner.guard()
            if kind == "compression":
                result = value if value else None
            elif type(value) is not int or value != len(data):
                raise TypeError("Native stream write receipt mismatch")
        elif command == "stream_flush":
            value = native.flush()
            result = value if kind == "compression" and value else None
        elif command == "stream_read":
            result = native.read_chunk()
        elif command == "stream_finalize":
            if kind == "compression":
                value = native.finalize()
                result = value if value else None
            elif direction == "encode":
                native.finalize()
                result = row["buffer"].getvalue()
            else:
                result = native.read_all()
                if native.authenticated_complete is not True:
                    raise TypeError("Native stream completion was not authenticated")
            owner.guard()
            close_native(row, abort=False)
            row["state"] = "finalized"
        else:
            raise MCPError("invalid_arguments")
        owner.guard()
        remaining(owner, budget)
        if result is not None:
            row["output"] = owner.retain(result, budget)
        return view(identity, row)
    except MCPError:
        row["state"] = "failed"
        raise
    except Exception:
        row["state"] = "failed"
        return NativeServiceResult(view(identity, row), "native_codec_incomplete")
