"""Explicit native storage utility commands inside one approved factory lifetime."""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from uuid import uuid4

from pydantic import TypeAdapter

from truthound.mcp.broker_worker import NativeServiceResult
from truthound.mcp.checkpoint_worker import binding_guard
from truthound.mcp.contracts import Name
from truthound.mcp.governance_worker import selected
from truthound.mcp.policy import MCPError
from truthound.mcp.store_utilities import (
    StoreCompressionStreamBinding,
    StoreEncryptionStreamBinding,
    StoreKeyBinding,
    StoreUtilityRuntime,
)
from truthound.mcp.tables import decode


class StoreUtilityOwner:
    def __init__(self, runtime, request):
        from truthound.stores.backpressure.base import BaseBackpressure
        from truthound.stores.compression.adaptive import AdaptiveCompressor
        from truthound.stores.compression.pipeline import CompressionPipeline
        from truthound.stores.compression.streaming import ChunkedCompressor
        from truthound.stores.encryption.keys import KeyManager
        from truthound.stores.encryption.pipeline import EncryptionPipeline
        from truthound.stores.encryption.streaming import ChunkedEncryptor

        if request["native_group"] != "store_utilities" or not isinstance(
            runtime, StoreUtilityRuntime
        ):
            raise MCPError("invalid_binding")
        mappings = (
            runtime.compression,
            runtime.encryption,
            runtime.adaptive,
            runtime.chunked_compression,
            runtime.chunked_encryption,
            runtime.backpressure,
            runtime.payloads,
            runtime.slots,
            runtime.keys,
            runtime.compression_streams,
            runtime.encryption_streams,
        )
        alias = TypeAdapter(Name)
        for mapping in mappings:
            for key in mapping:
                alias.validate_python(key)
        if (
            any(
                not isinstance(value, CompressionPipeline) for value in runtime.compression.values()
            )
            or any(
                not isinstance(value, EncryptionPipeline) for value in runtime.encryption.values()
            )
            or any(not isinstance(value, AdaptiveCompressor) for value in runtime.adaptive.values())
            or any(
                not isinstance(value, ChunkedCompressor)
                for value in runtime.chunked_compression.values()
            )
            or any(
                not isinstance(value, ChunkedEncryptor)
                for value in runtime.chunked_encryption.values()
            )
            or any(
                not isinstance(value, BaseBackpressure) for value in runtime.backpressure.values()
            )
            or any(
                not isinstance(value, StoreCompressionStreamBinding)
                or not callable(value.compressor_factory)
                or not callable(value.decompressor_factory)
                for value in runtime.compression_streams.values()
            )
            or any(
                not isinstance(value, StoreEncryptionStreamBinding)
                or not callable(value.encryptor_factory)
                or not callable(value.decryptor_factory)
                for value in runtime.encryption_streams.values()
            )
            or any(not isinstance(value, bytes) for value in runtime.payloads.values())
            or any(
                not isinstance(value, StoreKeyBinding)
                or not isinstance(value.manager, KeyManager)
                or not isinstance(value.native_id, str)
                or not value.native_id
                for value in runtime.keys.values()
            )
            or any(
                not isinstance(value, Path) or not value.is_absolute()
                for value in runtime.slots.values()
            )
        ):
            raise MCPError("invalid_binding")
        self.runtime = runtime
        self.guard = binding_guard(request)
        for value in (
            *runtime.compression.values(),
            *runtime.encryption.values(),
            *runtime.adaptive.values(),
            *runtime.chunked_compression.values(),
            *runtime.chunked_encryption.values(),
            *runtime.backpressure.values(),
        ):
            value.set_execution_guard(self.guard)
        self.streams = {}
        self.payloads = dict(runtime.payloads)
        self.uncertain = set()
        self.uncertain_keys = set()
        self.codec_keys = {}
        for value in runtime.keys.values():
            value.manager.set_execution_guard(self.guard)
        self.shutdown_requested = False
        if len(self.payloads) > 32 or sum(map(len, self.payloads.values())) > 16 * 1024 * 1024:
            raise MCPError("resource_exhausted")

    def retain(self, value, budget):
        if not isinstance(value, bytes):
            raise MCPError("unsupported_value")
        if len(self.payloads) >= 32 or len(value) + sum(
            map(len, self.payloads.values())
        ) + self.stream_buffer_bytes() > min(budget, 16 * 1024 * 1024):
            raise MCPError("resource_exhausted")
        identity = "payload_" + uuid4().hex
        self.payloads[identity] = value
        return self.payload_view(identity)

    def stream_buffer_bytes(self):
        from truthound.mcp.store_stream_codecs import buffered_bytes

        return sum(buffered_bytes(row) for row in self.streams.values())

    def payload_view(self, identity):
        value = selected(self.payloads, identity)
        return {
            "payload_id": identity,
            "size_bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }

    def pressure_view(self, identity, pressure):
        metrics = pressure.metrics.to_dict()
        # Native numeric metrics and enum states only; no callback/config reprs.
        values = {
            key: value
            for key, value in metrics.items()
            if type(value) in (int, float) and math.isfinite(value)
        }
        if (
            type(pressure.state.is_paused) is not bool
            or isinstance(pressure.state.current_rate, bool)
            or not isinstance(pressure.state.current_rate, (int, float))
            or not math.isfinite(pressure.state.current_rate)
            or pressure.state.current_rate < 0
        ):
            raise MCPError("unsupported_value")
        return {
            "pressure_id": identity,
            "uncertain": identity in self.uncertain,
            "paused": pressure.state.is_paused,
            "level": pressure.state.pressure_level.value,
            "current_rate": pressure.state.current_rate,
            "metrics": values,
        }

    def key_view(self, identity, value):
        from truthound.stores.encryption.base import EncryptionKey

        if (
            not isinstance(value, EncryptionKey)
            or type(value.version) is not int
            or value.version <= 0
        ):
            raise MCPError("unsupported_value")
        return {
            "key_id": identity,
            "version": value.version,
            "algorithm": value.algorithm.value,
            "key_type": value.key_type.value,
            "created_at": value.created_at.isoformat(),
            "expires_at": value.expires_at.isoformat() if value.expires_at else None,
            "expired": value.is_expired,
            "uncertain": identity in self.uncertain_keys,
        }

    def latch_keys(self, manager):
        self.uncertain_keys.update(
            alias for alias, value in self.runtime.keys.items() if value.manager is manager
        )

    async def execute(self, args, budget, *, inputs=None):
        self.guard()
        command = args["command"]
        if command == "inspect":
            limit = args.get("limit", 100)
            return {
                "compression": sorted(self.runtime.compression)[:limit],
                "encryption": sorted(self.runtime.encryption)[:limit],
                "backpressure": sorted(self.runtime.backpressure)[:limit],
                "slots": sorted(self.runtime.slots)[:limit],
                "adaptive": sorted(self.runtime.adaptive)[:limit],
                "chunked_compression": sorted(self.runtime.chunked_compression)[:limit],
                "chunked_encryption": sorted(self.runtime.chunked_encryption)[:limit],
                "keys": sorted(self.runtime.keys)[:limit],
                "compression_streams": sorted(self.runtime.compression_streams)[:limit],
                "encryption_streams": sorted(self.runtime.encryption_streams)[:limit],
                "payloads": [self.payload_view(key) for key in list(self.payloads)[:limit]],
            }
        if command == "shutdown":
            self.shutdown_requested = True
            return {"shutdown": True}
        if command.startswith("stream_"):
            from truthound.mcp.store_stream_codecs import execute_stream

            return execute_stream(self, args, budget)
        if command.startswith("key_"):
            from truthound.stores.encryption.base import KeyError_

            identity = args["key_id"]
            slot = selected(self.runtime.keys, identity)
            if identity in self.uncertain_keys and command not in {"key_status", "key_history"}:
                return NativeServiceResult(
                    {"key_id": identity, "uncertain": True}, "native_key_incomplete"
                )
            try:
                if command == "key_cleanup_expired":
                    deleted = slot.manager.cleanup_expired()
                    self.guard()
                    if type(deleted) is not int or deleted < 0:
                        raise TypeError("Native expired-key deletion receipt must be an integer")
                    return {"deleted": deleted}
                if command == "key_history":
                    values = [
                        key
                        for key in slot.manager.list_keys(include_expired=True)
                        if key.key_id == slot.native_id
                        or key.key_id == slot.native_id + "_v" + str(key.version)
                    ]
                    self.guard()
                    values.sort(key=lambda key: key.version, reverse=True)
                    return {
                        "keys": [
                            self.key_view(identity, value)
                            for value in values[: args.get("limit", 100)]
                        ],
                        "truncated": len(values) > args.get("limit", 100),
                    }
                if command == "key_delete" and args.get("key_version") is None:
                    deleted = slot.manager.delete_key(slot.native_id)
                    if type(deleted) is not bool:
                        raise TypeError("Native key deletion receipt must be boolean")
                    return {"key_id": identity, "deleted": deleted}
                if command == "key_create":
                    value = slot.manager.create_key(slot.native_id)
                elif command == "key_get_or_create":
                    value = slot.manager.get_or_create_key(slot.native_id)
                elif command == "key_rotate":
                    value = slot.manager.rotate_key(
                        slot.native_id, archive_old=args.get("archive_old", True)
                    )
                else:
                    version = args.get("key_version")
                    if version is not None:
                        try:
                            value = slot.manager.get_key(
                                slot.native_id + "_v" + str(version),
                                validate=command == "key_apply",
                            )
                        except KeyError_:
                            value = slot.manager.get_key(
                                slot.native_id, validate=command == "key_apply"
                            )
                            if value.version != version:
                                raise MCPError("invalid_arguments") from None
                    else:
                        value = slot.manager.get_key(
                            slot.native_id, validate=command == "key_apply"
                        )
                    if command == "key_apply":
                        codec = selected(self.runtime.encryption, args["codec_id"])
                        algorithms = codec.get_encryption_algorithms()
                        if not algorithms or any(
                            algorithm != value.algorithm for algorithm in algorithms
                        ):
                            raise MCPError("invalid_arguments")
                        codec.set_key(value.key_material)
                        self.codec_keys[args["codec_id"]] = (
                            slot.manager,
                            value.key_id,
                            value.version,
                            hashlib.sha256(value.key_material).digest(),
                        )
                    elif command == "key_delete":
                        deleted = slot.manager.delete_key(value.key_id)
                        self.guard()
                        if type(deleted) is not bool:
                            raise TypeError("Native key deletion receipt must be boolean")
                        return {"key_id": identity, "deleted": deleted}
                self.guard()
                return self.key_view(identity, value)
            except MCPError:
                raise
            except KeyError_:
                if command == "key_status":
                    return {
                        "key_id": identity,
                        "found": False,
                        "uncertain": identity in self.uncertain_keys,
                    }
                self.latch_keys(slot.manager)
                return NativeServiceResult(
                    {"key_id": identity, "uncertain": True}, "native_key_incomplete"
                )
            except Exception:
                self.guard()
                self.latch_keys(slot.manager)
                return NativeServiceResult(
                    {"key_id": identity, "uncertain": True}, "native_key_incomplete"
                )
        if command in {"payload_capture", "payload_import"}:
            if not inputs or set(inputs) != {"data_path"}:
                raise MCPError("invalid_handle")
            path = Path(inputs["data_path"])
            if path.stat().st_size > min(budget, 16 * 1024 * 1024):
                raise MCPError("resource_exhausted")
            data = path.read_bytes()
            if command == "payload_import":
                try:
                    value = decode(json.loads(data))
                except (ValueError, TypeError):
                    raise MCPError("invalid_handle") from None
                if (
                    not isinstance(value, dict)
                    or set(value) != {"data"}
                    or not isinstance(value["data"], bytes)
                ):
                    raise MCPError("invalid_handle")
                data = value["data"]
            self.guard()
            return self.retain(data, budget)
        if command in {"payload_status", "payload_release", "payload_export"}:
            value = selected(self.payloads, args["payload_id"])
            if command == "payload_status":
                return self.payload_view(args["payload_id"])
            if command == "payload_release":
                del self.payloads[args["payload_id"]]
                return {"released": True}
            return {"data": value}
        if command in {
            "codec_process",
            "codec_reverse",
            "codec_range",
            "codec_analyze",
            "codec_recommend",
        } and args["codec_kind"] in {"adaptive", "chunked_compression", "chunked_encryption"}:
            from truthound.mcp.store_codec_extensions import execute_codec

            return execute_codec(self, args, budget)
        if command in {"codec_process", "codec_reverse"}:
            data = selected(self.payloads, args["payload_id"])
            codec = selected(
                self.runtime.compression
                if args["codec_kind"] == "compression"
                else self.runtime.encryption,
                args["codec_id"],
            )
            if args["codec_kind"] == "encryption" and args["codec_id"] in self.codec_keys:
                manager, native_id, version, digest = self.codec_keys[args["codec_id"]]
                if any(
                    alias in self.uncertain_keys
                    for alias, bound in self.runtime.keys.items()
                    if bound.manager is manager
                ):
                    return NativeServiceResult(
                        {"codec_id": args["codec_id"], "uncertain_key": True},
                        "native_key_incomplete",
                    )
                key = manager.get_key(native_id)
                if key.version != version or hashlib.sha256(key.key_material).digest() != digest:
                    raise MCPError("invalid_handle")
            maximum = min(budget - sum(map(len, self.payloads.values())), 16 * 1024 * 1024)
            if maximum <= 0:
                raise MCPError("resource_exhausted")
            if args["codec_kind"] == "encryption":
                if command == "codec_process":
                    data = codec.process_to_bytes(data, max_output_bytes=maximum)
                else:
                    data = codec.reverse_from_bytes(data, max_output_bytes=maximum)
            elif command == "codec_process":
                data = codec.process(data, max_output_bytes=maximum).data
            else:
                data = codec.reverse(data, max_output_bytes=maximum)
            self.guard()
            return self.retain(data, budget)
        if command.startswith("slot_"):
            path = selected(self.runtime.slots, args["slot_id"])
            if command == "slot_delete":
                existed = path.exists()
                self.guard()
                path.unlink(missing_ok=True)
                return {"deleted": existed}
            if command == "slot_load":
                if path.stat().st_size > min(budget, 16 * 1024 * 1024):
                    raise MCPError("resource_exhausted")
                data = path.read_bytes()
                self.guard()
                return self.retain(data, budget)
            data = selected(self.payloads, args["payload_id"])
            descriptor, name = tempfile.mkstemp(prefix=".truthound-payload-", dir=path.parent)
            temporary = Path(name)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                self.guard()
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            return {"saved": True, "size_bytes": len(data)}
        if command.startswith("pressure_"):
            identity = args["pressure_id"]
            pressure = selected(self.runtime.backpressure, identity)
            if command == "pressure_status":
                return self.pressure_view(identity, pressure)
            if identity in self.uncertain and command != "pressure_reset":
                return NativeServiceResult(
                    self.pressure_view(identity, pressure), "native_pressure_incomplete"
                )
            try:
                if command == "pressure_acquire":
                    verdict = await pressure.acquire()
                    self.guard()
                    if type(verdict) is not bool:
                        raise TypeError("Native backpressure verdict must be boolean")
                    return {"acquired": verdict, "state": self.pressure_view(identity, pressure)}
                if command == "pressure_update":
                    pressure.update_metrics(
                        **{
                            key: args[key]
                            for key in ("memory_percent", "queue_depth", "latency_ms")
                            if args.get(key) is not None
                        }
                    )
                elif command == "pressure_release":
                    pressure.release()
                elif command == "pressure_reset":
                    pressure.reset()
                    self.guard()
                    self.uncertain.discard(identity)
                else:
                    raise MCPError("invalid_arguments")
                self.guard()
            except MCPError:
                raise
            except Exception:
                self.guard()
                self.uncertain.add(identity)
                return NativeServiceResult(
                    self.pressure_view(identity, pressure), "native_pressure_incomplete"
                )
            return self.pressure_view(identity, pressure)
        raise MCPError("invalid_arguments")

    async def close(self):
        from truthound.mcp.store_stream_codecs import release_all

        try:
            release_all(self)
        finally:
            # Cleanup never implicitly finalizes unfinished codec streams.
            self.payloads.clear()


if __name__ == "__main__":
    from truthound.mcp.broker_worker import main

    main(StoreUtilityOwner)
