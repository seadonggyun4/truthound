"""Operator-bound native storage codecs, private payload slots and flow control."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from truthound.mcp.approved_sources import SourceBindingOpen
from truthound.mcp.bound_services import BoundServiceSession
from truthound.mcp.checkpoint_services import ServiceProcess
from truthound.mcp.contracts import Name
from truthound.mcp.policy import MCPError
from truthound.mcp.session_contracts import Handle


@dataclass
class StoreKeyBinding:
    manager: Any
    native_id: str


@dataclass
class StoreCompressionStreamBinding:
    compressor_factory: Any
    decompressor_factory: Any


@dataclass
class StoreEncryptionStreamBinding:
    encryptor_factory: Any
    decryptor_factory: Any


@dataclass
class StoreUtilityRuntime:
    compression: dict[str, Any] = field(default_factory=dict)
    encryption: dict[str, Any] = field(default_factory=dict)
    adaptive: dict[str, Any] = field(default_factory=dict)
    chunked_compression: dict[str, Any] = field(default_factory=dict)
    chunked_encryption: dict[str, Any] = field(default_factory=dict)
    backpressure: dict[str, Any] = field(default_factory=dict)
    payloads: dict[str, bytes] = field(default_factory=dict)
    slots: dict[str, Path] = field(default_factory=dict)
    keys: dict[str, StoreKeyBinding] = field(default_factory=dict)
    compression_streams: dict[str, StoreCompressionStreamBinding] = field(default_factory=dict)
    encryption_streams: dict[str, StoreEncryptionStreamBinding] = field(default_factory=dict)


class StoreUtilityCommand(SourceBindingOpen):
    service_id: Handle
    command: Literal[
        "inspect",
        "payload_capture",
        "payload_import",
        "payload_export",
        "payload_release",
        "payload_status",
        "codec_process",
        "codec_reverse",
        "codec_range",
        "codec_analyze",
        "codec_recommend",
        "slot_save",
        "slot_load",
        "slot_delete",
        "pressure_status",
        "pressure_update",
        "pressure_acquire",
        "pressure_release",
        "pressure_reset",
        "key_status",
        "key_create",
        "key_get_or_create",
        "key_rotate",
        "key_delete",
        "key_apply",
        "key_history",
        "key_cleanup_expired",
        "stream_open",
        "stream_write",
        "stream_read",
        "stream_flush",
        "stream_finalize",
        "stream_abort",
        "stream_status",
        "stream_release",
        "shutdown",
    ]
    stream_id: Name | None = None
    direction: Literal["encode", "decode"] | None = None
    payload_id: Name | None = None
    codec_id: Name | None = None
    codec_kind: (
        Literal[
            "compression", "encryption", "adaptive", "chunked_compression", "chunked_encryption"
        ]
        | None
    ) = None
    start: Annotated[int, Field(ge=0, le=16_777_216)] | None = None
    end: Annotated[int, Field(ge=0, le=16_777_216)] | None = None
    pressure_id: Name | None = None
    slot_id: Name | None = None
    key_id: Name | None = None
    key_version: Annotated[int, Field(ge=1, le=1_000_000)] | None = None
    archive_old: bool = True
    asset_id: Handle | None = None
    artifact_id: Handle | None = None
    memory_percent: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)] | None = None
    queue_depth: Annotated[int, Field(ge=0, le=1_000_000_000)] | None = None
    latency_ms: Annotated[float, Field(ge=0, le=86_400_000, allow_inf_nan=False)] | None = None
    limit: Annotated[int, Field(ge=1, le=1000)] = 100

    @model_validator(mode="after")
    def shape(self):
        required = {
            "payload_id": {
                "payload_export",
                "payload_release",
                "payload_status",
                "codec_process",
                "codec_reverse",
                "codec_range",
                "codec_analyze",
                "codec_recommend",
                "stream_write",
                "slot_save",
            },
            "codec_id": {
                "codec_process",
                "codec_reverse",
                "codec_range",
                "codec_analyze",
                "codec_recommend",
                "key_apply",
                "stream_open",
            },
            "codec_kind": {
                "codec_process",
                "codec_reverse",
                "codec_range",
                "codec_analyze",
                "codec_recommend",
                "stream_open",
            },
            "direction": {"stream_open"},
            "stream_id": {
                "stream_write",
                "stream_read",
                "stream_flush",
                "stream_finalize",
                "stream_abort",
                "stream_status",
                "stream_release",
            },
            "start": {"codec_range"},
            "end": {"codec_range"},
            "pressure_id": {
                "pressure_status",
                "pressure_update",
                "pressure_acquire",
                "pressure_release",
                "pressure_reset",
            },
            "slot_id": {"slot_save", "slot_load", "slot_delete"},
            "key_id": {
                "key_status",
                "key_create",
                "key_get_or_create",
                "key_rotate",
                "key_delete",
                "key_apply",
                "key_history",
                "key_cleanup_expired",
            },
        }
        for name, commands in required.items():
            expected = self.command in commands
            if name == "payload_id" and self.command == "stream_open":
                expected = self.codec_kind == "encryption" and self.direction == "decode"
            if expected != (getattr(self, name) is not None):
                raise ValueError("Field does not match command")
        if self.command == "stream_open" and self.codec_kind not in {"compression", "encryption"}:
            raise ValueError("Stream codec kind is not supported")
        if self.command == "codec_range" and (
            self.codec_kind not in {"chunked_compression", "chunked_encryption"}
            or self.start > self.end
        ):
            raise ValueError("Invalid codec range")
        if self.command in {"codec_analyze", "codec_recommend"} and self.codec_kind != "adaptive":
            raise ValueError("Adaptive codec required")
        if self.command == "payload_capture":
            if (self.asset_id is None) == (self.artifact_id is None):
                raise ValueError("Capture requires exactly one input")
        elif self.command == "payload_import":
            if self.artifact_id is None or self.asset_id is not None:
                raise ValueError("Import requires an exported payload artifact")
        elif self.asset_id is not None or self.artifact_id is not None:
            raise ValueError("Input does not match command")
        if "limit" in self.model_fields_set and self.command not in {"inspect", "key_history"}:
            raise ValueError("Field does not match command")
        if "archive_old" in self.model_fields_set and self.command != "key_rotate":
            raise ValueError("Field does not match command")
        if self.key_version is not None and self.command not in {
            "key_status",
            "key_apply",
            "key_delete",
        }:
            raise ValueError("Field does not match command")
        fields = (self.memory_percent, self.queue_depth, self.latency_ms)
        if self.command == "pressure_update":
            if all(value is None for value in fields):
                raise ValueError("At least one native metric is required")
        elif any(value is not None for value in fields):
            raise ValueError("Metric does not match command")
        return self


class StoreUtilityProcess(ServiceProcess):
    worker_module = "truthound.mcp.store_utility_worker"

    def validate_reply(self, result):
        if result == {"ok": True, "native_failed": "native_pressure_incomplete"}:
            return result
        if result == {"ok": True, "native_failed": "native_codec_incomplete"}:
            return result
        if result == {"ok": True, "native_failed": "native_key_incomplete"}:
            return result
        return super().validate_reply(result)


class StoreUtilities(BoundServiceSession):
    group = "store_utilities"
    process_type = StoreUtilityProcess
    limit_commands = {"inspect", "key_history"}
    effects = (
        "installed_extension_execute",
        "operator_selected_native_compression_encryption_pipeline",
        "operator_selected_adaptive_and_chunked_codec_analysis_and_byte_ranges",
        "operator_selected_incremental_codec_stream_write_read_finalize_abort_and_cleanup",
        "operator_selected_key_use_without_key_disclosure",
        "operator_selected_native_key_create_rotate_archive_read_delete_and_apply",
        "selected_key_manager_expired_key_cleanup_including_unaliased_owned_records",
        "operator_selected_native_key_audit_callback",
        "pinned_asset_or_artifact_byte_read",
        "operator_selected_private_payload_slot_read_write_delete",
        "native_backpressure_provider_metrics_admission_wait_release_reset",
        "sensitive_data_artifact",
    )

    def input_resources(self, arguments):
        if arguments.command not in {"payload_capture", "payload_import"}:
            return {}
        if arguments.asset_id is not None:
            return {"data_path": self.session.resources.get(arguments.asset_id, "asset")}
        entry = self.session.resources.get(arguments.artifact_id, "artifact")
        if (
            arguments.command == "payload_import"
            and entry.media_type != "application/vnd.truthound.store-payload+json"
        ):
            raise MCPError("invalid_handle")
        return {"data_path": entry}

    def artifact_media_type(self, arguments):
        if arguments.command == "payload_export":
            return "application/vnd.truthound.store-payload+json"
        return super().artifact_media_type(arguments)

    def reply_error(self, arguments, reply):
        return reply.get("native_failed")
