"""Explicit approved native streaming store lifecycle, separate from BaseStore."""

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import Field, JsonValue, model_validator

from truthound.mcp.approved_sources import SourceBindingOpen
from truthound.mcp.bound_services import BoundServiceSession
from truthound.mcp.checkpoint_services import ServiceProcess
from truthound.mcp.contracts import Name, WireModel
from truthound.mcp.policy import MCPError
from truthound.mcp.session_contracts import Handle


@dataclass(frozen=True)
class StreamingRun:
    """Factory-selected native identity; the model selects only its alias."""

    run_id: str
    data_asset: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamingStoreRuntime:
    store: Any
    runs: dict[str, StreamingRun] = field(default_factory=dict)
    multipart_receipts: dict[str, dict[str, str]] = field(default_factory=dict, repr=False)


class StreamRecord(WireModel):
    validator_name: Annotated[str, Field(min_length=1, max_length=512)]
    success: bool
    column: Annotated[str, Field(max_length=512)] | None = None
    issue_type: Annotated[str, Field(max_length=512)] | None = None
    count: Annotated[int, Field(ge=0)] = 0
    severity: Literal["critical", "high", "medium", "low", "info"] | None = None
    message: Annotated[str, Field(max_length=4096)] | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)
    execution_time_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0


class StreamingStoreCommand(SourceBindingOpen):
    service_id: Handle
    command: Literal[
        "inspect",
        "session_create",
        "session_open",
        "session_resume",
        "session_status",
        "session_release",
        "writer_open",
        "writer_write",
        "writer_write_run",
        "writer_flush",
        "writer_close",
        "writer_abort",
        "writer_status",
        "reader_open",
        "reader_read",
        "reader_reset",
        "reader_close",
        "reader_status",
        "chunks_list",
        "chunk_get",
        "chunk_release",
        "chunks_delete",
        "finalize",
        "stats",
        "storage_size",
        "recovery_status",
        "recovery_reconcile",
        "recovery_release",
        "multipart_reconcile",
    ]
    run_id: Name | None = None
    session_id: Name | None = None
    writer_id: Name | None = None
    reader_id: Name | None = None
    chunk_id: Name | None = None
    result_id: Handle | None = None
    recovery_id: Name | None = None
    receipt_id: Name | None = None
    resolution: Literal["adopt", "discard"] | None = None
    records: Annotated[list[StreamRecord], Field(min_length=1, max_length=1000)] | None = None
    asynchronous: bool = False
    limit: Annotated[int, Field(ge=1, le=1000)] = 100
    offset: Annotated[int, Field(ge=0, le=1_000_000)] = 0

    @model_validator(mode="after")
    def shape(self):
        required = {
            "run_id": {
                "session_create",
                "session_open",
                "reader_open",
                "chunks_list",
                "chunks_delete",
                "stats",
                "storage_size",
                "recovery_status",
            },
            "session_id": {
                "session_resume",
                "session_status",
                "session_release",
                "writer_open",
                "finalize",
            },
            "writer_id": {
                "writer_write",
                "writer_write_run",
                "writer_flush",
                "writer_close",
                "writer_abort",
                "writer_status",
            },
            "reader_id": {"reader_read", "reader_reset", "reader_close", "reader_status"},
            "chunk_id": {"chunk_get", "chunk_release"},
            "result_id": {"writer_write_run"},
            "records": {"writer_write"},
            "recovery_id": {"recovery_reconcile", "recovery_release", "multipart_reconcile"},
            "resolution": {"recovery_reconcile"},
        }
        for key, commands in required.items():
            if (self.command in commands) != (getattr(self, key) is not None):
                raise ValueError("Field does not match command")
        for key, commands in {
            "asynchronous": {"writer_open", "reader_open"},
            "limit": {"inspect", "reader_read", "chunks_list"},
            "offset": {"inspect", "chunks_list"},
            "receipt_id": {"multipart_reconcile"},
        }.items():
            if key in self.model_fields_set and self.command not in commands:
                raise ValueError("Field does not match command")
        return self


class StreamingStoreProcess(ServiceProcess):
    worker_module = "truthound.mcp.streaming_store_worker"


class StreamingStoreServices(BoundServiceSession):
    group = "streaming_store_services"
    process_type = StreamingStoreProcess
    limit_commands = {"inspect", "reader_read", "chunks_list"}
    effects = (
        "installed_extension_execute",
        "operator_selected_native_streaming_store_io",
        "native_stream_session_create_resume_and_close",
        "native_stream_write_read_flush_finalize",
        "operator_selected_run_chunk_deletion",
        "sensitive_data_artifact",
        "retained_native_stream_chunk_adoption_or_deletion",
        "explicit_native_multipart_abort_with_owned_or_operator_asserted_receipts",
    )

    def input_resources(self, arguments):
        if arguments.command != "writer_write_run":
            return {}
        entry = self.session.resources.get(arguments.result_id, "artifact")
        if entry.media_type != "application/vnd.truthound.validation+json":
            raise MCPError("invalid_handle")
        return {"run_path": entry}
