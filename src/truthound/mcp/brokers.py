"""Explicit broker adapter, window and processing contracts; factories own transports."""

from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from truthound.mcp.approved_sources import SourceBindingOpen
from truthound.mcp.bound_services import BoundServiceSession
from truthound.mcp.checkpoint_services import ServiceProcess
from truthound.mcp.contracts import WireModel
from truthound.mcp.session_contracts import Handle


@dataclass
class BrokerRuntime:
    """Operator factory payload; all computations remain in native Core objects."""

    adapter: Any = None
    window: Any = None
    processor: Any = None
    output_topic: str | None = None
    validator: Any = None
    stream_source: Any = None
    file_source: Any = None
    pipeline: Any = None
    streaming_validators: Any = None


class BrokerMessage(WireModel):
    value: JsonValue
    key: Annotated[str, Field(max_length=4096)] | None = None
    partition: Annotated[int, Field(ge=0)] = 0
    timestamp: datetime

    @field_validator("timestamp", mode="before")
    @classmethod
    def from_wire(cls, value):
        return datetime.fromisoformat(value) if isinstance(value, str) else value


class BrokerCommand(SourceBindingOpen):
    service_id: Handle
    command: Literal[
        "connect",
        "disconnect",
        "status",
        "consume",
        "ack",
        "seek",
        "produce",
        "flush",
        "window_add",
        "window_flush",
        "process",
        "recover",
        "validate_batch",
        "validation_stats",
        "validation_reset",
        "source_status",
        "source_connect",
        "source_disconnect",
        "source_next",
        "stream_start",
        "stream_next",
        "stream_cancel",
        "pipeline_validate",
        "pipeline_stream",
        "stream_validate_many",
    ]
    stream_mode: Literal["sync", "async"] = "sync"
    max_batches: Annotated[int, Field(ge=1, le=10000)] = 100
    limit: Annotated[int, Field(ge=1, le=1000)] = 100
    timeout_ms: Annotated[int, Field(ge=1, le=30000)] = 1000
    batch_id: Annotated[str, Field(pattern=r"^batch_[0-9a-f]{32}$")] | None = None
    partition: Annotated[int, Field(ge=0)] | None = None
    offset: Annotated[int, Field(ge=0)] | None = None
    message: BrokerMessage | None = None

    @model_validator(mode="after")
    def shape(self):
        if (self.command in {"ack", "window_add", "process", "validate_batch", "pipeline_validate"}) != (self.batch_id is not None):
            raise ValueError("batch_id is required only for retained batch commands")
        if (self.command == "seek") != (self.partition is not None and self.offset is not None):
            raise ValueError("seek requires partition and offset")
        if self.command != "seek" and (self.partition is not None or self.offset is not None):
            raise ValueError("partition and offset are only supported by seek")
        if (self.command == "produce") != (self.message is not None):
            raise ValueError("message is required only for produce")
        if self.command not in {"consume", "stream_next", "source_next"} and "limit" in self.model_fields_set:
            raise ValueError("limit is only supported by paged reads")
        if self.command != "consume" and "timeout_ms" in self.model_fields_set:
            raise ValueError("timeout_ms is only supported by consume")
        if self.command != "stream_start" and self.model_fields_set & {"stream_mode", "max_batches"}:
            raise ValueError("stream settings are only supported by stream_start")
        return self


class BrokerProcess(ServiceProcess):
    worker_module = "truthound.mcp.broker_worker"


class BrokerServices(BoundServiceSession):
    group = "brokers"
    limit_commands = {"consume", "stream_next", "source_next"}
    process_type = BrokerProcess
    effects = (
        "installed_extension_execute",
        "broker_connect_consume_produce_ack_seek",
        "operator_configured_native_retries",
        "native_window_and_transaction_recovery",
        "native_batch_validation_and_operator_callbacks",
        "operator_bound_stream_connect_read_validate_and_cleanup",
    )
