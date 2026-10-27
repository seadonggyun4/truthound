"""Closed audit and telemetry administration over operator-selected Core owners."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from truthound.mcp.approved_sources import SourceBindingOpen
from truthound.mcp.bound_services import BoundServiceSession
from truthound.mcp.checkpoint_services import ServiceProcess
from truthound.mcp.contracts import Name
from truthound.mcp.session_contracts import Handle


@dataclass
class AuditRuntime:
    storage: Any


@dataclass
class TelemetryRuntime:
    registry: Any
    provider: Any
    exporters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)


class AuditCommand(SourceBindingOpen):
    service_id: Handle
    command: Literal["query", "read", "count", "record", "delete_before", "flush"]
    event_id: Name | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    before: datetime | None = None
    event_types: Annotated[list[Name], Field(max_length=100)] | None = None
    actor_id: Name | None = None
    resource_id: Name | None = None
    outcome: Literal["success", "failure", "partial", "unknown"] | None = None
    action: Name | None = None
    message: Annotated[str, Field(max_length=4096)] = ""
    data: Annotated[dict[Name, JsonValue], Field(max_length=100)] = Field(default_factory=dict)
    limit: Annotated[int, Field(ge=1, le=1000)] = 100
    offset: Annotated[int, Field(ge=0)] = 0

    @field_validator("start_time", "end_time", "before", mode="before")
    @classmethod
    def from_wire(cls, value):
        return datetime.fromisoformat(value) if isinstance(value, str) else value

    @field_validator("event_types")
    @classmethod
    def native_types(cls, value):
        from truthound.audit.core import AuditEventType

        if value is not None:
            for item in value:
                AuditEventType(item)
        return value

    @model_validator(mode="after")
    def shape(self):
        if (self.command == "read") != (self.event_id is not None):
            raise ValueError("event_id is required only for read")
        if (self.command == "record") != (self.action is not None):
            raise ValueError("action is required only for record")
        if (self.command == "delete_before") != (self.before is not None):
            raise ValueError("before is required only for delete_before")
        allowed = {
            "start_time": {"query", "count"},
            "end_time": {"query", "count"},
            "event_types": {"query", "count"},
            "actor_id": {"query"},
            "resource_id": {"query"},
            "outcome": {"query", "record"},
            "message": {"record"},
            "data": {"record"},
            "limit": {"query"},
            "offset": {"query"},
        }
        for key, commands in allowed.items():
            if key in self.model_fields_set and self.command not in commands:
                raise ValueError("Unsupported audit command field")
        for value in (self.start_time, self.end_time, self.before):
            if value is not None and value.utcoffset() is None:
                raise ValueError("Audit timestamps require a UTC offset")
        return self


class TelemetryCommand(SourceBindingOpen):
    service_id: Handle
    command: Literal["collect", "export", "metric", "span", "flush", "shutdown"]
    exporter_id: Name | None = None
    metric_id: Name | None = None
    update: Literal["inc", "set", "observe"] | None = None
    value: Annotated[float, Field(allow_inf_nan=False)] | None = None
    labels: Annotated[dict[Name, Name], Field(max_length=32)] = Field(default_factory=dict)
    span_name: Name | None = None
    attributes: Annotated[dict[Name, JsonValue], Field(max_length=100)] = Field(
        default_factory=dict
    )
    timeout_ms: Annotated[int, Field(ge=1, le=30000)] = 1000

    @model_validator(mode="after")
    def shape(self):
        if (self.command == "export") != (self.exporter_id is not None):
            raise ValueError("exporter_id is required only for export")
        metric_fields = {"metric_id", "update", "value"}
        if self.command == "metric":
            if any(getattr(self, key) is None for key in metric_fields):
                raise ValueError("metric requires metric_id, update and value")
        elif self.model_fields_set & (metric_fields | {"labels"}):
            raise ValueError("metric fields are only supported by metric")
        if (self.command == "span") != (self.span_name is not None):
            raise ValueError("span_name is required only for span")
        if self.command != "span" and "attributes" in self.model_fields_set:
            raise ValueError("attributes are only supported by span")
        if self.command != "flush" and "timeout_ms" in self.model_fields_set:
            raise ValueError("timeout_ms is only supported by flush")
        return self


class AdministrationProcess(ServiceProcess):
    worker_module = "truthound.mcp.administration_worker"


class AuditServices(BoundServiceSession):
    group = "audit_services"
    process_type = AdministrationProcess
    limit_commands = {"query"}
    effects = (
        "installed_extension_execute",
        "native_audit_read_write_flush",
        "explicit_native_audit_retention_delete",
    )


class TelemetryServices(BoundServiceSession):
    group = "telemetry_services"
    process_type = AdministrationProcess
    effects = (
        "installed_extension_execute",
        "native_metrics_and_spans",
        "operator_selected_exporter_network_or_storage_effects",
        "native_exporter_flush_shutdown",
    )
