"""Approved native tracker lifetimes, context jobs, history and private graph files."""

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from truthound.mcp.approved_sources import SourceBindingOpen
from truthound.mcp.bound_services import BoundServiceSession
from truthound.mcp.checkpoint_services import ServiceProcess
from truthound.mcp.contracts import Name
from truthound.mcp.lineage_contracts import RenderSettings, TrackingEvent
from truthound.mcp.session_contracts import Handle


@dataclass(frozen=True)
class LineageJob:
    operation: Any
    name: str
    operation_type: Any = "transform"


@dataclass
class LineageRuntime:
    tracker: Any
    jobs: dict[str, LineageJob] = field(default_factory=dict)
    files: dict[str, Any] = field(default_factory=dict)


class LineageCommand(SourceBindingOpen):
    service_id: Handle
    command: Literal[
        "inspect",
        "snapshot",
        "history",
        "track",
        "run",
        "lineage",
        "path",
        "save",
        "load",
        "clear",
        "export",
        "render",
        "shutdown",
    ]
    event: TrackingEvent | None = None
    job_id: Name | None = None
    file_id: Name | None = None
    node_id: Name | None = None
    source_id: Name | None = None
    target_id: Name | None = None
    direction: Literal["upstream", "downstream", "both"] = "both"
    rendering: RenderSettings | None = None
    offset: Annotated[int, Field(ge=0)] = 0
    limit: Annotated[int, Field(ge=0, le=1000)] = 100

    @model_validator(mode="after")
    def shape(self):
        for key, commands in {
            "event": {"track"},
            "job_id": {"run"},
            "file_id": {"save", "load"},
            "node_id": {"lineage"},
            "source_id": {"path"},
            "target_id": {"path"},
            "rendering": {"render"},
        }.items():
            if (self.command in commands) != (getattr(self, key) is not None):
                raise ValueError("Field does not match lineage command")
        if ({"limit", "offset"} & self.model_fields_set and self.command != "history") or (
            "direction" in self.model_fields_set and self.command != "lineage"
        ):
            raise ValueError("Selector does not match lineage command")
        return self


class LineageProcess(ServiceProcess):
    worker_module = "truthound.mcp.lineage_service_worker"

    def validate_reply(self, result):
        if result == {"ok": True, "native_failed": "native_lineage_incomplete"}:
            return result
        return super().validate_reply(result)


class LineageServices(BoundServiceSession):
    group = "lineage_services"
    process_type = LineageProcess
    limit_commands = {"history"}
    effects = (
        "installed_extension_execute",
        "native_lineage_context_job_and_history",
        "operator_selected_graph_file_read_write",
        "sensitive_lineage_artifact_disclosure",
    )

    def artifact_media_type(self, arguments):
        if arguments.command == "export":
            return "application/json"
        if arguments.command == "render":
            from truthound.mcp.lineage_worker import render_media_type

            return render_media_type(arguments.rendering)
        return super().artifact_media_type(arguments)

    def response_media_type(self, arguments, reply):
        return (
            "application/vnd.truthound.lineage_services+json"
            if reply.get("native_failed")
            else self.artifact_media_type(arguments)
        )

    def reply_error(self, arguments, reply):
        return reply.get("native_failed")
