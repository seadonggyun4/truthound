"""Approved native lineage delivery with operator-pinned transport and datasets."""

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from truthound.mcp.approved_sources import SourceBindingOpen
from truthound.mcp.bound_services import BoundServiceSession
from truthound.mcp.checkpoint_services import ServiceProcess
from truthound.mcp.contracts import Name
from truthound.mcp.session_contracts import Handle


@dataclass
class OpenLineageRuntime:
    emitter: Any
    datasets: dict[str, dict[str, Any]] = field(default_factory=dict)


class OpenLineageCommand(SourceBindingOpen):
    service_id: Handle
    command: Literal[
        "status", "start", "running", "complete", "fail", "abort", "graph", "resolve", "shutdown"
    ]
    job_name: Name | None = None
    run_id: Annotated[str, Field(pattern=r"^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$")] | None = (
        None
    )
    parent_run_id: (
        Annotated[str, Field(pattern=r"^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$")] | None
    ) = None
    datasets: Annotated[list[Name], Field(max_length=100)] = Field(default_factory=list)
    graph_id: Handle | None = None
    delivered: bool | None = None

    @model_validator(mode="after")
    def shape(self):
        required = {
            "job_name": {"start", "graph"},
            "run_id": {"running", "complete", "fail", "abort", "resolve"},
            "graph_id": {"graph"},
            "delivered": {"resolve"},
        }
        for name, commands in required.items():
            if (self.command in commands) != (getattr(self, name) is not None):
                raise ValueError("Field does not match OpenLineage command")
        if "datasets" in self.model_fields_set and self.command not in {"start", "complete"}:
            raise ValueError("Datasets apply only to start or complete")
        if "parent_run_id" in self.model_fields_set and self.command != "start":
            raise ValueError("Parent applies only to start")
        return self


class OpenLineageProcess(ServiceProcess):
    worker_module = "truthound.mcp.openlineage_worker"

    def validate_reply(self, result):
        if result == {"ok": True, "native_failed": "outcome_unknown"}:
            return result
        return super().validate_reply(result)


class OpenLineageServices(BoundServiceSession):
    group = "openlineage_services"
    process_type = OpenLineageProcess
    effects = (
        "installed_extension_execute",
        "operator_selected_openlineage_endpoint_request",
        "bound_dataset_or_graph_names_disclosure",
        "native_run_transition_or_explicit_unknown_delivery_resolution",
        "sensitive_data_artifact",
    )

    def input_resources(self, arguments):
        if arguments.command == "graph":
            return {"graph_path": self.session.resources.get(arguments.graph_id, "graph")}
        return {}

    def reply_error(self, arguments, reply):
        return reply.get("native_failed")
