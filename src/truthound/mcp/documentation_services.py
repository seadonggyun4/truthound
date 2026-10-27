"""Operator-selected native API documentation generation and retained file artifacts."""

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import model_validator

from truthound.mcp.approved_sources import SourceBindingOpen
from truthound.mcp.bound_services import BoundServiceSession
from truthound.mcp.checkpoint_services import ServiceProcess
from truthound.mcp.contracts import Name
from truthound.mcp.session_contracts import Handle


@dataclass
class DocumentationJob:
    generator: Any
    source: Any
    kind: Literal["source", "module", "info", "scaffold", "validate"] = "source"
    strict: bool = False


@dataclass
class DocumentationRuntime:
    jobs: dict[str, DocumentationJob]


class DocumentationCommand(SourceBindingOpen):
    service_id: Handle
    command: Literal["inspect", "generate", "result", "read", "release", "shutdown"]
    job_id: Name | None = None
    result_id: Name | None = None
    file_id: Name | None = None

    @model_validator(mode="after")
    def shape(self):
        for field, commands in {
            "job_id": {"generate"},
            "result_id": {"result", "read", "release"},
            "file_id": {"read"},
        }.items():
            if (self.command in commands) != (getattr(self, field) is not None):
                raise ValueError("Field does not match documentation command")
        return self


class DocumentationProcess(ServiceProcess):
    worker_module = "truthound.mcp.documentation_worker"

    def validate_reply(self, result):
        if result == {"ok": True, "native_failed": "native_documentation_incomplete"}:
            return result
        return super().validate_reply(result)


class DocumentationServices(BoundServiceSession):
    group = "documentation_services"
    process_type = DocumentationProcess
    effects = (
        "installed_extension_execute",
        "operator_selected_source_read_or_module_import",
        "native_documentation_render_and_owned_temporary_files",
        "sensitive_documentation_artifact_disclosure",
    )

    def artifact_media_type(self, arguments):
        if arguments.command == "read":
            return "application/octet-stream"
        return super().artifact_media_type(arguments)

    def reply_error(self, arguments, reply):
        return reply.get("native_failed")
