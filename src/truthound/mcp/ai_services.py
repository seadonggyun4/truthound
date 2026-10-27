"""Approved native AI proposal lifecycle; provider and source are operator-owned."""

from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from truthound.mcp.approved_sources import SourceBindingOpen
from truthound.mcp.bound_services import BoundServiceSession
from truthound.mcp.checkpoint_services import ServiceProcess
from truthound.mcp.contracts import Name
from truthound.mcp.session_contracts import Handle


@dataclass
class AIRuntime:
    context: Any
    provider: Any
    data: Any = None
    source: Any = None
    model: str | None = None
    run: Any = None


class AICommand(SourceBindingOpen):
    service_id: Handle
    command: Literal[
        "suggest",
        "explain",
        "list_proposals",
        "proposal",
        "approve",
        "reject",
        "apply",
        "history",
        "list_analyses",
        "analysis",
    ]
    prompt: Annotated[str, Field(min_length=1, max_length=16000)] | None = None
    proposal_id: Annotated[str, Field(min_length=1, max_length=256)] | None = None
    expected_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    analysis_id: Annotated[str, Field(min_length=1, max_length=256)] | None = None
    run_id: Name | None = None
    comment: Annotated[str, Field(max_length=4096)] = ""
    limit: Annotated[int, Field(ge=1, le=1000)] = 100
    sample_size: Annotated[int, Field(ge=1, le=10000)] = 1000
    include_history: bool = True

    @model_validator(mode="after")
    def shape(self):
        from truthound._ai_contract import (
            is_valid_analysis_artifact_id,
            is_valid_proposal_artifact_id,
        )

        if (self.command in {"approve", "reject", "apply"}) != (self.expected_sha256 is not None):
            raise ValueError(
                "expected_sha256 is required only for proposal decisions and application"
            )
        review = self.command in {"proposal", "approve", "reject", "apply", "history"}
        if review != (self.proposal_id is not None):
            raise ValueError("proposal_id is required only for proposal review commands")
        if self.proposal_id is not None and not is_valid_proposal_artifact_id(self.proposal_id):
            raise ValueError("Invalid proposal identity")
        if (self.command == "analysis") != (self.analysis_id is not None):
            raise ValueError("analysis_id is required only for analysis")
        if self.analysis_id is not None and not is_valid_analysis_artifact_id(self.analysis_id):
            raise ValueError("Invalid analysis identity")
        if (self.command == "suggest") != (self.prompt is not None):
            raise ValueError("prompt is required only for suggest")
        allowed = {
            "run_id": {"explain", "list_analyses"},
            "comment": {"approve", "reject", "apply"},
            "limit": {"list_proposals", "list_analyses"},
            "sample_size": {"suggest"},
            "include_history": {"explain"},
        }
        for field, commands in allowed.items():
            if field in self.model_fields_set and self.command not in commands:
                raise ValueError("Unsupported AI command field")
        if self.run_id is not None and any(token in self.run_id for token in ("/", "\\", "..")):
            raise ValueError("run_id must be an identity, not a path")
        return self


class AIProcess(ServiceProcess):
    worker_module = "truthound.mcp.ai_service_worker"


class AIServices(BoundServiceSession):
    group = "ai_services"
    limit_commands = {"list_proposals", "list_analyses"}
    process_type = AIProcess
    effects = (
        "installed_extension_execute",
        "operator_selected_ai_provider_request_and_possible_cost",
        "summary_only_redacted_source_disclosure",
        "native_ai_artifact_workspace_writes",
        "operator_approved_proposal_decision_or_suite_application",
    )
