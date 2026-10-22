"""Reviewed immutable tool descriptors; the catalog is not an execution escape hatch."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from pydantic import create_model

from truthound.mcp import contracts as wire
from truthound.mcp.session_catalog import operations as session_operations


@dataclass(frozen=True)
class Operation:
    operation_id: str
    description: str
    input_type: type[wire.WireModel]
    public_target: str
    data_type: type
    sensitive: bool = False
    route: str = "file"
    mutating: bool = False
    requires_workspaces: bool = False
    requires_control: bool = False

    def output_type(self):
        return create_model(
            self.operation_id + "Result",
            __base__=wire.ResultEnvelope,
            operation_id=(Literal[self.operation_id], ...),
            data=(self.data_type | None, ...),
        )

    def describe(self) -> dict:
        discovery = self.operation_id in {
            "capabilities_list",
            "capability_describe",
            "validators_list",
            "validators_describe",
            "models_catalog",
            "lineage_catalog",
            "drift_catalog",
            "profilers_catalog",
        }
        return {
            "operation_id": self.operation_id,
            "contract_version": 1,
            "description": self.description,
            "input_schema": self.input_type.model_json_schema(),
            "output_schema": self.output_type().model_json_schema(),
            "public_target": self.public_target,
            "effects": []
            if discovery
            else ["compute", "sensitive_read"]
            + (
                ["workspace_write"]
                if self.requires_control and self.mutating
                else ["session_state_write"]
                if self.mutating
                else []
            ),
            "required_grants": (
                []
                if discovery
                else [
                    "workspace_directories" if self.requires_workspaces else "allowed_directories"
                ]
            )
            + (["control_directory"] if self.requires_control else [])
            + (["operator_plan_approval"] if self.requires_control and self.mutating else [])
            + (["allow_sensitive_results"] if self.sensitive else []),
            "conditional_grants": {
                **(
                    {"evidence=full": ["allow_sensitive_results"]}
                    if "evidence" in self.input_type.model_fields
                    else {}
                ),
                **(
                    {
                        "context_id supplied": [
                            "workspace_directories",
                            "control_directory",
                            "operator_plan_approval",
                            "workspace_write",
                        ],
                        "any check.binding_id supplied": [
                            "control_directory",
                            "operator_plan_approval",
                            "execute_trusted_binding",
                        ],
                    }
                    if self.operation_id
                    in {"assets_validate", "suites_validate", "sources_validate"}
                    else {}
                ),
            },
            "execution_mode": "inline_discovery"
            if discovery
            else (
                "inline_session"
                if self.operation_id
                in {
                    "resources_describe",
                    "resources_release",
                    "artifacts_read",
                    "schemas_inspect",
                    "suites_create",
                    "suites_inspect",
                    "contexts_open",
                    "contexts_inspect",
                    "plans_status",
                }
                else "resident_worker"
                if self.operation_id.startswith(("models_", "sources_", "streams_"))
                else "disposable_worker"
            ),
            "context_mode": "persistent"
            if self.requires_workspaces
            else ("retained" if self.route == "session" else "transient"),
            "required_extras": ["mcp", "duckdb"]
            if self.operation_id == "sources_open_duckdb"
            else ["mcp"],
        }


_OPERATIONS = (
    Operation(
        "capabilities_list",
        "List the installed Core-profile catalog, with bounded paging.",
        wire.Page,
        "truthound.mcp.catalog",
        wire.CapabilityPage,
    ),
    Operation(
        "capability_describe",
        "Inspect one installed tool's exact input schema and effects.",
        wire.Describe,
        "truthound.mcp.catalog",
        wire.CapabilityDetail,
    ),
    Operation(
        "validation_check",
        "Validate a file with typed rules or Core's auto-suite. Full evidence requires operator consent.",
        wire.ValidationInput,
        "truthound.api.check",
        wire.ValidationSummary | wire.FullResult,
    ),
    Operation(
        "schema_learn",
        "Learn a schema including value constraints; sensitive output requires operator consent.",
        wire.LearnInput,
        "truthound.schema.learn",
        wire.FullResult,
        True,
    ),
    Operation(
        "data_read",
        "Read a bounded page of actual values; requires operator consent, not just directory access.",
        wire.ReadInput,
        "truthound.api.read",
        wire.DataPage,
        True,
    ),
    Operation(
        "data_mask",
        "Mask selected columns into a bounded returned page, never overwrite input. Unselected columns remain sensitive.",
        wire.MaskInput,
        "truthound.api.mask",
        wire.DataPage,
        True,
    ),
    Operation(
        "data_profile",
        "Profile a file with summary or explicitly authorized full Core statistics.",
        wire.EvidenceInput,
        "truthound.api.profile",
        wire.ProfileSummary | wire.FullResult,
    ),
    Operation(
        "privacy_scan",
        "Scan a file using Core's first-1000-non-null sampling. Counts are estimates.",
        wire.EvidenceInput,
        "truthound.api.scan",
        wire.ScanSummary | wire.FullResult,
    ),
)
_ALL_OPERATIONS = _OPERATIONS + session_operations(Operation)
CATALOG = MappingProxyType({op.operation_id: op for op in _ALL_OPERATIONS})
if len(CATALOG) != len(_ALL_OPERATIONS):
    raise RuntimeError("Duplicate MCP operation identity")


def needs_disclosure(operation: Operation, arguments: wire.WireModel) -> bool:
    return operation.sensitive or getattr(arguments, "evidence", "summary") == "full"


def discover(
    name: str,
    arguments: wire.WireModel,
    *,
    sensitive: bool,
    workspaces: bool = False,
    control: bool = False,
) -> dict:
    """Discovery cannot load optional providers or start workers."""
    if name == "capability_describe":
        operation = CATALOG.get(arguments.operation_id)
        if operation is None:
            from truthound.mcp.policy import MCPError

            raise MCPError("unknown_operation")
        return operation.describe()
    operations = list(CATALOG.values())
    page = operations[arguments.offset : arguments.offset + arguments.limit]
    end = arguments.offset + len(page)

    def availability(operation):
        if operation.operation_id == "sources_open_duckdb":
            from truthound.mcp.duckdb_source import dependency_available

            if not dependency_available():
                return "missing_dependency"
        if (operation.requires_workspaces and not workspaces) or (
            operation.requires_control and not control
        ):
            return "operator_configuration_required"
        if operation.sensitive and not sensitive:
            return "approval_required"
        return "available"

    return {
        "operations": [
            {
                "operation_id": op.operation_id,
                "description": op.description,
                "availability": availability(op),
            }
            for op in page
        ],
        "total": len(operations),
        "next_offset": end if end < len(operations) else None,
        "full_core_parity": False,
        "limitations": [
            "Retained assets and operator-approved workspaces are available; provider/advanced completeness remains unverified.",
            "File validation retains its six-rule compatibility contract; retained validation has closed constructor contracts for 260 concrete registry entries, including six native streaming validators. That inventory is not a claim of complete configuration or error-path coverage.",
            "Untrusted SQL, Python callables, filesystem paths and unrestricted regex require separate operator bindings; advanced subsystem parity remains incomplete.",
        ],
    }
