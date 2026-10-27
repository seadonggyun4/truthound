"""Selected native tenant administration; scope does not grant MCP data access."""

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from truthound.mcp.approved_sources import SourceBindingOpen
from truthound.mcp.bound_services import BoundServiceSession
from truthound.mcp.checkpoint_services import ServiceProcess
from truthound.mcp.contracts import Name, WireModel
from truthound.mcp.session_contracts import Handle

Tier = Literal["free", "starter", "professional", "enterprise", "custom"]
Isolation = Literal["shared", "row_level", "schema", "database", "instance"]
Resource = Literal[
    "validations",
    "rows_processed",
    "storage_bytes",
    "api_calls",
    "checkpoints",
    "concurrent_jobs",
    "retention_days",
    "users",
]
Limit = Annotated[int, Field(ge=-1, le=2**63 - 1)]


@dataclass
class TenantRuntime:
    manager: Any
    tenants: dict[str, str]
    settings: dict[str, Any] = field(default_factory=dict)
    policies: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, Any] = field(default_factory=dict, repr=False)
    contexts: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)


class TenantPolicyPatch(WireModel):
    enabled: bool | None = None
    priority: Annotated[int, Field(ge=-1_000_000, le=1_000_000)] | None = None

    @model_validator(mode="after")
    def supplied(self):
        if not self.model_fields_set or any(
            getattr(self, key) is None for key in self.model_fields_set
        ):
            raise ValueError("Supply non-null policy settings")
        return self


class TenantQuotaPatch(WireModel):
    validations_per_day: Limit | None = None
    validations_per_hour: Limit | None = None
    rows_per_validation: Limit | None = None
    storage_bytes: Limit | None = None
    api_calls_per_minute: Limit | None = None
    api_calls_per_day: Limit | None = None
    checkpoints_per_day: Limit | None = None
    max_retention_days: Limit | None = None
    max_concurrent_jobs: Limit | None = None
    max_users: Limit | None = None
    custom_limits: Annotated[dict[Name, Limit], Field(max_length=100)] | None = None

    @model_validator(mode="after")
    def supplied(self):
        if not self.model_fields_set or any(
            getattr(self, key) is None for key in self.model_fields_set
        ):
            raise ValueError("Supply non-null quota settings")
        return self


class TenantPatch(WireModel):
    name: Name | None = None
    slug: Name | None = None
    owner_id: Annotated[str, Field(max_length=512)] | None = None
    owner_email: Annotated[str, Field(max_length=512)] | None = None
    isolation_level: Isolation | None = None
    quota: TenantQuotaPatch | None = None
    settings_id: Name | None = None

    @model_validator(mode="after")
    def supplied(self):
        if not self.model_fields_set or any(
            getattr(self, key) is None for key in self.model_fields_set
        ):
            raise ValueError("Supply non-null tenant settings")
        return self


class TenantCommand(SourceBindingOpen):
    service_id: Handle
    command: Literal[
        "catalog",
        "resolver_catalog",
        "source_catalog",
        "source_read",
        "resolve",
        "resolve_or_create",
        "get",
        "create",
        "update",
        "delete",
        "suspend",
        "activate",
        "set_tier",
        "quota_check",
        "usage_track",
        "usage_summary",
        "context",
        "isolation_apply",
        "policy_catalog",
        "policy_add",
        "policy_remove",
        "policy_update",
    ]
    tenant_id: Name | None = None
    source_id: Name | None = None
    context_id: Name | None = None
    default_name: Name | None = None
    patch: TenantPatch | None = None
    asset_id: Handle | None = None
    columns: Annotated[list[str], Field(min_length=1, max_length=100)] | None = None
    policy_id: Name | None = None
    policy_patch: TenantPolicyPatch | None = None
    tier: Tier | None = None
    update_quota: bool = True
    reason: Annotated[str, Field(max_length=4096)] = ""
    resource: Resource | None = None
    amount: Annotated[int, Field(ge=1, le=1_000_000_000)] = 1
    window: Literal["hour", "day", "month"] = "day"
    limit: Annotated[int, Field(ge=1, le=1000)] = 100
    offset: Annotated[int, Field(ge=0)] = 0

    @model_validator(mode="after")
    def shape(self):
        if (self.command not in {"catalog", "resolver_catalog", "source_catalog", "resolve"}) != (
            self.tenant_id is not None
        ):
            raise ValueError("tenant_id does not match command")
        if (self.command in {"resolve", "resolve_or_create"}) != (self.context_id is not None):
            raise ValueError("context_id does not match command")
        if (self.command == "resolve_or_create") != (self.default_name is not None):
            raise ValueError("default_name does not match command")
        if (self.command == "source_read") != (self.source_id is not None):
            raise ValueError("source_id does not match command")
        if (self.command == "isolation_apply") != (self.asset_id is not None):
            raise ValueError("asset_id does not match command")
        if (self.command in {"policy_add", "policy_remove", "policy_update"}) != (
            self.policy_id is not None
        ):
            raise ValueError("policy_id does not match command")
        if (self.command == "policy_update") != (self.policy_patch is not None):
            raise ValueError("policy_patch does not match command")
        if (self.command in {"create", "update"}) != (self.patch is not None):
            raise ValueError("patch does not match command")
        if self.command == "create" and self.patch.name is None:
            raise ValueError("Tenant creation requires a name")
        if self.command == "create" and "slug" in self.patch.model_fields_set:
            raise ValueError("Native creation generates the slug; edit it through update")
        if self.command == "set_tier" and self.tier is None:
            raise ValueError("set_tier requires tier")
        if (self.command in {"quota_check", "usage_track", "usage_summary"}) != (
            self.resource is not None
        ):
            raise ValueError("resource does not match command")
        for key, commands in {
            "tier": {"create", "set_tier"},
            "update_quota": {"set_tier"},
            "reason": {"suspend"},
            "amount": {"quota_check", "usage_track"},
            "window": {"usage_summary"},
            "limit": {"catalog", "resolver_catalog", "source_catalog", "source_read", "policy_catalog", "isolation_apply"},
            "offset": {"catalog", "resolver_catalog", "source_catalog", "source_read", "policy_catalog", "isolation_apply"},
            "columns": {"isolation_apply", "source_read"},
        }.items():
            if key in self.model_fields_set and self.command not in commands:
                raise ValueError("Field does not match command")
        return self


class TenantProcess(ServiceProcess):
    worker_module = "truthound.mcp.tenancy_worker"


class TenantServices(BoundServiceSession):
    group = "tenant_services"
    process_type = TenantProcess
    limit_commands = {"catalog", "resolver_catalog", "source_catalog", "source_read", "policy_catalog", "isolation_apply"}
    effects = (
        "installed_extension_execute",
        "operator_selected_tenant_store_admin",
        "native_tenant_status_quota_usage_context",
        "operator_configured_tenant_hooks",
        "native_tenant_resolver_on_selected_private_context_and_fenced_creation",
        "native_tenant_row_isolation_on_pinned_input_and_bound_policy_edits",
        "native_tenant_sql_source_namespace_verified_on_reading_connection",
        "sensitive_data_artifact",
    )

    def input_resources(self, arguments):
        if arguments.asset_id is not None:
            return {"asset_path": self.session.resources.get(arguments.asset_id, "asset")}
        return {}
