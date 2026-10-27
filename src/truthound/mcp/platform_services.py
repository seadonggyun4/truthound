"""Typed native RBAC and plugin control; no credentials or code in wire input."""

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import Field, JsonValue, model_validator

from truthound.mcp.approved_sources import SourceBindingOpen
from truthound.mcp.bound_services import BoundServiceSession
from truthound.mcp.checkpoint_services import ServiceProcess
from truthound.mcp.contracts import Name, WireModel
from truthound.mcp.policy import MCPError
from truthound.mcp.session_contracts import Handle


@dataclass
class RBACRuntime:
    manager: Any
    tenant_id: str | None = None
    attribute_keys: tuple[str, ...] = ()
    condition_fields: tuple[str, ...] = ()


@dataclass
class PluginRuntime:
    manager: Any
    classes: dict[str, Any] = field(default_factory=dict)
    configs: dict[str, Any] = field(default_factory=dict)
    hooks: dict[str, Any] = field(default_factory=dict)
    providers: dict[str, Any] = field(default_factory=dict)
    reporters: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)
    check_factories: dict[str, Any] = field(default_factory=dict)
    hook_types: dict[str, str] = field(default_factory=dict)
    reload_factories: dict[str, Any] = field(default_factory=dict)


class PermissionSpec(WireModel):
    resource: Name
    action: Name
    scope: Annotated[str, Field(max_length=512)] = ""
    effect: Literal["allow", "deny"] = "allow"

    @model_validator(mode="after")
    def canonical(self):
        if any(":" in part for part in (self.resource, self.action, self.scope)):
            raise ValueError("Permission components cannot contain the delimiter")
        return self


class ConditionSpec(WireModel):
    field: Name
    operator: Literal[
        "eq",
        "neq",
        "contains",
        "not_contains",
        "starts_with",
        "ends_with",
        "matches",
        "in",
        "not_in",
        "gt",
        "lt",
        "gte",
        "lte",
        "exists",
        "not_exists",
    ]
    value: JsonValue
    description: Annotated[str, Field(max_length=4096)] = ""


class RolePatch(WireModel):
    name: Name | None = None
    description: Annotated[str, Field(max_length=4096)] | None = None
    enabled: bool | None = None
    permissions: Annotated[list[Name | PermissionSpec], Field(max_length=1000)] | None = None
    parent_roles: Annotated[list[Name], Field(max_length=100)] | None = None
    conditions: Annotated[list[ConditionSpec], Field(max_length=100)] | None = None

    @model_validator(mode="after")
    def supplied(self):
        if not self.model_fields_set or any(
            getattr(self, key) is None for key in self.model_fields_set
        ):
            raise ValueError("Supply non-null role settings")
        from truthound.rbac.core import Permission

        for value in self.permissions or ():
            if isinstance(value, PermissionSpec):
                continue
            if (
                not value
                or len(value.split(":")) not in {2, 3}
                or Permission.parse(value).to_string() != value
            ):
                raise ValueError("Use a native resource:action[:scope] permission")
        return self


class PrincipalPatch(WireModel):
    name: Name | None = None
    email: Annotated[str, Field(max_length=512)] | None = None
    enabled: bool | None = None
    attributes: Annotated[dict[Name, JsonValue], Field(max_length=100)] | None = None
    direct_permissions: Annotated[list[Name | PermissionSpec], Field(max_length=1000)] | None = None

    @model_validator(mode="after")
    def supplied(self):
        if not self.model_fields_set or any(
            getattr(self, key) is None for key in self.model_fields_set
        ):
            raise ValueError("Supply non-null principal settings")
        RolePatch(permissions=self.direct_permissions or [])
        return self


class SecurityCommand(SourceBindingOpen):
    service_id: Handle
    command: Literal[
        "roles",
        "role",
        "role_create",
        "role_update",
        "role_delete",
        "permission_add",
        "permission_remove",
        "principals",
        "principal",
        "principal_create",
        "principal_update",
        "principal_delete",
        "assign_role",
        "revoke_role",
        "permissions",
        "check",
        "cache_clear",
    ]
    role_id: Name | None = None
    principal_id: Name | None = None
    role_patch: RolePatch | None = None
    principal_patch: PrincipalPatch | None = None
    permission: Name | PermissionSpec | None = None
    resource: Name | None = None
    action: Name | None = None
    resource_attributes: Annotated[dict[Name, JsonValue], Field(max_length=100)] = Field(
        default_factory=dict
    )
    limit: Annotated[int, Field(ge=1, le=1000)] = 100
    offset: Annotated[int, Field(ge=0)] = 0

    @model_validator(mode="after")
    def shape(self):
        role_commands = {
            "role",
            "role_create",
            "role_update",
            "role_delete",
            "permission_add",
            "permission_remove",
            "assign_role",
            "revoke_role",
        }
        principal_commands = {
            "principal",
            "principal_create",
            "principal_update",
            "principal_delete",
            "assign_role",
            "revoke_role",
            "permissions",
            "check",
        }
        if (self.command in role_commands) != (self.role_id is not None):
            raise ValueError("role_id does not match command")
        if (self.command in principal_commands) != (self.principal_id is not None):
            raise ValueError("principal_id does not match command")
        if (self.command in {"role_create", "role_update"}) != (self.role_patch is not None):
            raise ValueError("role_patch does not match command")
        if (self.command in {"principal_create", "principal_update"}) != (
            self.principal_patch is not None
        ):
            raise ValueError("principal_patch does not match command")
        if self.command == "role_create" and self.role_patch.name is None:
            raise ValueError("New roles require a name")
        if self.command == "principal_create" and self.principal_patch.name is None:
            raise ValueError("New principals require a name")
        if (self.command in {"permission_add", "permission_remove"}) != (
            self.permission is not None
        ):
            raise ValueError("permission does not match command")
        if self.permission is not None:
            RolePatch(permissions=[self.permission])
        if (self.command == "check") != (self.resource is not None and self.action is not None):
            raise ValueError("check requires resource and action")
        if self.command != "check" and self.model_fields_set & {
            "resource",
            "action",
            "resource_attributes",
        }:
            raise ValueError("access fields are only supported by check")
        if self.command not in {"roles", "principals"} and self.model_fields_set & {
            "limit",
            "offset",
        }:
            raise ValueError("pagination is only supported by lists")
        return self


class PluginCommand(SourceBindingOpen):
    service_id: Handle
    command: Literal[
        "catalog",
        "shutdown",
        "load",
        "reload",
        "activate",
        "deactivate",
        "enable",
        "disable",
        "unload",
        "status",
        "health",
        "ports",
        "hook_register",
        "hook_unregister",
        "hook_fire",
        "hooks",
        "provider_register",
        "reporter_register",
        "capability_register",
        "check_factory_register",
        "provider_unregister",
        "reporter_unregister",
        "capability_unregister",
        "check_factory_unregister",
        "hook_fire_detailed",
        "validate",
        "render",
        "run_get",
        "run_export",
        "run_release",
        "provider_inspect",
    ]
    alias: Name | None = None
    reload_strategy: Literal["graceful", "immediate"] = "graceful"
    priority: Annotated[int, Field(ge=0, le=10000)] = 100
    payload: Annotated[dict[Name, JsonValue], Field(max_length=100)] = Field(default_factory=dict)
    hook_type: Name = "custom"
    asset_id: Handle | None = None
    provider_id: Name | None = None
    result_id: Handle | None = None
    run_id: Name | None = None
    check_aliases: Annotated[list[Name], Field(min_length=1, max_length=1000)] | None = None
    suite_name: Name = "plugin-suite"
    result_format: Literal["boolean_only", "basic", "summary", "complete"] = "summary"
    parallel: bool = False
    max_workers: Annotated[int, Field(ge=1, le=8)] | None = None
    pushdown: bool | None = None

    @model_validator(mode="after")
    def shape(self):
        if (
            self.command
            not in {
                "catalog",
                "shutdown",
                "ports",
                "hooks",
                "hook_fire",
                "hook_fire_detailed",
                "validate",
                "run_get",
                "run_export",
                "run_release",
            }
        ) != (self.alias is not None):
            raise ValueError("alias does not match command")
        if self.command != "hook_register" and "priority" in self.model_fields_set:
            raise ValueError("priority is only supported by hook_register")
        if (
            self.command not in {"hook_fire", "hook_fire_detailed"}
            and "payload" in self.model_fields_set
        ):
            raise ValueError("payload is only supported by hook_fire")
        if (self.command == "validate") != (self.check_aliases is not None):
            raise ValueError("check_aliases does not match command")
        if self.command == "validate" and (self.asset_id is None) == (self.provider_id is None):
            raise ValueError("Select one retained asset or registered provider")
        if self.command == "render" and (self.run_id is None) == (self.result_id is None):
            raise ValueError("Select one retained native run or canonical result artifact")
        if self.command in {"run_get", "run_export", "run_release"} and self.run_id is None:
            raise ValueError("run_id required")
        for key, commands in {
            "reload_strategy": {"reload"},
            "hook_type": {"hook_register", "hook_unregister", "hook_fire", "hook_fire_detailed"},
            "asset_id": {"validate"},
            "provider_id": {"validate"},
            "result_id": {"render"},
            "run_id": {"render", "run_get", "run_export", "run_release"},
            "suite_name": {"validate"},
            "result_format": {"validate"},
            "parallel": {"validate"},
            "max_workers": {"validate"},
            "pushdown": {"validate"},
        }.items():
            if key in self.model_fields_set and self.command not in commands:
                raise ValueError("Field does not match command")
        return self


class PlatformProcess(ServiceProcess):
    worker_module = "truthound.mcp.platform_worker"


class SecurityServices(BoundServiceSession):
    group = "security_services"
    process_type = PlatformProcess
    limit_commands = {"roles", "principals"}
    effects = (
        "installed_extension_execute",
        "native_role_principal_permission_changes",
        "operator_configured_rbac_hooks_and_storage",
        "explicit_role_or_principal_deletion",
    )


class PluginServices(BoundServiceSession):
    group = "plugin_services"
    process_type = PlatformProcess
    effects = (
        "installed_extension_execute",
        "operator_selected_plugin_lifecycle",
        "native_plugin_reload_and_observed_rollback",
        "native_plugin_cleanup_and_acknowledged_shutdown",
        "native_hook_provider_reporter_capability_registration",
        "operator_selected_custom_hook_execution",
        "native_registered_provider_validation_reporting_and_lifecycle_hooks",
        "sensitive_data_artifact",
    )

    def input_resources(self, arguments):
        if (
            arguments.check_aliases is not None
            and len(arguments.check_aliases) > self.session.policy.max_items
        ):
            raise MCPError("resource_exhausted")
        if arguments.asset_id is not None:
            return {"asset_path": self.session.resources.get(arguments.asset_id, "asset")}
        if arguments.result_id is not None:
            entry = self.session.resources.get(arguments.result_id, "artifact")
            if entry.media_type != "application/vnd.truthound.validation+json":
                raise MCPError("invalid_handle")
            return {"run_path": entry}
        return {}

    def artifact_media_type(self, arguments):
        if arguments.command == "run_export":
            return "application/vnd.truthound.validation+json"
        return super().artifact_media_type(arguments)
