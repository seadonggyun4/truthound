"""Native RBAC/plugin manager ownership with closed dispatch and safe projections."""

import asyncio
from dataclasses import asdict

from truthound.mcp.checkpoint_worker import binding_guard
from truthound.mcp.platform_services import PluginRuntime, RBACRuntime
from truthound.mcp.policy import MCPError


def permission_view(permission):
    return permission.to_dict()


def native_permission(value):
    from truthound.rbac.core import Permission

    return Permission.parse(value) if isinstance(value, str) else Permission.from_dict(value)


class PlatformOwner:
    def __init__(self, runtime, request):
        from truthound.plugins import Plugin, PluginManager
        from truthound.rbac import RBACManager

        if request["native_group"] == "security_services":
            if not isinstance(runtime, RBACRuntime) or not isinstance(runtime.manager, RBACManager):
                raise MCPError("invalid_binding")
        elif request["native_group"] == "plugin_services":
            if not isinstance(runtime, PluginRuntime) or not isinstance(
                runtime.manager, PluginManager
            ):
                raise MCPError("invalid_binding")
            if any(
                not isinstance(cls, type) or not issubclass(cls, Plugin)
                for cls in runtime.classes.values()
            ):
                raise MCPError("invalid_binding")
            if any(not callable(handler) for handler in runtime.hooks.values()):
                raise MCPError("invalid_binding")
        else:
            raise MCPError("invalid_binding")
        self.runtime = runtime
        self.loaded = {}
        self.native_runs = {}
        self.shutdown_requested = False
        self.reloads = None
        self.guard = binding_guard(request)
        if isinstance(runtime, PluginRuntime):
            runtime.manager.set_execution_guard(self.guard)
            if any(
                alias not in runtime.classes or not callable(factory)
                for alias, factory in runtime.reload_factories.items()
            ):
                raise MCPError("invalid_binding")
            if runtime.reload_factories:
                from truthound.plugins.lifecycle.hot_reload import HotReloadManager

                def load_replacement(name):
                    self.guard()
                    alias = next(
                        (alias for alias, loaded in self.loaded.items() if loaded == name), None
                    )
                    if alias not in runtime.reload_factories:
                        raise MCPError("invalid_binding")
                    return runtime.reload_factories[alias](name)

                self.reloads = HotReloadManager(runtime.manager, plugin_loader=load_replacement)

    def hook_event(self, args):
        from truthound.plugins.hooks import HookType

        alias = args.get("hook_type", "custom")
        if alias in self.runtime.hook_types:
            return self.runtime.hook_types[alias]
        try:
            return HookType(alias)
        except ValueError:
            raise MCPError("invalid_binding") from None

    def role(self, identity):
        role = self.runtime.manager.get_role(identity)
        if (
            role is not None
            and self.runtime.tenant_id is not None
            and role.tenant_id != self.runtime.tenant_id
        ):
            raise MCPError("access_denied")
        return role

    def principal(self, identity):
        principal = self.runtime.manager.get_principal(identity)
        if (
            principal is not None
            and self.runtime.tenant_id is not None
            and principal.tenant_id != self.runtime.tenant_id
        ):
            raise MCPError("access_denied")
        return principal

    def role_view(self, role):
        if role is None:
            return None
        return {
            "id": role.id,
            "name": role.name,
            "description": role.description,
            "enabled": role.enabled,
            "tenant_id": role.tenant_id,
            "parent_roles": sorted(role.parent_roles),
            "conditions": [
                condition.to_dict()
                for condition in role.conditions
                if self.condition_allowed(condition)
            ],
            "conditions_redacted": any(not self.condition_allowed(c) for c in role.conditions),
            "permissions": [
                permission_view(value)
                for value in sorted(
                    role.permissions, key=lambda value: (value.to_string(), value.effect.value)
                )
            ],
        }

    def condition_allowed(self, condition):
        def allowed(path):
            return path.startswith("resource.") or path in {
                "environment.timestamp",
                "principal.id",
                "principal.tenant_id",
                "principal.name",
                "principal.email",
                "principal.type",
                "principal.enabled",
                "principal.roles",
                *self.runtime.condition_fields,
                *("principal.attributes." + key for key in self.runtime.attribute_keys),
            }

        value = condition.value
        return allowed(condition.field) and (
            not isinstance(value, str)
            or not (value.startswith("${") and value.endswith("}"))
            or allowed(value[2:-1])
        )

    def principal_view(self, principal):
        if principal is None:
            return None
        return {
            "id": principal.id,
            "name": principal.name,
            "email": principal.email,
            "type": principal.type.value,
            "enabled": principal.enabled,
            "tenant_id": principal.tenant_id,
            "roles": sorted(principal.roles),
            "direct_permissions": [
                permission_view(value)
                for value in sorted(
                    principal.direct_permissions,
                    key=lambda value: (value.to_string(), value.effect.value),
                )
            ],
            "attributes": {
                key: principal.attributes[key]
                for key in self.runtime.attribute_keys
                if key in principal.attributes
            },
        }

    def security(self, args):
        from truthound.rbac.core import Condition

        runtime, command = self.runtime, args["command"]
        manager = runtime.manager
        role_id, principal_id = args.get("role_id"), args.get("principal_id")
        role = self.role(role_id) if role_id is not None else None
        principal = self.principal(principal_id) if principal_id is not None else None
        if command == "roles":
            values = manager.list_roles(tenant_id=runtime.tenant_id)
            offset, limit = args.get("offset", 0), args.get("limit", 100)
            return {"roles": [self.role_view(value) for value in values[offset : offset + limit]]}
        if command == "principals":
            values = manager.list_principals(tenant_id=runtime.tenant_id)
            offset, limit = args.get("offset", 0), args.get("limit", 100)
            return {
                "principals": [
                    self.principal_view(value) for value in values[offset : offset + limit]
                ]
            }
        if command == "role":
            return {"role": self.role_view(role)}
        if command == "principal":
            return {"principal": self.principal_view(principal)}
        if command in {"role_create", "role_update"}:
            from truthound.mcp.platform_services import RolePatch

            patch = RolePatch.model_validate(args["role_patch"]).model_dump(exclude_unset=True)
            if "permissions" in patch:
                patch["permissions"] = {native_permission(value) for value in patch["permissions"]}
            if "conditions" in patch:
                patch["conditions"] = [Condition.from_dict(value) for value in patch["conditions"]]
                if not all(self.condition_allowed(c) for c in patch["conditions"]):
                    raise MCPError("access_denied")
            if "parent_roles" in patch:
                for parent in patch["parent_roles"]:
                    if self.role(parent) is None:
                        raise MCPError("invalid_arguments")
                patch["parent_roles"] = set(patch["parent_roles"])
            if command == "role_create":
                role = manager.create_role(role_id=role_id, tenant_id=runtime.tenant_id, **patch)
            else:
                role = manager.update_role(role_id, **patch)
            return {"role": self.role_view(role)}
        if command in {"principal_create", "principal_update"}:
            from truthound.mcp.platform_services import PrincipalPatch

            patch = PrincipalPatch.model_validate(args["principal_patch"]).model_dump(
                exclude_unset=True
            )
            if set(patch.get("attributes", {})) - set(runtime.attribute_keys):
                raise MCPError("access_denied")
            if "direct_permissions" in patch:
                patch["direct_permissions"] = {
                    native_permission(value) for value in patch["direct_permissions"]
                }
            if command == "principal_create":
                permissions = patch.pop("direct_permissions", set())
                principal = manager.create_principal(
                    principal_id=principal_id,
                    tenant_id=runtime.tenant_id,
                    permissions=permissions,
                    **patch,
                )
            else:
                principal = manager.update_principal(principal_id, **patch)
            return {"principal": self.principal_view(principal)}
        if command == "role_delete":
            return {"deleted": manager.delete_role(role_id)}
        if command == "principal_delete":
            return {"deleted": manager.delete_principal(principal_id)}
        if command in {"assign_role", "revoke_role"}:
            if role is None or principal is None:
                raise MCPError("invalid_arguments")
            if command == "assign_role":
                principal = manager.assign_role(principal_id, role_id)
            else:
                principal = manager.revoke_role(principal_id, role_id)
            return {"principal": self.principal_view(principal)}
        if command in {"permission_add", "permission_remove"}:
            if command == "permission_add":
                role = manager.add_permission_to_role(
                    role_id, native_permission(args["permission"])
                )
            else:
                role = manager.remove_permission_from_role(
                    role_id, native_permission(args["permission"])
                )
            return {"role": self.role_view(role)}
        if command == "permissions":
            values = manager.get_principal_permissions(principal_id)
            return {
                "permissions": [
                    permission_view(value)
                    for value in sorted(
                        values, key=lambda value: (value.to_string(), value.effect.value)
                    )
                ]
            }
        if command == "check":
            if principal is None:
                raise MCPError("invalid_arguments")
            result = manager.check(
                principal_id, args["resource"], args["action"], args.get("resource_attributes", {})
            )
            return {
                "allowed": result.allowed,
                "effect": result.effect.value,
                "reason": result.reason,
                "matching_permissions": [
                    permission_view(value) for value in result.matching_permissions
                ],
                "matching_policies": result.matching_policies,
            }
        if command == "cache_clear":
            manager.engine.invalidate_cache()
            return {"cleared": True}
        raise MCPError("invalid_arguments")

    def plugin(self, args):
        self.guard()
        runtime, command = self.runtime, args["command"]
        manager, alias = runtime.manager, args.get("alias")
        if command == "catalog":
            return {"aliases": sorted(runtime.classes)}
        if command == "ports":
            return {
                "capabilities": manager.list_capabilities(),
                "providers": manager.list_data_asset_providers(),
                "reporters": manager.list_reporter_contracts(),
                "check_factories": sorted(manager.iter_check_factories()),
            }
        if command == "hooks":
            return {
                "hooks": [
                    {
                        "type": hook.hook_type,
                        "priority": hook.priority,
                        "source": hook.source,
                        "enabled": hook.enabled,
                        "async": hook.async_handler,
                    }
                    for hook in manager.hooks.get_hooks()
                ]
            }
        if command in {"hook_register", "hook_unregister"}:
            handler = runtime.hooks.get(alias)
            if handler is None:
                raise MCPError("invalid_binding")
            if command == "hook_register":
                if any(
                    hook.handler is handler
                    for hook in manager.hooks.get_hooks(self.hook_event(args))
                ):
                    raise MCPError("invalid_arguments")
                manager.hooks.register(
                    self.hook_event(args), handler, priority=args.get("priority", 100), source=alias
                )
                return {"registered": True}
            return {"removed": manager.hooks.unregister(self.hook_event(args), handler=handler)}
        registrations = {
            "provider_register": (runtime.providers, manager.register_data_asset_provider),
            "reporter_register": (runtime.reporters, manager.register_reporter),
            "capability_register": (runtime.capabilities, manager.register_capability),
            "check_factory_register": (runtime.check_factories, manager.register_check_factory),
        }
        if command in registrations:
            values, register = registrations[command]
            if alias not in values:
                raise MCPError("invalid_binding")
            register(alias, values[alias])
            return {"registered": True}
        removals = {
            "provider_unregister": manager.unregister_data_asset_provider,
            "reporter_unregister": manager.unregister_reporter,
            "capability_unregister": manager.unregister_capability,
            "check_factory_unregister": manager.unregister_check_factory,
        }
        if command in removals:
            return {"removed": removals[command](alias)}
        if alias not in runtime.classes:
            raise MCPError("invalid_binding")
        if command == "load":
            plugin = manager.load_from_class(
                runtime.classes[alias], config=runtime.configs.get(alias), activate=False
            )
            self.loaded[alias] = plugin.name
        else:
            name = self.loaded.get(alias)
            if name is None:
                raise MCPError("invalid_handle")
            if command == "unload":
                receipt = manager.unload_plugin(name, detailed=True)
                del self.loaded[alias]
                if not receipt.success:
                    return {"loaded": False, "native_failed": True, "cleanup": receipt.to_dict()}
                return {"loaded": False}
            if command == "activate":
                manager.activate_plugin(name)
            elif command == "deactivate":
                manager.deactivate_plugin(name)
            elif command == "enable":
                manager.enable_plugin(name)
            elif command == "disable":
                manager.disable_plugin(name)
            elif command == "health":
                healthy = manager.get_plugin(name).health_check()
                if type(healthy) is not bool:
                    raise MCPError("unsupported_value")
                return {"healthy": healthy}
            elif command != "status":
                raise MCPError("invalid_arguments")
            plugin = manager.get_plugin(name)
        return {"loaded": True, "state": plugin.state.value, "info": asdict(plugin.info)}

    async def execute(self, args, budget, *, inputs=None):
        if isinstance(self.runtime, RBACRuntime):
            return await asyncio.to_thread(self.security, args)
        if args["command"] == "shutdown":
            self.guard()
            self.shutdown_requested = True
            return {"shutdown": True}
        if args["command"] == "reload":
            from truthound.plugins.lifecycle.hot_reload import ReloadStrategy

            self.guard()
            alias = args["alias"]
            name = self.loaded.get(alias)
            if self.reloads is None or alias not in self.runtime.reload_factories:
                raise MCPError("unsupported_operation")
            if name is None:
                raise MCPError("invalid_handle")
            result = await self.reloads.reload(
                name, ReloadStrategy[args.get("reload_strategy", "graceful").upper()]
            )
            loaded = self.runtime.manager.is_plugin_loaded(name)
            if not loaded:
                self.loaded.pop(alias, None)
            return {
                "reloaded": result.success,
                "loaded": loaded,
                "rolled_back": result.rolled_back,
                "error_type": result.error,
                "native_failed": not result.success,
            }
        if args["command"] == "hook_fire":
            return {
                "results": await self.runtime.manager.hooks.trigger_async(
                    self.hook_event(args), payload=args.get("payload", {})
                )
            }
        if args["command"] == "hook_fire_detailed":
            from truthound.mcp.plugin_workflows import hook_rows

            rows = await self.runtime.manager.hooks.trigger_detailed_async(
                self.hook_event(args), payload=args.get("payload", {})
            )
            return {"hooks": hook_rows(rows), "native_failed": any(not row.success for row in rows)}
        if args["command"] in {
            "validate",
            "render",
            "run_get",
            "run_export",
            "run_release",
            "provider_inspect",
        }:
            from truthound.mcp.plugin_workflows import execute

            return await execute(self, args, budget, inputs=inputs)
        return await asyncio.to_thread(self.plugin, args)

    async def close(self):
        self.native_runs.clear()
        errors = []
        if self.reloads is not None:
            try:
                await self.reloads.aclose()
            except Exception as error:
                errors.append(error)
        if isinstance(self.runtime, PluginRuntime):
            for alias, name in tuple(self.loaded.items()):
                try:
                    if self.runtime.manager.is_plugin_loaded(name):
                        receipt = self.runtime.manager.unload_plugin(name, detailed=True)
                        if not receipt.success:
                            errors.append(RuntimeError("Native plugin cleanup failed"))
                except Exception as error:
                    errors.append(error)
                finally:
                    if not self.runtime.manager.is_plugin_loaded(name):
                        self.loaded.pop(alias, None)
        if errors:
            raise ExceptionGroup("Plugin owner cleanup failed", errors)


if __name__ == "__main__":
    from truthound.mcp.broker_worker import main

    main(PlatformOwner)
