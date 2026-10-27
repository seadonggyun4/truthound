"""Native tenant metadata/quota administration over explicit operator aliases."""

from copy import deepcopy
from dataclasses import asdict, replace

from truthound.mcp.governance_worker import selected
from truthound.mcp.policy import MCPError
from truthound.mcp.tenancy import TenantPatch, TenantRuntime


def tenant_view(alias, value):
    if value is None:
        return None
    return {
        "tenant_id": alias,
        "name": value.name,
        "slug": value.slug,
        "status": value.status.value,
        "tier": value.tier.value,
        "isolation_level": value.isolation_level.value,
        "owner_id": value.owner_id,
        "owner_email": value.owner_email,
        "quota": value.quota.to_dict(),
        "created_at": value.created_at,
        "updated_at": value.updated_at,
        "suspended_at": value.suspended_at,
    }


class TenantOwner:
    def __init__(self, runtime, request):
        from truthound.multitenancy.core import TenantSettings
        from truthound.multitenancy.isolation import RowLevelPolicy
        from truthound.multitenancy.manager import TenantManager

        if (
            request["native_group"] != "tenant_services"
            or not isinstance(runtime, TenantRuntime)
            or not isinstance(runtime.manager, TenantManager)
        ):
            raise MCPError("invalid_binding")
        if any(not isinstance(value, str) for value in runtime.tenants.values()) or any(
            not isinstance(value, TenantSettings) for value in runtime.settings.values()
        ):
            raise MCPError("invalid_binding")
        if (
            len(runtime.policies) > 4096
            or any(not isinstance(policy, RowLevelPolicy) for policy in runtime.policies.values())
            or len({policy.name for policy in runtime.policies.values()}) != len(runtime.policies)
        ):
            raise MCPError("invalid_binding")
        if len(runtime.contexts) > 4096 or any(
            not isinstance(value, dict) for value in runtime.contexts.values()
        ):
            raise MCPError("invalid_binding")
        from truthound.multitenancy.sources import TenantSQLSource

        if len(runtime.sources) > 4096 or any(
            not isinstance(value, TenantSQLSource) for value in runtime.sources.values()
        ):
            raise MCPError("invalid_binding")
        self.contexts = deepcopy(runtime.contexts)
        self.runtime = runtime

    async def execute(self, args, budget, *, inputs=None):
        from truthound.multitenancy.core import (
            IsolationLevel,
            ResourceType,
            TenantQuota,
            TenantTier,
        )

        runtime, command = self.runtime, args["command"]
        manager = runtime.manager
        if command == "catalog":
            offset, limit = args.get("offset", 0), args.get("limit", 100)
            return {
                "tenants": sorted(runtime.tenants)[offset : offset + limit],
                "settings": sorted(runtime.settings)[offset : offset + limit],
            }
        if command == "source_catalog":
            offset, limit = args.get("offset", 0), args.get("limit", 100)
            return {"sources": sorted(runtime.sources)[offset:offset + limit], "total": len(runtime.sources)}
        if command == "resolver_catalog":
            offset, limit = args.get("offset", 0), args.get("limit", 100)
            return {
                "contexts": sorted(self.contexts)[offset : offset + limit],
                "total": len(self.contexts),
            }
        if command in {"resolve", "resolve_or_create"}:
            context = deepcopy(selected(self.contexts, args["context_id"]))
            if command == "resolve_or_create":
                alias = args["tenant_id"]
                value = manager.resolve_or_create(
                    context, args["default_name"], expected_tenant_id=selected(runtime.tenants, alias)
                )
            else:
                value = manager.resolve(context)
                if value is None:
                    return {"resolved": False, "tenant_id": None}
                aliases = [key for key, native in runtime.tenants.items() if native == value.id]
                if len(aliases) != 1:
                    raise MCPError("invalid_binding")
                alias = aliases[0]
            return {"resolved": True, "tenant_id": alias, "tenant": tenant_view(alias, value)}
        alias = args["tenant_id"]
        native_id = selected(runtime.tenants, alias)
        if command == "source_read":
            source = selected(runtime.sources, args["source_id"])
            frame = source.read(manager, native_id)
            columns = args.get("columns")
            if columns is not None:
                if any(column not in frame.collect_schema().names() for column in columns):
                    raise MCPError("invalid_arguments")
                frame = frame.select(columns)
            limit, offset = args.get("limit", 100), args.get("offset", 0)
            page = frame.slice(offset, limit + 1).collect()
            return {"tenant_id": alias, "source_id": args["source_id"], "columns": page.columns,
                    "rows": page.head(limit).to_dicts(), "returned_rows": min(page.height, limit),
                    "truncated": page.height > limit}
        if command in {
            "isolation_apply",
            "policy_catalog",
            "policy_add",
            "policy_remove",
            "policy_update",
        }:
            from truthound.mcp.tenant_isolation import execute

            return execute(runtime, alias, native_id, args, inputs=inputs)
        if command == "get":
            return {"tenant": tenant_view(alias, manager.get(native_id))}
        if command in {"create", "update"}:
            patch = TenantPatch.model_validate(args["patch"]).model_dump(exclude_unset=True)
            if "settings_id" in patch:
                patch["settings"] = selected(runtime.settings, patch.pop("settings_id"))
            if "isolation_level" in patch:
                patch["isolation_level"] = IsolationLevel(patch["isolation_level"])
            if "quota" in patch:
                base = (
                    TenantQuota.for_tier(
                        TenantTier(args["tier"])
                        if args.get("tier")
                        else manager._config.default_tier
                    )
                    if command == "create"
                    else manager.require(native_id).quota
                )
                patch["quota"] = replace(base, **patch["quota"])
            if command == "create":
                if args.get("tier"):
                    patch["tier"] = TenantTier(args["tier"])
                value = manager.create(tenant_id=native_id, **patch)
            else:
                value = manager.update(native_id, **patch)
        elif command == "delete":
            return {"deleted": manager.delete(native_id)}
        elif command == "suspend":
            value = manager.suspend(native_id, reason=args.get("reason", ""))
        elif command == "activate":
            value = manager.activate(native_id)
        elif command == "set_tier":
            value = manager.set_tier(
                native_id, TenantTier(args["tier"]), update_quota=args.get("update_quota", True)
            )
        elif command == "context":
            with manager.context(native_id):
                active = manager.require_current()
                return {
                    "tenant_id": alias,
                    "current": active.id == native_id,
                    "status": active.status.value,
                }
        elif command in {"quota_check", "usage_track", "usage_summary"}:
            tenant = manager.require(native_id)
            resource = ResourceType(args["resource"])
            if command == "quota_check":
                return {
                    "allowed": manager.check_quota(resource, args.get("amount", 1), tenant=tenant)
                }
            if command == "usage_track":
                return {
                    "usage": manager.track_usage(resource, args.get("amount", 1), tenant=tenant)
                }
            summary = manager.get_usage_summary(resource, args.get("window", "day"), tenant=tenant)
            return {
                "usage": {
                    **asdict(summary),
                    "tenant_id": alias,
                    "is_exceeded": summary.is_exceeded,
                    "is_warning": summary.is_warning,
                }
            }
        else:
            raise MCPError("invalid_arguments")
        return {"tenant": tenant_view(alias, value)}

    async def close(self):
        # TenantManager has no shutdown API. The approved context factory owns
        # the selected native store, tracker, resolver and hook cleanup.
        pass


if __name__ == "__main__":
    from truthound.mcp.broker_worker import main

    main(TenantOwner)
