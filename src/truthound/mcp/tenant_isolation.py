"""Tenant row filtering and bound policy edits, delegated to native Core."""

from dataclasses import replace

from truthound.mcp.governance_worker import selected
from truthound.mcp.policy import MCPError


def execute(runtime, alias, native_id, args, *, inputs):
    from truthound.multitenancy.core import IsolationLevel
    from truthound.multitenancy.isolation import RowLevelIsolation

    tenant = runtime.manager.require(native_id)
    strategy = runtime.manager.isolation_manager.get_strategy(tenant)
    strategy.validate_isolation(tenant)
    command = args["command"]
    if command != "isolation_apply":
        if not isinstance(strategy, RowLevelIsolation):
            raise MCPError("unsupported_operation")
        if command == "policy_catalog":
            active = {policy.name for policy in strategy.policies}
            offset, limit = args.get("offset", 0), args.get("limit", 100)
            aliases = sorted(runtime.policies)[offset:offset + limit]
            return {"policies": [{"policy_id": key, "active": runtime.policies[key].name in active,
                                   "enabled": runtime.policies[key].enabled,
                                   "priority": runtime.policies[key].priority} for key in aliases],
                    "total": len(runtime.policies)}
        policy = selected(runtime.policies, args["policy_id"])
        if command == "policy_add":
            strategy.add_policy(policy)
            return {"added": True, "policy_id": args["policy_id"]}
        if command == "policy_remove":
            return {"removed": strategy.remove_policy(policy.name)}
        patched = replace(policy, **args["policy_patch"])
        active = any(row.name == policy.name for row in strategy.policies)
        if active:
            strategy.remove_policy(policy.name)
            strategy.add_policy(patched)
        runtime.policies[args["policy_id"]] = patched
        return {"policy_id": args["policy_id"], "active": active,
                "enabled": patched.enabled, "priority": patched.priority}
    # A caller-supplied Arrow table cannot prove provenance in a separate
    # tenant schema/database. Only the native column-filtering levels apply.
    if tenant.isolation_level not in {IsolationLevel.SHARED, IsolationLevel.ROW_LEVEL}:
        raise MCPError("unsupported_operation")
    if not inputs or set(inputs) != {"asset_path"}:
        raise MCPError("invalid_handle")
    import polars as pl

    frame = pl.scan_ipc(inputs["asset_path"], memory_map=False)
    columns = args.get("columns")
    if columns is not None and any(column not in frame.collect_schema().names() for column in columns):
        raise MCPError("invalid_arguments")
    filtered = runtime.manager.apply_isolation(frame, tenant)
    if columns is not None:
        filtered = filtered.select(columns)
    limit, offset = args.get("limit", 100), args.get("offset", 0)
    page = filtered.slice(offset, limit + 1).collect()
    return {"tenant_id": alias, "isolation_level": tenant.isolation_level.value,
            "columns": page.columns, "rows": page.head(limit).to_dicts(),
            "returned_rows": min(page.height, limit), "truncated": page.height > limit}
