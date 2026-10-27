"""Database pool projections without connection URLs, SQL or provider messages."""

from truthound.mcp.policy import MCPError


def execute(owner, command):
    from truthound.stores.backends.database import DatabaseStore

    store = owner.base
    if not isinstance(store, DatabaseStore):
        raise MCPError("unsupported_operation")
    owner.guard()
    if command == "pool_recycle":
        return {"recycled": store.recycle_connections()}
    native = store.get_pool_status()
    value = {key: native[key] for key in (
        "initialized", "disposed", "healthy", "using_pool_manager", "circuit_state", "dialect",
    ) if key in native}
    metrics = store.pool_metrics
    if metrics is not None:
        rows = metrics.to_dict()
        value["metrics"] = {
            group: {key: rows[group][key] for key in keys if key in rows[group]}
            for group, keys in {
                "connections": ("total_created", "total_closed", "current_size", "current_checked_out", "current_overflow", "peak", "idle"),
                "operations": ("checkouts", "checkins", "checkout_failures", "recycles", "invalidations"),
                "timing": ("avg_checkout_ms", "max_checkout_ms", "total_connection_time_ms"),
                "health": ("checks_passed", "checks_failed", "last_check"),
                "circuit_breaker": ("opens", "closes", "requests_rejected"),
            }.items() if group in rows
        }
    return value
