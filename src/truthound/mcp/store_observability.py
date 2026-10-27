"""Explicit projections over an actual native observable store."""

from datetime import datetime

from truthound.mcp.governance_worker import selected
from truthound.mcp.policy import MCPError


def execute(owner, args):
    from truthound.stores.observability.metrics import InMemoryMetricsBackend
    from truthound.stores.observability.store import ObservableStore

    store, command = owner.runtime.store, args["command"]
    if not isinstance(store, ObservableStore):
        raise MCPError("invalid_binding")
    if command == "observability_flush":
        store.flush_observability()
        return {"flushed": True}
    if command == "metric_read":
        query = selected(owner.runtime.metrics, args["metric_id"])
        metrics = store.observability.metrics
        if metrics is None or not isinstance(metrics.backend, InMemoryMetricsBackend):
            raise MCPError("invalid_binding")
        value = (
            metrics.backend.get_counter(query.name, query.labels)
            if query.kind == "counter"
            else metrics.backend.get_gauge(query.name, query.labels)
        )
        return {"metric_id": args["metric_id"], "kind": query.kind, "value": value}
    filters = dict(args.get("audit") or {})
    for key in ("start_time", "end_time"):
        if filters.get(key) is not None:
            filters[key] = datetime.fromisoformat(filters[key])
    if args.get("record_id"):
        filters["resource_id"] = selected(owner.records, args["record_id"])
    limit, offset = args.get("limit", 100), args.get("offset", 0)
    if offset + limit > 10_000:
        raise MCPError("resource_exhausted")
    values = store.get_audit_events(**filters, limit=offset + limit)
    return {
        "events": [
            {
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type.value,
                "status": event.status.value,
                "record_id": owner.reference(event.resource_id) if event.resource_id else None,
                "duration_ms": event.duration_ms,
                "failed": event.status.value in {"failure", "partial", "denied"},
            }
            for event in values[offset : offset + limit]
        ],
        "next_offset": offset + limit if len(values) == offset + limit else None,
    }
