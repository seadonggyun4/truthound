"""A bounded native batch is completed within one exactly approved operation."""

from dataclasses import asdict
from uuid import uuid4

from truthound.mcp.governance_worker import selected
from truthound.mcp.policy import MCPError


def execute(owner, args):
    from truthound.stores.backends.concurrent_filesystem import ConcurrentFileSystemStore
    from truthound.stores.results import ValidationResult

    store = owner.runtime.store
    if not isinstance(store, ConcurrentFileSystemStore):
        raise MCPError("invalid_binding")
    if args["command"] == "concurrent_recovery_list":
        rows = store.list_batch_recovery(args.get("limit", 100), args.get("offset", 0))
        output = []
        for row in rows:
            pinned = (row["recovery_id"], row["fingerprint"])
            alias = next((key for key, value in owner.recoveries.items() if value == pinned), None)
            if alias is None:
                if len(owner.recoveries) >= 4096:
                    raise MCPError("resource_exhausted")
                alias = "recovery_" + uuid4().hex
                owner.recoveries[alias] = pinned
            output.append(
                {"recovery_id": alias, "state": row["state"], "file_count": row["file_count"]}
            )
        return {"recoveries": output}
    if args["command"] == "concurrent_recovery_cleanup":
        identity, fingerprint = selected(owner.recoveries, args["recovery_id"])
        result = store.cleanup_batch_recovery(identity, expected_fingerprint=fingerprint)
        owner.recoveries.pop(args["recovery_id"], None)
        return result
    if args["command"] == "concurrent_reconcile":
        identity, fingerprint = selected(owner.recoveries, args["recovery_id"])
        result = store.reconcile_batch(
            identity, resolution=args["resolution"], expected_fingerprint=fingerprint
        )
        owner.recoveries.pop(args["recovery_id"], None)
        return result
    if args["command"] == "concurrent_stats":
        store.initialize()
        stats = store.lock_statistics
        return {"statistics": asdict(stats) if stats is not None else None}
    prepared = []
    for action in args["actions"]:
        identity = selected(owner.records, action["record_id"])
        value = None
        if action["kind"] == "save":
            value = owner.objects.get(action["record_id"])
            if value is None:
                value = store.get(identity)
            if not isinstance(value, ValidationResult):
                raise MCPError("invalid_arguments")
        prepared.append((action["kind"], identity, value))
    with store.batch() as batch:
        for kind, identity, value in prepared:
            owner.guard()
            if kind == "save":
                batch.save(value)
            else:
                batch.delete(identity)
        if args.get("dry_run", True):
            count = batch.pending_count
            batch.rollback()
            return {"dry_run": True, "buffered": count}
        return {"dry_run": False, "applied": batch.commit()}
