"""Native retention, tiering and replica operations with explicit projections."""

from truthound.mcp.governance_worker import selected
from truthound.mcp.policy import MCPError


def result_view(value):
    result = value.to_dict() if hasattr(value, "to_dict") else dict(value)
    errors = result.pop("errors", [])
    result["error_count"] = len(errors)
    result["native_failed"] = bool(errors)
    return result


def public_alias(mapping, value):
    return next((alias for alias, native in mapping.items() if native == value), None)


async def execute_extension(owner, args):
    from truthound.stores.migration.store import MigratableStore
    from truthound.stores.replication.store import ReplicatedStore
    from truthound.stores.retention.store import RetentionStore
    from truthound.stores.tiering.store import TieredStore

    store, runtime, command = owner.runtime.store, owner.runtime, args["command"]
    identity = selected(owner.records, args["record_id"]) if args.get("record_id") else None
    limit = args.get("limit", 100)
    if command.startswith("migration_"):
        if not isinstance(store, MigratableStore):
            raise MCPError("invalid_binding")
        if command == "migration_status":
            status = store.get_migration_status()
            return {"status": status, "native_failed": bool(status["failed_items"])}
        if command == "migration_validate":
            result = store.validate_item(identity)
            return {
                "validation": {
                    "record_id": args["record_id"],
                    **{
                        key: result[key]
                        for key in (
                            "current_version",
                            "target_version",
                            "needs_migration",
                            "can_migrate",
                            "migration_path",
                        )
                        if key in result
                    },
                },
                "native_failed": "error" in result,
            }
        if command == "migration_run":
            result = store.migrate_all(
                dry_run=args.get("dry_run", True),
                batch_size=args.get("batch_size", 100),
            )
            values = result.to_dict()
            return {
                "migration": {
                    key: values[key]
                    for key in (
                        "start_time",
                        "end_time",
                        "duration_seconds",
                        "from_version",
                        "to_version",
                        "items_migrated",
                        "items_failed",
                        "items_skipped",
                        "dry_run",
                        "success",
                    )
                },
                "applied_count": len(result.migrations_applied),
                "error_count": len(result.errors),
                "native_failed": not result.success,
            }
    elif command.startswith("retention_"):
        if not isinstance(store, RetentionStore):
            raise MCPError("invalid_binding")
        if command == "retention_cleanup":
            return {"cleanup": result_view(store.run_cleanup(dry_run=args.get("dry_run", True)))}
        if command == "retention_history":
            return {"history": [result_view(row) for row in store.get_cleanup_history(limit)]}
        if command == "retention_estimate":
            estimate = store.estimate_cleanup()
            history = store.get_cleanup_history(1)
            errors = history[-1]["errors"] if history else []
            return {
                "estimate": {**estimate, "native_failed": bool(errors), "error_count": len(errors)}
            }
        if command == "retention_stats":
            return {"stats": store.get_cleanup_stats()}
        if command == "retention_info":
            row = store.get_retention_info(identity)
            return {
                "retention": {
                    "record_id": args["record_id"],
                    **{key: row[key] for key in ("will_be_retained", "excluded", "expiry_time")},
                }
            }
    elif command.startswith("tier_"):
        if not isinstance(store, TieredStore):
            raise MCPError("invalid_binding")
        if command == "tier_list":
            return {
                "tiers": [
                    {
                        "tier_id": public_alias(runtime.tiers, row["name"]),
                        **{
                            key: row[key]
                            for key in ("tier_type", "priority", "cost_per_gb", "retrieval_time_ms")
                        },
                    }
                    for row in store.list_tiers()[:limit]
                ]
            }
        if command == "tier_stats":
            return {
                "tiers": {
                    public_alias(runtime.tiers, name): value
                    for name, value in list(store.get_tier_stats().items())[:limit]
                }
            }
        if command == "tier_info":
            info = store.get_item_info(identity)
            if info is None:
                return {"tier": None}
            fields = info.to_dict()
            fields.pop("item_id")
            fields["tier_id"] = public_alias(runtime.tiers, fields.pop("tier_name"))
            return {"tier": {"record_id": args["record_id"], **fields}}
        if command == "tier_move":
            return {"moved": store.move_to_tier(identity, selected(runtime.tiers, args["tier_id"]))}
        if command == "tier_run":
            value = result_view(store.run_tiering(dry_run=args.get("dry_run", True)))
            moves = value.pop("migrations")
            value["migrations"] = [
                {
                    "record_id": owner.reference(row["item_id"]),
                    "from_tier_id": public_alias(runtime.tiers, row["from_tier"]),
                    "to_tier_id": public_alias(runtime.tiers, row["to_tier"]),
                    "size_bytes": row["size_bytes"],
                }
                for row in moves[:limit]
            ]
            value["truncated"] = len(moves) > limit
            return {"tiering": value}
    elif command.startswith("replication_"):
        if not isinstance(store, ReplicatedStore):
            raise MCPError("invalid_binding")
        if command == "replication_sync":
            name = (
                selected(runtime.replicas, args["replica_id"]) if args.get("replica_id") else None
            )
            value = await store.full_sync(name)
            return {
                "replicas": {public_alias(runtime.replicas, key): row for key, row in value.items()}
            }
        if command == "replication_start":
            await store.start()
        elif command == "replication_stop":
            await store.stop()
        elif command == "replication_flush":
            await store.flush()
        elif command != "replication_stats":
            raise MCPError("invalid_arguments")
        stats = store.get_stats()
        metrics = {
            key: value
            for key, value in stats["metrics"].items()
            if key not in {"replica_success_rates", "current_lag_ms"}
        }
        for field in ("replica_success_rates", "current_lag_ms"):
            metrics[field] = {
                public_alias(runtime.replicas, key): value
                for key, value in stats["metrics"][field].items()
            }
        return {
            "stats": {
                "mode": stats["mode"],
                "read_preference": stats["read_preference"],
                "queue_size": stats["queue_size"],
                "pending_tasks": stats["pending_tasks"],
                "background_failures": stats["background_failures"],
                "monitor": {
                    "running": stats["health"]["running"],
                    "failures": stats["health"]["monitor_failures"],
                    "handler_failures": stats["health"]["handler_failures"],
                    "errors": [
                        {"phase": error["phase"],
                         "replica_id": public_alias(runtime.replicas, error["replica"]),
                         "handler_index": error["handler_index"], "timestamp": error["timestamp"]}
                        for error in stats["health"]["errors"][-limit:]
                    ],
                },
                "metrics": metrics,
                "replicas": [
                    {
                        "replica_id": public_alias(runtime.replicas, target.name),
                        "health": target.health.value,
                        "state": target.state.value,
                        "lag_ms": target.replication_lag_ms,
                    }
                    for target in store.config.targets[:limit]
                ],
            }
        }
    raise MCPError("invalid_arguments")
