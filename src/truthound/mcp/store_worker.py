"""Selected native stores, with backend configuration owned by the factory."""

import json
from datetime import datetime
from itertools import islice
from pathlib import Path
from uuid import uuid4

from truthound.mcp.checkpoint_worker import binding_guard
from truthound.mcp.governance_worker import selected
from truthound.mcp.policy import MCPError
from truthound.mcp.store_services import StoreMetric, StoreRuntime
from truthound.mcp.tables import decode


class StoreOwner:
    def __init__(self, runtime, request):
        from truthound.stores.backends._object_store import ObjectValidationStore
        from truthound.stores.backends.concurrent_filesystem import (
            ConcurrentFileSystemExpectationStore,
            ConcurrentFileSystemStore,
        )
        from truthound.stores.backends.filesystem import FileSystemExpectationStore, FileSystemStore
        from truthound.stores.backends.memory import MemoryExpectationStore, MemoryStore
        from truthound.stores.base import BaseStore
        from truthound.stores.batching.store import BatchedStore
        from truthound.stores.caching.store import CachedStore
        from truthound.stores.expectations import ExpectationSuite
        from truthound.stores.migration.store import MigratableStore
        from truthound.stores.replication.store import ReplicatedStore
        from truthound.stores.results import ValidationResult
        from truthound.stores.tiering.store import TieredStore

        if request["native_group"] != "store_services" or not isinstance(runtime, StoreRuntime):
            raise MCPError("invalid_binding")
        store = runtime.store
        if not isinstance(store, (BaseStore, CachedStore, BatchedStore, ReplicatedStore)):
            raise MCPError("invalid_binding")
        self.base = store
        self.wrapper_layers = []
        seen = set()
        while isinstance(self.base, (CachedStore, BatchedStore, ReplicatedStore)):
            if id(self.base) in seen or len(self.wrapper_layers) >= 64:
                raise MCPError("invalid_binding")
            seen.add(id(self.base))
            self.wrapper_layers.append(self.base)
            self.base = (
                self.base.primary if isinstance(self.base, ReplicatedStore) else self.base.store
            )
        if not isinstance(self.base, BaseStore) or any(
            not isinstance(value, (ValidationResult, ExpectationSuite))
            for value in runtime.records.values()
        ):
            raise MCPError("invalid_binding")
        if any(
            not isinstance(value, StoreMetric) or value.kind not in {"counter", "gauge"}
            for value in runtime.metrics.values()
        ):
            raise MCPError("invalid_binding")
        self.runtime = runtime
        self.shutdown_requested = False
        self.records = {}
        self.recoveries = {}
        self.objects = dict(runtime.records)
        for alias, item in runtime.records.items():
            self.records[alias] = item.id
        self.guard = binding_guard(request)
        if isinstance(
            self.base,
            (
                MigratableStore,
                ConcurrentFileSystemStore,
                ConcurrentFileSystemExpectationStore,
                FileSystemStore,
                FileSystemExpectationStore,
                MemoryStore,
                MemoryExpectationStore,
                ObjectValidationStore,
            ),
        ):
            self.base.set_io_guard(self.guard)
        if isinstance(store, ReplicatedStore):
            if set(runtime.replicas.values()) != {target.name for target in store.config.targets}:
                raise MCPError("invalid_binding")
        if isinstance(store, TieredStore) and set(runtime.tiers.values()) != set(store.tiers):
            raise MCPError("invalid_binding")
        for layer in self.wrapper_layers:
            if isinstance(layer, (CachedStore, ReplicatedStore)):
                layer.set_io_guard(self.guard)
            if isinstance(layer, BatchedStore):
                write = layer._write_batch

                def guarded_write(items, native_write=write):
                    self.guard()
                    return native_write(items)

                layer._write_batch = guarded_write

    def reference(self, native_id):
        for alias, value in self.records.items():
            if value == native_id:
                return alias
        if len(self.records) >= 4096:
            raise MCPError("resource_exhausted")
        alias = "record_" + uuid4().hex
        self.records[alias] = native_id
        return alias

    def view(self, item):
        from truthound.stores.expectations import ExpectationSuite
        from truthound.stores.results import ValidationResult

        if item is None:
            return None
        alias = self.reference(item.id)
        asset = next(
            (key for key, value in self.runtime.data_assets.items() if value == item.data_asset),
            None,
        )
        if isinstance(item, ValidationResult):
            return {
                "record_id": alias,
                "data_asset_id": asset,
                "run_time": item.run_time.isoformat(),
                "status": item.status.value,
                "statistics": item.statistics.to_dict(),
                "tags": item.tags,
                "results": [
                    {
                        key: value
                        for key, value in result.to_dict().items()
                        if key not in {"details", "message"}
                    }
                    for result in item.results
                ],
            }
        if isinstance(item, ExpectationSuite):
            return {
                "record_id": alias,
                "data_asset_id": asset,
                "expectations": [
                    {key: value for key, value in row.to_dict().items() if key != "meta"}
                    for row in item.expectations
                ],
                "tags": item.tags,
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat(),
            }
        raise MCPError("unsupported_value")

    def version_view(self, info):
        if info is None:
            return None
        return {
            "record_id": self.reference(info.item_id),
            "version": info.version,
            "created_at": info.created_at.isoformat(),
            "parent_version": info.parent_version,
            "checksum": info.checksum,
            "size_bytes": info.size_bytes,
        }

    def query(self, args, *, bounded=True):
        from truthound.stores.base import StoreQuery

        fields = dict(args.get("query") or {})
        if fields.get("data_asset_id") is not None:
            fields["data_asset"] = selected(self.runtime.data_assets, fields.pop("data_asset_id"))
        else:
            fields.pop("data_asset_id", None)
        for key in ("start_time", "end_time"):
            if fields.get(key) is not None:
                fields[key] = datetime.fromisoformat(fields[key])
        return StoreQuery(
            **fields,
            limit=args.get("limit", 100) if bounded else None,
            offset=args.get("offset", 0),
        )

    async def execute(self, args, budget, *, inputs=None):
        from truthound.core.results import ValidationRunResult
        from truthound.stores.base import ExpectationStore, StoreConfig, ValidationStore
        from truthound.stores.batching.store import BatchedStore
        from truthound.stores.caching.store import CachedStore
        from truthound.stores.expectations import ExpectationSuite
        from truthound.stores.replication.store import ReplicatedStore
        from truthound.stores.results import ValidationResult
        from truthound.stores.versioning.store import VersionedStore

        store, command = self.runtime.store, args["command"]
        limit, offset = args.get("limit", 100), args.get("offset", 0)
        identity = selected(self.records, args["record_id"]) if args.get("record_id") else None
        asset = (
            selected(self.runtime.data_assets, args["data_asset_id"])
            if args.get("data_asset_id")
            else None
        )
        if command == "inspect":
            return {
                "store_type": type(store).__name__,
                "base_type": type(self.base).__name__,
                "validation": isinstance(self.base, ValidationStore),
                "expectation": isinstance(self.base, ExpectationStore),
                "versioned": isinstance(store, VersionedStore),
                "cached": isinstance(store, CachedStore),
                "batched": isinstance(store, BatchedStore),
                "records": sorted(self.records)[:limit],
                "metrics": sorted(self.runtime.metrics)[:limit],
                "data_assets": sorted(self.runtime.data_assets)[:limit],
                "serialization_format": self.base.config.serialization_format
                if isinstance(self.base.config, StoreConfig)
                else None,
                "compression": self.base.config.compression
                if isinstance(self.base.config, StoreConfig)
                else None,
            }
        if command == "shutdown":
            self.shutdown_requested = True
            return {"shutdown": True}
        if command in {"get_access_tier", "set_access_tier"}:
            from truthound.stores.backends.azure_blob import AzureBlobStore

            if not isinstance(store, AzureBlobStore):
                raise MCPError("invalid_binding")
            acknowledged = None
            if command == "set_access_tier":
                acknowledged = store.set_access_tier(identity, args["access_tier"])
            state = store.get_access_tier(identity)
            if state["archive_status"] not in {
                None,
                "rehydrate-pending-to-hot",
                "rehydrate-pending-to-cool",
                "rehydrate-pending-to-cold",
            }:
                raise MCPError("unsupported_value")
            return {
                "record_id": args["record_id"],
                **state,
                **({"acknowledged": acknowledged} if acknowledged is not None else {}),
            }
        if command in {"pool_status", "pool_recycle"}:
            from truthound.mcp.store_database import execute

            return execute(self, command)
        if command in {
            "concurrent_apply",
            "concurrent_stats",
            "concurrent_recovery_list",
            "concurrent_reconcile",
            "concurrent_recovery_cleanup",
        }:
            from truthound.mcp.store_concurrency import execute

            return execute(self, args)
        if command in {"audit_query", "metric_read", "observability_flush"}:
            from truthound.mcp.store_observability import execute

            return execute(self, args)
        if command.startswith(("retention_", "tier_", "replication_", "migration_")):
            from truthound.mcp.store_extensions import execute_extension

            return await execute_extension(self, args)
        if command == "initialize":
            self.base.initialize()
            return {"initialized": True}
        if command in {"save", "save_suite", "batch_add"}:
            if args.get("result_id"):
                if not inputs or set(inputs) != {"run_path"}:
                    raise MCPError("invalid_handle")
                run = ValidationRunResult.from_dict(
                    decode(json.loads(Path(inputs["run_path"]).read_text(encoding="utf-8")))
                )
                item = ValidationResult.from_validation_run_result(run)
            elif args.get("suite") is not None:
                fields = dict(args["suite"])
                fields["data_asset"] = selected(
                    self.runtime.data_assets, fields.pop("data_asset_id")
                )
                item = ExpectationSuite.from_dict(fields)
            else:
                item = self.objects.get(args["record_id"])
                if item is None:
                    item = store.get(identity)
            if isinstance(self.base, ExpectationStore) != isinstance(item, ExpectationSuite):
                raise MCPError("invalid_arguments")
            if command == "batch_add":
                if not isinstance(store, BatchedStore):
                    raise MCPError("invalid_binding")
                return {"added": await store.add(item), "record_id": self.reference(item.id)}
            if command == "save_suite":
                if not isinstance(store, ExpectationStore):
                    raise MCPError("invalid_binding")
                saved = store.save_suite(item)
            elif isinstance(store, VersionedStore):
                saved = store.save(
                    item,
                    message=args.get("message"),
                    created_by=self.runtime.actor,
                    expected_version=args.get("expected_version"),
                )
            elif isinstance(store, ReplicatedStore):
                saved = await store.save_async(item)
            else:
                if args.get("message") is not None or args.get("expected_version") is not None:
                    raise MCPError("invalid_arguments")
                saved = store.save(item)
            return {"record_id": self.reference(saved)}
        if command in {"get", "exists", "delete", "get_suite"}:
            if args.get("version") is not None and not isinstance(store, VersionedStore):
                raise MCPError("invalid_arguments")
            kwargs = {"version": args["version"]} if args.get("version") is not None else {}
            if command == "get":
                return {"record": self.view(store.get(identity, **kwargs))}
            if command == "exists":
                return {"exists": store.exists(identity, **kwargs)}
            if command == "delete":
                return {"deleted": store.delete(identity, **kwargs)}
            if not isinstance(store, ExpectationStore):
                raise MCPError("invalid_binding")
            return {"record": self.view(store.get_suite(identity))}
        if command.startswith("cache_"):
            if not isinstance(store, CachedStore):
                raise MCPError("invalid_binding")
            if command == "cache_flush":
                store.flush()
                return {"stats": store.get_stats()}
            if command == "cache_reconcile":
                store.reconcile_write(
                    args["write_token"],
                    args["cache_resolution"],
                    accept_duplicate_risk=args["accept_duplicate_risk"],
                )
                return {"stats": store.get_stats()}
            if command == "cache_stats":
                return {"stats": store.get_stats()}
            if command == "cache_invalidate_all":
                return {"invalidated": store.invalidate_all()}
            if command == "cache_invalidate":
                return {"invalidated": store.invalidate(identity)}
            if command == "cache_refresh":
                return {"record": self.view(store.refresh(identity))}
            ids = [selected(self.records, alias) for alias in args["record_ids"]]
            if command == "cache_warm":
                return {"warmed": store.warm(ids)}
            return {"records": [self.view(item) for item in store.get_many(ids).values()]}
        if command.startswith("batch_"):
            if not isinstance(store, BatchedStore):
                raise MCPError("invalid_binding")
            if command == "batch_stats":
                return {"stats": store.get_stats()}
            if command == "batch_flush":
                count = await store.flush()
                return {"batches": count, "stats": store.get_stats()}
            if command == "batch_start":
                await store.start()
            else:
                await store.stop()
            return {"stats": store.get_stats()}
        if command in {
            "version_history",
            "version_info",
            "latest_version",
            "count_versions",
            "rollback",
            "diff",
        }:
            if not isinstance(store, VersionedStore):
                raise MCPError("invalid_binding")
            if command == "version_history":
                return {
                    "versions": [
                        self.version_view(value)
                        for value in store.get_version_history(identity, limit, offset)
                    ]
                }
            if command == "version_info":
                return {
                    "version": self.version_view(store.get_version_info(identity, args["version"]))
                }
            if command == "latest_version":
                return {"version": self.version_view(store.get_latest_version_info(identity))}
            if command == "count_versions":
                return {"count": store.count_versions(identity)}
            if command == "rollback":
                return {
                    "record_id": self.reference(
                        store.rollback(
                            identity,
                            args["version"],
                            message=args.get("message"),
                            created_by=self.runtime.actor,
                        )
                    )
                }
            diff = store.diff(identity, args["version"], args.get("version_b"))
            # Native diff computes all changes; expose structural paths/types only.
            return {
                "record_id": args["record_id"],
                "version_a": diff.version_a,
                "version_b": diff.version_b,
                "changes": [
                    {"path": row["path"], "type": row["type"]}
                    for row in diff.changes
                    if not row["path"].startswith(("metadata", "runtime_environment"))
                ],
            }
        if command == "rebuild_index":
            from truthound.stores.backends._object_store import ObjectValidationStore
            from truthound.stores.backends.concurrent_filesystem import ConcurrentFileSystemStore
            from truthound.stores.backends.filesystem import FileSystemStore

            if not isinstance(
                store, (FileSystemStore, ConcurrentFileSystemStore, ObjectValidationStore)
            ):
                raise MCPError("invalid_binding")
            return {"indexed": store.rebuild_index()}
        # Wrappers have different public contracts. Never silently bypass their
        # cache/buffer/version behavior by querying an inner store.
        if (
            isinstance(store, (CachedStore, BatchedStore, ReplicatedStore))
            and command != "list_ids"
        ):
            raise MCPError("unsupported_operation")
        if command == "list_ids":
            return {
                "record_ids": [
                    self.reference(value) for value in store.list_ids(query=self.query(args))
                ]
            }
        if command == "query":
            return {"records": [self.view(value) for value in store.query(self.query(args))]}
        if command == "iter_query":
            iterator = store.iter_query(
                self.query(args), batch_size=min(limit, args.get("batch_size", 100))
            )
            try:
                return {"records": [self.view(value) for value in islice(iterator, limit)]}
            finally:
                iterator.close()
        if command == "count":
            return {"count": store.count(self.query(args, bounded=False))}
        if command == "clear":
            return {"deleted": store.clear(self.query(args))}
        if command == "get_latest":
            return {"record": self.view(store.get_latest(asset))}
        if command in {"list_runs", "get_history", "get_failures"}:
            if not isinstance(store, ValidationStore):
                raise MCPError("invalid_binding")
            if command == "list_runs":
                return {
                    "record_ids": [
                        self.reference(value)
                        for value in store.list_runs(asset)[offset : offset + limit]
                    ]
                }
            values = (
                store.get_history(asset, limit)
                if command == "get_history"
                else store.get_failures(asset, limit)
            )
            return {"records": [self.view(value) for value in values]}
        if command == "list_suites":
            if not isinstance(store, ExpectationStore):
                raise MCPError("invalid_binding")
            return {
                "record_ids": [
                    self.reference(value)
                    for value in store.list_suites(asset)[offset : offset + limit]
                ]
            }
        raise MCPError("invalid_arguments")

    async def close(self):
        from truthound.stores.batching.store import BatchedStore
        from truthound.stores.caching.store import CachedStore
        from truthound.stores.replication.store import ReplicatedStore

        if isinstance(self.runtime.store, BatchedStore):
            await self.runtime.store.stop()
        if isinstance(self.runtime.store, ReplicatedStore):
            await self.runtime.store.stop()
        if isinstance(self.runtime.store, CachedStore):
            self.runtime.store.close()
        self.recoveries.clear()
        self.objects.clear()
        self.records.clear()
        # The approved context factory closes the selected stores and their
        # shared providers exactly once, including wrapper-specific teardown.


if __name__ == "__main__":
    from truthound.mcp.broker_worker import main

    main(StoreOwner)
