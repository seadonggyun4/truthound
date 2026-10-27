"""Native store lifecycle with retained run artifacts and opaque record references."""

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import Field, JsonValue, model_validator

from truthound.mcp.approved_sources import SourceBindingOpen
from truthound.mcp.bound_services import BoundServiceSession
from truthound.mcp.checkpoint_services import ServiceProcess
from truthound.mcp.contracts import Name, WireModel
from truthound.mcp.policy import MCPError
from truthound.mcp.session_contracts import Handle


@dataclass(frozen=True)
class StoreMetric:
    name: str
    labels: dict[str, str] = field(default_factory=dict)
    kind: Literal["counter", "gauge"] = "counter"


@dataclass
class StoreRuntime:
    store: Any
    records: dict[str, Any] = field(default_factory=dict)
    data_assets: dict[str, str] = field(default_factory=dict)
    actor: str = "mcp-operator"
    tiers: dict[str, str] = field(default_factory=dict)
    replicas: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, StoreMetric] = field(default_factory=dict)


class StoredExpectation(WireModel):
    expectation_type: Name
    column: Name | None = None
    kwargs: dict[str, JsonValue] = Field(default_factory=dict)
    meta: dict[str, JsonValue] = Field(default_factory=dict)
    enabled: bool = True
    mostly: Annotated[float, Field(ge=0, le=1)] = 1.0


class StoredSuite(WireModel):
    name: Annotated[
        str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    ]
    data_asset_id: Name
    expectations: Annotated[list[StoredExpectation], Field(max_length=1000)] = Field(
        default_factory=list
    )
    tags: Annotated[dict[Name, Name], Field(max_length=100)] = Field(default_factory=dict)
    meta: dict[str, JsonValue] = Field(default_factory=dict)


class StoreFilter(WireModel):
    data_asset_id: Name | None = None
    start_time: Annotated[str, Field(max_length=64)] | None = None
    end_time: Annotated[str, Field(max_length=64)] | None = None
    status: Literal["success", "failure", "error"] | None = None
    tags: Annotated[dict[Name, Name], Field(max_length=100)] | None = None
    order_by: Literal["run_time", "data_asset", "status", "name"] = "run_time"
    ascending: bool = False

    @model_validator(mode="after")
    def dates(self):
        from datetime import datetime

        for key in ("start_time", "end_time"):
            if getattr(self, key) is not None:
                datetime.fromisoformat(getattr(self, key))
        return self


class StoreAuditFilter(WireModel):
    start_time: Annotated[str, Field(max_length=64)] | None = None
    end_time: Annotated[str, Field(max_length=64)] | None = None
    event_type: (
        Literal[
            "create",
            "read",
            "update",
            "delete",
            "query",
            "list",
            "count",
            "initialize",
            "close",
            "flush",
            "batch_create",
            "batch_delete",
            "replicate",
            "sync",
            "migrate",
            "rollback",
            "access_denied",
            "access_granted",
            "error",
            "validation_error",
        ]
        | None
    ) = None

    @model_validator(mode="after")
    def dates(self):
        from datetime import datetime

        for key in ("start_time", "end_time"):
            if getattr(self, key) is not None:
                datetime.fromisoformat(getattr(self, key))
        return self


class StoreBatchAction(WireModel):
    kind: Literal["save", "delete"]
    record_id: Name


class StoreCommand(SourceBindingOpen):
    service_id: Handle
    command: Literal[
        "inspect",
        "initialize",
        "save",
        "get",
        "get_access_tier",
        "set_access_tier",
        "exists",
        "delete",
        "list_ids",
        "query",
        "iter_query",
        "count",
        "clear",
        "get_latest",
        "get_history",
        "get_failures",
        "list_runs",
        "list_suites",
        "get_suite",
        "save_suite",
        "version_history",
        "version_info",
        "latest_version",
        "count_versions",
        "rollback",
        "diff",
        "cache_stats",
        "cache_flush",
        "cache_reconcile",
        "cache_invalidate",
        "cache_invalidate_all",
        "cache_refresh",
        "cache_warm",
        "cache_get_many",
        "batch_add",
        "batch_start",
        "batch_stop",
        "batch_flush",
        "batch_stats",
        "rebuild_index",
        "retention_cleanup",
        "retention_info",
        "retention_stats",
        "retention_history",
        "retention_estimate",
        "tier_list",
        "tier_info",
        "tier_move",
        "tier_run",
        "tier_stats",
        "replication_start",
        "replication_stop",
        "replication_flush",
        "replication_stats",
        "replication_sync",
        "migration_status",
        "migration_validate",
        "migration_run",
        "audit_query",
        "metric_read",
        "observability_flush",
        "concurrent_apply",
        "concurrent_stats",
        "concurrent_recovery_list",
        "concurrent_reconcile",
        "concurrent_recovery_cleanup",
        "pool_status",
        "pool_recycle",
        "shutdown",
    ]
    record_id: Name | None = None
    record_ids: Annotated[list[Name], Field(min_length=1, max_length=1000)] | None = None
    result_id: Handle | None = None
    suite: StoredSuite | None = None
    query: StoreFilter | None = None
    data_asset_id: Name | None = None
    version: Annotated[int, Field(ge=1)] | None = None
    version_b: Annotated[int, Field(ge=1)] | None = None
    expected_version: Annotated[int, Field(ge=0)] | None = None
    message: Annotated[str, Field(max_length=4096)] | None = None
    limit: Annotated[int, Field(ge=1, le=1000)] = 100
    offset: Annotated[int, Field(ge=0, le=1_000_000)] = 0
    batch_size: Annotated[int, Field(ge=1, le=1000)] = 100
    dry_run: bool = True
    tier_id: Name | None = None
    access_tier: Literal["Hot", "Cool", "Cold", "Archive"] | None = None
    replica_id: Name | None = None
    metric_id: Name | None = None
    audit: StoreAuditFilter | None = None
    actions: Annotated[list[StoreBatchAction], Field(min_length=1, max_length=1000)] | None = None
    recovery_id: Name | None = None
    resolution: Literal["rollback", "keep_current"] | None = None
    cache_resolution: Literal["confirm", "discard", "retry"] | None = None
    write_token: Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")] | None = None
    accept_duplicate_risk: bool = False

    @model_validator(mode="after")
    def shape(self):
        if (self.command == "cache_reconcile") != (
            self.cache_resolution is not None and self.write_token is not None
        ):
            raise ValueError("Cache reconciliation requires resolution and write token")
        if self.command != "cache_reconcile" and (
            self.cache_resolution is not None or self.write_token is not None
        ):
            raise ValueError("Unexpected cache reconciliation fields")
        if self.accept_duplicate_risk != (self.cache_resolution == "retry"):
            raise ValueError("Cache retry requires explicit duplicate-risk acceptance")
        save = self.command in {"save", "save_suite", "batch_add"}
        if save:
            if (
                sum(value is not None for value in (self.record_id, self.result_id, self.suite))
                != 1
            ):
                raise ValueError(
                    "Saving requires exactly one native record, retained result or suite"
                )
            if self.command == "save_suite" and self.result_id is not None:
                raise ValueError("save_suite requires an expectation suite")
        else:
            required = {
                "get",
                "get_access_tier",
                "set_access_tier",
                "exists",
                "delete",
                "get_suite",
                "version_history",
                "version_info",
                "latest_version",
                "count_versions",
                "rollback",
                "diff",
                "cache_invalidate",
                "cache_refresh",
                "retention_info",
                "tier_info",
                "tier_move",
                "migration_validate",
            }
            if self.command != "audit_query" and (self.command in required) != (
                self.record_id is not None
            ):
                raise ValueError("record_id does not match command")
            if self.result_id is not None or self.suite is not None:
                raise ValueError("Input does not match command")
        if (
            self.command in {"get_latest", "get_history", "list_runs"}
            and self.data_asset_id is None
        ):
            raise ValueError("data_asset_id required")
        if self.command in {"version_info", "rollback", "diff"} and self.version is None:
            raise ValueError("version required")
        if (self.command in {"cache_warm", "cache_get_many"}) != (self.record_ids is not None):
            raise ValueError("record_ids does not match command")
        if (self.command in {"concurrent_reconcile", "concurrent_recovery_cleanup"}) != (
            self.recovery_id is not None
        ):
            raise ValueError("recovery_id does not match command")
        if (self.command == "concurrent_reconcile") != (self.resolution is not None):
            raise ValueError("resolution does not match command")
        if (self.command == "concurrent_apply") != (self.actions is not None):
            raise ValueError("actions do not match command")
        if (self.command == "metric_read") != (self.metric_id is not None):
            raise ValueError("metric_id does not match command")
        for key, commands in {
            "audit": {"audit_query"},
            "query": {"query", "iter_query", "list_ids", "count", "clear"},
            "data_asset_id": {
                "get_latest",
                "get_history",
                "get_failures",
                "list_runs",
                "list_suites",
            },
            "version": {"get", "exists", "delete", "version_info", "rollback", "diff"},
            "version_b": {"diff"},
            "expected_version": {"save"},
            "message": {"save", "rollback"},
            "batch_size": {"iter_query", "migration_run"},
            "dry_run": {"retention_cleanup", "tier_run", "migration_run", "concurrent_apply"},
            "tier_id": {"tier_move"},
            "access_tier": {"set_access_tier"},
            "replica_id": {"replication_sync"},
            "limit": {
                "concurrent_recovery_list",
                "audit_query",
                "inspect",
                "list_ids",
                "query",
                "iter_query",
                "clear",
                "get_history",
                "get_failures",
                "list_runs",
                "list_suites",
                "version_history",
                "retention_history",
                "tier_list",
                "tier_stats",
                "replication_stats",
                "tier_run",
            },
            "offset": {
                "concurrent_recovery_list",
                "audit_query",
                "list_ids",
                "query",
                "iter_query",
                "clear",
                "list_runs",
                "list_suites",
                "version_history",
            },
        }.items():
            if key in self.model_fields_set and self.command not in commands:
                raise ValueError("Field does not match command")
        if self.command == "tier_move" and self.tier_id is None:
            raise ValueError("tier_id required")
        if self.command == "set_access_tier" and self.access_tier is None:
            raise ValueError("access_tier required")
        return self


class StoreProcess(ServiceProcess):
    worker_module = "truthound.mcp.store_worker"


class StoreServices(BoundServiceSession):
    group = "store_services"
    process_type = StoreProcess
    limit_commands = {
        "concurrent_recovery_list",
        "audit_query",
        "inspect",
        "list_ids",
        "query",
        "iter_query",
        "clear",
        "get_history",
        "get_failures",
        "list_runs",
        "list_suites",
        "version_history",
        "retention_history",
        "tier_list",
        "tier_stats",
        "replication_stats",
        "tier_run",
    }
    effects = (
        "installed_extension_execute",
        "operator_selected_native_store_read_write_delete",
        "native_version_cache_batch_lifecycle",
        "native_cache_write_behind_and_explicit_uncertain_write_reconciliation",
        "native_migration_callbacks_backup_lazy_read_and_rewrite",
        "native_store_audit_metric_query_and_observability_flush",
        "native_concurrent_file_batch_commit_rollback_and_index_writes",
        "native_retained_batch_rollback_or_keep_current_and_backup_deletion",
        "native_replica_background_io_and_retention_tier_actions",
        "native_database_connection_pool_recycle_and_acknowledged_shutdown",
        "native_azure_access_tier_change_and_observed_rehydration_status",
        "sensitive_data_artifact",
    )

    def input_resources(self, arguments):
        if arguments.result_id is None:
            return {}
        entry = self.session.resources.get(arguments.result_id, "artifact")
        if entry.media_type != "application/vnd.truthound.validation+json":
            raise MCPError("invalid_handle")
        return {"run_path": entry}
