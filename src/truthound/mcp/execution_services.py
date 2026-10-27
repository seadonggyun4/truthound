"""Approved native execution engines and retained native child engines."""

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import Field, JsonValue, model_validator

from truthound.mcp.approved_sources import SourceBindingOpen
from truthound.mcp.bound_services import BoundServiceSession
from truthound.mcp.checkpoint_services import ServiceProcess
from truthound.mcp.contracts import Name
from truthound.mcp.session_contracts import Handle

Aggregation = Literal[
    "count",
    "sum",
    "mean",
    "median",
    "min",
    "max",
    "std",
    "var",
    "first",
    "last",
    "count_distinct",
    "null_count",
]


@dataclass
class ExecutionRuntime:
    engine: Any
    predicates: dict[str, str] = field(default_factory=dict)
    queries: dict[str, str] = field(default_factory=dict)
    expressions: dict[str, Any] = field(default_factory=dict)
    row_filters: dict[str, Any] = field(default_factory=dict)
    plans: dict[str, Any] = field(default_factory=dict)
    validators: dict[str, Any] = field(default_factory=dict)
    adapter_config: Any = None


class ExecutionCommand(SourceBindingOpen):
    service_id: Handle
    engine_id: Name = "root"
    command: Literal[
        "inspect",
        "count_rows",
        "count_nulls",
        "count_nulls_all",
        "count_distinct",
        "get_stats",
        "get_completeness_ratio",
        "get_distinct_values",
        "get_column_values",
        "get_sample_values",
        "get_quantiles",
        "get_value_counts",
        "aggregate",
        "aggregate_column",
        "aggregate_plan",
        "aggregate_stream",
        "validate",
        "count_matching",
        "count_not_matching",
        "filter_by_condition",
        "count_matching_regex",
        "count_not_matching_regex",
        "count_in_range",
        "count_outside_range",
        "count_in_set",
        "count_not_in_set",
        "count_duplicates",
        "get_duplicate_values",
        "sample",
        "to_polars",
        "to_numpy",
        "to_numpy_safe",
        "select",
        "filter",
        "with_columns",
        "execute_query",
        "clear_cache",
        "release",
        "shutdown",
        "partitions",
        "repartition",
        "coalesce",
        "metrics",
        "clear_metrics",
        "persist",
        "unpersist",
        "materialize",
        "checkpoint",
        "native_filter",
        "filter_rows",
        "head",
        "tail",
        "take",
        "compute",
        "to_arrow",
        "arrow_stream_open",
        "arrow_stream_read",
        "arrow_stream_close",
    ]
    stream_id: Name | None = None
    aggregation_plan_id: Name | None = None
    validator_ids: Annotated[list[Name], Field(min_length=1, max_length=32)] | None = None
    batch_size: Annotated[int, Field(ge=1, le=100_000)] = 10_000
    column: Annotated[str, Field(min_length=1, max_length=1024)] | None = None
    columns: Annotated[list[str], Field(min_length=1, max_length=100)] | None = None
    aggregations: Annotated[dict[str, Aggregation], Field(min_length=1, max_length=100)] | None = (
        None
    )
    aggregation: Aggregation | None = None
    quantiles: (
        Annotated[list[Annotated[float, Field(ge=0, le=1)]], Field(min_length=1, max_length=100)]
        | None
    ) = None
    pattern: Annotated[str, Field(max_length=4096)] | None = None
    min_value: JsonValue | None = None
    max_value: JsonValue | None = None
    inclusive: bool = True
    values: Annotated[list[JsonValue], Field(max_length=1000)] | None = None
    predicate_id: Name | None = None
    query_id: Name | None = None
    row_filter_id: Name | None = None
    num_partitions: Annotated[int, Field(ge=1, le=1000)] | None = None
    storage_level: Literal["MEMORY_ONLY", "MEMORY_AND_DISK", "DISK_ONLY", "MEMORY_ONLY_SER"] = (
        "MEMORY_AND_DISK"
    )
    expression_ids: Annotated[list[Name], Field(min_length=1, max_length=100)] | None = None
    limit: Annotated[int, Field(ge=1, le=1000)] = 100
    offset: Annotated[int, Field(ge=0, le=1_000_000)] = 0
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] | None = None

    @model_validator(mode="after")
    def shape(self):
        if self.command == "shutdown" and self.engine_id != "root":
            raise ValueError("Shutdown applies to the root service lifetime")
        column = {
            "count_nulls",
            "count_distinct",
            "get_stats",
            "get_completeness_ratio",
            "get_distinct_values",
            "get_column_values",
            "get_sample_values",
            "get_quantiles",
            "get_value_counts",
            "aggregate_column",
            "count_matching_regex",
            "count_not_matching_regex",
            "count_in_range",
            "count_outside_range",
            "count_in_set",
            "count_not_in_set",
        }
        required = {
            "stream_id": {"arrow_stream_read", "arrow_stream_close"},
            "aggregation_plan_id": {"aggregate_plan", "aggregate_stream"},
            "validator_ids": {"validate"},
            "column": column,
            "aggregations": {"aggregate"},
            "aggregation": {"aggregate_column"},
            "quantiles": {"get_quantiles"},
            "pattern": {"count_matching_regex", "count_not_matching_regex"},
            "values": {"count_in_set", "count_not_in_set"},
            "predicate_id": {
                "count_matching",
                "count_not_matching",
                "filter_by_condition",
                "native_filter",
            },
            "row_filter_id": {"filter_rows"},
            "num_partitions": {"repartition", "coalesce"},
            "query_id": {"execute_query"},
            "expression_ids": {"filter", "with_columns"},
        }
        for key, commands in required.items():
            if (self.command in commands) != (getattr(self, key) is not None):
                raise ValueError("Field does not match command")
        if (
            self.command in {"count_duplicates", "get_duplicate_values", "select"}
            and not self.columns
        ):
            raise ValueError("columns required")
        allowed = {
            "batch_size": {"aggregate_stream", "arrow_stream_open"},
            "storage_level": {"persist"},
            "columns": {
                "count_duplicates",
                "get_duplicate_values",
                "to_numpy",
                "to_numpy_safe",
                "select",
                "validate",
            },
            "min_value": {"count_in_range", "count_outside_range"},
            "max_value": {"count_in_range", "count_outside_range"},
            "inclusive": {"count_in_range"},
            "offset": {"get_column_values"},
            "seed": {"sample", "to_numpy_safe"},
            "limit": {
                "inspect",
                "get_distinct_values",
                "get_column_values",
                "get_sample_values",
                "get_value_counts",
                "get_duplicate_values",
                "sample",
                "to_polars",
                "to_numpy",
                "to_numpy_safe",
                "execute_query",
                "metrics",
                "aggregate_stream",
                "head",
                "tail",
                "take",
                "compute",
                "to_arrow",
                "arrow_stream_read",
            },
        }
        for key, commands in allowed.items():
            if key in self.model_fields_set and self.command not in commands:
                raise ValueError("Field does not match command")
        if (
            self.command in {"count_in_range", "count_outside_range"}
            and self.min_value is None
            and self.max_value is None
        ):
            raise ValueError("A range requires a bound")
        if self.command == "filter" and len(self.expression_ids) != 1:
            raise ValueError("A filter requires one selected expression")
        return self


class ExecutionProcess(ServiceProcess):
    worker_module = "truthound.mcp.execution_worker"


class ExecutionServices(BoundServiceSession):
    group = "execution_services"
    process_type = ExecutionProcess
    limit_commands = {
        "inspect",
        "aggregate_stream",
        "get_distinct_values",
        "get_column_values",
        "get_sample_values",
        "get_value_counts",
        "get_duplicate_values",
        "sample",
        "to_polars",
        "to_numpy",
        "to_numpy_safe",
        "execute_query",
        "metrics",
        "head",
        "tail",
        "take",
        "compute",
        "to_arrow",
        "arrow_stream_read",
    }
    effects = (
        "installed_extension_execute",
        "operator_selected_native_execution_engine",
        "native_data_query_and_selected_statements",
        "retained_native_child_engines",
        "native_arrow_iterator_read_and_owned_cleanup",
        "operator_configured_engine_cache_checkpoint_writes",
        "sensitive_data_artifact",
    )
