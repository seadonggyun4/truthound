"""Delegate computations to Core engines; never evaluate caller code or SQL."""

from contextlib import ExitStack
from uuid import uuid4

from truthound.mcp.execution_services import ExecutionRuntime
from truthound.mcp.governance_worker import selected
from truthound.mcp.policy import MCPError
from truthound.mcp.tables import decode


class ExecutionOwner:
    def __init__(self, runtime, request):
        from truthound.execution.base import BaseExecutionEngine

        if (
            request["native_group"] != "execution_services"
            or not isinstance(runtime, ExecutionRuntime)
            or not isinstance(runtime.engine, BaseExecutionEngine)
        ):
            raise MCPError("invalid_binding")
        if any(
            not isinstance(value, str)
            for value in (*runtime.predicates.values(), *runtime.queries.values())
        ):
            raise MCPError("invalid_binding")
        from truthound.execution.distributed.aggregations import AggregationPlan
        from truthound.execution.distributed.validator_adapter import AdapterConfig
        from truthound.validators.base import Validator

        if (
            any(not isinstance(value, AggregationPlan) for value in runtime.plans.values())
            or any(not isinstance(value, Validator) for value in runtime.validators.values())
            or (
                runtime.adapter_config is not None
                and not isinstance(runtime.adapter_config, AdapterConfig)
            )
        ):
            raise MCPError("invalid_binding")
        self.request = request
        self.runtime = runtime
        self.stack = ExitStack()
        self.engines = {"root": self.stack.enter_context(runtime.engine)}
        self.children = {}
        self.arrow_streams = {}
        self.shutdown_requested = False

    def retain(self, engine):
        from truthound.execution.base import BaseExecutionEngine

        if not isinstance(engine, BaseExecutionEngine):
            raise MCPError("unsupported_value")
        for identity, current in self.engines.items():
            if current is engine:
                return {"engine_id": identity, "engine_type": engine.engine_type}
        identity = "engine_" + uuid4().hex
        stack = ExitStack()
        self.engines[identity] = stack.enter_context(engine)
        self.children[identity] = stack
        return {"engine_id": identity, "engine_type": engine.engine_type}

    async def execute(self, args, budget):
        from truthound.execution._protocols import AggregationType
        from truthound.execution.distributed.dask_engine import DaskExecutionEngine
        from truthound.execution.distributed.ray_engine import RayExecutionEngine
        from truthound.execution.distributed.spark_engine import SparkExecutionEngine
        from truthound.execution.polars_engine import PolarsExecutionEngine
        from truthound.execution.sql_engine import SQLExecutionEngine

        runtime, command = self.runtime, args["command"]
        identity = args.get("engine_id", "root")
        engine = selected(self.engines, identity)
        if command == "shutdown":
            if identity != "root":
                raise MCPError("invalid_arguments")
            self.shutdown_requested = True
            return {"shutdown": True}
        limit = args.get("limit", 100)
        if command in {"arrow_stream_open", "arrow_stream_read", "arrow_stream_close"}:
            from truthound.mcp.execution_arrow_streams import execute

            return execute(self, engine, args, budget)
        if command == "inspect":
            return {
                "engine_id": identity,
                "engine_type": engine.engine_type,
                "supports_sql_pushdown": engine.supports_sql_pushdown,
                "columns": engine.get_columns()[:limit],
                "config": {
                    key: getattr(engine.config, key)
                    for key in (
                        "max_rows_for_numpy",
                        "max_rows_for_polars",
                        "sample_size_for_stats",
                        "cache_results",
                        "parallel",
                        "num_partitions",
                        "partition_strategy",
                        "partition_columns",
                        "execution_mode",
                        "arrow_batch_size",
                        "collect_metrics",
                        "checkpoint_enabled",
                        "timeout_seconds",
                        "retry_count",
                        "memory_fraction",
                    )
                    if hasattr(engine.config, key)
                },
                "predicates": sorted(runtime.predicates)[:limit],
                "queries": sorted(runtime.queries)[:limit],
                "expressions": sorted(runtime.expressions)[:limit],
                "row_filters": sorted(runtime.row_filters)[:limit],
                "engines": sorted(self.engines)[:limit],
                "aggregation_plans": sorted(runtime.plans)[:limit],
                "validators": sorted(runtime.validators)[:limit],
                "arrow_streams": [
                    key
                    for key, cursor in self.arrow_streams.items()
                    if cursor.engine_id == identity
                ][:limit],
            }
        column, columns = args.get("column"), args.get("columns")
        supplied = (
            ([column] if column is not None else [])
            + (columns or [])
            + list(args.get("aggregations", {}))
        )
        if supplied and any(value not in engine.get_columns() for value in supplied):
            raise MCPError("invalid_arguments")
        if (
            command
            in {
                "sample",
                "filter_by_condition",
                "select",
                "filter",
                "with_columns",
                "repartition",
                "coalesce",
                "checkpoint",
                "native_filter",
                "filter_rows",
            }
            and len(self.children) >= 16
        ):
            raise MCPError("resource_exhausted")
        if command in {"aggregate_plan", "aggregate_stream", "validate"}:
            from truthound.mcp.execution_plans import execute_plan

            return execute_plan(self, engine, args, budget)
        if command in {
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
        }:
            from truthound.mcp.execution_extensions import execute_extension

            return execute_extension(self, engine, args)
        if command == "release":
            if identity == "root":
                raise MCPError("invalid_arguments")
            for stream_id, cursor in list(self.arrow_streams.items()):
                if cursor.engine_id == identity:
                    cursor.close()
                    del self.arrow_streams[stream_id]
            self.children.pop(identity).close()
            del self.engines[identity]
            return {"released": True}
        if command == "clear_cache":
            engine.clear_cache()
            return {"cleared": True}
        if command == "sample":
            return self.retain(engine.sample(n=limit, seed=args.get("seed")))
        if command in {"count_matching", "count_not_matching", "filter_by_condition"}:
            predicate = selected(runtime.predicates, args["predicate_id"])
            if command == "filter_by_condition":
                return self.retain(engine.filter_by_condition(predicate))
            value = (
                engine.count_matching(predicate)
                if command == "count_matching"
                else engine.count_not_matching(predicate)
            )
        elif command in {"select", "filter", "with_columns"}:
            if command == "select":
                if isinstance(engine, DaskExecutionEngine):
                    return self.retain(engine.select(columns))
                if isinstance(engine, RayExecutionEngine):
                    return self.retain(engine.select_columns(columns))
            if not isinstance(engine, PolarsExecutionEngine):
                raise MCPError("unsupported_operation")
            expressions = [
                selected(runtime.expressions, key) for key in args.get("expression_ids", [])
            ]
            if command == "select":
                return self.retain(engine.select(*columns))
            if command == "filter":
                return self.retain(engine.filter(expressions[0]))
            return self.retain(engine.with_columns(*expressions))
        elif command == "execute_query":
            if isinstance(engine, SparkExecutionEngine):
                if len(self.children) >= 16:
                    raise MCPError("resource_exhausted")
                return self.retain(engine.sql(selected(runtime.queries, args["query_id"])))
            if not isinstance(engine, SQLExecutionEngine):
                raise MCPError("unsupported_operation")
            # The complete statement is pinned by the operator. No string interpolation.
            rows = engine.execute_sql(selected(runtime.queries, args["query_id"]))
            value = {"rows": rows[:limit], "truncated": len(rows) > limit}
        elif command == "to_polars":
            value = engine.to_polars_lazyframe().limit(limit).collect().to_dicts()
        elif command in {"to_numpy", "to_numpy_safe"}:
            if command == "to_numpy":
                if engine.count_rows() > limit:
                    raise MCPError("resource_exhausted")
                value = engine.to_numpy(columns).tolist()
            else:
                value = engine.to_numpy_safe(
                    columns, max_rows=limit, seed=args.get("seed", 42)
                ).tolist()
        elif command == "aggregate":
            value = engine.aggregate(
                {key: AggregationType(value) for key, value in args["aggregations"].items()}
            )
        elif command == "aggregate_column":
            value = engine.aggregate_column(column, AggregationType(args["aggregation"]))
        elif command == "get_quantiles":
            value = engine.get_quantiles(column, args["quantiles"])
        elif command == "get_value_counts":
            # Preserve non-string native keys without JSON key coercion.
            value = [
                {"value": key, "count": count}
                for key, count in engine.get_value_counts(column, limit).items()
            ]
        elif command in {"count_in_set", "count_not_in_set"}:
            values = {decode(value) for value in args["values"]}
            value = (
                engine.count_in_set(column, values)
                if command == "count_in_set"
                else engine.count_not_in_set(column, values)
            )
        elif command in {"count_in_range", "count_outside_range"}:
            bounds = [decode(args.get("min_value")), decode(args.get("max_value"))]
            value = (
                engine.count_in_range(column, *bounds, inclusive=args.get("inclusive", True))
                if command == "count_in_range"
                else engine.count_outside_range(column, *bounds)
            )
        else:
            calls = {
                "count_rows": lambda: engine.count_rows(),
                "count_nulls_all": lambda: engine.count_nulls_all(),
                "count_nulls": lambda: engine.count_nulls(column),
                "count_distinct": lambda: engine.count_distinct(column),
                "get_stats": lambda: engine.get_stats(column),
                "get_completeness_ratio": lambda: engine.get_completeness_ratio(column),
                "get_distinct_values": lambda: engine.get_distinct_values(column, limit),
                "get_column_values": lambda: engine.get_column_values(
                    column, limit, args.get("offset", 0)
                ),
                "get_sample_values": lambda: engine.get_sample_values(column, n=limit),
                "count_matching_regex": lambda: engine.count_matching_regex(
                    column, args["pattern"]
                ),
                "count_not_matching_regex": lambda: engine.count_not_matching_regex(
                    column, args["pattern"]
                ),
                "count_duplicates": lambda: engine.count_duplicates(columns),
                "get_duplicate_values": lambda: engine.get_duplicate_values(columns, limit),
            }
            if command not in calls:
                raise MCPError("invalid_arguments")
            value = calls[command]()
        return {"value": value}

    async def close(self):
        errors = []
        for cursor in self.arrow_streams.values():
            try:
                cursor.close()
            except BaseException as error:
                errors.append(error)
        self.arrow_streams.clear()
        try:
            for stack in reversed(list(self.children.values())):
                try:
                    stack.close()
                except BaseException as error:
                    errors.append(error)
        finally:
            self.children.clear()
            self.engines.clear()
            try:
                self.stack.close()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise BaseExceptionGroup("Native engine cleanup failed", errors)


if __name__ == "__main__":
    from truthound.mcp.broker_worker import main

    main(ExecutionOwner)
