"""Closed distributed execution commands, delegating to native public methods."""

from truthound.mcp.governance_worker import selected
from truthound.mcp.policy import MCPError


def execute_extension(owner, engine, args):
    from truthound.execution.distributed.base import BaseDistributedEngine
    from truthound.execution.distributed.dask_engine import DaskExecutionEngine
    from truthound.execution.distributed.ray_engine import RayExecutionEngine
    from truthound.execution.distributed.spark_engine import SparkExecutionEngine

    command, limit = args["command"], args.get("limit", 100)
    if not isinstance(engine, BaseDistributedEngine):
        raise MCPError("unsupported_operation")
    if command == "partitions":
        return {"backend": engine.backend_type.value, "num_partitions": engine.num_partitions}
    if command == "repartition":
        return owner.retain(engine.repartition(args["num_partitions"]))
    if command == "coalesce":
        return owner.retain(engine.coalesce(args["num_partitions"]))
    if command == "clear_metrics":
        engine.clear_metrics()
        return {"cleared": True}
    if command == "metrics":
        return {
            "metrics": [
                {
                    "operation": row.operation,
                    "duration_ms": row.duration_ms,
                    "partitions_processed": row.partitions_processed,
                    "rows_processed": row.rows_processed,
                    "bytes_processed": row.bytes_processed,
                    "error_count": len(row.errors),
                    "warning_count": len(row.warnings),
                    "success": row.success,
                }
                for row in engine.get_metrics()[-limit:]
            ]
        }
    if command == "persist":
        if isinstance(engine, SparkExecutionEngine):
            return owner.retain(engine.persist(args.get("storage_level", "MEMORY_AND_DISK")))
        if isinstance(engine, DaskExecutionEngine) and "storage_level" not in args:
            return owner.retain(engine.persist())
    if command == "unpersist" and isinstance(engine, SparkExecutionEngine):
        return owner.retain(engine.unpersist())
    if command == "checkpoint" and isinstance(
        engine, (DaskExecutionEngine, RayExecutionEngine, SparkExecutionEngine)
    ):
        return owner.retain(engine.checkpoint())
    if command == "materialize" and isinstance(engine, RayExecutionEngine):
        return owner.retain(engine.materialize())
    if command == "native_filter" and isinstance(engine, DaskExecutionEngine):
        return owner.retain(engine.filter(selected(owner.runtime.predicates, args["predicate_id"])))
    if command == "filter_rows" and isinstance(engine, RayExecutionEngine):
        return owner.retain(
            engine.filter(selected(owner.runtime.row_filters, args["row_filter_id"]))
        )
    if command == "head" and isinstance(engine, DaskExecutionEngine):
        return {"rows": engine.head(limit).to_dict(orient="records")}
    if command == "tail" and isinstance(engine, DaskExecutionEngine):
        return {"rows": engine.tail(limit).to_dict(orient="records")}
    if command == "take" and isinstance(engine, RayExecutionEngine):
        return {"rows": engine.take(limit)}
    if command in {"compute", "to_arrow"}:
        if engine.count_rows() > limit:
            raise MCPError("resource_exhausted")
        if command == "compute" and isinstance(engine, DaskExecutionEngine):
            return {"rows": engine.compute().to_dict(orient="records")}
        if command == "to_arrow":
            return {"rows": engine.to_arrow_table().to_pylist()}
    raise MCPError("unsupported_operation")
