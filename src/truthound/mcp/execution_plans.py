"""Factory-selected Core plans and validators; no caller-authored code or SQL."""

import json

from truthound.mcp.checkpoint_worker import binding_guard
from truthound.mcp.encoding import encode
from truthound.mcp.governance_worker import selected
from truthound.mcp.policy import MCPError


def execute_plan(owner, engine, args, budget):
    from truthound.execution.distributed.aggregations import DistributedAggregator
    from truthound.execution.distributed.base import BaseDistributedEngine
    from truthound.execution.distributed.dask_engine import DaskExecutionEngine
    from truthound.execution.distributed.ray_engine import RayExecutionEngine
    from truthound.execution.distributed.spark_engine import SparkExecutionEngine
    from truthound.execution.distributed.validator_adapter import DistributedValidatorAdapter
    from truthound.execution.polars_engine import PolarsExecutionEngine

    guard = binding_guard(owner.request)
    guard()
    if args["command"] == "validate":
        if not isinstance(engine, BaseDistributedEngine):
            raise MCPError("unsupported_operation")
        validators = [selected(owner.runtime.validators, key) for key in args["validator_ids"]]
        adapter = DistributedValidatorAdapter(engine, owner.runtime.adapter_config)
        adapter.set_execution_guard(guard)
        # Keep each caller-visible alias; the native name is part of the issue
        # contract. Raw failure values remain governed by the approved binding.
        value = [issue.to_dict() for issue in adapter.validate_all(validators, args.get("columns"))]
        guard()
        if len(json.dumps(encode(value), allow_nan=False).encode()) > budget:
            raise MCPError("resource_exhausted")
        return {"issues": value}
    if isinstance(engine, DaskExecutionEngine):
        data = engine.dask_dataframe
    elif isinstance(engine, RayExecutionEngine):
        data = engine.dataset
    elif isinstance(engine, SparkExecutionEngine):
        data = engine.spark_dataframe
    elif isinstance(engine, PolarsExecutionEngine):
        data = engine.lazyframe
    else:
        raise MCPError("unsupported_operation")
    plan = selected(owner.runtime.plans, args["aggregation_plan_id"])
    aggregator = DistributedAggregator(
        data,
        dask_scheduler=engine._scheduler() if isinstance(engine, DaskExecutionEngine) else None,
    )
    aggregator.set_execution_guard(guard)
    if args["command"] == "aggregate_plan":
        value = aggregator.execute(plan)
        guard()
        return {"value": value}
    stream = aggregator.execute_streaming(plan, batch_size=args.get("batch_size", 10_000))
    snapshots = []
    size = 0
    complete = True
    try:
        for snapshot in stream:
            guard()
            if len(snapshots) >= args.get("limit", 100):
                complete = False
                break
            encoded_size = len(json.dumps(encode(snapshot), allow_nan=False).encode())
            if size + encoded_size > budget:
                raise MCPError("resource_exhausted")
            size += encoded_size
            snapshots.append(snapshot)
    finally:
        stream.close()
    return {"snapshots": snapshots, "complete": complete}
