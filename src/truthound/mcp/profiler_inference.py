"""Native type inference and feedback with factory-owned models and contexts."""

from copy import deepcopy

from truthound.mcp.governance_worker import selected
from truthound.mcp.policy import MCPError


def view(column, result):
    return {"column": column, "inferred_type": result.inferred_type.value,
            "confidence": result.confidence,
            "alternatives": [{"type": dtype.value, "confidence": confidence}
                             for dtype, confidence in result.alternatives]}


def execute(owner, args, inputs):
    import polars as pl

    from truthound.profiler.base import DataType

    owner.guard()
    native = selected(owner.runtime.inferrers, args["inferrer_id"])
    if args["command"] == "inference_cache_clear":
        native.clear_cache()
        return {"cleared": True}
    if not inputs or set(inputs) != {"asset_path"}:
        raise MCPError("invalid_handle")
    frame = pl.read_ipc(inputs["asset_path"], memory_map=False)
    if args["command"] == "inference_table":
        results = native.infer_table(frame)
        owner.guard()
        return {"columns": [view(column, result) for column, result in results.items()]}
    column = args["column"]
    if column not in frame.columns:
        raise MCPError("invalid_arguments")
    context = deepcopy(selected(owner.inference_contexts, args["inference_context_id"])) if args.get("inference_context_id") else {}
    if args["command"] == "inference_feedback":
        native.provide_feedback(frame[column], DataType(args["true_type"]), context)
        owner.guard()
        return {"learning_enabled": native.config.enable_learning,
                "pending_feedback": len(native._feedback_buffer)}
    result = native.infer(frame[column], context)
    owner.guard()
    return view(column, result)
