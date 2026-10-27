"""Native sampling, bounded retained snapshots and downstream native profiling."""

from uuid import uuid4

from truthound.mcp.governance_worker import selected
from truthound.mcp.policy import MCPError


def execute(owner, args, inputs, budget):
    import polars as pl

    owner.guard()
    command = args["command"]
    if command == "sampling_sample":
        if not inputs or set(inputs) != {"asset_path"}:
            raise MCPError("invalid_handle")
        if len(owner.samples) >= 16:
            raise MCPError("resource_exhausted")
        sampler = selected(owner.runtime.samplers, args["sampler_id"])
        result = sampler.sample(pl.scan_ipc(inputs["asset_path"], memory_map=False))
        frame = result.data.collect()
        owner.guard()
        if frame.height != result.metrics.sample_size:
            raise ValueError("Native sample metrics do not match the observed sample")
        from truthound.profiler.parallel_sampling import SketchSamplingResult
        sketches = None
        size = frame.estimated_size() + 256
        if isinstance(result, SketchSamplingResult):
            sketches = (result.cardinality_sketches, result.frequency_sketches)
            size += sum(sketch.memory_bytes() for group in sketches for sketch in group.values())
        if size + sum(owner.sizes.values()) > budget:
            raise MCPError("resource_exhausted")
        identity = "sample_" + uuid4().hex
        if sketches is not None:
            owner.sample_sketches[identity] = sketches
        owner.samples[identity] = frame
        owner.sizes[identity] = size
        return {"sample_id": identity, "metrics": result.metrics.to_dict(),
                "is_sampled": result.is_sampled, "columns": frame.columns}
    identity = args["sample_id"]
    frame = selected(owner.samples, identity)
    if command == "sample_release":
        owner.sample_sketches.pop(identity, None)
        del owner.samples[identity]
        del owner.sizes[identity]
        return {"released": True}
    if command == "sample_sketch_query":
        sketches = selected(owner.sample_sketches, identity)
        cardinality = selected(sketches[0], args["column"])
        frequency = selected(sketches[1], args["column"])
        return {"sample_id": identity, "column": args["column"],
                "estimated_distinct": cardinality.estimate(),
                "estimated_frequency": frequency.estimate_frequency(args["sketch_value"]),
                "sample_rows": frame.height}
    if command == "sample_profile":
        native = selected(owner.runtime.profilers, args["profiler_id"])
        value = native.profile(frame.lazy())
        owner.guard()
        return owner.profile_view(owner.retain(value, "profile", budget), value)
    columns = args.get("columns")
    if columns is not None:
        if any(column not in frame.columns for column in columns):
            raise MCPError("invalid_arguments")
        frame = frame.select(columns)
    limit, offset = args.get("limit", 100), args.get("offset", 0)
    page = frame.slice(offset, limit)
    return {"sample_id": identity, "columns": page.columns, "rows": page.to_dicts(),
            "returned_rows": page.height, "truncated": offset + page.height < frame.height}
