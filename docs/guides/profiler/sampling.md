# Sampling

`truthound.profiler.sampling` selects rows from a Polars `LazyFrame`. Use a stable
source for the count and sampling passes. Results contain a `data` LazyFrame,
`metrics` and `is_sampled`; collecting the result evaluates any remaining plan.

## SamplingConfig

```python
import polars as pl
from truthound.profiler.sampling import Sampler, SamplingConfig, SamplingMethod

frame = pl.DataFrame({"x": range(1000)}).lazy()
sampler = Sampler(SamplingConfig(
    strategy=SamplingMethod.RANDOM, max_rows=100, min_sample_size=100,
    seed=7, confidence_level=0.95, margin_of_error=0.05,
    max_memory_mb=16, enable_caching=True,
    fallback_strategy=SamplingMethod.HEAD,
))
result = sampler.sample(frame)
print(result.data.collect().height, result.metrics.sample_size)
```

`max_rows=0` removes the row cap; the statistical sample-size calculation still
applies. `min_sample_size` is bounded by population and any positive row cap.
`max_memory_mb=0` disables the output memory check. Positive values bound the
materialized sample's estimated bytes before `Sampler.sample()` returns; this is
not a native heap reservation or protection against every source allocation.
`enable_caching` controls the adaptive strategy's bounded decision cache, not a
data/result cache. Each sampler copies its default configuration.

`SamplingConfig.for_accuracy("low" | "medium" | "high" | "maximum")`,
`for_speed()` and `for_memory(max_memory_mb=100)` provide presets.
`calculate_sample_size(population_size, confidence_level=0.95,
margin_of_error=0.05, min_sample_size=1)` uses Cochran's formula with finite
population correction. Its assumptions do not guarantee accuracy for biased or
non-independent observations.

## Strategies

| Method | Behavior |
| --- | --- |
| `NONE` | Explicitly returns all rows, ignoring the sampling row cap. |
| `HEAD` | First calculated number of rows; ordering can bias the sample. |
| `RANDOM` | Seeded row-index hash ordering selects exactly the calculated size. |
| `SYSTEMATIC` | Local-RNG starting offset and bounded interval selection. |
| `HASH` | Stable hash ordering of the configured column, or row index by default; equal hashes use input order. |
| `STRATIFIED` | Proportional per-group allocation with largest-remainder rounding, then seeded selection within each group. |
| `RESERVOIR` | Algorithm R over native chunks retains the selected rows plus one input chunk. |
| `ADAPTIVE` | Chooses a strategy from dataset-size thresholds; a positive row cap still applies to small datasets. |

Random and hash ordering may require native sorting. Stratification collects group
counts and joins quotas; tiny strata can receive zero rows when the budget cannot
represent them. Reservoir retention is O(sample size), not O(1). The framework
counts rows before selection; reservoir's selection itself makes one pass. Current
Polars uses `collect_batches`; older Polars without that iterator materializes the
query result before slicing, so it cannot provide the same streaming memory bound.
Temporary columns never replace user columns. Fixed seeds leave process-global
random state unchanged.

Stratification and hash columns belong to strategy objects, not `SamplingConfig`:

```python
from truthound.profiler.sampling import (
    SamplingStrategyRegistry, StratifiedSamplingStrategy,
)
registry = SamplingStrategyRegistry()
registry.register(StratifiedSamplingStrategy(stratify_column="category"))
sampler = Sampler(SamplingConfig(strategy=SamplingMethod.STRATIFIED, seed=7), registry)
```

A registry also accepts `HashSamplingStrategy(hash_column="id")` or trusted custom
`SamplingStrategy` objects. Other strategy failures invoke the configured fallback;
`MemoryError`, `TimeoutError` and `PermissionError` propagate without another read.
The fallback does not make external callback effects safe to replay.

## SamplingMetrics

Metrics expose `original_size`, `sample_size`, `sampling_ratio`, `confidence_level`,
`margin_of_error`, `strategy_used`, `sampling_time_ms` and
`memory_saved_estimate_mb`. `is_full_scan` and `reduction_factor` are derived
properties. Timing for lazy strategies measures plan construction; reservoir
selection and an enabled output check also execute work. Memory savings are an
estimate, not measured process peak memory.

`sample_data(lf, max_rows=100_000, strategy="adaptive")` is a convenience entry.
`Sampler.sample_column(lf, column, config=None)` selects one column before sampling.

## Approved MCP Sampling

A private `ProfilerRuntime.samplers` map binds native `Sampler` objects, including
selected registry/configuration. `sampling_sample(sampler_id, asset_id)` samples a
pinned asset once and retains the concrete snapshot, checking native row metrics
and the owner memory budget. It returns `sample_id` and metrics.
`sample_get(sample_id, columns?, offset=0, limit=100)` pages rows;
`sample_profile(sample_id, profiler_id)` passes that snapshot to a bound native
profiler; `sample_release(sample_id)` releases it. Samples are owner-local and
capped at 16; they do not survive worker restart. These operations require exact
approval and sensitive-result consent. Wire requests cannot supply custom code,
source paths or registry imports.

## Next Steps

- [Quality Scoring](quality-scoring.md)
- [ML Inference](ml-inference.md)

## Parallel block execution

`ParallelBlockSampler(ParallelSamplingConfig(...)).sample(frame, SamplingConfig(...))`
collects selected rows inside owned workers before returning a retained snapshot.
Block quotas add up to the requested sample size; a short last block receives its
proportional quota. Seed `0` is deterministic, result order follows block order,
and input columns are preserved. Use a stable input throughout count and reads.

`execution_mode=ExecutionMode.PROCESS` uses spawned processes with a serializable
Polars plan; guard the application entry point with `if __name__ == "__main__":`.
`THREAD` and `ADAPTIVE` use threads. Submitted work is bounded by
`max_workers * (1 + prefetch_blocks)`. With `enable_work_stealing=True`, completed
work frees capacity immediately; false waits for each submitted window. This
uses a shared executor queue, so `blocks_stolen` is zero rather than an estimated
steal count. Priority scheduling sorts block priority before submission; equal
priorities preserve block order. Process execution does not inherit custom owner
method overrides.

Block failures propagate without sequential replay or partial-success results.
Deadlines reject late results and drain running work; they cannot interrupt a
running Polars query. Retained sample bytes, including full scans, must fit
`memory_budget.available_memory_mb`. This check happens after allocation and is
not a process-heap reservation. Monitoring may request garbage collection when
RSS exceeds the budget; query sorting and process serialization can use more
memory than the retained sample. Calls on one sampler are serialized.

An approved profiler factory can place
`NativeSamplingBinding(strategy, SamplingConfig(...))` from
`truthound.mcp.profiler_services` in `ProfilerRuntime.samplers`. This invokes the
selected native strategy directly without fallback. `sampling_sample` retains
its concrete result for `sample_get`, `sample_profile`, and `sample_release`;
strategy, execution mode, and paths remain factory configuration.

## Enterprise strategies and sample sketches

`BlockSamplingStrategy` reuses the block executor with one worker and exposes
`BlockSamplingMetrics`. `MultiStageSamplingStrategy` performs exact ranked
selection at each reduction stage (`num_stages >= 1`).
`ColumnAwareSamplingStrategy` adjusts its requested size for column complexity
without exceeding a positive `SamplingConfig.max_rows`. All three materialize
the returned snapshot and check output memory and late results.

`ProgressiveSamplingStrategy` evaluates increasing nested samples with the same
seed. Between stages it compares each column's null fraction, numeric mean and
population standard deviation, or nonnumeric unique fraction. The maximum
`abs(new-old) / max(1, abs(old))` controls early stopping at
`convergence_threshold`; `enable_progressive=False` completes every distinct
stage size. `ProgressiveSamplingMetrics` exposes actual `stages_completed`,
`converged`, and `convergence_delta`. Full scans retain ordinary `SamplingMetrics`.
This convergence heuristic is not a statistical accuracy guarantee.

`SketchBasedSampler` returns `SketchSamplingResult`, including populated native
`cardinality_sketches[column]` (HyperLogLog) and
`frequency_sketches[column]` (CountMinSketch). These describe the retained sample,
not the original population. Each supports its native estimate/query API.
The MCP sampling port retains these native sketches under the sample ID.
`sample_sketch_query(sample_id, column, sketch_value)` returns estimated distinct
and frequency counts for an explicit JSON scalar (including null), without
echoing the query value. Release removes rows and sketches together; owner
restart expires both. Admission accounts for native sketch memory estimates
in addition to retained rows. Sketch construction
also checks the elapsed deadline; the retained-row budget does not include
the Python sketch tables or temporary query allocations.
