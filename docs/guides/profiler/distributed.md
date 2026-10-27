# Distributed profiling

The `truthound.profiler.distributed` package exports `DistributedProfiler`,
`DistributedProfileConfig`, `BackendType`, `PartitionStrategy`, and the native
`LocalBackend`, `DaskBackend`, `RayBackend`, and `SparkBackend` classes.
`DistributedBackend` is the backend interface, not an enum.

```python
import polars as pl
from truthound.profiler.distributed import DistributedProfiler, DaskConfig

with DistributedProfiler.create(
    backend="dask",
    backend_config=DaskConfig(scheduler="threads"),
    num_partitions=2,
) as profiler:
    profile = profiler.profile(pl.DataFrame({"x": [1.0, None, 3.0]}), name="measurements")
    print(profile.row_count, profile.columns[0].null_count)
```

Pass configuration through `backend_config`, and high-level settings through
`DistributedProfileConfig` or `create` keyword arguments. Backend names are
`local`, `dask`, `ray`, `spark`, or `auto`. Auto selection prefers installed Dask,
then Ray, Spark, and local; choose explicitly for predictable resource ownership.
Install the corresponding `dask`, `ray`, or `spark` package extra and the native
runtime prerequisites. A successful import is not a cluster readiness check.

## Data and global statistics

Input is a Polars DataFrame/LazyFrame or a backend-supported file path.
In-memory Polars inputs are materialized before distribution; this API does not
promise bounded driver memory for arbitrarily large inputs. Local supports CSV
and Parquet; Dask supports CSV and Parquet; Ray and Spark path loading uses
Parquet. The source must remain stable across partition and global reductions.

Local column partitions count table rows once. Row partitions never report more
rows merely because the requested partition count exceeds the input size.
Global distinct counts include a null category. Numeric mean and sample standard
deviation exclude nulls; string length extrema and mean use all non-null rows.
Local uses native Polars global reductions, Dask uses native global expressions,
Ray performs global Polars reductions in an owned remote task, and Spark uses
native global SQL aggregations. Distinct counts and string lengths are not
copied from an arbitrary first partition. These operations may need additional
passes and backend shuffle state.

Partition failures, missing results, or mismatched identities raise instead of
producing a successful partial `TableProfile`. `timeout_seconds` rejects late
results after native work returns; it is not hard query preemption. Explicit
cluster cancellation and transport failures remain native backend behavior.

## Runtime ownership

Use the context manager to initialize and shut down owned resources. Dask uses
its selected scheduler per computation without changing the global Dask
scheduler. Shutdown attempts both its client and owned cluster and reports
cleanup errors together. A body failure and cleanup failure are both preserved.
Ray shuts down the Ray runtime only when this backend started it. Spark stops
its context only when this backend created it, preserving an already active
context. Driver subprocess isolation is still recommended for independent JVM
or Ray lifetimes.

## Sampling, patterns, callbacks, and MCP

`sample_size` limits the input to a native leading subset before profiling; the
returned row count describes that subset. Spark's distributed row order is not
stable unless the source supplies ordering. `include_patterns=True` runs the
Core `PatternAnalyzer` on at most its default 1,000 non-null values per string
column. False omits pattern sampling. Pattern matches include native sample
values and must be treated as sensitive profile content.

`map_partitions(callback, partitions, data)` invokes the operator's serializable
callback with `PartitionInfo` and an already selected Polars DataFrame. It must
return `WorkerResult` with the same partition ID and observed row count. Native
profiling uses this same callback route. Exceptions remain failures; the native
global-statistics reducer is not a general-purpose reducer for custom metadata.
Calls on a single profiler serialize access to its active backend input.

An approved profiler factory can register an initialized `DistributedProfiler`
in `ProfilerRuntime.profilers`. The existing MCP `profile`, profile save/restore,
retained sample profiling, and release paths use the native object. The factory
owns backend initialization and cleanup; wire requests select only the alias
and pinned asset, not cluster addresses, credentials or callback code. Native
execution guards check source authority between profiling phases.

## Partition and resource policies

`ROW_BASED` partitions rows; `COLUMN_BASED` distributes disjoint column groups;
`HYBRID` forms row/column groups; `HASH` assigns equal complete rows to the same
partition. Hybrid partition counts are a planning hint and may differ from the
requested count. Native hashes and row order differ across engines and versions;
this is not a cross-engine byte-identical sampling promise. Local and Ray use a
shared Polars partition planner; Dask and Spark use native shuffle/column plans.
All policies preserve cells without adding or removing records in the profile.

`BackendConfig.parallelism` selects the default partition count when explicit
`num_partitions` is absent. `memory_per_worker` accepts positive byte sizes such
as `512m` or `2g`; each materialized callback input is checked before invocation.
This is not a heap reservation: input loading, sorting, shuffle and global
reductions may allocate more. The backend `timeout_seconds` bounds each callback
result cooperatively; high-level timeout bounds the complete profile result.
Zero removes that deadline. Running native work is drained, not forcibly killed.

Core replays a callback only after `RetryableExecutionError` and at most
`retry_count` times. Ordinary failures and late results do not request Core
replay. Ray callback tasks disable implicit task retries; Spark cluster-level
retry behavior remains governed by the operator's native Spark configuration.
Do not put uncertain external side effects in profiling callbacks.

## Durable native checkpoints

Set `checkpoint_enabled=True` and a local `checkpoint_dir` in the selected
backend config. Native Parquet serialization uses the shared execution-engine
checkpoint publisher with SHA-256 inventory, file/directory synchronization and
atomic directory publication. `backend.checkpoint_path` identifies the committed
snapshot. Failed staging directories remain available for operator inspection.

`backend.restore_checkpoint(location, num_partitions=..., strategy=...)` validates
backend identity and the complete inventory before loading native Parquet, then
returns partition handles for `map_partitions` and `aggregate_results`. A fresh
backend can reopen a checkpoint; corruption is rejected. If checkpointing is
still enabled on that backend, distribution publishes a new snapshot. Checkpoint
paths contain sensitive source data and remain factory-owned; they are not
included in MCP profile replies or released automatically with a session.


## Explicit Arrow conversion policy

The advanced `truthound.execution.distributed.arrow_bridge.ArrowBridge` helper captures a copy of `ArrowBridgeConfig`. Generic and backend-specific conversions apply the same index, null, temporal and byte-budget settings. Pandas/Dask index retention uses `preserve_index`; it is not a new row identity for other backends. Positive integer `batch_size`, explicit chunk sizes and `max_memory_bytes` are required. Dask converts one partition before splitting into bounded row chunks; Spark collects through the selected strategy before splitting; Ray consumes native Arrow batches. These paths do not promise a hard process-memory ceiling. The byte budget checks Arrow `nbytes`, after native materialization and after policy conversion; complete-result conversion also bounds the combined result.

`null_handling="mask"` preserves top-level nulls, `"error"` rejects them, and `"sentinel"` requires an explicit, non-null, type-compatible `sentinel_values` entry for each nullable column. Temporal coercion applies to top-level timestamps/durations/times as microseconds and date64 as date32. Casts are safe: sub-microsecond values fail rather than truncate; choose `coerce_temporal=False` to retain original precision. Nested temporal/null fields retain their native representation. Field and schema metadata are retained. `collect=False` selects a LazyFrame return type after conversion checks; it does not defer source materialization.

Spark conversion no longer changes the caller's Arrow session setting or retries arbitrary native failures through another conversion path. An explicitly selected native strategy fails if the collector is unavailable; automatic selection can use Pandas when that capability is absent. Polars-to-Spark uses one Pandas-backed creation call after policy validation. External clusters and remote source services require their own runtime verification.
