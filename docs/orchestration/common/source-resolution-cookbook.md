---
title: Source Resolution Cookbook
---

# Source Resolution Cookbook

`resolve_data_source(...)` is one of the most important shared-runtime entry
points because every adapter uses it to decide whether an input is a file, SQL
statement, dataframe, stream, callable, or URI-like reference.

## Who This Is For

- platform teams normalizing data inputs across multiple hosts
- users moving the same quality intent between Python adapters and dbt
- operators diagnosing source-shape mismatches

## When To Use It

Use this page when you need a practical mapping from “what I handed the adapter”
to “what the shared runtime thinks this source is”.

## Prerequisites

- familiarity with [Source Resolution](source-resolution.md)
- knowledge of whether your workflow starts from a file, SQL relation,
  dataframe, or stream

## Minimal Quickstart

```python
from common.runtime import resolve_data_source

local_source = resolve_data_source(data_path="data/users.parquet")
sql_source = resolve_data_source(sql="SELECT * FROM users")
```

The first resolves as a local-path style source. The second resolves as a SQL
source that requires a host-native connection path.

## Cookbook

### Local Paths

Use local or mounted paths for the fastest onboarding path:

```python
source = resolve_data_source(data_path="/opt/data/users.parquet")
```

Best for:

- Airflow operators reading mounted files
- local Prefect and Mage development
- Kestra script tasks with staged files

### SQL Statements

Use SQL when the host already has a native connection story:

```python
source = resolve_data_source(sql="SELECT * FROM users WHERE ds = CURRENT_DATE")
```

Best for:

- Airflow + `connection_id`
- warehouse-backed quality tasks in host adapters

Not ideal for:

- local zero-config runs with no connection surface

### Dataframes

If the host already loaded data into memory, the shared runtime accepts the
frame directly:

```python
result = data_quality.check(dataframe, rules=[{"column": "id", "type": "not_null"}])
```

Best for:

- Dagster assets and ops
- Prefect tasks and flows
- Mage transformers

### Streams

Streaming and bounded-memory checks use the orchestration streaming surfaces:

```python
from common.orchestration import StreamRequest, run_stream_check

request = StreamRequest(stream=my_iterable)
```

Best for:

- Airflow streaming operators
- Kestra stream scripts
- shared runtime streaming envelopes

### Warehouse Relations In dbt

The dbt adapter does not resolve local files or in-memory frames. The runtime
boundary is always the compiled relation:

- `ref("model_name")`
- `source("pkg", "table")`
- adapter-dispatched SQL relation names

## Production Pattern

- Keep source shape explicit in runbooks and examples.
- Use local paths only when the runtime environment guarantees those paths.
- Use SQL only when credentials, retry policy, and timeout semantics are
  already host-native.
- Prefer dataframes in Dagster, Prefect, and Mage when upstream code already
  materialized the dataset.
- Use streaming APIs intentionally; do not fake streaming by chunking
  unbounded iterators without checkpoint semantics.

## Failure Modes And Troubleshooting

| Symptom | Likely Cause | Read Next |
|---------|--------------|-----------|
| SQL preflight says a connection is required | zero-config does not cover SQL execution | [Preflight and Compatibility](preflight-compatibility.md) |
| local path works on laptop but fails in the host | runtime filesystem does not expose that path | host deployment docs and secret/volume config |
| dataframe metadata is missing | host bypassed result serialization helpers | [Result Serialization](result-serialization.md) |
| streaming job reprocesses rows after restart | checkpointing or resume state is not wired in | [Streaming Validation](../engines/streaming.md) |

## Related Pages

- [Source Resolution](source-resolution.md)
- [Failure Catalog](failure-catalog.md)
- [Streaming Validation](../engines/streaming.md)
