---
title: Source Resolution
---

# Source Resolution

Every host accepts different input shapes, but they all rely on the same shared runtime normalization path. Source resolution turns whatever the host provides into a `ResolvedDataSource` with a canonical `kind`, a human-readable `reference`, and a flag describing whether the source still needs a host-managed connection.

## Supported Source Kinds

The shared runtime recognizes these canonical source kinds:

| Source Kind | Typical Input | Requires Connection |
|-------------|---------------|---------------------|
| `dataframe` | in-memory DataFrame object | No |
| `local_path` | local file path such as `users.parquet` | No |
| `remote_uri` | `s3://`, `gs://`, `abfs://`, `https://` | Often yes, depending on the host/runtime |
| `sql` | SQL text or query-like source | Usually yes |
| `sync_stream` | iterable or stream source | No, but execution model matters |
| `async_stream` | async iterable source | No, but capability support matters |
| `callable` | host-provided function returning data | Depends on what it loads |
| `object` | opaque host object | Depends on host wrapper behavior |

## What Hosts Usually Pass In

| Platform | Common Input Shapes |
|----------|---------------------|
| Airflow | local paths, remote URIs, SQL paired with a connection, hook-managed reads |
| Dagster | DataFrames, resource-loaded objects, local paths |
| Prefect | function inputs, DataFrames, paths, explicit block-driven configuration |
| dbt | model refs and warehouse execution context rather than Python-side data loading |
| Mage | DataFrames and project-config-backed sources |
| Kestra | input URIs, task output paths, script-provided arguments |

## Why Resolution Happens Before Execution

Normalization happens early so the runtime can answer operational questions before a job starts:

- does this source need a connection or profile?
- is the source compatible with the selected engine?
- can the result be serialized safely for this host?
- will the current host be able to explain a failure clearly?

## Local Paths Versus SQL

This is the most common source-resolution misunderstanding.

- Local paths are the easiest zero-config path and usually resolve without host credentials.
- SQL is not "just another string path". It usually implies a host connection, warehouse profile, or adapter-specific execution surface.
- Remote URIs can look simple, but their credentials story is still host-dependent.

If a workflow works for local paths but fails for SQL, that usually means the source is normalized correctly and preflight is correctly telling you the missing piece is a real connection contract.

## Best Practices

- Prefer local path or DataFrame inputs for onboarding and smoke tests.
- Make SQL sources explicit in Airflow, dbt, and other connection-aware hosts.
- Keep remote URI credential handling inside the host platform or secret backend, not hidden inside ad hoc loader code.
- Use `ResolvedDataSource` as the debugging frame of reference when a workflow behaves differently across hosts.

## Related Reading

- [Preflight and Compatibility](preflight-compatibility.md)
- [Zero-Config](../zero-config.md)
- platform-specific install and troubleshooting pages
