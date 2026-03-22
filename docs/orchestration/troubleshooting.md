---
title: Troubleshooting
---

# Troubleshooting

Use this page when a first run, CI job, or production workflow fails and you need to quickly determine whether the problem lives in host wiring, shared runtime normalization, engine support, or result serialization.

## Troubleshooting By Layer

Start with the layer that matches the failure:

| Symptom | Likely Layer | Read Next |
|---------|--------------|-----------|
| package install or resolver conflict | host compatibility or security surface | [Compatibility](compatibility.md) |
| source cannot be resolved | shared runtime input normalization | [Source Resolution](common/source-resolution.md) |
| source requires a connection or profile unexpectedly | host discovery or preflight | [Preflight and Compatibility](common/preflight-compatibility.md) |
| result shape is present but the host cannot consume it | serialization boundary | [Result Serialization](common/result-serialization.md) |
| workflow runs but observability, retries, or health signals are unclear | shared runtime operations | [Observability and Resilience](common/observability-resilience.md) |
| one platform behaves differently from another | host boundary | the relevant platform section |

## Common Failures

## Install Or Resolver Conflicts

Typical causes:

- using an unsupported host-plus-Python tuple
- mixing release-blocking surfaces with unconstrained convenience extras
- pinning a transitive dependency outside the host's supported dependency graph

What to do:

1. Verify the tuple in [Compatibility](compatibility.md).
2. Install the platform-specific extra instead of `all` when debugging.
3. For Airflow, respect the documented constraints story for the pinned version.

## Preflight Fails Before Execution Starts

Typical causes:

- a SQL source needs a host connection or dbt profile
- the selected engine does not support the requested operation
- the runtime context is missing platform metadata the host package normally supplies

What to do:

1. Inspect the normalized source path in [Source Resolution](common/source-resolution.md).
2. Re-run the same source with explicit host configuration.
3. Confirm the engine capability in [Engines](engines/index.md).

## Zero-Config Does Not Behave The Way You Expected

Typical causes:

- expecting persistence when zero-config is intentionally ephemeral
- expecting automatic credential discovery for a source that truly needs secrets
- assuming dbt or Airflow can infer remote profiles and connections without explicit configuration

What to do:

1. Re-read [Zero-Config](zero-config.md).
2. Make persistence or connection details explicit.
3. Treat zero-config as an onboarding mode, not as a replacement for production configuration.

## Host-Specific Output Handling Fails

Typical causes:

- the host wrapper is trying to consume a payload outside the shared wire contract
- the result is too large for the host's preferred exchange pattern
- custom metadata was added in a host-specific way that hides the shared result fields

What to do:

1. Check [Result Serialization](common/result-serialization.md).
2. Verify the platform's documented output surface.
3. Prefer the first-party serializer path instead of ad hoc conversion.

## dbt Compiles But Execution Or Summary Fails

Typical causes:

- package-qualified tests were not used consistently
- `profiles.yml` or project profile names do not match
- the chosen model/rule pair is not valid for the macro being exercised

What to do:

1. Confirm `dbt deps` completed against the expected package source.
2. Verify package-qualified examples in [dbt Generic Tests](dbt/generic-tests.md).
3. Use the documented first-party suite guidance in [dbt CI and First-Party Suite](dbt/ci-first-party-suite.md).

## Airflow Orchestration Can Read Files But Not SQL

Typical causes:

- local file zero-config is being confused with SQL zero-config
- the DAG has no usable Airflow connection for the SQL source

What to do:

1. Use local paths only for the pure zero-config path.
2. For SQL, make the connection explicit and validate the hook path.
3. See [Airflow Hooks](airflow/hooks.md) and [Airflow Install and Compatibility](airflow/install-compatibility.md).

## Prefect Or Dagster Config Reuse Is Inconsistent

Typical causes:

- ephemeral defaults are being mixed with persisted configuration
- teams are switching between in-memory and saved block/resource patterns without documenting it

What to do:

1. Decide whether the deployment should stay ephemeral or reusable.
2. Use saved Prefect blocks or explicit Dagster resource config for shared environments.
3. See the platform-specific deployment or recipes pages.

## Before Filing A Bug

Capture these details:

- package version and host version
- Python version
- which platform extra is installed
- source shape: DataFrame, file path, URI, SQL, stream, or callable
- selected engine or default engine
- whether the failure happened during install, preflight, execution, or serialization

That context usually makes the failure class obvious.
