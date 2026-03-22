---
title: dbt Overview
---

# Truthound for dbt

Truthound's dbt package is the SQL-first member of the `truthound-orchestration 3.x`
family. It keeps the authoring model native to dbt: generic tests in `schema.yml`,
adapter-dispatched SQL, and `run-operation` entry points for ad hoc validation and
summary workflows.

## Who This Is For

- dbt teams that want first-party data quality rules without introducing a second
  orchestration runtime inside the project
- analytics engineers who prefer declarative YAML tests over Python task code
- platform operators who need repeatable CI validation across warehouses

## When To Use The dbt Adapter

Use the dbt package when:

- your validation boundary is already model- or source-centric
- you want dbt-native test failures and warnings in CI
- you need warehouse-aware SQL generation through `adapter.dispatch`
- you want to keep Truthound rule vocabulary aligned with the other orchestration
  adapters in this repository

Prefer Airflow, Dagster, or Prefect when you need multi-step orchestration, external
task retries, scheduling, or cross-system alert routing outside dbt itself.

## What The Package Provides

- package-qualified generic tests such as `truthound.truthound_check`
- convenience column tests such as `truthound.truthound_not_null`
- macros for summary and profile style execution
- warehouse-specific SQL behavior routed through adapter dispatch
- first-party compile and execution fixtures used in CI

## Minimal Quickstart

1. Add the package to `packages.yml`.
2. Run `dbt deps`.
3. Attach Truthound tests to a model or source.
4. Run `dbt test`.

```yaml
# packages.yml
packages:
  - package: truthound/truthound
    version: ">=3.0.0,<4.0.0"
```

```yaml
# models/marts/schema.yml
version: 2

models:
  - name: dim_users
    tests:
      - truthound.truthound_check:
          arguments:
            rules:
              - column: id
                check: not_null
              - column: id
                check: unique
              - column: email
                check: email_format
              - column: status
                check: in_set
                values: ["active", "inactive", "pending"]
```

## Production Pattern

The most reliable production layout is:

- package-qualified test names everywhere
- model-level `truthound_check` for grouped quality gates
- column-level convenience tests for obvious constraints
- `severity: warn` only for known-bad fixtures, migration windows, or soft rollout
- CI that runs both `dbt test` and targeted `run-operation` smoke checks

## Data Shapes And Rule Semantics

The dbt adapter works on compiled SQL relations, so it is best suited for:

- dbt models
- dbt sources
- warehouse tables and views referenced by `ref()` and `source()`

Unlike the Python adapters, dbt does not resolve local file paths or in-memory
dataframes at runtime. The validation target is always the relation produced by dbt.

## What Is Shared With The Other Adapters

The dbt package uses the same Truthound rule vocabulary as the Python orchestration
adapters. That means a team can keep the same high-level quality intent while choosing
different execution hosts for different pipelines.

For cross-cutting runtime behavior, see:

- [Shared Runtime Overview](../common/index.md)
- [Preflight and Compatibility](../common/preflight-compatibility.md)
- [Result Serialization](../common/result-serialization.md)

## Recommended Reading Order

- [Package Setup](package-setup.md)
- [Generic Tests](generic-tests.md)
- [Macros and Operations](macros.md)
- [Adapter Behavior](adapter-behavior.md)
- [CI and First-Party Suite](ci-first-party-suite.md)
- [Troubleshooting](troubleshooting.md)
