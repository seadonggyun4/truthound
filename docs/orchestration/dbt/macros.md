---
title: dbt Macros and Operations
---

# dbt Macros and Operations

Truthound ships both generic tests and callable macros. The generic tests are the
normal entry point for models and sources. The macros are most useful for smoke tests,
debugging, operational summaries, and ad hoc execution in CI or local development.

## Who Should Use This Page

- analytics engineers who need `dbt run-operation` workflows
- maintainers debugging compiled SQL or warehouse-specific behavior
- CI operators wiring first-party dbt smoke checks

## Core Macros

The package exposes three primary macros in `packages/dbt/macros/truthound_check.sql`.

### `truthound_check(model, rules, options={})`

Runs grouped validation rules against a relation.

```sql
{{ truthound_check(
    ref('dim_users'),
    [
      {"column": "id", "check": "not_null"},
      {"column": "id", "check": "unique"},
      {"column": "email", "check": "email_format"}
    ],
    {"limit": 100}
) }}
```

Use this when you want one model-level quality gate that bundles multiple rules.

### `truthound_summary(model, rules, options={})`

Produces summary-style output for one or more rules, which is useful for operator
smokes and human inspection.

```sql
{{ truthound_summary(
    ref('dim_users'),
    [
      {"column": "email", "check": "not_null"}
    ],
    {"limit": 50}
) }}
```

### `truthound_profile(model, columns=none, options={})`

Builds profile-style output for a relation or a selected column set.

```sql
{{ truthound_profile(
    ref('fct_orders'),
    columns=["order_id", "status", "total_amount"],
    options={"limit": 100}
) }}
```

## `run-operation` Entry Points

These macros are intended for operational smoke tests.

### `run_truthound_check`

```bash
dbt run-operation run_truthound_check --args '{
  "model_name": "dim_users",
  "rules": [{"column": "email", "check": "email_format"}],
  "options": {"limit": 50}
}'
```

### `run_truthound_summary`

```bash
dbt run-operation run_truthound_summary --args '{
  "model_name": "dim_users",
  "rules": [{"column": "email", "check": "not_null"}],
  "options": {"limit": 50}
}'
```

The first-party suite uses these macros to prove that package logic works in a real
warehouse, not only in compilation.

## Windowed And Incremental Helpers

The package also exposes window helpers in `truthound_window.sql`.

- `truthound_window_predicate(options={})`
- `truthound_windowed_check(model, rules, options={})`
- `truthound_incremental_check(model, rules, options={})`

These helpers are for projects that want to validate only a bounded time window or a
subset of recent rows instead of the full relation every time.

## Options You Will Commonly Pass

Exact option handling depends on the target macro, but the most common patterns are:

- `limit`: cap result output
- `where`: validate only a filtered slice
- `sample_size`: sample a subset for large relations
- `fail_on_first`: short-circuit on early failures

Keep options deterministic in CI so that validation is reproducible.

## Adapter Dispatch

Truthound uses `adapter.dispatch` heavily inside `truthound_utils.sql` and related
macros. This matters because SQL details vary by warehouse:

- regex functions differ
- safe casts differ
- percentage rounding differs
- sampling and limiting differ

As a result, the public macro names stay stable while the compiled SQL changes by
adapter.

## Production Guidance

- Prefer generic tests in `schema.yml` for ordinary project authoring.
- Use `run-operation` macros for operator smoke checks, scheduled summaries, and
  CI proof-of-life runs.
- Keep rule dictionaries close to the relation they validate; avoid burying business
  critical rules in deeply indirect macro chains.

## Failure Modes To Watch

- `test_truthound_* is undefined`
  The package was not installed or the test was not package-qualified consistently.
- compilation succeeds but execution fails
  The warehouse-specific SQL path compiled but hit adapter-specific runtime behavior.
- summary or profile SQL fails on one warehouse only
  Check the adapter-dispatched utility macro, not just the top-level macro.

## Related Pages

- [Package Setup](package-setup.md)
- [Generic Tests](generic-tests.md)
- [Adapter Behavior](adapter-behavior.md)
- [CI and First-Party Suite](ci-first-party-suite.md)
