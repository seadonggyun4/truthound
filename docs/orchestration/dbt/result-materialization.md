---
title: dbt Result Materialization
---

# dbt Result Materialization

Truthound's dbt package exposes two result surfaces: standard dbt test outcomes and macro-driven summary output. Understanding that split is the key to using dbt as both a build gate and a reporting surface.

## Who This Is For

- teams using both `dbt test` and `run-operation`
- CI owners deciding which commands belong in a verification suite
- analytics engineers who want summaries without losing test semantics

## When To Use It

Use this page when:

- the build should fail on quality rules but also emit a readable summary
- a pipeline needs an ad hoc quality report from a model
- operators want to separate gate behavior from summary behavior

## Prerequisites

- the Truthound package installed in dbt
- familiarity with `run_truthound_check` and `run_truthound_summary`
- a compiled model available in the target warehouse

## Minimal Quickstart

Use `dbt test` for build gating:

```bash
dbt test --select test_model_valid
```

Use `run-operation` for summary-style output:

```bash
dbt run-operation run_truthound_summary --args '{
  "model_name": "test_reference_model",
  "rules": [{"column": "name", "check": "not_null"}],
  "options": {"limit": 50}
}'
```

## Production Pattern

Use both surfaces intentionally:

| Surface | Best Fit |
|--------|----------|
| `dbt test` | hard build gates, warnings, CI status |
| `run_truthound_check` | targeted smoke checks or scripted validation |
| `run_truthound_summary` | human-readable summaries and operator diagnostics |

Recommended checklist:

- use `dbt test` as the canonical pass/fail contract
- use summary macros for diagnostics, not for the only gate
- keep macro smoke coverage in CI when package behavior changes

## Failure Modes and Troubleshooting

| Symptom | Likely Cause | What To Do |
|--------|--------------|------------|
| summary works but build gate does not | only macros are being exercised | add package-qualified tests to `schema.yml` |
| build fails but summary is fine | the severity policy differs between tests and macros | align the rollout intent explicitly |
| summary macro breaks on one warehouse | runtime SQL behavior is backend-specific | validate with the first-party suite on the supported target |

## Related Pages

- [dbt Overview](index.md)
- [Macros and Operations](macros.md)
- [CI and First-Party Suite](ci-first-party-suite.md)
