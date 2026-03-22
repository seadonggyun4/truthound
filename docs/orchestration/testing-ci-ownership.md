---
title: Testing and CI Ownership
---

# Testing and CI Ownership

Use this guide when you are adding or moving tests in Truthound Orchestration.

## Who This Is For

- contributors adding adapter coverage
- maintainers deciding where a new regression test should live
- CI owners updating workflow routing or artifact boundaries

## When To Use It

Use this page whenever you need to decide:

- whether a test belongs under `packages/<adapter>/tests` or `tests/<adapter>`
- which fixtures should stay shared versus adapter-local
- which workflow should prove the behavior in CI

## Prerequisites

- a local checkout of `truthound-orchestration`
- the development environment from the repository README
- a clear understanding of whether the behavior is adapter-native or repo-contract level

## Canonical Ownership Model

| Location | Owns | Examples |
|----------|------|----------|
| `packages/<adapter>/tests/` | adapter-native unit and integration tests | host primitives, adapter config, result shaping, host-specific utilities |
| `tests/<adapter>/` | repo-level contract and harness tests | first-party suite runners, support-matrix contracts, CI harness behavior, multi-package wiring |
| `tests/common/` | monorepo-wide CI and shared runtime contracts | support-matrix exports, workflow routing, docs contract checks |

Default rule:

- if a test imports only adapter package code and shared fixtures, it belongs package-local
- if a test verifies repo wiring, CI contracts, install surfaces, or multi-package behavior, it belongs at the repo root

## Minimal Quickstart

For a new adapter-native test:

```bash
PYTHONPATH=.:packages/mage/src pytest --import-mode=importlib packages/mage/tests
```

For a repo-level dbt contract test:

```bash
PYTHONPATH=. pytest --import-mode=importlib tests/dbt/test_first_party_suite.py
```

## Production Pattern

Treat every adapter family as a self-contained product surface:

1. keep happy-path and failure-path tests in `packages/<adapter>/tests`
2. keep repo-level contracts thin and intentional
3. route CI by adapter workflow so failures are visible per host
4. preserve shared fixtures only when they reduce duplication across multiple adapters

### Decision Table

| Question | If yes | If no |
|----------|--------|-------|
| does the test exercise only one adapter package? | place it under `packages/<adapter>/tests` | continue evaluating |
| does the test validate a repo script, CI harness, or support-matrix contract? | keep it under `tests/<adapter>` or `tests/common` | continue evaluating |
| does the test need shared runtime fixtures but no repo harness behavior? | keep shared fixtures in `common.testing`, but place the test package-local | continue evaluating |
| does the test verify multiple packages working together? | keep it at the repo root | package-local is usually correct |

## Fixture Placement

Use this fixture policy:

- shared fixtures in `common.testing` or top-level `conftest.py` for host-agnostic engines and payloads
- adapter-local `conftest.py` for host contexts, adapter config shapes, or host-native expectations
- root contract fixtures only when the contract itself is repo-owned

## CI Workflow Ownership

Each adapter family should report health independently.

| Workflow | Owns |
|----------|------|
| `airflow.yml` | Airflow package-native suites |
| `dagster.yml` | Dagster package-native suites |
| `prefect.yml` | Prefect package-native suites |
| `dbt.yml` | dbt package-native suites plus repo-level first-party suite contracts |
| `mage.yml` | Mage package-native suites |
| `kestra.yml` | Kestra package-native suites |
| `shared-runtime.yml` | shared runtime and cross-adapter common behavior |
| `ci-foundation.yml` | lint, build, doc contracts, support-matrix sync |

### Result Surface

- package-local adapter failures should appear in the adapter workflow, not inside a bundled multi-host lane
- root contract failures should remain visible in their contract lane, such as `dbt` or `foundation`
- summary jobs should aggregate adapter results, not hide ownership boundaries

## Failure Modes And Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| a Mage-only change triggers Kestra CI | PR routing still groups adapters together | update `scripts/ci/pr_change_filter.py` and workflow wiring |
| a dbt unit test still lives at root | adapter-native test was never migrated | move it to `packages/dbt/tests` unless it validates repo harness behavior |
| package-local tests cannot find fixtures | shared fixtures are still root-only | move generic fixtures to `common.testing` or adapter-local `conftest.py` |
| CI artifacts bundle multiple adapters together | workflow boundaries are still shared | split the workflow and artifact names by adapter |
| support-matrix docs drift from workflow reality | matrix export logic or docs sync is stale | run `python scripts/ci/export_support_matrix.py sync-docs --path docs/compatibility.md --write` |

## Adding A New Adapter-Native Test

Use this checklist:

1. put the test under `packages/<adapter>/tests`
2. keep fixtures shared only if another adapter genuinely reuses them
3. ensure the adapter workflow runs the new file by path or directory
4. add or adjust repo-level contract tests only if package behavior alone is not enough
5. verify PR routing still points the change at the correct workflow

## Related Pages

- [Compatibility](compatibility.md)
- [Production Readiness](production-readiness.md)
- [Troubleshooting](troubleshooting.md)
- [Releasing](releasing.md)
