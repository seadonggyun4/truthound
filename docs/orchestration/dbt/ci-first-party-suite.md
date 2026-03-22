---
title: dbt CI and First-Party Suite
---

# dbt CI and First-Party Suite

Truthound keeps a checked-in dbt integration project so that package behavior is tested
the way users actually consume it.

## What The Suite Covers

The first-party project under `packages/dbt/integration_tests` proves:

- package installation through `dbt deps`
- seed loading
- model compilation and execution
- generic test discovery
- `run_truthound_check`
- `run_truthound_summary`

## Why This Matters

Compile-only validation is not enough for dbt packages. Real regressions often happen
in one of these layers:

- package resolution
- adapter-dispatched SQL
- seed typing and warehouse coercion
- `run-operation` entry points

The first-party suite is there to catch those issues before release.

## Local Commands

Compile-oriented verification:

```bash
uv run --extra dbt python packages/dbt/scripts/run_first_party_suite.py \
  --mode compile \
  --target postgres
```

Warehouse execution verification:

```bash
uv run --extra dbt python packages/dbt/scripts/run_first_party_suite.py \
  --mode execute \
  --target postgres
```

## Expected Signal

The execute lane intentionally includes warning-producing fixtures for invalid data.
That means a healthy run is not necessarily "all passes." The healthy shape is:

- expected warnings for intentionally invalid fixtures
- zero unexpected errors
- successful `run_truthound_check`
- successful `run_truthound_summary`

## CI Usage Pattern

The repository's dbt workflows split responsibilities:

- compile matrix for packaging and resolution
- Python/dbt package checks
- first-party execution against PostgreSQL

That split is deliberate. It narrows failures quickly and keeps adapter regressions
separate from packaging regressions.

## Maintaining The Fixtures

When you change the package:

- keep the valid fixtures truly valid under the current rule semantics
- keep invalid fixtures intentionally invalid and explicitly marked as warning cases
- update the regression tests if you change runner assumptions

## Related Pages

- [Package Setup](package-setup.md)
- [Generic Tests](generic-tests.md)
- [Troubleshooting](troubleshooting.md)
