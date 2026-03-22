---
title: dbt Singular vs Generic Tests
---

# dbt Singular vs Generic Tests

Truthound's dbt package is optimized around generic tests, but teams often need to decide when grouped model-level quality gates are better than many individual column checks. This page explains that trade-off using the package-qualified tests shipped in the repository.

## Who This Is For

- analytics engineers authoring `schema.yml`
- dbt platform maintainers standardizing test style
- teams migrating from ad hoc SQL tests to Truthound rules

## When To Use It

Use this page when:

- a model already has many single-column assertions
- the team needs a reusable grouped quality gate
- warning vs failure semantics should be standardized across tests

## Prerequisites

- `dbt-core` with the Truthound package installed via `packages.yml`
- `dbt deps` already run
- package-qualified test names such as `truthound.truthound_check`

## Minimal Quickstart

Use a grouped generic test for a model-level quality contract:

```yaml
models:
  - name: test_model_valid
    tests:
      - truthound.truthound_check:
          rules:
            - column: id
              check: not_null
            - column: id
              check: unique
            - column: email
              check: email_format
```

Use convenience tests when the rule is obvious and local to one column:

```yaml
columns:
  - name: id
    tests:
      - truthound.truthound_not_null
      - truthound.truthound_unique
```

## Production Pattern

Use this decision table:

| Pattern | Best Fit |
|--------|----------|
| `truthound.truthound_check` | grouped model-level contracts and mixed rule sets |
| convenience tests such as `truthound.truthound_not_null` | simple column-local assertions |
| singular SQL tests | highly custom logic that does not map cleanly to Truthound rule vocabulary |

Recommended style:

- prefer package-qualified generic tests for the main contract
- keep convenience tests for obvious column rules
- use singular SQL only when the rule cannot be expressed cleanly with the package

## Failure Modes and Troubleshooting

| Symptom | Likely Cause | What To Do |
|--------|--------------|------------|
| test names are undefined | package-qualified names were omitted | use `truthound.truthound_*` consistently |
| quality rules are duplicated | the same logic exists as both grouped and column tests | choose one owner for each constraint |
| grouped tests become unreadable | too many unrelated checks live in one model-level test | split by domain or ownership |

## Related Pages

- [dbt Overview](index.md)
- [Generic Tests](generic-tests.md)
- [Package Setup and Profiles](package-setup.md)
