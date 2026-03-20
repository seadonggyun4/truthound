---
title: dbt
---

# dbt

The dbt package is the first-party Truthound 3.x SQL validation package in this repository. It follows dbt-native patterns: generic tests, macros, and adapter dispatch.

## Install

```yaml
packages:
  - package: truthound/truthound
    version: ">=3.0.0,<4.0.0"
```

## Standard Path

```yaml
models:
  - name: stg_users
    tests:
      - truthound_check:
          rules:
            - column: user_id
              check: not_null
            - column: email
              check: email_format
```

## Structural Strengths In This Package

- generic tests for dbt-native authoring
- adapter dispatch for warehouse-specific SQL
- shared rule vocabulary and package defaults
- integration test scaffolding for compile and execution validation

## Role In The 3.x Line

dbt is not a thin Python wrapper around Truthound internals. It is the SQL-first first-party package for bringing Truthound rule semantics into dbt projects with minimal ceremony.
