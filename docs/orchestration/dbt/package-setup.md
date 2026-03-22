---
title: dbt Package Setup
---

# dbt Package Setup

This page covers the supported setup path for installing Truthound into a dbt project.

## Who This Is For

- teams adopting Truthound in an existing dbt repository
- platform engineers standardizing package and adapter installation
- CI maintainers who need reproducible compile and execution lanes

## Prerequisites

- a dbt project with a working `profiles.yml`
- a supported dbt core and adapter combination
- warehouse connectivity if you plan to run execution tests locally

See [Compatibility](../compatibility.md) for the current orchestration support matrix.

## Install From dbt Hub

```yaml
# packages.yml
packages:
  - package: truthound/truthound
    version: ">=3.0.0,<4.0.0"
```

Then install:

```bash
dbt deps
```

## Local Development Install

For work inside this repository or for testing a local checkout:

```yaml
packages:
  - local: "../truthound-orchestration/packages/dbt"
```

This is the same style used by the first-party integration project under
`packages/dbt/integration_tests`.

## Project Conventions That Matter

Truthound follows normal dbt conventions:

- tests are declared in `schema.yml`
- models are referenced with `ref()`
- sources are referenced with `source()`
- package macros are available after `dbt deps`

The package does not need a custom adapter plugin of its own. It runs on top of the
warehouse adapter already used by the dbt project.

## Recommended First Validation

Start with one model-level grouped test:

```yaml
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
```

Then run:

```bash
dbt test --select dim_users
```

## Profiles And Execution Environments

Truthound does not replace dbt profile handling. It expects:

- the project profile name to exist
- the selected target to be valid
- the warehouse adapter to be installed

The repository's first-party execute suite runs with a project-local `profiles.yml` so
that CI can validate compile and warehouse execution deterministically.

## Package-Qualified Test Names

The safest pattern is to qualify generic and convenience tests with the package name:

- `truthound.truthound_check`
- `truthound.truthound_not_null`
- `truthound.truthound_unique`

This makes test resolution explicit and avoids confusion in multi-package projects.

## Production Pattern

For mature deployments:

- pin the package version range intentionally
- keep `dbt deps` in CI, not only local workflows
- test both compile-only and execute-against-warehouse paths
- prefer package-qualified test names consistently across the project

## Related Pages

- [Generic Tests](generic-tests.md)
- [Macros and Operations](macros.md)
- [CI and First-Party Suite](ci-first-party-suite.md)
