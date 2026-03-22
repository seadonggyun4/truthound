---
title: dbt Troubleshooting
---

# dbt Troubleshooting

This page focuses on the issues most likely to appear in real dbt package adoption and
CI maintenance.

## `test_truthound_* is undefined`

This usually means one of three things:

- `dbt deps` was not run
- the package was not installed from the expected location
- test names were not resolved consistently in a multi-package environment

Recommended fix:

- run `dbt deps`
- use package-qualified names such as `truthound.truthound_check`

## Compile Works, Execution Fails

This is usually an adapter-path problem rather than a YAML authoring problem.

Check:

- the compiled SQL in `target/compiled`
- seed typing and warehouse coercion
- adapter-dispatched utility macros

## Warning Count Looks High In The First-Party Suite

That can be healthy. The suite includes intentionally invalid fixtures to prove failure
reporting. Treat warnings as expected only when they correspond to those fixtures.

Unexpected `ERROR` output is the real release blocker.

## URL, UUID, Email, Or Numeric Rules Behave Differently Than Expected

Treat the package's current tested semantics as canonical. If a fixture that is meant
to be valid fails, update the fixture or the shared macro logic deliberately. Do not
patch random downstream YAML first.

## `run_truthound_summary` Or `run_truthound_check` Fails

Validate the operation arguments first:

- does the target model exist?
- do the referenced columns exist on that model?
- are the rules valid for the current adapter?

Then inspect the macro and adapter utility path rather than only the wrapper script.

## dbt Deprecation Warning About `arguments`

Current dbt versions warn when generic test arguments are supplied as top-level keys.
Prefer:

```yaml
- truthound.truthound_range:
    arguments:
      min: 0
      max: 150
```

over the older top-level shape.

## Related Pages

- [Package Setup](package-setup.md)
- [Macros and Operations](macros.md)
- [CI and First-Party Suite](ci-first-party-suite.md)
