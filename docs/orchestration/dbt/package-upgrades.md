---
title: dbt Package Upgrade Guidance
---

# dbt Package Upgrade Guidance

The safest way to upgrade the Truthound dbt package is to treat it like any other tested adapter surface: keep package-qualified names, rerun `dbt deps`, validate compile parity, and then rerun the execution suite on the canonical warehouse target.

## Who This Is For

- maintainers upgrading `truthound` package versions
- CI owners guarding dbt compatibility
- teams moving between Truthound minor or major releases

## When To Use It

Use this page when:

- moving from one Truthound package version to another
- package macros or generic tests changed
- CI should prove that both compile and execute lanes still work

## Prerequisites

- versioned `packages.yml`
- access to the first-party dbt verification suite
- a canonical target such as PostgreSQL for execution checks

## Minimal Quickstart

Update the package and refresh dependencies:

```yaml
packages:
  - package: truthound/truthound
    version: ">=3.0.0,<4.0.0"
```

```bash
dbt deps
dbt parse
```

Then rerun the first-party suite contract:

```bash
python packages/dbt/scripts/run_first_party_suite.py --mode all --target postgres
```

## Production Pattern

Recommended upgrade checklist:

1. keep package-qualified generic test names
2. rerun `dbt deps`
3. verify `dbt parse` or `dbt ls` still resolves all tests
4. run execution checks on the canonical warehouse target
5. inspect summary macro output and warning-only fixtures

## Failure Modes and Troubleshooting

| Symptom | Likely Cause | What To Do |
|--------|--------------|------------|
| tests become undefined after upgrade | package names or macro paths changed | keep `truthound.truthound_*` names and rerun `dbt deps` |
| compile passes but execution fails | warehouse-specific SQL changed | rerun the first-party execution lane |
| existing invalid fixtures now fail the build | warning policy changed or fixture expectations drifted | reassert `severity: warn` where intentional |

## Related Pages

- [dbt Overview](index.md)
- [CI and First-Party Suite](ci-first-party-suite.md)
- [Warehouse Runtime Behavior](warehouse-runtime-behavior.md)
