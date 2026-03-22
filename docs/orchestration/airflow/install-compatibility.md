---
title: Airflow Install And Compatibility
---

# Airflow Install And Compatibility

## Supported Airflow Line

Use the canonical compatibility page for the exact CI-backed tuples. For Airflow, the important support anchors are:

- minimum supported: Airflow `2.6.0` on Python `3.11`
- primary supported: Airflow `3.1.8` on Python `3.12`

See [Compatibility](../compatibility.md) for the generated support matrix.

## Install

```bash
pip install truthound-orchestration[airflow] "truthound>=3.0,<4.0"
```

If you are targeting one of the CI-backed Airflow lines, keep your runtime close to those tuples rather than assuming the newest transitive dependency set is safe.

## Constraints Story

Airflow is the most dependency-sensitive host in this repository.

- minimum lanes should follow the documented Airflow constraints story
- primary lanes should stay close to the tested compatibility matrix
- if a dependency resolver conflict appears, debug it as an Airflow surface issue first, not as a generic Truthound problem

## What To Validate After Install

Confirm all of the following:

- Airflow can import `truthound_airflow`
- your DAG can instantiate a `DataQualityCheckOperator`
- local-path validation works before you move to SQL or remote URI sources
- Airflow connections exist for any SQL-backed sources

## Upgrade Guidance

When upgrading Airflow:

1. upgrade Airflow and Python as a tested tuple
2. re-run DAG imports
3. verify one local-path quality operator
4. verify one connection-backed quality operator or hook path
5. verify any sensors or SLA callbacks that gate downstream tasks

## Common Gotchas

- local path success does not prove SQL connectivity
- Airflow provider and connection behavior still belong to Airflow, not to Truthound
- legacy host versions may need constraints-compatible dependency resolution

## Related Reading

- [Airflow Overview](index.md)
- [Hooks](hooks.md)
- [Troubleshooting](troubleshooting.md)
