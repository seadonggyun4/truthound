# Guides

Guides are the task-oriented part of the **Truthound Core** docs. Use them when
you already know the job to be done and want implementation guidance,
operational patterns, and troubleshooting for the main `truthound` repository.

If your task is primarily about scheduler-native execution or workflow-host
integration, jump to [Truthound Orchestration](../orchestration/index.md). If
your task is primarily about reviewable AI proposals or run analysis, jump to
[Truthound AI](../ai/index.md). If your task is primarily about sessions,
ownership, incidents, secrets, or artifact operations in a web UI, jump to
[Truthound Dashboard](../dashboard/index.md).

> **Looking for CLI documentation?** See [CLI Reference](../cli/index.md).
>
> **Looking for Python API lookup?** See [Python API Reference](../python-api/index.md).

## What Belongs Here

| If you need to... | Use these guides |
|------|--------------|
| validate data and design suites | [Validators](validators/index.md) |
| connect files, SQL systems, or warehouses | [Datasources](datasources/index.md) |
| automate runs and policies inside the core repo | [Checkpoint family](checkpoint/index.md) and [Checkpoints overview](checkpoints.md) |
| render output or persist artifacts | [Reporters](reporters/index.md), [Data Docs](datadocs/index.md), [Stores](stores/index.md) |
| profile data and generate rules | [Profiler](profiler/index.md) |
| tune execution and benchmark behavior | [Performance](performance.md), [Benchmark Methodology](benchmark-methodology.md), [Benchmark Workloads](benchmark-workloads.md) |

## What Does Not Belong Here

- **Host-native execution layers**: use [Truthound Orchestration](../orchestration/index.md)
- **AI review-layer workflows**: use [Truthound AI](../ai/index.md)
- **Operational control-plane workflows**: use [Truthound Dashboard](../dashboard/index.md)
- **Command lookup**: use [Reference](../reference/index.md)
- **Sequential first-run learning**: use [Tutorials](../tutorials/index.md)

## Common Core Entry Points

```python
import truthound as th
from truthound.drift import compare

# Basic validation through the core kernel
run = th.check("data.csv")
print(f"Found {len(run.issues)} issues")

# Explicit validators when you need them
run = th.check("data.csv", validators=["null", "duplicate", "range"])

# Learn a baseline schema from trusted data
schema = th.learn("baseline.csv")

# Compare two datasets for drift
drift = compare("baseline.csv", "current.csv", method="auto")
```

## Guide Families

### Validation and Data Access

- [Validators](validators/index.md)
- [Datasources](datasources/index.md)
- [Data Masking](data-masking.md)
- [Privacy](privacy.md)

### Automation and Operations Inside Core

- [Checkpoints Overview](checkpoints.md)
- [Checkpoint Family](checkpoint/index.md)
- [Configuration](configuration/index.md)
- [CI/CD](ci-cd/index.md)
- [Notifications](notifications.md)

### Reporting, Artifacts, and Persistence

- [Reporters](reporters/index.md)
- [Data Docs](datadocs/index.md)
- [Reporter SDK](reporter-sdk.md)
- [Stores](stores/index.md)

### Profiling and Extended Core Workflows

- [Profiler](profiler/index.md)
- [Performance](performance.md)
- [Benchmark Methodology](benchmark-methodology.md)
- [Benchmark Workloads](benchmark-workloads.md)
- [Great Expectations Comparison](gx-parity.md)
- [Migration to 3.0](migration-3.0.md)

## Suggested Reading Paths

### New adopter path

1. [Validators](validators/index.md)
2. [Datasources](datasources/index.md)
3. [Reporters](reporters/index.md)
4. [Checkpoints Overview](checkpoints.md)

### Platform team path

1. [Configuration](configuration/index.md)
2. [Checkpoint Family](checkpoint/index.md)
3. [Stores](stores/index.md)
4. [Performance](performance.md)

### Extended workflow path

1. [Profiler](profiler/index.md)
2. [Data Docs](datadocs/index.md)
3. [Great Expectations Comparison](gx-parity.md)
4. [Truthound AI](../ai/index.md) if the next step is human-reviewed proposals or run analysis
5. [Truthound Orchestration](../orchestration/index.md) if the next step is scheduler-native rollout

## Related Reading

- [Getting Started](../getting-started/index.md)
- [Tutorials](../tutorials/index.md)
- [Reference](../reference/index.md)
- [Concepts & Architecture](../concepts/index.md)
- [Truthound AI](../ai/index.md)
- [Truthound Orchestration](../orchestration/index.md)
- [Truthound Dashboard](../dashboard/index.md)
