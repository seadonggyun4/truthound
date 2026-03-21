# CI/CD Validation Guide

Integrate Truthound into pull requests, scheduled pipelines, and deployment gates without turning your data quality workflow into a custom test harness.

## Who This Is For

- Data engineers adding validation to GitHub Actions, GitLab CI, Jenkins, or similar pipelines
- Platform teams standardizing Truthound across multiple repositories
- Teams moving from ad-hoc scripts to repeatable checkpoint-based validation

## When To Use This Guide

Use this guide when you want one of the following outcomes:

- run a first validation in CI in less than 10 minutes
- publish machine-readable outputs for pipeline annotations and artifacts
- turn one-off checks into reusable checkpoints
- route failures to the right team with escalation, throttling, and metadata

## Canonical First Task

Start with a single repository-level validation job and only then add checkpoints, routing, and release gating.

```yaml
name: data-quality

on:
  pull_request:
  push:
    branches: [main]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install "truthound[dev]"
      - run: truthound check data/customers.csv --format github
```

Expected result:

- pull requests show pass/fail status from Truthound
- CI logs include issue summaries
- the same command can be promoted into a checkpoint later

## Recommended Reading Path

1. [Getting Started: First Validation](../../getting-started/first-validation.md) to choose your first validation workflow
2. [Checkpoint Guide](../checkpoints.md) to understand reusable execution units
3. [CI Platform Integration](../checkpoint/ci-platforms.md) for environment detection and metadata capture
4. [Checkpoint CLI](../../cli/checkpoint/index.md) if your team prefers pipeline-first operations
5. [Docs Deployment Verification](../docs-deployment-verification.md) if the pipeline also publishes Data Docs or docs artifacts

## Choose Your Path

### Path 1: First Validation In CI

Best for teams that want a fast adoption path.

- Install Truthound in the pipeline
- Run `truthound check ...`
- Emit console, JSON, or GitHub/GitLab-friendly output
- Fail the job only on the severities you want to block

Start here:

- [CLI check](../../cli/core/check.md)
- [Reporters Guide](../reporters/index.md)
- [Getting Started](../../getting-started/index.md)

### Path 2: Checkpoint-Driven Pipelines

Best for teams with multiple datasets, environments, or approval gates.

- Define named checkpoints
- Store metadata about branch, commit, actor, and run ID
- Route failures and add escalation logic
- Reuse the same checkpoint locally and in CI

Start here:

- [Checkpoint Basics](../checkpoint/basics.md)
- [Checkpoint CLI](../../cli/checkpoint/index.md)
- [Checkpoint Routing](../checkpoint/routing.md)
- [Checkpoint Escalation](../checkpoint/escalation.md)

### Path 3: Advanced Platform Automation

Best for central platform teams and larger estates.

- standardize config and secrets
- attach artifacts and HTML reports
- integrate profiling and rule generation jobs
- support multiple CI vendors without rewriting the workflow model

Start here:

- [Configuration Guide](../configuration/index.md)
- [Profiler Guide](../profiler/index.md)
- [Data Docs Guide](../datadocs/index.md)
- [Benchmark Methodology](../benchmark-methodology.md)

## Implementation Pattern

### Step 1: Pick The Execution Mode

- `truthound check ...` for direct file or datasource validation
- `truthound checkpoint run ...` for reusable named executions
- Python API when the pipeline already orchestrates data access in code

### Step 2: Pick Output Contracts

- console output for fast operator feedback
- JSON or YAML for downstream parsing
- GitHub or JUnit-style reporters for CI-native annotations and test views
- HTML reports when you want human-readable release artifacts

### Step 3: Capture Run Metadata

Truthound can detect CI providers automatically and attach environment context.

```python
from truthound.checkpoint.ci import get_ci_environment

env = get_ci_environment()
print(env.platform, env.branch, env.commit_sha, env.run_id)
```

See [CI Platform Integration](../checkpoint/ci-platforms.md) for supported providers and available metadata.

### Step 4: Decide What Blocks A Merge

Common choices:

- fail on any issue for critical tables
- fail only on high/critical severity in pull requests
- warn on drift or profiling deltas but do not block
- publish full artifacts even when validation fails

### Step 5: Add Operational Guardrails

Use these once the first job is stable:

- [Throttling](../checkpoint/throttling.md) for noisy schedules
- [Deduplication](../checkpoint/deduplication.md) for repeated alerts
- [Routing](../checkpoint/routing.md) for ownership-based escalation
- [Escalation](../checkpoint/escalation.md) for incident workflows

## Example: GitHub Actions With Checkpoint CLI

```yaml
name: checkpoint-validation

on:
  pull_request:

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install "truthound[dev]"
      - run: truthound checkpoint init orders-pr --datasource data/orders.parquet
      - run: truthound checkpoint validate orders-pr
      - run: truthound checkpoint run orders-pr --format github
```

Artifacts to expect:

- job logs with validation summary
- checkpoint metadata enriched with CI context
- optional JSON, JUnit, or HTML outputs for downstream systems

## Common Failure Modes

### Pipeline Passes Locally But Fails In CI

- confirm the same Truthound version is installed in both environments
- check file paths, cloud credentials, and environment-variable-backed configs
- verify that sampled local runs are not masking full-data issues

### CI Output Is Hard To Consume

- switch reporters instead of rewriting the job first
- use [CI Reporters](../reporters/ci-reporters.md) or [Common Output Formats](../../cli/common/output-formats.md)
- publish JSON/JUnit artifacts for machine processing

### Too Many Noisy Failures

- move repeated runs into checkpoints
- add throttling and deduplication
- route failures by owner or dataset instead of using one global notification path

### Team Uses Multiple CI Vendors

- rely on [CI Platform Integration](../checkpoint/ci-platforms.md) for environment detection
- keep the validation command stable and vary only the runner wrapper

## Related Docs

- [Checkpoint Guide](../checkpoints.md)
- [Checkpoint CI Platforms](../checkpoint/ci-platforms.md)
- [Checkpoint CLI](../../cli/checkpoint/index.md)
- [Configuration Guide](../configuration/index.md)
- [Reporters Guide](../reporters/index.md)
- [Data Docs Guide](../datadocs/index.md)
