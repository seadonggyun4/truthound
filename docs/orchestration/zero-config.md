---
title: Zero-Config
---

# Zero-Config

Zero-config is one of the most important terms in the Truthound 3.x orchestration line, and it is also one of the easiest to misunderstand.

In this repository, zero-config means:

- a fast first run with Truthound-first defaults
- read-only discovery wherever possible
- no hidden workspace creation or persistent run history
- no fake "auto magic" for sources that still need real credentials

It does **not** mean that every host can infer every connection, warehouse profile, or credential without explicit input.

## What Zero-Config Means Here

Truthound's zero-config strength shows up in orchestration as:

- no engine name required for the normal path
- no rule set required for a first Truthound `check()`
- no project workspace required for the default runtime
- no hidden persistence unless the caller opts into it

## Safe Auto Policy

The default policy is `safe_auto`.

It prefers predictable behavior over aggressive inference:

- Truthound as the default engine
- ephemeral runtime contexts
- read-only discovery of host configuration
- explicit connection use only when a source genuinely needs credentials

Use `safe_auto` when you want new projects and CI jobs to be easy to start but hard to surprise.

## Platform Discovery Rules

### Airflow

- local file paths do not require a connection
- SQL sources require a connection
- shared preflight reports explain why a source is blocked
- hooks can normalize connection-backed sources, but orchestration does not invent missing credentials

### Prefect

- omitting a block creates an in-memory Truthound-backed block
- saved blocks remain the right choice for repeated or shared deployments

### Dagster

- `DataQualityResource()` is the canonical zero-config resource
- explicit configuration is still available for parallelism or lifecycle tuning

### Mage

- `io_config.yaml` is discovered from the current directory upward to the project root
- discovery is read-only

### Kestra

- script helpers default to Truthound
- flow templates are designed for YAML-first adoption

### dbt

- install the package
- add YAML tests
- rely on adapter dispatch and package defaults
- dbt profiles stay explicit; orchestration does not try to infer a warehouse target outside the dbt contract

## What Still Requires Explicit Configuration

Zero-config deliberately stops short when the risk of guessing is too high.

You still need explicit configuration for:

- SQL execution that requires a real Airflow connection or dbt target
- Prefect blocks you want to persist and reuse across deployments
- custom engine selection when you do not want the Truthound-first path
- non-default persistence, docs generation, or workspace creation
- SLA thresholds, notification hooks, or secret backends in production
- advanced observability endpoints such as OpenLineage emitters

## Persistence Guarantees

The default Truthound runtime path does not persist:

- runs
- generated docs
- auto-created workspace state

To opt into persistence, configure the Truthound engine context explicitly.

## Practical Guidance

Use zero-config for:

- a developer's first local run
- onboarding examples
- smoke tests
- one-off DAG, flow, or asset experiments
- dbt package adoption before warehouse-specific rollout hardening

Move to explicit configuration when:

- the workflow becomes shared or production-facing
- credentials must be managed centrally
- result persistence or docs artifacts become a requirement
- you need deterministic rollout behavior across environments

## Related Reading

- [Getting Started](getting-started.md)
- [Shared Runtime: Source Resolution](common/source-resolution.md)
- [Shared Runtime: Preflight and Compatibility](common/preflight-compatibility.md)
- [Troubleshooting](troubleshooting.md)
