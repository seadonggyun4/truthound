---
title: Zero-Config
---

# Zero-Config

## What Zero-Config Means Here

Truthound's zero-config strength shows up in orchestration as:

- no engine name required for the normal path
- no rule set required for a first Truthound `check()`
- no project workspace required for the default runtime
- no hidden persistence unless the caller opts into it

## Safe Auto Policy

The default policy is `safe_auto`.

It prefers:

- Truthound as the default engine
- ephemeral runtime contexts
- read-only discovery of host configuration
- explicit connection use only when a source genuinely needs credentials

## Platform Discovery Rules

### Airflow

- local file paths do not require a connection
- SQL sources require a connection
- shared preflight reports explain why a source is blocked

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

## Persistence Guarantees

The default Truthound runtime path does not persist:

- runs
- generated docs
- auto-created workspace state

To opt into persistence, configure the Truthound engine context explicitly.
