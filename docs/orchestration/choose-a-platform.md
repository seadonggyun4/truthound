---
title: Choose A Platform
---

# Choose a Platform

Truthound Orchestration supports several host environments, but they solve different problems. This page helps you choose the host that best matches your team, execution model, and operational constraints.

## Start With The Host You Already Operate Well

Truthound does not ask you to adopt a new orchestration style just to run validation. The best default is usually:

- Airflow if your team already ships DAG-based scheduled pipelines
- Dagster if your core abstraction is software-defined assets
- Prefect if your team prefers Python-first flow code and optional persisted configuration
- dbt if your primary need is warehouse-native data quality checks authored in YAML
- Mage if you want lightweight block-level validation inside Mage projects
- Kestra if you want Python-script entry points and generated YAML flow templates

## Decision Table

| If you need... | Choose... | Why |
|----------------|-----------|-----|
| scheduled data quality tasks inside a mature DAG estate | [Airflow](airflow/index.md) | operators, sensors, hooks, and SLA callbacks stay Airflow-native |
| resource-centric checks attached to assets and metadata | [Dagster](dagster/index.md) | `DataQualityResource` fits naturally into assets, ops, and asset checks |
| reusable configuration with Python-native flows and deployments | [Prefect](prefect/index.md) | blocks persist config while tasks and flows stay ergonomic |
| validation authored directly in SQL project configuration | [dbt](dbt/index.md) | generic tests and macros keep the whole workflow dbt-native |
| project-local checks in Mage blocks with minimal ceremony | [Mage](mage/index.md) | transformers, sensors, and `io_config.yaml` discovery fit Mage projects |
| generated flow YAML or Python Script task entry points | [Kestra](kestra/index.md) | script helpers and flow generators fit Kestra's execution model |

## Comparison By Boundary

| Platform | Integration Boundary | Best For | Operational Shape |
|----------|----------------------|----------|-------------------|
| Airflow | Operators, Sensors, Hooks | DAGs, SLAs, task gating | scheduler + workers + connections |
| Dagster | Resources, Ops, Assets, Asset Checks | asset-aware quality controls | definitions + resources + metadata |
| Prefect | Blocks, Tasks, Flows | persisted config with Python-first authoring | deployments + work pools + blocks |
| dbt | Generic Tests, Macros, Run Operations | SQL validation and warehouse enforcement | package management + adapters + CI |
| Mage | Transformers, Sensors, Conditions | simple project-level quality blocks | local project config + Mage runtime |
| Kestra | Python scripts, flow generators, outputs | YAML workflow generation and script tasks | namespaces + flows + task outputs |

## When Zero-Config Matters Most

If you want the lowest-friction first run:

- Dagster and Prefect are the smoothest Python-first host integrations.
- dbt is the smoothest SQL-first path if your data quality contract belongs in model YAML.
- Mage and Kestra are the lightest "embed inside an existing host project" paths.
- Airflow is straightforward, but you still need to think about connections for SQL-backed sources.

## When Explicit Configuration Matters More

Choose the host that gives your team the strongest operational story once the first run is behind you:

- Airflow for DAG dependencies, task retries, and existing scheduler governance
- Dagster for metadata-rich assets and asset checks that stay visible in the Dagster catalog
- Prefect for reusable saved blocks and deployment-level configuration reuse
- dbt for environment-specific profiles, adapters, and warehouse governance
- Kestra for flow-as-YAML generation and Kestra-native outputs
- Mage for project-local configuration discovery with minimal external dependencies

## Engine Considerations

All first-party hosts are Truthound-first by default, but advanced engines remain available.

- If you want the smoothest docs path, stay on Truthound.
- If you already operate Great Expectations or Pandera, use the shared runtime and [Engines](engines/index.md) guidance before standardizing host-level patterns.
- If you need streaming, drift, anomaly detection, or engine lifecycle hooks, make sure the selected host plus engine pair supports the operation you need before rollout.

## Operational Questions To Ask Before Committing

Answer these before choosing the host:

1. Where does your team already manage runtime credentials and secrets?
2. Do you need persisted host configuration, or is per-run ephemeral configuration enough?
3. Is your quality contract better expressed in Python, DAG config, or SQL/YAML?
4. Do you need host-native metadata and UI visibility for quality results?
5. Will the first rollout be CI-only, batch-only, or attached to production workflows?

## Recommended Reading By Platform

- [Airflow](airflow/index.md)
- [Dagster](dagster/index.md)
- [Prefect](prefect/index.md)
- [dbt](dbt/index.md)
- [Mage](mage/index.md)
- [Kestra](kestra/index.md)
- [Shared Runtime](common/index.md)
