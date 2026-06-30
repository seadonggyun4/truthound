<div align="center">
  <img width="500" alt="Truthound Banner" src="docs/assets/truthound_banner.png?v=a4bd297" />
</div>

<h1 align="center">Truthound — Data Quality Workflow</h1>

<p align="center">
  <strong>Polars 기반 제로 설정 데이터 품질 프레임워크</strong> <br/>
  <strong>Zero-Configuration Data Quality Framework Powered by Polars</strong>
</p>

<p align="center">
  <em>Sniffs out bad data.</em>
</p>

<p align="center">
  <a href="https://truthound.netlify.app/"><img src="https://img.shields.io/badge/docs-truthound.netlify.app-blue" alt="Documentation"></a>
  <a href="https://pypi.org/project/truthound/"><img src="https://img.shields.io/pypi/v/truthound.svg" alt="PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-orange.svg" alt="License"></a>
  <a href="https://pola.rs/"><img src="https://img.shields.io/badge/Powered%20by-Polars-2563EB?logo=polars&logoColor=white" alt="Powered by Polars"></a>
  <a href="https://github.com/ddotta/awesome-polars"><img src="https://awesome.re/badge.svg" alt="Awesome Polars"></a>
  <a href="https://pepy.tech/project/truthound">
    <img src="https://img.shields.io/pepy/dt/truthound?color=brightgreen" alt="Downloads">
  </a>
</p>

<!--
README HEADER FORMAT LOCK:
The exact header format above MUST be preserved.
Do not rewrite it as Markdown-only syntax, do not remove the centered HTML,
do not change the banner image block, title, bilingual subtitle, slogan, or
badge block, and do not reorder those elements.
This required format starts at:
  <div align="center">
and ends at the closing </p> of the badge block above.
-->

---

## Overview

Truthound is an open-source framework for data validation and data quality workflows. Built around a Polars-first validation kernel, it lets teams declare schema checks, custom rules, quality checks, and anomaly detection in code, then manage the results reproducibly.

Truthound는 데이터 품질 검증(Data Validation)과 데이터 워크플로우(Data Workflow)를 위한 오픈소스 프레임워크입니다.

**Documentation**: [truthound.netlify.app](https://truthound.netlify.app/)

---

## Motivation

Data quality validation is often implemented differently in every project, which makes reuse and standardization difficult. Truthound aims to standardize data validation and quality management as a code-based framework and workflow so anyone can build a consistent, reproducible validation environment.

---

## Introduction

Truthound provides an open-source data validation framework and workflow orchestration layer built on Polars. It supports schema validation, custom rules, quality checks, anomaly detection, and related validation capabilities. It can also be connected to major workflow environments such as Airflow, Dagster, and Prefect to automate validation steps. With a code-first declarative validation style, Truthound improves reproducibility and maintainability for data quality management and is designed to fit naturally into many data pipelines.

Truthound is an open-source ecosystem composed of two repositories.

| Component | Repository | Role |
| --- | --- | --- |
| **Truthound** | [`truthound`](https://github.com/seadonggyun4/truthound) | Validation kernel — `th.check()`, `ValidationRunResult`, planner/runtime, zero-configuration workspace, reporters, checkpoints |
| **Truthound Orchestration** | [`truthound-orchestration`](https://github.com/seadonggyun4/truthound-orchestration) | Workflow integration layer for Airflow, Dagster, Prefect, dbt, Mage, Kestra, and similar environments |

---

## Impact

By standardizing and automating data validation logic, Truthound reduces repetitive quality-management work and improves the reliability and operational efficiency of data pipelines. Its integration-friendly open-source structure can also provide a reusable data quality foundation for data analytics, ETL, AI/ML, and other data-driven domains.

---

## Key Features

<!--
FACT-CHECK LOCK, 2026-07-01:
The default th.check() local runtime is Polars/LazyFrame based, but it executes
validators sequentially through ValidationRuntime._execute_sequential().
ExpressionBatchExecutor and SharedMetricStore provide batched/deduplicated
metric execution utilities, but ScanPlanner does not automatically fuse all
validator metrics into one collect() on the default path.
-->
- **Polars-first execution**: Local validation uses Polars `LazyFrame` as the reference path. Batched expression and shared-metric utilities are available, but the default `th.check()` path does not automatically fuse every validator metric into one aggregate query.
- **Zero-Configuration**: `th.check(data)` automatically creates and reuses a local `.truthound/` workspace.
- **Deterministic auto validation suites**: Selects only relevant checks using schema, nullability, type, range, and key heuristics instead of "run everything."
- **Single result model**: Checkpoints, reporters, validation docs, and plugins all share one `ValidationRunResult`.
- **Explicit contracts**: Provides stable interfaces for contexts, check factories, backends, and artifact generation.
- **Workflow integration**: Truthound Orchestration enables host-native execution inside schedulers and workflow systems.

> AI review features are optional helper layers. Truthound's core is a data quality workflow; AI only assists with human-reviewable prompt-based validation suggestions and run analysis. All core features work without AI.

---

## Benchmark

In fixed-runner release-grade benchmarks, Truthound measured faster execution time and lower memory usage than Great Expectations across the eight comparable workloads in the published release artifact set while preserving correctness.

| Workload | Truthound Warm (s) | GX Warm (s) | Speedup | Memory Ratio |
| --- | --- | --- | --- | --- |
| local-mixed-core-suite | 0.028240 | 0.075232 | 2.66x | 44.29% |
| local-null | 0.016487 | 0.024964 | 1.51x | 43.62% |
| local-range | 0.002470 | 0.013219 | 5.35x | 43.84% |
| local-schema | 0.001479 | 0.017303 | 11.70x | 35.88% |
| local-unique | 0.002023 | 0.013785 | 6.81x | 42.28% |
| sqlite-null | 0.007370 | 0.032909 | 4.47x | 48.16% |
| sqlite-range | 0.006053 | 0.022355 | 3.69x | 43.80% |
| sqlite-unique | 0.002066 | 0.015655 | 7.58x | 42.12% |

This comparison is limited to deterministic core checks and SQLite pushdown workloads. It is not a generalized claim about every feature area. The repository-local `.truthound/benchmarks/artifacts` directory may contain only a subset of raw observations; treat the release artifact set and the [Latest Verified Benchmark Summary](https://github.com/seadonggyun4/truthound/blob/main/docs/releases/latest-benchmark-summary.md) as the official source for these numbers.

Primary reasons for the performance difference:
- Polars `LazyFrame` based local execution and SQL pushdown paths
- batched expression and shared-metric optimization utilities used by some validators and advanced execution paths
- deterministic auto-suite selection that keeps default work exact and relevant
- a lightweight zero-configuration context that preserves baselines and artifacts without heavy project bootstrapping
- a single result contract shared by reporters, checkpoints, and validation docs

---

## Quick Start

### Installation

```bash
pip install truthound
```

```bash
# Optional AI review features
pip install truthound[ai]
```

```bash
# Development/docs workflow in this repository
uv sync --extra dev --extra docs
```

### Python API

```python
import truthound as th
from truthound.datadocs import generate_validation_report
from truthound.reporters import get_reporter
from truthound.drift import compare

run = th.check(
    {"customer_id": [1, 2, 2], "email": ["a@example.com", None, "c@example.com"]},
)

print(run.execution_mode)
print([check.name for check in run.checks])
print(run.metadata["context_root"])

json_report = get_reporter("json").render(run)
validation_docs = generate_validation_report(run, title="Customer Quality Overview")

context = th.get_context()
schema = th.learn({"id": [1, 2], "status": ["active", "inactive"]})
masked = th.mask(
    {"email": ["a@example.com", "b@example.com"]},
    columns=["email"],
    strategy="hash",
)
drift = compare({"score": [0.1, 0.2]}, {"score": [0.1, 0.8]})
```

### CLI

```bash
truthound check data.csv --validators null,unique
truthound check --connection "sqlite:///warehouse.db" --table users --pushdown
truthound scan pii.csv
truthound profile data.csv
truthound doctor . --workspace
truthound plugins list --json
```

```bash
# Optional AI review workflow
truthound ai suggest-suite data.csv --prompt "Require customer_id to be unique"
truthound ai proposals list
truthound ai explain-run --run-id <run_id>
```

---

## Zero-Config Workflow

Truthound automatically creates a `.truthound/` workspace at the project root. By default it manages:

- `.truthound/config.yaml`: resolved project defaults
- `.truthound/catalog/`: asset fingerprints and source signatures
- `.truthound/baselines/`: learned schemas and metric history
- `.truthound/runs/`: persisted `ValidationRunResult` metadata
- `.truthound/docs/`: generated validation docs
- `.truthound/plugins/`: resolved plugin manifest and trust metadata

If you do nothing except call `th.check(data)`, Truthound will:

1. detect the asset/backend
2. resolve the active `TruthoundContext`
3. load or create a baseline
4. synthesize an auto validation suite
5. plan and execute validation
6. persist the run and validation docs when enabled

Use `truthound doctor . --workspace` to check the structural integrity of the local `.truthound/` layout, indexes, baselines, and persisted run artifacts.

---

## Public Surface

The root package intentionally exposes a small API.

- Stable facade: `check`, `scan`, `mask`, `profile`, `learn`, `read`, `get_context`
- Core types: `TruthoundContext`, `ValidationSuite`, `CheckSpec`, `SchemaSpec`, `ValidationRunResult`, `CheckResult`
- `th.check()` returns `ValidationRunResult` directly.
- Checkpoint results: `CheckpointResult.validation_run` is canonical, while `CheckpointResult.validation_view` is a legacy compatibility projection.
- Reporter types: `truthound.reporters.RunPresentation`, `truthound.reporters.ReporterContext`
- Validation docs entry points: `truthound.datadocs.ValidationDocsBuilder`, `truthound.datadocs.generate_validation_report`
- Drift comparison: `truthound.drift.compare`
- Advanced systems: import by namespace, for example `truthound.ml`, `truthound.lineage`, `truthound.realtime`, `truthound.datadocs`
- Optional AI layer: install `truthound[ai]`, then import `truthound.ai`

---

## Plugin Platform

Truthound uses a single lifecycle runtime.

- `PluginManager` is the canonical plugin manager.
- `EnterprisePluginManager` is an async, capability-driven facade over the same runtime.
- Plugins register through stable ports such as `register_check_factory`, `register_data_asset_provider`, `register_reporter`, `register_hook`, and `register_capability`.
- Reporter plugins should target the contract-v3 surface, where `ValidationRunResult` is the canonical render input and `RunPresentation` is the shared render projection.

---

## Documentation

- Main docs portal: [truthound.netlify.app](https://truthound.netlify.app/)
- Overview: [docs/index.md](https://github.com/seadonggyun4/truthound/blob/main/docs/index.md)
- Getting started: [docs/getting-started/index.md](https://github.com/seadonggyun4/truthound/blob/main/docs/getting-started/index.md)
- Architecture: [docs/concepts/architecture.md](https://github.com/seadonggyun4/truthound/blob/main/docs/concepts/architecture.md)
- Zero-config context: [docs/concepts/zero-config.md](https://github.com/seadonggyun4/truthound/blob/main/docs/concepts/zero-config.md)
- Guides: [docs/guides/index.md](https://github.com/seadonggyun4/truthound/blob/main/docs/guides/index.md)
- Reference: [docs/reference/index.md](https://github.com/seadonggyun4/truthound/blob/main/docs/reference/index.md)
- AI docs: [docs/ai/index.md](https://github.com/seadonggyun4/truthound/blob/main/docs/ai/index.md)
- Orchestration: [truthound.netlify.app/orchestration/](https://truthound.netlify.app/orchestration/)
- Orchestration getting started: [docs/orchestration/getting-started.md](https://github.com/seadonggyun4/truthound/blob/main/docs/orchestration/getting-started.md)
- Latest verified benchmark summary: [docs/releases/latest-benchmark-summary.md](https://github.com/seadonggyun4/truthound/blob/main/docs/releases/latest-benchmark-summary.md)

---

## Development

```bash
uv run --frozen --extra dev python -m pytest -q
uv run --frozen --extra dev python -m pytest --collect-only -q tests
uv run --frozen --extra dev python -m pytest -q -m "contract or fault or e2e" -p no:cacheprovider
uv run --frozen --extra benchmarks python -m truthound.cli benchmark parity --suite pr-fast --frameworks truthound --backend local --strict
uv run --frozen --extra benchmarks python -m truthound.cli benchmark parity --suite nightly-core --frameworks both --backend local --strict
uv run --frozen --extra dev --extra docs mkdocs build --strict
```

Tests follow a failure-first lane model.

- `contract`: stable public API and compatibility boundaries
- `fault`: deterministic failure injection, timeout, corruption, and concurrency scenarios
- `integration`: opt-in backend and external-service coverage
- `soak` / `stress`: nightly-only load and chaos coverage

Official performance numbers must only be cited from the verified release-grade parity artifacts under `.truthound/benchmarks/release/`. Nightly output is for trend visibility, not public benchmark positioning.

---

## License

Apache License 2.0. See [LICENSE](https://github.com/seadonggyun4/truthound/blob/main/LICENSE) for details.
