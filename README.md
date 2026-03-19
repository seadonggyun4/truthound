<div align="center">
  <img width="500" alt="Truthound Banner" src="docs/assets/truthound_banner.png" />
</div>

<h1 align="center">Truthound</h1>

<p align="center">
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
  <a href="https://pepy.tech/project/truthound"><img src="https://static.pepy.tech/badge/truthound?color=green" alt="Downloads"></a>
</p>

> Truthound 2.0 keeps the familiar `th.check()`, `th.scan()`, `th.mask()`, `th.profile()`, and `th.learn()` experience while routing validation through a smaller kernel with explicit suite, planning, runtime, result, and plugin boundaries.

---

## Abstract

<p align="center">
  <img width="200" alt="Truthound Icon" src="docs/assets/truthound_icon_banner.png" />
</p>

Truthound is a Polars-first data validation framework for modern data engineering systems. It combines an approachable zero-configuration workflow with a more disciplined internal architecture: a compact validation kernel, backend-aware planning, structured runtime results, unified plugin lifecycle management, and direct validation-doc/reporter composition over the same canonical result model.

**Documentation**: [truthound.netlify.app](https://truthound.netlify.app/)

**Related Projects**

| Project | Description | Status |
| --- | --- | --- |
| [truthound-orchestration](https://github.com/seadonggyun4/truthound-orchestration) | Workflow integration for Airflow, Dagster, Prefect, and dbt | Alpha |
| [truthound-dashboard](https://github.com/seadonggyun4/truthound-dashboard) | Web-based data quality monitoring dashboard | Alpha |

## Why Truthound

- Polars-first execution with a small validation kernel instead of a monolithic validator runtime
- Zero-configuration entry points for validation, profiling, masking, and schema learning
- Canonical `ValidationRunResult` model shared by checkpoints, reporters, validation docs, and plugins
- Unified plugin runtime with stable contracts for check factories, reporters, data asset providers, and hooks
- Failure-first test strategy built around deterministic contract and fault lanes

## What Changed in 2.0

Truthound 2.0 introduces five internal layers:

| Layer | Responsibility |
| --- | --- |
| `contracts` | Stable ports such as `DataAsset`, `ExecutionBackend`, `MetricRepository`, `ArtifactStore`, and plugin capabilities |
| `suite` | Immutable validation intent via `ValidationSuite`, `CheckSpec`, `SchemaSpec`, evidence policy, and severity policy |
| `planning` | Scan planning, backend routing, duplicate metric handling, and pushdown eligibility |
| `runtime` | Session lifecycle, retries, timeout-safe execution, exception isolation, and evidence capture |
| `results` | `CheckResult`, `ValidationRunResult`, and `ExecutionIssue` as the canonical output model |

This redesign is intentionally informed by several mature systems:

- Great Expectations: suite, checkpoint, and artifact separation
- Soda: scan planning and backend-aware execution
- Deequ: analyzer, constraint, verification, and repository decomposition
- Pandera: schema-first modeling and lazy validation ergonomics

The migration is controlled rather than disruptive:

- `th.check()` still returns a legacy `Report` facade for compatibility
- `report.validation_run` exposes the structured 2.0 result model
- `CheckpointResult.validation_run` is now the canonical in-memory checkpoint result
- legacy compatibility aliases remain for one migration cycle, but advanced subsystems are expected to be imported via their namespaces

## Quick Start

### Installation

```bash
pip install truthound
```

```bash
# Development and docs workflows in this repository
uv sync --extra dev --extra docs
```

### Python API

```python
import truthound as th
from truthound.datadocs import generate_validation_report
from truthound.reporters import get_reporter

report = th.check(
    {"id": [1, 2, 2], "email": ["a@example.com", None, "c@example.com"]},
    validators=["null", "unique"],
)

print(report)
print(report.validation_run.execution_mode)
print([check.name for check in report.validation_run.checks])

json_report = get_reporter("json").render(report.validation_run)
validation_docs = generate_validation_report(report.validation_run)

schema = th.learn({"id": [1, 2], "status": ["active", "inactive"]})
masked = th.mask(
    {"email": ["a@example.com", "b@example.com"]},
    columns=["email"],
    strategy="hash",
)
```

### CLI

```bash
truthound check data.csv --validators null,unique
truthound check --connection "sqlite:///warehouse.db" --table users --pushdown
truthound scan pii.csv
truthound profile data.csv
truthound plugins list --json
```

## Public Surface

The root package intentionally exports a smaller API:

- Stable facade: `check`, `scan`, `mask`, `profile`, `learn`, `read`
- Core types: `ValidationSuite`, `CheckSpec`, `SchemaSpec`, `ValidationRunResult`, `CheckResult`
- Checkpoint runtime results: `CheckpointResult.validation_run` is canonical; `validation_result` remains as a deprecated compatibility alias
- Reporter-facing types: `truthound.reporters.RunPresentation`, `truthound.reporters.ReporterContext`
- Validation docs entry points: `truthound.datadocs.ValidationDocsBuilder`, `truthound.datadocs.generate_validation_report`
- Advanced systems: import by namespace, for example `truthound.ml`, `truthound.lineage`, `truthound.realtime`, or `truthound.datadocs`

The experimental `use_engine` and `--use-engine` switches were removed in the 2.0 cleanup.

## Plugin Platform

Truthound now uses one lifecycle runtime:

- `PluginManager` is the canonical plugin manager
- `EnterprisePluginManager` is an async, capability-driven facade over the same runtime
- Plugins register through stable ports such as `register_check_factory`, `register_data_asset_provider`, `register_reporter`, `register_hook`, and `register_capability`
- Reporter plugins should target the contract-v2 surface where `ValidationRunResult` is the canonical render input

## Documentation

- Documentation site: [truthound.netlify.app](https://truthound.netlify.app/)
- Getting started: [docs/getting-started/index.md](docs/getting-started/index.md)
- Quickstart: [docs/getting-started/quickstart.md](docs/getting-started/quickstart.md)
- Architecture: [docs/concepts/architecture.md](docs/concepts/architecture.md)
- Plugin platform: [docs/concepts/plugins.md](docs/concepts/plugins.md)
- Reporter SDK: [docs/guides/reporter-sdk.md](docs/guides/reporter-sdk.md)
- Checkpoints: [docs/guides/checkpoints.md](docs/guides/checkpoints.md)
- Migration guide: [docs/guides/migration-2.0.md](docs/guides/migration-2.0.md)
- Legacy archive: [docs/legacy/index.md](docs/legacy/index.md)
- Release notes: [docs/releases/truthound-2.0.md](docs/releases/truthound-2.0.md)
- ADRs: [docs/adr/001-validation-kernel.md](docs/adr/001-validation-kernel.md), [docs/adr/002-plugin-platform.md](docs/adr/002-plugin-platform.md), [docs/adr/003-result-model.md](docs/adr/003-result-model.md), [docs/adr/004-migration-compatibility.md](docs/adr/004-migration-compatibility.md)

## Development

```bash
uv run --frozen --extra dev python -m pytest -q
uv run --frozen --extra dev python -m pytest --collect-only -q tests
uv run --frozen --extra dev python -m pytest -q -m "contract or fault or e2e" -p no:cacheprovider
uv run --frozen --extra dev python -m pytest -q -m "contract or fault or integration or soak or stress or scale_100m or e2e" --run-integration --run-expensive --run-soak -p no:cacheprovider
uv run --frozen --extra dev python docs/scripts/check_links.py --mkdocs mkdocs.yml README.md CLAUDE.md
uv run --frozen --extra dev --extra docs mkdocs build --strict
```

Tests now follow a failure-first lane model:

- `contract`: stable public API and compatibility boundaries
- `fault`: deterministic failure injection, timeout, corruption, and concurrency scenarios
- `integration`: opt-in backend and external-service coverage
- `soak` and `stress`: nightly-only load and chaos coverage

The default local run is intentionally fast. Manual verification artifacts live under `verification/phase6` and are intentionally kept out of pytest discovery.

When adding tests, prefer scenarios that protect public contracts or operational failure modes. Avoid adding default-value, getter/setter, enum-literal, `to_dict()` round-trip, or CSS-string existence tests unless they prove a compatibility boundary that has failed before.
