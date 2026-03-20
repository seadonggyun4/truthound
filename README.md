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
  <a href="https://pepy.tech/project/truthound">
    <img src="https://img.shields.io/pepy/dt/truthound?color=brightgreen" alt="Downloads">
  </a>
</p>

> Truthound 3.0 turns the familiar `th.check()`, `th.scan()`, `th.mask()`, `th.profile()`, and `th.learn()` facade into a native zero-configuration validation platform built around `TruthoundContext`, `ValidationRunResult`, deterministic auto-suites, and a Polars-first planning/runtime kernel.

---

## Abstract

<p align="center">
  <img width="200" alt="Truthound Icon" src="docs/assets/Truthound_icon_banner.png" />
</p>

Truthound is a Polars-first data validation framework for modern data engineering systems. Version 3.0 keeps the easy first-run experience, but the runtime is now organized around a smaller and more durable kernel: a zero-config project context, deterministic auto-suite selection, backend-aware planning, exact-by-default execution, a single canonical `ValidationRunResult`, and one plugin/reporting surface shared across checkpoints, docs, and automation.

**Documentation**: [truthound.netlify.app](https://truthound.netlify.app/)
Orchestration integrations now live inside the main docs site under [`/orchestration/`](https://truthound.netlify.app/orchestration/).

<!--
Temporary comment-out: keep the related-projects section hidden until these
projects are sufficiently mature for the public README again.

**Related Projects**

| Project | Description | Status |
| --- | --- | --- |
| [truthound-orchestration](https://github.com/seadonggyun4/truthound-orchestration) | Workflow integration for Airflow, Dagster, Prefect, and dbt | Alpha |
| [truthound-dashboard](https://github.com/seadonggyun4/truthound-dashboard) | Web-based data quality monitoring dashboard | Alpha |
-->

## Why Truthound

- Polars-first execution and planner-driven aggregation instead of repeated validator-side scans
- Extreme zero-configuration by default: `th.check(data)` creates and reuses a local `.truthound/` workspace automatically
- Deterministic auto-suite selection that starts with schema/nullability/type/range/key heuristics instead of "run everything"
- Canonical `ValidationRunResult` shared by checkpoints, reporters, validation docs, and plugins
- Explicit contracts for contexts, check factories, backends, and artifact generation
- Failure-first test lanes and migration diagnostics that make framework upgrades safer in production

## Measured Advantages Over Great Expectations

The latest fixed-runner release-grade benchmark artifact set shows Truthound ahead of Great Expectations on every comparable workload in the current comparison catalog while preserving correctness parity.

| Workload | Truthound Warm (s) | GX Warm (s) | Speedup | Memory Ratio |
| --- | ---: | ---: | ---: | ---: |
| local-mixed-core-suite | 0.028240 | 0.075232 | 2.66x | 44.29% |
| local-null | 0.016487 | 0.024964 | 1.51x | 43.62% |
| local-range | 0.002470 | 0.013219 | 5.35x | 43.84% |
| local-schema | 0.001479 | 0.017303 | 11.70x | 35.88% |
| local-unique | 0.002023 | 0.013785 | 6.81x | 42.28% |
| sqlite-null | 0.007370 | 0.032909 | 4.47x | 48.16% |
| sqlite-range | 0.006053 | 0.022355 | 3.69x | 43.80% |
| sqlite-unique | 0.002066 | 0.015655 | 7.58x | 42.12% |

The practical reasons behind that result are straightforward:

- a Polars-first planner/runtime that deduplicates metric work instead of re-scanning through validator loops
- deterministic auto-suite selection that keeps default work relevant and exact
- a smaller zero-config context model that persists baselines and artifacts without forcing a heavy project bootstrap
- one canonical result contract shared by reporters, checkpoints, and validation docs

This comparison is intentionally bounded. It covers comparable deterministic core checks and SQLite pushdown workloads. It is not a blanket claim over every Great Expectations feature area.

Read the published evidence in [Latest Verified Benchmark Summary](docs/releases/latest-benchmark-summary.md).

## What 3.0 Stabilizes

Truthound 3.0 centers the public contract around a smaller and more durable kernel:

| Layer | Responsibility |
| --- | --- |
| `TruthoundContext` | Auto-discovered project workspace, baselines, run history, docs artifacts, plugin runtime, and resolved defaults |
| `contracts` | Stable ports such as `DataAsset`, `ExecutionBackend`, `MetricRepository`, `ArtifactStore`, and plugin capabilities |
| `suite` | Immutable validation intent via `ValidationSuite`, `CheckSpec`, `SchemaSpec`, evidence policy, and severity policy |
| `planning` | Scan planning, backend routing, metric deduplication, and pushdown eligibility |
| `runtime` | Session lifecycle, retries, timeout-safe execution, exception isolation, and evidence capture |
| `results` | `CheckResult`, `ValidationRunResult`, and `ExecutionIssue` as the canonical output model |

The design is grounded in proven ideas from Great Expectations, Soda, Deequ, and Pandera, but optimized for a simpler zero-config starting point and a Polars-first execution path.

The practical 3.0 changes are:

- `th.check()` returns `ValidationRunResult` directly
- the local `.truthound/` workspace is auto-created and reused
- `validators=None` now means deterministic `AutoSuiteBuilder`, not "run every built-in validator"
- `compare` moved to `truthound.drift.compare`
- checkpoints standardize on `CheckpointResult.validation_run` and `CheckpointResult.validation_view`
- reporters and validation docs consume `ValidationRunResult` directly through reporter contract v3

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
truthound doctor . --migrate-2to3
truthound doctor . --workspace
truthound plugins list --json
```

## Public Surface

The root package intentionally exports a smaller API:

- Stable facade: `check`, `scan`, `mask`, `profile`, `learn`, `read`, `get_context`
- Core types: `TruthoundContext`, `ValidationSuite`, `CheckSpec`, `SchemaSpec`, `ValidationRunResult`, `CheckResult`
- `th.check()` returns `ValidationRunResult` directly
- Checkpoint runtime results: `CheckpointResult.validation_run` is canonical and `CheckpointResult.validation_view` is the compatibility projection for legacy action formatting
- Reporter-facing types: `truthound.reporters.RunPresentation`, `truthound.reporters.ReporterContext`
- Validation docs entry points: `truthound.datadocs.ValidationDocsBuilder`, `truthound.datadocs.generate_validation_report`
- Drift comparison: import from `truthound.drift.compare`
- Advanced systems: import by namespace, for example `truthound.ml`, `truthound.lineage`, `truthound.realtime`, or `truthound.datadocs`

The experimental `use_engine` and `--use-engine` switches remain removed.

## Zero-Config Workflow

Truthound 3.0 auto-creates a `.truthound/` workspace at your project root. By default it manages:

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
4. synthesize an auto-suite
5. plan and execute the validation
6. persist the run and validation docs when persistence is enabled

Use `truthound doctor . --workspace` to verify that the local `.truthound/` layout, indexes, baselines, and persisted run artifacts are still structurally healthy.

## Plugin Platform

Truthound now uses one lifecycle runtime:

- `PluginManager` is the canonical plugin manager
- `EnterprisePluginManager` is an async, capability-driven facade over the same runtime
- Plugins register through stable ports such as `register_check_factory`, `register_data_asset_provider`, `register_reporter`, `register_hook`, and `register_capability`
- Reporter plugins should target the contract-v3 surface where `ValidationRunResult` is the canonical render input and `RunPresentation` is the shared render projection

## Documentation

- Documentation site: [truthound.netlify.app](https://truthound.netlify.app/)
- Orchestration overview: [truthound.netlify.app/orchestration/](https://truthound.netlify.app/orchestration/)
- Orchestration getting started: [docs/orchestration/getting-started.md](docs/orchestration/getting-started.md)
- Getting started: [docs/getting-started/index.md](docs/getting-started/index.md)
- Quickstart: [docs/getting-started/quickstart.md](docs/getting-started/quickstart.md)
- Architecture: [docs/concepts/architecture.md](docs/concepts/architecture.md)
- Zero-config context: [docs/concepts/zero-config.md](docs/concepts/zero-config.md)
- Plugin platform: [docs/concepts/plugins.md](docs/concepts/plugins.md)
- Reporter SDK: [docs/guides/reporter-sdk.md](docs/guides/reporter-sdk.md)
- Checkpoints: [docs/guides/checkpoints.md](docs/guides/checkpoints.md)
- Performance and benchmarks: [docs/guides/performance.md](docs/guides/performance.md)
- Benchmark methodology: [docs/guides/benchmark-methodology.md](docs/guides/benchmark-methodology.md)
- Workload catalog: [docs/guides/benchmark-workloads.md](docs/guides/benchmark-workloads.md)
- Great Expectations comparison: [docs/guides/gx-parity.md](docs/guides/gx-parity.md)
- Docs deployment verification: [docs/guides/docs-deployment-verification.md](docs/guides/docs-deployment-verification.md)
- Migration guide: [docs/guides/migration-3.0.md](docs/guides/migration-3.0.md)
- Legacy archive: [docs/legacy/index.md](docs/legacy/index.md)
- Release notes: [docs/releases/truthound-3.0.md](docs/releases/truthound-3.0.md)
- Latest verified benchmark summary: [docs/releases/latest-benchmark-summary.md](docs/releases/latest-benchmark-summary.md)
- ADRs: [docs/adr/001-validation-kernel.md](docs/adr/001-validation-kernel.md), [docs/adr/002-plugin-platform.md](docs/adr/002-plugin-platform.md), [docs/adr/003-result-model.md](docs/adr/003-result-model.md), [docs/adr/004-migration-compatibility.md](docs/adr/004-migration-compatibility.md)

## Development

```bash
uv run --frozen --extra dev python -m pytest -q
uv run --frozen --extra dev python -m pytest --collect-only -q tests
uv run --frozen --extra dev python -m pytest -q -m "contract or fault or e2e" -p no:cacheprovider
uv run --frozen --extra dev python -m pytest -q -m "contract or fault or integration or soak or stress or scale_100m or e2e" --run-integration --run-expensive --run-soak -p no:cacheprovider
uv run --frozen --extra dev python -m pytest -q tests/test_truthound_3_0_contract.py tests/test_api.py tests/test_public_surface.py tests/test_checkpoint.py -p no:cacheprovider
uv run --frozen --extra benchmarks python -m truthound.cli benchmark parity --suite pr-fast --frameworks truthound --backend local --strict
uv run --frozen --extra benchmarks python -m truthound.cli benchmark parity --suite nightly-core --frameworks both --backend local --strict
uv run --frozen --extra benchmarks python -m truthound.cli benchmark parity --suite nightly-sql --frameworks both --backend sqlite --strict
uv run --frozen --extra benchmarks python -m truthound.cli benchmark parity --suite release-ga --frameworks both --strict
uv run --frozen --extra dev python docs/scripts/check_links.py --mkdocs mkdocs.yml README.md CLAUDE.md
uv run --frozen --extra dev --extra docs mkdocs build --strict
truthound doctor . --migrate-2to3
```

Official benchmark comparisons should cite the published fixed-runner artifact set: `release-ga.json`, `env-manifest.json`, and `latest-benchmark-summary.md`.

Tests now follow a failure-first lane model:

- `contract`: stable public API and compatibility boundaries
- `fault`: deterministic failure injection, timeout, corruption, and concurrency scenarios
- `integration`: opt-in backend and external-service coverage
- `soak` and `stress`: nightly-only load and chaos coverage

The default local run is intentionally fast. Manual verification artifacts live under `verification/phase6` and are intentionally kept out of pytest discovery.

Official performance claims should come only from the verified release-grade parity artifacts under `.truthound/benchmarks/release/`. Nightly outputs are for trend visibility, not public benchmark positioning.

When adding tests, prefer scenarios that protect public contracts or operational failure modes. Avoid adding default-value, getter/setter, enum-literal, `to_dict()` round-trip, or CSS-string existence tests unless they prove a compatibility boundary that has failed before.
