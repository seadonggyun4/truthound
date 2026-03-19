# Truthound

Truthound is a Polars-first data validation framework. The 2.0 redesign keeps the familiar `th.check()`, `th.scan()`, `th.mask()`, `th.profile()`, and `th.learn()` entry points, but routes validation through a smaller core kernel with explicit suite, planning, runtime, result, and plugin boundaries.

## Why 2.0

Truthound now centers on five internal layers:

| Layer | Responsibility |
| --- | --- |
| `contracts` | Stable ports such as `DataAsset`, `ExecutionBackend`, `MetricRepository`, and plugin capabilities |
| `suite` | Immutable validation intent via `ValidationSuite`, `CheckSpec`, `SchemaSpec`, evidence policy, and severity policy |
| `planning` | Scan planning, backend routing, duplicate check accounting, and pushdown eligibility |
| `runtime` | Session lifecycle, retries, timeout-safe execution, exception isolation, and evidence capture |
| `results` | `CheckResult`, `ValidationRunResult`, and `ExecutionIssue` as the canonical output model |

This structure is intentionally informed by several mature validation systems:

- Great Expectations: separation of suite definition, execution, and artifacts
- Soda: scan planning and backend-aware execution
- Deequ: analyzer, constraint, verification, and repository decomposition
- Pandera: schema-first modeling and lazy validation ergonomics

## Quick Start

```bash
pip install truthound
```

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
```

```bash
truthound check data.csv --validators null,unique
truthound check --connection "sqlite:///warehouse.db" --table users --pushdown
truthound plugins list --json
```

## Public Surface

The root package intentionally exports a smaller API:

- Stable facade: `check`, `scan`, `mask`, `profile`, `learn`, `read`
- Core types: `ValidationSuite`, `CheckSpec`, `SchemaSpec`, `ValidationRunResult`, `CheckResult`
- Checkpoint runtime results: `CheckpointResult.validation_run` is canonical; `validation_result` remains as a deprecated compatibility alias
- Reporter-facing types: `truthound.reporters.RunPresentation`, `truthound.reporters.ReporterContext`
- Validation docs entry points: `truthound.datadocs.ValidationDocsBuilder`, `truthound.datadocs.generate_validation_report`
- Advanced systems: import by namespace, for example `truthound.ml`, `truthound.lineage`, or `truthound.datadocs`

The experimental `use_engine` and `--use-engine` switches were removed as part of the 2.0 cleanup.

## Plugin Platform

Truthound now uses one lifecycle runtime:

- `PluginManager` is the canonical plugin manager
- `EnterprisePluginManager` is an async, capability-driven facade over the same runtime
- Plugins register through stable ports such as `register_check_factory`, `register_data_asset_provider`, `register_reporter`, `register_hook`, and `register_capability`
- Reporter plugins should target the contract-v2 surface where `ValidationRunResult` is the canonical input

## Documentation

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
