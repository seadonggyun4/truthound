<div align="center">
  <img width="500" alt="Truthound Banner" src="docs/assets/truthound_banner.png?v=a4bd297" />
</div>

<h1 align="center">Truthound — Data Quality Workflow</h1>

<!-- mcp-name: io.github.seadonggyun4/truthound -->

<p align="center">
  <strong>Polars 기반 제로 설정 데이터 품질 프레임워크</strong> <br/>
  <strong>Zero-Configuration Data Quality Framework Powered by Polars</strong>
</p>

<p align="center">
  <em>Sniffs out bad data.</em>
</p>

[![Documentation](https://img.shields.io/badge/docs-truthound.netlify.app-blue)](https://truthound.netlify.app/)
[![PyPI](https://img.shields.io/pypi/v/truthound.svg)](https://pypi.org/project/truthound/)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)
[![Powered by Polars](https://img.shields.io/badge/Powered%20by-Polars-2563EB?logo=polars&logoColor=white)](https://pola.rs/)
[![Awesome Polars](https://awesome.re/badge.svg)](https://github.com/ddotta/awesome-polars)

[![Total Downloads](https://api.pepy.tech/personalized-badge/truthound?period=total&units=abbreviation&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/truthound)
[![Monthly Downloads](https://api.pepy.tech/personalized-badge/truthound?period=month&units=abbreviation&left_color=grey&right_color=green&left_text=downloads%2Fmonth)](https://pepy.tech/project/truthound)
[![Weekly Downloads](https://api.pepy.tech/personalized-badge/truthound?period=week&units=abbreviation&left_color=grey&right_color=red&left_text=downloads%2Fweek)](https://pepy.tech/project/truthound)

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

## Plugin Platform



Process and container workers return bounded data-only values; await execution or cleanup to confirm owned worker termination. Container images must be installed before execution; selected canonical host paths can be granted read-only or explicit write access. See the advanced plugin guide for enforced limits and remaining policy restrictions.

The development Core profile also exposes native manual lineage tracking, dependency/path queries, impact summaries and level/type filters, JSON import/export, table/column profile-based rule generation and validator configuration export (without automatic application), two-asset schema evolution and configurable column-rename candidate/confidence analysis, ML prediction score/anomaly/drift-column queries and size-based, ALL/ANY composite retention; complete Core parity remains unverified.

Development sources also provide separately approved persistent cron recurrence and automatic retention, driven by an operator-started foreground worker, built-in schema compatibility analysis, schema history with version save/query/diff/rollback/deletion, approved file-history checkpoints and cross-session recovery, request-driven schema monitoring and local alerts, separately approved session-owned sync/async polling with operator-pinned notifier delivery, approved-lifetime monitoring recovery after worker restart with a durable notification outbox (no automatic retry of unknown sends), and connection/validation/cleanup of operator-pinned and approved sync/async native datasource factories. Live remote services remain unverified; these are not in the published 3.2.0 package. Approved durable watch controls now include pause/resume, operator-confirmed unknown-alert reconciliation and duplicate-risk retry, native age/count retention for resolved queue entries and checkpoint history, explicit expiry renewal and failed-watch recovery, and approved definition deletion. Operator-pinned, approval-gated native Checkpoint runs now preserve configured actions, routing and retries in sensitive result artifacts; native async execution and separately approved session-owned CheckpointRunner scheduling and EscalationEngine lifecycle commands are also available. Approved immediate runs/removal, policy and incident queries, and new policy, trigger, level and notification-target editing are also supported. Quiescent runners support interval/cron/shared-trigger and checkpoint-definition edits that preserve execution history. Signed runner/policy configurations can be restored into a stopped service in a new session. Operator-configured durable native runners and escalation services recover pending work after restart, fence concurrent owners and quarantine uncertain effects without automatic resend; this is not an exactly-once external delivery guarantee.

The 3.3.0 development candidate extends the existing three MCP tools with an opt-in Core profile: closed validator settings, retained assets/results/schemas, native Polars/SQLite/DuckDB datasource lifecycle, local file-stream cursors, declarative kernel suites, advanced profiling, drift/ML/lineage (including trained-model save/restore, ensemble editing, approved cross-session snapshots, learned-rule queries/conversion and graph/subgraph JSON/DOT/Mermaid and whole-graph HTML/Markdown/SVG/PNG artifacts), native cron previews, approved durable checkpoint recovery and outcome reconciliation, native streaming-store sessions, writers/readers and finalization, execution-engine/general-store lifecycles with checked atomic filesystem records and captured concurrent-store codecs/locking, captured memory settings, conditional S3/GCS/Azure receipts and redacted SQL pool controls, selected grouped/streaming aggregation plans, null-aware duplicate groups, complete bounded SQL statistics and parameter-bound range/set operands, native validator ports, native engine checkpoints and paged Arrow cursors, mode-aware cached stores with owned write-behind and explicit uncertain-write reconciliation, and acknowledged shutdown, native model monitoring with pinned prediction/reference/feedback assets, periodic collection, real alert history and visible unconfirmed delivery, factory-owned native adaptive/chunked compression/encryption, range reads, incremental codec sessions and backpressure with private payload slots, migration preview/apply/reopen and explicit batch reconciliation/cleanup, live-source read/profile/PII/mask/schema facades and bound local dashboard/broker/AI/audit/telemetry and approved native tracker context jobs/history and graph save/reopen, plus OpenLineage delivery with retained uncertain outcomes, pinned native benchmark workloads with retained failure artifacts, cooperative time/RSS budgets and lossless baseline restoration, seeded native generators with paged/Arrow output, approved API documentation generation, pinned scaffolds and strict docstring checks with hashed artifacts, shared native reporter controls and partial-failure retention, native tenant row-policy filtering/edits and security/rate-budget controls and redacted secret-provider/cache lifecycles, approved durable finite validation series, native age/count retention previews and approved deletion, schema audits, reporting and separately approved persistent context lifecycle/native extension bindings, same-owner provider/check/reporter workflows, approved native plugin replacement with observed rollback, and detailed hook outcomes, plus approved custom and distributed profiler compositions, retained native samples and sample-sketch queries, rule quality/history, local type-inference feedback, native weighted sklearn classifier training/recovery and factory-bound inference, and HTML/PDF reports, with profile/suite save/restore and generated-suite execution that preserves analysis and conversion failures. Sensitive reads require operator disclosure consent. **This is not full Core parity or part of the published 3.2.0 release.** [Current implementation, usage and limitations](docs/guides/mcp.md#core-profile-unreleased-source-checkout)

- **Codex and ChatGPT desktop (plugin 0.1.4)**: Add the [Truthound repository marketplace](.agents/plugins/marketplace.json), then choose **Plugins → Truthound Plugins → Truthound Local → Install → Set up** and approve a data directory. One-time marketplace setup: `codex plugin marketplace add seadonggyun4/truthound --ref main`. [Installation, consent and official submission](docs/guides/mcp.md)
- **Find it in a marketplace**: Search for `io.github.seadonggyun4/truthound` in the [official MCP Registry](https://registry.modelcontextprotocol.io/). If a connected client's marketplace lists it, select **Install** and specify the data directory you allow it to access. [Published 3.2.0 metadata](https://registry.modelcontextprotocol.io/v0.1/servers/io.github.seadonggyun4%2Ftruthound/versions/3.2.0)
- **Claude Desktop**: [Download the MCPB](https://github.com/seadonggyun4/truthound/releases/download/v3.2.0/truthound-3.2.0.mcpb) → **Settings → Extensions → Advanced settings → Install Extension…** → select the file and allowed directory. [Official installation guide](https://support.claude.com/en/articles/10949351-getting-started-with-local-mcp-servers-on-claude-desktop)

---

Plugin admission checks installed dependencies and versions, and selected MCP owners support acknowledged shutdown. Plugin lifecycle admission can require authenticated source signatures; see the [signing contract](docs/guides/advanced/plugins.md#authenticated-signatures-and-lifecycle-admission).

## Overview

Truthound is an open-source framework for data validation and data quality workflows. Built around a Polars-first validation kernel, it lets teams declare schema checks, custom rules, quality checks, and anomaly detection in code, then manage the results reproducibly.

Truthound is an open-source framework for data validation and data quality
workflows, built on a Polars-first validation kernel.

`th.scan()` inspects the first 1,000 non-null values per string column and estimates finding counts. Local Parquet inputs use 1,024-row batches when PyArrow is installed; complete Parquet footer counts avoid reading already-sampled columns further, while missing statistics retain full batched counting. Other inputs use lazy scalar counts and bounded samples. This is not a full-row inspection guarantee. See the [PII API](docs/python-api/core-functions.md#thscan).

The benchmark CLI supports `--size stress` (100 million rows); `--rows` overrides the preset. See the [benchmark command](docs/cli/benchmark/run.md). Console and JSON benchmark output work without the optional HTML reporting dependencies.

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

The table below preserves historical measurements from the previously published fixed-runner release artifact set. Under the conditions recorded for those eight comparable workloads, Truthound measured faster execution time and lower memory usage than Great Expectations while preserving correctness. These results do not establish a performance pass for later versions or a changed methodology.

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

New benchmarks default to `truthound-parity-thread-budget-v2`: an equal worker thread budget of `1` for both frameworks, one cold iteration, and seven warm iterations, retaining correctness for every iteration and every warm sample. Historical results are not overwritten; direct comparison requires matching methodology, workload contract, dataset, backend, and exactness. When a corrected harness verifies a published wheel, its revision is recorded separately from the wheel version and digest. Authoritative release verification requires the complete default methodology and unchanged thresholds; custom methodologies are advisory. A new performance pass requires actual verified artifacts. See [Benchmark Methodology](docs/guides/benchmark-methodology.md).

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

Data Docs produces A4-style HTML/PDF reports. The public themes are `light`,
`dark`, and `minimal`, and reports include numbered tables, figures,
appendices, quality-dimension mapping, and a methodology appendix for diagnostic
criteria and thresholds. Advanced integrations can inspect the report
architecture with `ReportDocument` while keeping `generate_html_report`,
`HTMLReportBuilder`, and `export_to_pdf` compatible.
Printed alert cards that fit on one page keep their title, message, and suggestion together at page boundaries.
Long JSON pattern examples also wrap within the PDF page without omitting their content.

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

Lineage sample metadata now requires explicit opt-in and is copied with row/byte bounds; source assets are never sampled automatically.

Advanced Arrow conversion now applies captured index/null/temporal policies and row/byte bounds across generic and distributed conversions; see the distributed profiling guide for materialization limits.

Filesystem suite queries honor selected ordering before pagination; ambiguous JSON records cannot be adopted by index rebuild.

The root package intentionally exposes a small API.

- Stable facade: `check`, `scan`, `mask`, `profile`, `learn`, `read`, `get_context`
- Core types: `TruthoundContext`, `ValidationSuite`, `CheckSpec`, `SchemaSpec`, `ValidationRunResult`, `CheckResult`
- `th.check()` returns `ValidationRunResult` directly.
- Checkpoint results: `CheckpointResult.validation_run` is canonical, while `CheckpointResult.validation_view` is a legacy compatibility projection.
- Reporter types: `truthound.reporters.RunPresentation`, `truthound.reporters.ReporterContext`
- Validation docs entry points: `truthound.datadocs.ValidationDocsBuilder`, `truthound.datadocs.generate_validation_report`
- Drift comparison: `truthound.drift.compare`; explicit zero thresholds are preserved, and invalid settings or nonfinite numeric inputs fail visibly.
- ERD audit (in development): [`truthound.schema_audit`](docs/python-api/schema-audit.md) provides immutable models, JSON, digests, standards-to-design audit A, design-to-catalog audit B and bounded PG14/17 read-only collection. Console, connection authorization and approval authority remain separate.
- Advanced systems: import by namespace, for example `truthound.ml`, `truthound.lineage`, `truthound.realtime`, `truthound.datadocs`. Native model constructors copy configuration and reject unknown overrides. Training samples use seeded priorities, and built-in detectors preserve trained state when refitting fails. Concept drift provides a correlation default and native River DDM, ADWIN, and Page-Hinkley stream comparisons ([contract](docs/cli/ml/drift.md)). Lineage tracking supports explicit row positions, context edges, durable tracker snapshots, and explicit bounded samples; OpenLineage delivery retains uncertain transitions and joins owned requests on shutdown ([contract](docs/guides/advanced/lineage.md)).
- Optional AI layer: install `truthound[ai]`, then import `truthound.ai`

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

Expression validation keeps failure samples, counts and rows attached to the reported column after passing checks are omitted.

The `ai`, `mcp` and `all` extras share a Pydantic 2.11 minimum in 3.3.0.
Build a Core MCPB with `truthound-mcp bundle --profile core --output truthound-core.mcpb`. The separate `truthound-core` plugin candidate pins Core 3.3.0 and uses explicit data, workspace, control and disclosure grants; reconnect after changes.

## Development

Source distributions exclude generated `dist/` verification output.
PR and full Nightly distribute independently executable algorithm cases across the same four contract shards without changing coverage or deadlines.
Benchmark workers establish their own workspace boundary so parent projects cannot absorb their run, history and documentation artifacts; persistence remains measured.

Documentation deployments record the public source commit and verification-page hashes in `deployment-manifest.json`.

The development dashboard runtime contracts use Node/npm and actual local servers.
The `dev` extra includes the MCP SDK for integration contracts; dashboard production-build
and cleanup checks run as ordinary contract tests. Release checks also verify a fresh
base installation without the SDK. The `all` extra includes the MCP runtime.
Native distributed contracts also run actual local Ray and Spark SDKs from the
`dev` extra. Native SDK verification closes and reaps its owned Spark launcher before the worker exits.
Ray global scalar reductions reuse cached native statistics; `clear_cache()` and `cache_results=False` control recomputation. Provision Java 17+ for Spark and select the matching Python worker;
these contracts do not connect to remote clusters.
Native Parquet checkpoints and atomic stores flush writable staged files on Windows as well.
Ray checkpoints create the staging destination and explicitly select the local filesystem.
Ray 2.59.0 is excluded for its local-path resolver regression. Windows Spark local
checkpoints require the matching native Hadoop SDK; private worker sinks use UTF-8.

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
