# Truthound 3.3.0

This release extends the optional local MCP integration with an opt-in
Core profile. Publication, client marketplace listing and deployment are separate
verification steps; this page does not assert that the candidate has been published.
The existing three default tools remain available. See the [MCP guide](../guides/mcp.md)
for commands, operator bindings, consent and service-specific limits.

## Install and use the core workflow

```bash
python -m pip install --upgrade "truthound[mcp]==3.3.0"
truthound-mcp serve --profile core --allow-dir /absolute/path/to/data
```

- Validate nulls, duplicates, ranges and an agreed schema.
- Profile columns, detect PII and mask approved columns without changing the input.
- Learn schemas and compare retained baseline/current assets for drift.
- Read and release result handles. Schema contents and masked rows require the
  explicit `--allow-sensitive-results` operator grant.

See the [core workflow](../guides/mcp.md#core-data-quality-workflow) for operations
and result interpretation. Execution success and data-quality success are distinct;
incomplete schema target accounting remains `unknown`, not a pass.

## SQL query limits and recovery

- SQLite applies a query deadline through its native progress handler for each
  borrowed connection scope; it cannot interrupt a Python UDF.
- PostgreSQL applies native `statement_timeout`.
- MySQL applies `max_execution_time` to supported read-only table SELECTs and
  honors server-side cursor settings; this is not a universal I/O deadline.
- Oracle applies a per-round-trip call timeout. Its pool replaces unhealthy
  connections before lending them again, without replaying the failed statement.

Existing Core APIs, the three default MCP tools and advanced capabilities remain
available. Full Core/MCP equivalence and live cloud services are not certified by
this release. Marketplace bundles pinned to 3.2.0 remain separate distributions.

## Core operations and native services

The Core profile uses retained data, result, schema and model handles for validation,
profiling, drift, privacy, rule generation, reports and lineage. Approved native
factories expose datasource and execution-engine lifetimes, general and streaming
stores, caches, codecs, migrations, tenant policies, plugins, checkpoints, dashboards,
brokers, AI, audit, telemetry and administration. Computation stays in Core.
Sensitive artifacts and effects require the configured disclosure policy and explicit
operation approval; wire requests cannot select arbitrary Python imports or callbacks.

Durable scheduling records intent before effects, fences concurrent owners and
preserves uncertain outcomes across restart. Recovery requires explicit reconciliation
where delivery is unknown. A native tracking job retains one tracker/context for the
whole synchronous callback, keeps completed and failed in-memory history, and supports
operator-selected graph save/load. Native graph files do not persist context history.

## Native correctness and generated artifacts

This candidate repairs bounded SQL statistics and bound range/set operands, null-aware
duplicate groups, copied store configuration, checked conditional object-store writes,
cache cleanup and uncertain-write handling. Native scaffolds preserve quoted metadata,
execute against the current reporter contract and apply shared reporter controls.
Benchmark comparisons reject mismatched suites and preserve raw measurement failures.

Lineage renderers preserve distinct node identities and escape embedded labels and
JSON. Graphviz SVG/PNG needs `truthound[lineage-visualization]` and the system `dot`
program. HTML rendering uses the documented renderer libraries; parser/native-image
verification does not establish every browser's interactive behavior.

## Verification scope

A complete release requires the same commit's full CI, native provider probes,
strict English/Korean documentation, privacy checks and exact wheel/sdist hashes,
followed by fresh installed clients and deployment checks. Historical or partial
successes do not satisfy those gates. Live cloud, broker, paid AI and external
notification accounts have their own verification requirements. The frozen OpenAI
plugin pinned to Core 3.2.0 remains a separate distribution and cannot prove the
new Core candidate's installation or feature coverage.
