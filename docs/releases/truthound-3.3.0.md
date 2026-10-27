# Truthound 3.3.0 candidate

This source candidate extends the optional local MCP integration with an opt-in
Core profile. Publication, client marketplace listing and deployment are separate
verification steps; this page does not assert that the candidate has been published.
The existing three default tools remain available. See the [MCP guide](../guides/mcp.md)
for commands, operator bindings, consent and service-specific limits.

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
