---
title: Production Readiness
---

# Production Readiness

This page is the operator-facing checklist for deciding whether a Truthound
orchestration integration is ready to move from local validation to shared
production workflows.

## Who This Is For

- platform engineers preparing a first rollout
- teams reviewing a host package before broad adoption
- operators building release gates, promotion checks, or runbooks

## When To Use It

Use this page when a project has already passed a local quickstart and now
needs to answer operational questions:

- is the compatibility tuple supported?
- is the selected host the right execution boundary?
- are secrets, retries, and alerting defined explicitly?
- do result payloads land in a host-native surface that downstream users can
  rely on?

## Prerequisites

- a chosen host package or shared-runtime entry point
- a supported tuple from [Compatibility](compatibility.md)
- a known source shape such as a local path, SQL query, dataframe, stream, or
  warehouse relation
- a decision about whether Truthound, Pandera, or Great Expectations is the
  execution engine

## Minimal Readiness Review

Before rollout, confirm all of the following:

| Layer | What Must Be True |
|-------|-------------------|
| Host | the adapter is installed from a supported host-plus-Python tuple |
| Runtime | `run_preflight(...)` passes with the same inputs used in production |
| Source | local, SQL, stream, and warehouse inputs are resolved intentionally |
| Engine | the engine name and optional fallback strategy are explicit |
| Results | the team knows where results land: XCom, metadata, artifacts, outputs, or dbt test status |
| Operations | retries, caching, rate limiting, notifications, and SLA handling are configured where needed |

## Production Pattern

The most reliable rollout sequence is:

1. Start with the host-native quickstart for one narrow validation task.
2. Keep the default Truthound engine unless you have a concrete reason to route
   to Pandera or Great Expectations.
3. Add shared runtime preflight checks before the first production release.
4. Move source credentials or connection details into the host's native secret
   surface.
5. Add result consumers that read the documented host-specific payload rather
   than parsing logs.
6. Add SLA, notification, or incident-routing hooks only after the core
   validation behavior is stable.

## Readiness Checklist

- [ ] Host package is installed from a supported tuple
- [ ] Rules use the shared Truthound rule vocabulary consistently
- [ ] Source resolution behavior is documented for this workflow
- [ ] Preflight compatibility is part of the promotion path
- [ ] Result payload consumers are documented and tested
- [ ] Secrets are not embedded in DAGs, flow code, YAML, or checked-in files
- [ ] Soft-fail versus hard-fail semantics are explicit
- [ ] Retry, cache, rate limiting, and timeout behavior are intentional
- [ ] Operators know where to look for alerts, metrics, and audit evidence

## Failure Modes And Troubleshooting

| Symptom | Likely Cause | First Page To Read |
|---------|--------------|--------------------|
| quality task fails before execution | preflight incompatibility or source mismatch | [Preflight and Compatibility](common/preflight-compatibility.md) |
| host sees a result but downstream consumers disagree | result payload contract not understood | [Result Serialization](common/result-serialization.md) |
| local onboarding worked but production SQL fails | missing host-native connection or secret configuration | host-specific install and operations pages |
| the same rule behaves differently across hosts | source shape or engine selection changed | [Choose a Platform](choose-a-platform.md) and [Engine Selection Guide](engines/selection-guide.md) |

## Related Pages

- [Choose a Platform](choose-a-platform.md)
- [Shared Runtime](common/index.md)
- [Data Quality Engines](engines/index.md)
- [Enterprise Operations](enterprise/index.md)
- [Troubleshooting](troubleshooting.md)
