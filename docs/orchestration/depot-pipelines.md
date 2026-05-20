---
title: Depot Pipelines
---

# Depot Pipelines

Depot pipelines are the shared orchestration surface for branch validation,
scheduled sync, release tagging, rollback triggers, and approval-aware
execution inside `truthound-orchestration`. They project Truthound Depot's
repository workflow into host-native systems without copying Depot business
state into adapters.

The most important boundary is:

- Truthound Orchestration is the **pipeline execution layer**.
- Truthound Depot remains the **business-state owner**.
- Truthound Core remains the **data-plane primitive owner**.

That means orchestration can submit operations, wait, poll, normalize results,
and project host metadata. It does not decide approval, release safety,
rollback safety, or Depot business state on its own.

## What Depot Pipelines Own

Depot pipelines give Airflow, Prefect, Dagster, dbt, Mage, and Kestra one
shared execution contract for:

- snapshot pull requests
- branch validation
- merge-after-approval submission
- release tag requests
- rollback-to-snapshot requests
- scheduled sync and scheduled validation flows

The shared layer keeps those operations compact, host-safe, and observable
without forcing each adapter to invent its own Depot semantics.

## Responsibility Split

| Layer | Owns | Must Not Own |
|-------|------|--------------|
| Core | validation semantics, rule execution, artifact generation | Depot business state, host-specific projection |
| Depot | approval, release safety, rollback safety, business state, canonical operation state | host-native retries, task metadata, adapter payload shaping |
| Orchestration | pipeline execution, submit/poll/wait, compact result emission, host-native projection | approval decisions, rollback safety decisions, business-state persistence |

## Shared Architecture

Depot support is layered the same way across every host:

```text
host-native entrypoint
  -> shared Depot runtime request normalization
  -> Depot client submit/read/wait
  -> artifact ref attachment
  -> failure normalization and redaction
  -> compact operation or flow payload
  -> host-native metadata wrapper
```

The canonical implementation lives in:

- `common/depot/*` for models, failure taxonomy, idempotency, client, polling,
  serialization, and observability
- `common.runtime` for runtime and flow envelopes
- `common.orchestration` for Depot operation and flow facades
- `common.serializers` for compact runtime and flow payload composition

## Supported Operation Surfaces

| Operation | Purpose | Typical terminal states |
|-----------|---------|-------------------------|
| `pull_snapshot` | Synchronize target branch or release state into a host run | `succeeded`, `failed`, `waiting` |
| `validate_branch` | Execute branch validation and publish quality-gate status through the shared Depot contract | `succeeded`, `failed`, `waiting` |
| `merge_after_approval` | Submit merge execution only after Depot approval | `succeeded`, `failed`, `waiting` |
| `release_tag` | Request an immutable release tag through Depot-owned release policy | `succeeded`, `failed`, `waiting`, `no_op` |
| `rollback_to_snapshot` | Request rollback execution against a Depot-owned rollback target | `succeeded`, `failed`, `waiting` |
| `scheduled_sync` | Run the single-operation scheduled sync wrapper around one Depot operation | `succeeded`, `failed`, `waiting`, `no_op` |

## Supported Flow Surfaces

Flows are intentionally thin wrappers over the shared operation layer. They are
not a second workflow engine.

Supported flow shapes:

- submit-only
- submit and wait
- `no_op` terminal flows
- `waiting` flows that propagate approval or external hold states unchanged
- failed terminal flows with compact failure summaries

Current shared flow entrypoints:

| Flow | Shared behavior |
|------|-----------------|
| `scheduled_sync` | Submit scheduled sync and optionally wait |
| `scheduled_validation` | Submit branch validation and optionally wait |
| `release_tag` | Submit release tag request and optionally wait |
| `rollback` | Submit rollback request and optionally wait |

## Result Semantics

Depot operation and flow payloads are compact by design.

`WAITING` means:

- the operation is still owned by Depot
- orchestration may poll and propagate the state
- adapters must not reinterpret it as success or failure

`NO_OP` means:

- the request reached a valid terminal state
- no mutation was required
- the payload is still a successful shared contract surface

`FAILED` means:

- the failure code and compact error message are part of the contract
- adapters should preserve the full shared payload
- business inference stays with the caller or Depot, not the adapter

## Failure And Observability Contract

All hosts share the same Depot failure taxonomy and redacted observability
surface. The shared layer guarantees:

- common `DepotFailureCode` values across runtime, flow, and adapter projections
- retryable vs. non-retryable classification at the shared layer
- compact payloads only
- redacted links, metadata, and execution context for observability outputs

The observability surface keeps operation IDs, flow types, host run metadata,
artifact refs, and failure summaries. It intentionally avoids raw snapshot
bodies, raw evidence blobs, dataset payloads, and secret-bearing headers or
tokens.

## Platform Mapping

| Platform | Native entrypoint | Best fit |
|----------|-------------------|----------|
| Airflow | Depot operators and flow operators | DAG-driven validation, scheduled sync, release, rollback |
| Prefect | Depot block and Depot tasks | Python-first flows with optional persisted config |
| Dagster | Depot resource and Depot ops | Metadata-rich graph execution and scheduled validation |
| dbt | `run-operation` Depot hooks/macros | SQL-first branch validation and release requests |
| Mage | Depot blocks | Pipeline-local validation and scheduled sync paths |
| Kestra | Depot scripts and generated flow templates | YAML flow generation with shared payload outputs |

## When To Use Depot Pipelines

Use ordinary validation surfaces when you need engine-focused quality behavior
without Depot coordination.

Use Depot pipelines when you need:

- branch or snapshot-aware orchestration
- approval-aware merge or release requests
- rollback triggers owned by Depot policy
- scheduled sync or scheduled validation as first-class orchestration events

## Operational Boundaries

Depot pipelines are not intended to replace the host scheduler, persist Depot
business state locally, redefine Core validation semantics, guess approval or
rollback safety, or emit raw runtime payloads for downstream parsing.

The goal is deterministic execution, deterministic status propagation, and
consistent host-native projection.
