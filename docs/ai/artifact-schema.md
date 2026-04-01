# Artifact Schema

Truthound AI persists typed JSON artifacts rather than opaque prompt logs.

## Core Artifact Types

| Artifact | Purpose |
| --- | --- |
| `SuiteProposalArtifact` | persisted proposal generated from prompt compilation |
| `RunAnalysisArtifact` | persisted operational analysis for one validation run |
| `ApprovalLogEvent` | append-only review and apply event record |

## Versioning Model

Two version fields are kept separate:

- `schema_version`: structural compatibility for stored artifacts
- `compiler_version`: lineage of the proposal or analysis compiler

That lets Truthound evolve proposal and analysis compilers independently while
keeping schema compatibility reviewable.

## Path Policy

| Path | Contract |
| --- | --- |
| `.truthound/ai/proposals/{artifact_id}.json` | proposal persistence |
| `.truthound/ai/analyses/{artifact_id}.json` | run analysis persistence |
| `.truthound/ai/approvals/approval-log.jsonl` | approval and apply event log |

`artifact_id` and filename stem must match, and path escape is rejected.

## Compatibility

Proposal artifacts remain backward-readable across:

- `phase0-schema-v1`
- `phase1-suggest-suite-v1`
- `phase1-suggest-suite-v2`

Run analysis artifacts remain backward-readable across:

- `phase0-schema-v1`
- `phase2-explain-run-v1`

Doctor checks validate known compiler-version allowlists rather than assuming a
single forever-fixed version.
