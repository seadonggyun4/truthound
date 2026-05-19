# Depot Engine Primitives

Truthound Core includes private Depot engine primitives so the future
Truthound Depot console and Truthound Orchestration adapters can share the same
machine-readable dataset repository artifacts. This page documents that
internal contract boundary before public API promotion.

These primitives live under `truthound._datasets`. The leading underscore is
intentional: this is a private namespace for Core-owned contracts, not a stable
public package. Truthound does not expose `truthound.datasets`,
`truthound.depot`, or root-level `Dataset*` exports.

## Purpose

The Depot engine primitive layer gives Core three narrow responsibilities:

- deterministic dataset fingerprint and summary diff primitives
- `ValidationRunResult`-based quality gate projection runtime
- redacted artifact bundles that Truthound Depot and Truthound Orchestration
  can store, exchange, or review

Everything else stays outside Core. Truthound Depot owns repository UI,
branch/merge/review/rollback decisions, approval state, and operator workflows.
Truthound Orchestration owns host-native pull, validate, merge, release,
rollback, and scheduled sync pipelines. Core does not query a Depot database,
operate approval state, execute merge policy, or transport artifacts between
systems.

## Artifact Envelope

Every dataset repository artifact is wrapped in a versioned
`DatasetArtifactEnvelope`.

| Field | Meaning |
| --- | --- |
| `artifact_schema_version` | Internal dataset artifact schema version. Current bootstrap value: `0.1`. |
| `artifact_type` | Machine-readable payload type such as `dataset_fingerprint`, `dataset_diff`, or `quality_gate_bundle`. |
| `payload` | Redacted mapping payload for the specific artifact type. |
| `fingerprint_policy_version` | Version of the deterministic fingerprint policy. |
| `sampling_policy_version` | Version of the deterministic sampling policy. |
| `created_at` | Envelope creation timestamp. |
| `metadata` | Redaction-checked metadata. |

Unsupported or missing envelope versions fail explicitly. Silent coercion is
not allowed because Depot Console and Orchestration must be able to reject
unknown artifact contracts before storing or replaying them.

## Primitive Scope

The MVP scope is intentionally summary-oriented.

| Primitive | Core owns | Core does not promise |
| --- | --- | --- |
| `DatasetAssetManifest` | asset identity and logical source reference | Depot ownership, approval state, branch lifecycle |
| `DatasetSnapshotManifest` | single-parent snapshot metadata and validation refs | multi-parent merge commits or release policy |
| `DatasetFingerprint` | deterministic schema/profile/sample summary hashes | full content addressability by default |
| `DatasetDiff` | schema/profile/row-count/sample digest categories | row-level diff, conflict resolution, automatic merge decisions |
| `QualityGateResult` | projected gate status and redacted failure summaries | business approval, release promotion, rollback execution |

Diff output remains summary-level. `row_level_diff_available` is `false`, and
conflict resolution is outside the Core primitive layer.

## Quality Gate Projection

Quality gate runtime consumes an existing `ValidationRunResult` and projects it
into `QualityGateResult`. It does not run validation suites, planners,
checkpoints, or external adapters.

The projection supports upload, branch, merge, release, and rollback gate
types. Policy classification is supplied as input, and Core returns a
deterministic status: `passed`, `failed`, `warning`, `skipped`, or `error`.

Rollback checks use context supplied by Depot or Orchestration. Core can mark
missing rollback evidence as unsafe, but it does not decide whether an operator
may execute the rollback.

## Bundle Exchange

Bundle artifacts combine primitive payloads into machine-readable exchange
units:

- `DatasetSnapshotBundle`
- `DatasetDiffBundle`
- `QualityGateBundle`
- `DatasetEvidenceInputPayload`

Bundles are designed for storage and transport by Truthound Depot and
Truthound Orchestration. JSON export is limited to string/dict serialization;
file storage adapters, database persistence, and orchestration transport remain
outside Core.

## Redaction Boundary

Dataset artifacts share the same summary-only redaction baseline used by the
AI review surface. Artifact payloads, bundle summaries, quality gate failures,
and evidence input payloads must not contain raw rows, sample values, example
rows, or PII-like literals.

Execution errors are projected without raw messages. Validation issues are
summarized by source, check, validator, issue type, column, severity, count,
and disposition. This keeps AI Evidence payloads and operator-facing artifacts
safe by default.

## What This Is Not

Depot engine primitives are not:

- a broad repository product claim
- a replacement claim for external dataset versioning systems
- company-wide lake management
- a public `truthound.datasets` API
- a public `truthound.depot` API
- a branch/merge/approval/rollback business state manager

The purpose is narrower: Core provides deterministic private artifact
contracts so first-party layers can build repository workflows without
inventing incompatible meanings for snapshots, diffs, gates, and evidence.

## Release Readiness Guardrails

The private primitive surface is protected by contract tests before any public
promotion:

- artifact envelope round-trip and version failure tests
- deterministic fingerprint tests across DataFrame, CSV, and Parquet inputs
- summary diff regression tests
- quality gate projection tests
- public/private namespace smoke tests
- `datasets-private` ruff ratchet coverage in CI

Any future public API promotion should preserve those semantics or introduce an
explicit migration path.
