<div align="center">
  <img width="720" alt="Truthound Depot Banner" src="../assets/truthound-dashboard-banner.png" />
</div>

# Truthound Depot

Truthound Depot is the dataset repository console for the broader Truthound
system. It sits above the core validation kernel, above the additive
`truthound.ai` review layer, and beside Truthound Orchestration, giving
operators one place to manage branch, push, compare, merge request, quality
gate, release, rollback, evidence, access, and validation outcomes.

This public docs portal keeps Depot visible at the system-boundary level, but
it does not reproduce the full console manual or deployment bundle here. Depot
is treated as an installation-managed repository console rather than part of
the `pip install truthound` developer workflow.

## What Depot Owns

- repository business state for Depot, branch, snapshot, merge request,
  release, rollback, evidence, audit, and access workflows
- operator-facing navigation for dataset repository work
- RBAC, membership, ownership, and approval workflows
- source registration, artifact browsing, and evidence surfaces
- AI Evidence panels that support merge, release, and rollback decisions
- observability and incident handling around Depot operations

## What Depot Does Not Own

- validation execution semantics
- planner/runtime logic
- `ValidationRunResult` as the canonical result model
- provider contracts, redaction policy, or AI artifact schema definitions
- host-native submit, poll, wait, retry, or scheduler projection behavior

Those contracts remain fixed in `truthound` and, for AI, in the public
`truthound.ai` namespace. Host-native execution behavior remains in Truthound
Orchestration.

## Relationship To Truthound AI

Depot is where human review happens, not where AI rules are invented.
The underlying proposal and analysis lifecycle is intentionally defined in the
core repository first:

- `suggest_suite(...)` produces reviewable suite proposal artifacts
- `explain_run(...)` produces reviewable run analysis artifacts
- `approve_proposal(...)`, `reject_proposal(...)`, and `apply_proposal(...)`
  keep mutation behind explicit approval

That boundary lets the repository console evolve without forcing the validation
kernel, AI provider contract, or orchestration adapter contract to move with it.

## Relationship To Depot Engine Primitives

Truthound Core owns the private data-plane primitives that Depot consumes:
redacted artifact envelopes, deterministic fingerprints, summary diffs, and
`ValidationRunResult`-based quality gate projections. Depot owns the business
state and decision workflow around those primitives. Orchestration owns the
host-native execution request and status projection surface.

## Operational Direction

The current product direction is an installation-managed deployment package for
managed environments. Public docs keep the boundary and capability story
visible, while deeper deployment, packaging, and runtime specifics are handled
through the Depot delivery channel itself.

## Related Reading

- [Truthound AI Overview](../ai/index.md)
- [Truthound 3.x Architecture](../concepts/architecture.md)
- [Truthound Orchestration](../orchestration/index.md)
