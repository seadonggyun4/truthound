# AI System Boundary

Truthound AI is intentionally outside the kernel hot path.

## Boundary Rules

1. `import truthound` must continue to work without AI dependencies.
2. `truthound.core` must not import `truthound.ai`.
3. AI artifacts live under `.truthound/ai/` and applied suites live under
   `.truthound/suites/`.
4. The review layer is additive: core validation continues to work without any
   provider, dashboard, or approval workflow.
5. Privacy policy is summary-only by default.

## Why This Matters

These constraints keep the kernel stable even when providers, prompt
contracts, or dashboard workflows evolve. They also make the AI layer easier to
audit because path policy, artifact schema, and redaction rules are explicit.

## Workspace Areas

| Area | Purpose |
| --- | --- |
| `.truthound/ai/proposals/` | persisted suite proposal artifacts |
| `.truthound/ai/analyses/` | persisted run analysis artifacts |
| `.truthound/ai/approvals/approval-log.jsonl` | approval/apply event log |
| `.truthound/suites/` | applied suite records used at runtime |

## Feature Probes

The root package exposes a lightweight feature probe so downstream systems do
not have to import `truthound.ai` just to decide whether AI features are ready:

- `has_ai_support() -> bool`
- `get_ai_support_status() -> AISupportStatus`

That keeps dashboards and services loosely coupled to the public contract
instead of internal provider modules.
