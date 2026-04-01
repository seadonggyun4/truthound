# Truthound 3.1 Release Notes

## Highlights

Truthound 3.1 adds a new additive AI review surface while keeping the core
validation kernel and zero-config workflow intact.

Key changes:

- `truthound.ai` is now the canonical optional AI namespace
- `truthound.has_ai_support()` and `truthound.get_ai_support_status()` expose a
  root-level feature probe
- `suggest_suite(...)` produces persisted, reviewable proposal artifacts
- `explain_run(...)` produces canonical run analysis artifacts
- approval, rejection, and apply now have typed lifecycle facades
- live smoke runners exist for both proposal and run-analysis OpenAI paths
- public docs now describe the dashboard through a boundary-level overview
  instead of mirroring a full dashboard manual

## Public Surface

The stable AI additions are:

- root probe: `has_ai_support()`, `get_ai_support_status()`
- review APIs: `suggest_suite(...)`, `explain_run(...)`
- lifecycle APIs: `approve_proposal(...)`, `reject_proposal(...)`,
  `apply_proposal(...)`
- review helpers: `list_proposals(...)`, `show_proposal(...)`,
  `list_analyses(...)`, `show_analysis(...)`

These remain additive. Importing `truthound` without AI dependencies continues
to work.

## Docs and Product Boundary

Truthound 3.1 also tightens the public documentation boundary:

- orchestration remains fully documented as a public first-party layer
- dashboard remains visible in the docs portal through a concise overview page
- the public docs no longer depend on a mirrored dashboard manual
- a new `AI` section explains the technical boundary, artifact schema,
  redaction policy, provider contract, and release gates

## Upgrade Guidance

Install the additive AI surface only when you need it:

```bash
pip install truthound[ai]
```

For the rest of the core workflow, existing `pip install truthound` behavior
is unchanged.
