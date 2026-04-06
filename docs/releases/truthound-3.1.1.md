# Truthound 3.1.1 Release Notes

## Highlights

Truthound 3.1.1 is a patch release on top of the 3.1 line.

This release keeps the public AI review surface additive and stable while
exposing one small but important helper for downstream integrations that need
canonical source identity.

## What's New

- `truthound.ai.resolve_source_key(...)` is now a root-level public helper
- dashboard and other review-surface consumers can resolve canonical source
  keys without reaching into `TruthoundContext` internals
- the rest of the `truthound.ai` lifecycle remains unchanged:
  `suggest_suite(...)`, `explain_run(...)`, `approve_proposal(...)`,
  `reject_proposal(...)`, `apply_proposal(...)`

## Compatibility

Truthound 3.1.1 is additive.

- importing `truthound` without AI dependencies still works
- `truthound[ai]` remains the optional installation path
- no core hot-path or workspace contract is broken by this patch release

## Upgrade Guidance

```bash
pip install --upgrade truthound
```

Optional AI support remains:

```bash
pip install --upgrade truthound[ai]
```
