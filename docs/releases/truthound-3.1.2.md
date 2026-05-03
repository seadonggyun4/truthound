# Truthound 3.1.2 Release Notes

## Highlights

Truthound 3.1.2 is a patch release on top of the 3.1 line.

This release keeps the additive AI review surface stable while tightening the
core runtime and release hygiene around resource-constrained execution and
package-version consistency.

It also promotes the AI prompt path from a simple prompt-to-provider flow to a
hardened review pipeline with deterministic normalization, compiler rejection,
evaluation gates, and redacted rollout metrics.

## What's New

- validation runtime now falls back to sequential execution when thread-pool
  startup is unavailable on constrained CI runners
- SQLite null-check parity uses the intended pushdown path again in the
  benchmarked SQL comparison lane
- reporter and checkpoint outer-layer cleanup is now protected by folder-level
  `ruff` ratchets in CI
- the core package version surface now resolves from one canonical source
  across runtime, plugin scaffolding, benchmark metadata, and current-release
  docs links
- AI suite proposal prompts now pass through a Korean/English normalization IR
  before provider guidance and compiler validation
- Unicode normalization keeps raw prompt text out of telemetry while preserving
  reproducible hash and warning metadata
- provider output hardening prefers strict structured output, falls back to JSON
  mode only for supported cases, and records fallback/repair/refusal counters
- the prompt acceptance gate now tracks 200+ Korean golden prompts, mixed
  Korean/English prompts, ambiguous prompts, and crash-free malformed fixtures
- prompt rollout is observable through redacted reason-code metrics and the
  `TRUTHOUND_AI_PROMPT_NORMALIZATION=off|shadow|enforce` mode switch

## Compatibility

Truthound 3.1.2 is additive.

- importing `truthound` without AI dependencies still works
- `truthound[ai]` remains the optional installation path
- no core workspace, checkpoint, reporter, or AI review contract is broken by
  this patch release
- prompt hardening changes rejected or ambiguous AI suggestions into reviewable
  artifacts instead of public API shape changes

## Upgrade Guidance

```bash
pip install --upgrade truthound
```

Optional AI support remains:

```bash
pip install --upgrade truthound[ai]
```
