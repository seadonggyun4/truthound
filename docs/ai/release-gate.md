# Smoke and Release Gates

Truthound AI uses explicit gates before new functionality is treated as release
ready.

## Release Discipline

The core release boundary is:

- proposal compilation stays reviewable
- run analysis stays read-only
- approval/apply remains explicit
- provider integration is exercised through opt-in smoke paths
- core hot-path contracts continue to pass without AI extras installed

## Smoke Coverage

Two live OpenAI smoke paths exist as opt-in checks:

- proposal smoke for `suggest_suite(...)`
- run-analysis smoke for `explain_run(...)`

Both use temporary workspaces, explicit environment gates, and minimum viable
contract assertions rather than brittle exact-output expectations.

Prompt hardening also has a deterministic acceptance gate that does not call a
provider:

- 200+ Korean golden prompts
- 50+ mixed Korean/English prompts
- 50+ ambiguous prompts
- 90%+ ready-or-partial acceptance for golden and mixed prompts
- 95%+ clarification behavior for ambiguous prompts
- zero route/compiler crashes for unsupported or malformed fixtures

Live provider checks stay manual-only. Use the single-model smoke for a quick
canary, or `TRUTHOUND_AI_SMOKE_MODEL_MATRIX` when comparing structured-output
and JSON-mode behavior across models.

```bash
TRUTHOUND_AI_RUN_LIVE_SMOKE=1 \
TRUTHOUND_AI_SMOKE_MODEL=gpt-5.4-mini \
truthound ai smoke openai
```

```bash
TRUTHOUND_AI_RUN_LIVE_SMOKE=1 \
TRUTHOUND_AI_SMOKE_MODEL_MATRIX='[{"model":"gpt-5.4-mini","expected_format":"json_schema"}]' \
truthound ai smoke openai-prompt-canary --json
```

## Why Manual Gates Matter

AI functionality is more temporally unstable than the kernel. Provider SDKs,
models, and structured-output behavior change. The release process therefore
leans on:

- contract tests
- doctor validation
- deterministic prompt acceptance tests
- live smoke runners
- explicit dependency probes

This is how the additive AI surface stays evolvable without turning the kernel
or docs portal into a moving target.
