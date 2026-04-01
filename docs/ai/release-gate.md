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

## Why Manual Gates Matter

AI functionality is more temporally unstable than the kernel. Provider SDKs,
models, and structured-output behavior change. The release process therefore
leans on:

- contract tests
- doctor validation
- live smoke runners
- explicit dependency probes

This is how the additive AI surface stays evolvable without turning the kernel
or docs portal into a moving target.
