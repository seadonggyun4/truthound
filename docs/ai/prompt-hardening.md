# Prompt Hardening

Truthound AI treats natural-language prompts as review inputs, not as direct
runtime mutations. The prompt pipeline is hardened around deterministic
normalization, structured provider output, compiler validation, and explicit
human review.

This page describes the public contract. It intentionally avoids deployment-
specific dashboard runtime details.

## Contract Summary

Prompt-to-proposal flow has four safety layers:

| Layer | Purpose |
| --- | --- |
| Normalization | Normalize Korean, English, and mixed prompts into canonical validation intent candidates. |
| Provider output | Prefer strict structured output where supported, with JSON-mode fallback and bounded repair. |
| Compiler gateway | Canonicalize intent names, reject unsupported parameters, and avoid unsafe automatic approximation. |
| Review workflow | Persist proposals for human review before approval or apply. |

The LLM is therefore a parser assistant, not the final authority. The
canonicalizer, compiler, and review workflow remain the safety boundary.

## Korean And Mixed Prompt Support

Truthound 3.1.2 adds a deterministic prompt normalization path for common
Korean and mixed Korean/English quality requests, including:

- required values such as `이메일은 비어 있으면 안 됩니다`
- uniqueness such as `고객ID 중복 없어야 해`
- numeric ranges such as `score는 0 이상 100 이하`
- enum membership such as `상태는 대기/승인/반려 중 하나만 허용`
- common formats such as email, URL, phone, UUID, and IP
- ratio-style checks such as missing-rate and uniqueness-rate thresholds

Ambiguous prompts are not guessed into validators. They are routed to
clarification or rejected proposal items with reason codes.

## Unicode Normalization

Parsing text is normalized with Unicode-aware handling for full-width ASCII,
full-width numbers, spacing variants, and canonical Hangul composition.

Raw prompt text is not stored in provider telemetry or observability metrics.
The pipeline records redacted metadata such as hashes, reason codes, and
normalization warnings so that operators can reproduce behavior without
persisting prompt contents.

When normalization could change meaning, the pipeline prefers clarification
over automatic compilation.

## Structured Output And Fallbacks

Provider integrations prefer strict structured output for models that support
it. If a model does not support that response format, Truthound can fall back to
JSON mode and perform one bounded repair attempt for malformed JSON.

The fallback policy separates failure classes:

| Failure | Behavior |
| --- | --- |
| Structured schema unsupported | Fall back to JSON mode. |
| Malformed JSON in JSON mode | Attempt one repair. |
| Provider refusal | Return a provider response error reason. |
| Auth, quota, or transport failure | Do not repair or fallback. |
| Unsupported intent or unsafe params | Produce rejected proposal items, not route crashes. |

## Evaluation Gate

Truthound keeps deterministic prompt acceptance separate from live provider
smoke tests. The deterministic gate uses repo-tracked fixtures and does not call
external model APIs.

Current acceptance policy:

- Korean golden prompt set: at least 200 cases with at least 90% ready or partial
  acceptance
- mixed Korean/English prompt set: at least 50 cases with at least 90% ready or
  partial acceptance
- ambiguous prompt set: at least 50 cases with at least 95% clarification
  behavior
- unsupported and malformed fixtures: zero route/compiler crashes

Live provider checks remain manual-only because model behavior and provider
availability are temporally unstable.

## Observability And Rollout

Prompt hardening exposes redacted operational counters for:

- normalization mode, language, candidate count, and unresolved terms
- clarification and Unicode warning reason codes
- compile statuses and rejection sources
- provider response format, fallback, repair, and refusal counters

`TRUTHOUND_AI_PROMPT_NORMALIZATION` supports three rollout modes:

| Mode | Purpose |
| --- | --- |
| `enforce` | Production default after deterministic acceptance passes. Non-actionable prompts avoid provider execution. |
| `shadow` | Incident mitigation or comparison mode. Records normalization metadata while keeping the provider path active. |
| `off` | Emergency bypass for prompt normalization. |

Metrics store hashes, counts, modes, and reason codes. They do not store raw
prompts, raw provider outputs, sample rows, or API keys.
