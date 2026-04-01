# Proposal Compiler

`suggest_suite(...)` turns natural-language intent into a reviewable
`SuiteProposalArtifact`.

## Pipeline

1. Build a read-only context bundle from schema summary, baseline summary,
   history summary, and current deterministic suite state.
2. Send a summary-only provider request.
3. Parse a structured model response.
4. Compile supported intents into normalized proposal checks.
5. Materialize a formal `ValidationSuite` diff preview.
6. Persist the proposal artifact for review.

## Supported Intent Model

The compiler accepts a curated DSL rather than arbitrary free-form code
generation. That means proposal breadth is deliberately constrained to safe,
reviewable check families such as nullability, uniqueness, ranges, lengths,
sets, regex formats, and simple aggregates.

## Compile Status

| Status | Meaning |
| --- | --- |
| `ready` | every compiled item is usable and nothing was rejected |
| `partial` | at least one check compiled, but some items were rejected |
| `rejected` | no executable checks were produced or the model output was malformed |

Rejected items are not dropped silently. They are preserved in the artifact so
reviewers can see what was unsafe, unsupported, or ambiguous.

## Diff Model

Phase 1.1 promoted the proposal diff into a typed `ValidationSuite`-based
comparison:

- `current_suite`
- `proposed_suite`
- `added`
- `already_present`
- `conflicts`
- `rejected`
- `counts`

That makes review/apply surfaces deterministic and dashboard-safe.
