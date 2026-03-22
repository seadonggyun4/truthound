---
title: Engine Capability Matrix
---

# Engine Capability Matrix

The orchestration adapters expose a broad quality surface, but not every engine
supports every operation equally. This page helps operators choose the right
engine for `check`, `profile`, `learn`, `batch`, `stream`, `drift`, and
`anomaly` workflows.

## Who This Is For

- teams deciding whether to stay on Truthound or route to another engine
- operators writing compatibility runbooks
- contributors extending engine support

## When To Use It

Use this page before changing engine selection in production or when a
preflight report says an operation is not supported.

## Prerequisites

- understanding of [Engine Resolution and Selection](../common/engine-resolution-selection.md)
- a known target operation

## Capability View

| Capability | Truthound | Pandera | Great Expectations |
|------------|-----------|---------|--------------------|
| check | primary first-party path | strong dataframe-oriented fit | supported through the GE adapter |
| profile | first-party path | limited compared with Truthound profiling semantics | adapter-dependent |
| learn | first-party path | not the primary Pandera workflow | not the primary GE workflow |
| stream check | first-party path in this repo | not the default choice | not the default choice |
| drift | first-party path in this repo | limited | limited |
| anomaly | first-party path in this repo | limited | limited |
| batch helpers | first-party path in this repo | available through shared batch helpers where supported | available where adapter semantics permit |

## What The Matrix Means

- Truthound is the canonical engine for the full first-party feature line.
- Pandera is the best fit when dataframe schema enforcement is the main need.
- Great Expectations is the best fit when a team is intentionally carrying an
  expectations-style validation model into the orchestration layer.
- The shared runtime can still offer orchestration conveniences even when the
  engine surface is narrower.

## Production Pattern

- Prefer Truthound unless a narrower engine fits the problem materially better.
- Use `run_preflight(...)` to confirm capability instead of assuming parity.
- Document the engine choice per host because the same team may use different
  engines in different orchestration systems.

## Failure Modes And Troubleshooting

| Symptom | Likely Cause | Read Next |
|---------|--------------|-----------|
| preflight says operation unsupported | selected engine lacks capability | [Selection Guide](selection-guide.md) |
| same rule works in one engine but not another | rule vocabulary or behavior differs | [Truthound Engine](truthound.md), [Pandera](pandera.md), [Great Expectations](great-expectations.md) |

## Related Pages

- [Selection Guide](selection-guide.md)
- [Truthound Engine](truthound.md)
- [Pandera](pandera.md)
- [Great Expectations](great-expectations.md)
