# Run Analysis Evidence Model

`explain_run(...)` produces a canonical analysis artifact for a validation run.

## Canonical Inputs

The analysis pipeline resolves exactly one run at a time by `run` object or
`run_id`. It then assembles a read-only evidence bundle from:

- persisted run metadata
- deterministic failed-check extraction
- deterministic top-column extraction
- history window summary
- baseline summary when unambiguous
- docs artifact reference when available

## Deterministic vs Model-Owned Fields

Truthound intentionally splits analysis output into deterministic and model
authored parts.

Deterministic fields:

- `run_id`
- `failed_checks`
- `top_columns`
- `history_window`
- `input_refs`

Model-authored fields:

- `summary`
- `recommended_next_actions`
- `evidence_refs`

This division keeps operational facts reviewable even when provider behavior
changes.

## Canonical One-Per-Run Rule

Phase 2 fixed analysis persistence to one canonical artifact per run:

- artifact id: `run-analysis-<run_id>`
- rerun behavior: overwrite the same canonical path
- no ad-hoc question variants in the public contract yet

That choice simplifies dashboards, indexing, and approval-adjacent review
surfaces.
