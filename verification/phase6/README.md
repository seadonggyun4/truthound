# Phase 6 Manual Verification

These scripts are manual verification artifacts for the phase 6 checkpoint and saga rollout.

- They are intentionally stored under `verification/phase6` instead of `tests/`.
- They are not part of the pytest contract/fault/integration/soak lanes.
- Some scripts print verification summaries or inspect package state directly, so they should be run manually when auditing historical rollout behavior.

Use them as ad hoc verification tools, not as CI regression tests.
