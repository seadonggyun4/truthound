# Truthound AI

Truthound AI is an additive review-layer namespace on top of the Truthound
validation kernel. It does not replace `th.check()`, `ValidationRunResult`, or
the deterministic suite/runtime model. Instead, it turns prompts and run
evidence into reviewable artifacts that a human can inspect, approve, reject,
or apply.

## Design Intent

The AI surface is deliberately narrow:

- proposal compilation: prompt to persisted suite proposal artifact
- run analysis: validation run to persisted operational analysis artifact
- approval and apply: explicit human-reviewed state transitions
- provider and privacy boundaries: summary-only, reviewable, and testable

The goal is not autonomous mutation. The goal is controlled operational
assistance that stays subordinate to the core validation contract.

## Public Entry Points

Install the additive namespace with:

```bash
pip install truthound[ai]
```

Then use the root probe and AI namespace:

```python
import truthound as th
import truthound.ai as thai

status = th.get_ai_support_status()
if status.ready:
    proposal = thai.suggest_suite(prompt="Require customer_id to be unique")
```

Key public entry points:

- `truthound.has_ai_support()`
- `truthound.get_ai_support_status()`
- `truthound.ai.suggest_suite(...)`
- `truthound.ai.explain_run(...)`
- `truthound.ai.approve_proposal(...)`
- `truthound.ai.reject_proposal(...)`
- `truthound.ai.apply_proposal(...)`

## Documentation Map

- [System Boundary](system-boundary.md)
- [Proposal Compiler](proposal-compiler.md)
- [Run Analysis Evidence Model](run-analysis-evidence.md)
- [Artifact Schema](artifact-schema.md)
- [Approval and Apply Semantics](approval-apply.md)
- [Privacy and Redaction](privacy-redaction.md)
- [Provider Contract](provider-contract.md)
- [Smoke and Release Gates](release-gate.md)

## Related Reading

- [Python API Reference](../python-api/index.md)
- [Truthound Dashboard](../dashboard/index.md)
- [Truthound 3.1.1 Release Notes](../releases/truthound-3.1.1.md)
