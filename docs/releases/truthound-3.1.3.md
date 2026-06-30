# Truthound 3.1.3 Release Notes

## Highlights

Truthound 3.1.3 is a patch release on top of the 3.1 line.

This release keeps the core validation runtime and AI review contracts stable
while aligning the public package, README, and documentation surface with the
completed Truthound product direction.

## What's New

- public product-line wording now identifies Truthound as the first-party
  dataset repository console
- Core is documented as the data-plane primitive owner for validation
  semantics, deterministic fingerprints, summary diffs, and quality gate
  projections
- Workflow is documented as the business-state owner for branch, merge request,
  release, rollback, approval, evidence, audit, and access workflows
- Orchestration is documented as the execution owner for host-native submit,
  poll, wait, retry, and projection behavior
- Data Docs dashboard wording now distinguishes the local dashboard UI from the
  Workflow repository console product
- release workflow defaults now point at the `3.1.3` package version

## Compatibility

Truthound 3.1.3 is additive.

- importing `truthound` without AI dependencies still works
- `truthound[ai]` remains the optional installation path
- no core workspace, checkpoint, reporter, or AI review contract is broken by
  this patch release
- the private Workflow engine primitive namespace remains private and does not
  introduce a public `truthound.datasets` or `truthound.workflow` API

## Upgrade Guidance

```bash
pip install --upgrade truthound
```

Optional AI support remains:

```bash
pip install --upgrade truthound[ai]
```
