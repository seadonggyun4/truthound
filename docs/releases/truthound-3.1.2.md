# Truthound 3.1.2 Release Notes

## Highlights

Truthound 3.1.2 is a patch release on top of the 3.1 line.

This release keeps the additive AI review surface stable while tightening the
core runtime and release hygiene around resource-constrained execution and
package-version consistency.

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

## Compatibility

Truthound 3.1.2 is additive.

- importing `truthound` without AI dependencies still works
- `truthound[ai]` remains the optional installation path
- no core workspace, checkpoint, reporter, or AI review contract is broken by
  this patch release

## Upgrade Guidance

```bash
pip install --upgrade truthound
```

Optional AI support remains:

```bash
pip install --upgrade truthound[ai]
```
