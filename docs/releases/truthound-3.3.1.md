# Truthound 3.3.1

This patch ships clearer local MCP tool descriptions and JSON Schema field
explanations. The tool IDs remain `check_file`, `profile_file` and `scan_pii`.
No validation capability, directory grant, output disclosure or network transport
is added. `check_file` checks nulls/duplicate rows; profiling observes types and
aggregate statistics; PII findings remain sampled estimates.

## Installation and publication

The release candidate is versioned 3.3.1. Publication is established by the exact
same-commit Release verification and Release PyPI workflows and a fresh PyPI
installation, not by this page. Registry publication is a separate step.

```bash
python -m pip install "truthound[mcp]==3.3.1"
```

The frozen Local plugin remains 0.1.6/Core 3.3.0 during candidate verification.
It is repinned only after 3.3.1 exists on PyPI. Existing Core packaging tests use
the exact candidate wheel before publication. The Styxis public catalog still
contains only the Local `truthound` plugin; no product or directory listing is added.

## Verification and limits

Release gates retain complete same-revision tests, native compatibility probes,
English/Korean documentation and exact wheel/sdist provenance. Core metadata
improvements alone do not establish Agent selection accuracy. Record actual
client/model observations separately; no universal discovery guarantee is made.
Installation grants no data access and raw values are not returned by the Local
summary tools. Paths, column names and aggregates can reach the host model.
