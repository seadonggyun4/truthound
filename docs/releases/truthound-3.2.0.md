# Truthound 3.2.0

This release adds the optional `truthound[mcp]` extra, local stdio
tools `check_file`, `profile_file` and `scan_pii`, and deterministic Claude
Desktop MCPB generation. Core quality computations and public Python signatures
are reused. See the [MCP guide](../guides/mcp.md) for limits and data handling.

The minor version marks a new optional integration surface. Consumers pinned
to `<3.2` continue to require a deliberate compatibility update. This document
describes the product changes; Claude Desktop marketplace approval and bundle
signing are separate from package publication. Both a PyPI-pinned MCPB and an
embedded-wheel MCPB are provided; their first dependency resolution needs network
access. Both the base package and MCP extra require Polars 1.x for the existing
Core PII scanner API. The previous open-ended base constraint could select Polars
2 and fail on its removed streaming argument; default wheel/sdist installs now
exercise Core check, profile and PII calls without installing the MCP SDK.

Request cancellation, stdin EOF and worker timeout terminate and reap the owned
worker, remove private inputs and release the busy lock. Initialization includes
configured safety instructions. The mandatory integration lane covers Core-only
source changes, rejects missing SDKs/empty/skipped suites, and records actual
Python 3.11–3.13 runtimes and dependency versions for fresh wheel/sdist and UV
launcher tests. These tests do not establish a universal process-memory limit,
OS sandbox or termination deadline for pathological kernel/filesystem hangs.

MCP profile `dtype` returns only an allowlisted type family: Enum category values,
nested type details and timezone/precision parameters are omitted. Unknown type
text becomes `Unknown`; the Core Python API retains full type metadata. Real
stdio and fresh installed-runtime regressions cover Enum Parquet value privacy.

Release verification now preserves the original full nightly marker universe in
disjoint contract, fault/e2e, integration, soak/stress and resource-checked scale
lanes. Its final gate checks actual same-SHA execution manifests and JUnit, not
just job exit codes. See [repository CI](../guides/ci-testing.md).
The package README also carries the ownership marker for the version-matched
[official Registry metadata](../guides/mcp.md#registry-metadata-and-installation).
Valid metadata does not by itself establish a published Registry/marketplace listing.
