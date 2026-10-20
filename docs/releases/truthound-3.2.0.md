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
