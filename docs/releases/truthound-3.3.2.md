# Truthound 3.3.2

This patch repairs MCP client discovery and Claude Desktop packaging without changing Core quality calculations or data grants.

## Fixes

- Recursive model tool schemas expose the object root required by MCP and preserve nested validation constraints.
- Empty optional Desktop workspace/control placeholders mean no permission; unresolved required data paths are rejected.
- Local and Core bundles have distinct extension IDs and compact icons.

## Installation and release status

Start at [Styxis Plugins](https://github.com/seadonggyun4/styxis-plugin#install) for Codex, Claude Code and separate Claude Desktop extensions. A Code plugin update does not update Desktop. Existing permissions remain unchanged.

Core 3.3.2 and matching plugin updates are being prepared. Publication and actual update installation must be verified before declaring release complete. The local `3.3.1+desktop.3` repair was an unpublished build.

## Verification

Regression tests cover recursive schema equivalence, object roots, bundle identity and optional settings. Before publication, an isolated fixture runs the candidate wheel through both client launchers separately from the frozen public plugin. After publication, regenerate plugin locks from PyPI artifacts, verify hashes and execution, and update the Styxis catalog's fixed source commits.

Full Core parity through MCP remains unverified. Tool discovery, execution success and data quality verdicts are separate outcomes.
