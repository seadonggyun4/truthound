# Truthound 3.3.2

This patch repairs MCP client discovery and Claude Desktop packaging while preserving Core quality calculations and existing data grants.

## Fixes

- Recursive model tool schemas expose the object root required by MCP and preserve nested validation constraints.
- Empty optional Desktop workspace/control placeholders mean no permission; unresolved required data paths are rejected.
- Local and Core Desktop bundles have distinct extension IDs and compact icons.
- Local plugin archives include the Claude manifest. Both client launchers are verified separately, and mismatched manifest versions prevent packaging.

## Installation

Start at [Styxis Plugins](https://github.com/seadonggyun4/styxis-plugin#install) for Codex, Claude Code and separate Claude Desktop extensions. Local 0.1.8 and Core 0.2.3 pin the published Core 3.3.2 runtime. The catalog records the current fixed source commits. A Code plugin update does not update Desktop. Installation grants no data access.

## Publication and verification

[PyPI 3.3.2](https://pypi.org/project/truthound/3.3.2/) was published from commit `30a65a581b6dbd075f64e547f1a69c0957c24aab` after [release verification](https://github.com/seadonggyun4/truthound/actions/runs/41954271544). The [publication workflow](https://github.com/seadonggyun4/truthound/actions/runs/41961838401) reused those artifacts without rebuilding. Public PyPI hashes match:

| Artifact | SHA-256 |
| --- | --- |
| Wheel | `6ea961ec86d7bcc0a5baf8fa7a8521d3bd8638d627e10e1d1819f22fec7608cc` |
| sdist | `f51e7bd750ea28a469a2375e2cfa8f6d6b176361f9ad0319e8b22ead6505c763` |

Regression coverage includes recursive schema equivalence, object roots, bundle identity, optional settings and isolated installed-runtime execution. Plugin locks use public PyPI artifacts; comparison wheels do not act as fallback installation sources. The local `3.3.1+desktop.3` repair was an unpublished build.

## Remaining limits

The first Windows Python 3.12 Dask Arrow-stream test timed out during native session initialization. The same commit passed all 214 checks on a fresh runner with unchanged limits, but the original startup-delay cause remains unproven. This patch does not claim to resolve it.

Full Core parity through MCP remains unverified. Tool discovery, execution success and data quality verdicts are separate outcomes. Real external deployments require their own configuration, authorization and verification.
