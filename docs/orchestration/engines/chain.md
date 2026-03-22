---
title: Engine Chains
---

# Engine Chains

Engine chains let you layer resolution and fallback behavior instead of assuming one
engine must always serve every dataset the same way.

## When Engine Chains Help

- one engine is the primary path and another is a compatibility fallback
- you need phased migration between engines
- different workloads justify different engines under one orchestration host

## Practical Use Cases

- Truthound-first default with a Pandera fallback during migration
- host-specific rollout where legacy datasets stay on Great Expectations temporarily

## Operational Guidance

- keep the primary engine explicit
- treat fallbacks as deliberate policy, not silent magic
- expose the selected engine in logs and metrics so operators can reason about results

## Related Pages

- [Engines Overview](index.md)
- [Lifecycle Management](lifecycle.md)
