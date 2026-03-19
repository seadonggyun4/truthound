# Plugin Platform

## One Runtime, Multiple Capabilities

Truthound 2.0 removes the old split between a standard plugin manager and a separate enterprise lifecycle runtime.

The current design is:

- `PluginManager`: canonical lifecycle runtime
- `EnterprisePluginManager`: async facade over the same runtime
- optional capability services: versioning, trust store, security policy, hot reload, signature verification

This keeps lifecycle semantics in one place while still allowing enterprise-oriented behavior to be attached when needed.

## Stable Extension Ports

Plugins should integrate through stable manager ports:

- `register_check_factory()`
- `register_validator()`
- `register_data_asset_provider()`
- `register_reporter()`
- `register_hook()`
- `register_capability()`

These ports provide an abstraction boundary between plugin authors and Truthound internals.

For reporter plugins, the current canonical contract is reporter contract version 2:

- canonical input: `ValidationRunResult`
- shared projection: `RunPresentation`
- compatibility adapters still accept legacy `Report` and persisted validation DTOs for one migration cycle

`PluginManager` records reporter contract versions so mixed plugin fleets can be audited during migration.

## Capability Model

Capabilities are deliberately modeled as optional services rather than hard-coded branches in the plugin lifecycle:

- version compatibility
- trust-store metadata
- security policy metadata
- hot reload intent
- signature verification requirements

This design lets the manager remain small while still supporting richer deployment policies.

## CLI Surface

```bash
truthound plugins list
truthound plugins info my-plugin
truthound plugins load my-plugin
truthound plugins unload my-plugin
```

The public command group is `plugins`, not `plugin`.

## Authoring Guidance

For validator-style plugins, older registry-based approaches still work, but new plugins should prefer manager ports and `CheckSpec`-oriented extension points. Reporter plugins should likewise target `ValidationRunResult` rather than `truthound.stores.results.ValidationResult`. The long-term target is plugin registration against durable kernel contracts rather than direct mutation of internal registries.

## Design Rationale

This platform mirrors the broader 2.0 architecture:

- stable contracts in the center
- adapters at the edge
- optional services instead of manager duplication

The formal decisions are captured in [ADR 002](../adr/002-plugin-platform.md) and the compatibility implications are captured in [ADR 004](../adr/004-migration-compatibility.md).
