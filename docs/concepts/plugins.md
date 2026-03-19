# Plugin Platform

## One Runtime, Multiple Capabilities

Truthound 3.0 removes the old split between a standard plugin manager and a separate enterprise lifecycle runtime.

The current design is:

- `PluginManager`: canonical lifecycle runtime
- `EnterprisePluginManager`: async facade over the same runtime
- optional capability services: versioning, trust store, security policy, hot reload, signature verification

This keeps lifecycle semantics in one place while still allowing enterprise-oriented behavior to be attached when needed.

## Stable Extension Ports

Plugins should integrate through stable manager ports:

- `register_check_factory()`
- `register_data_asset_provider()`
- `register_reporter()`
- `register_hook()`
- `register_capability()`

These ports provide an abstraction boundary between plugin authors and Truthound internals.

For reporter plugins, the canonical 3.0 contract is reporter contract version 3:

- canonical input: `ValidationRunResult`
- shared projection: `RunPresentation`
- render entry point: `render(run_result, *, context)`

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

Truthound 3.0 standardizes plugin authoring around declarative and stable contracts:

- validation extensions should register `CheckSpecFactory` implementations
- reporter extensions should target `ValidationRunResult` and `RunPresentation`
- datasource extensions should register `DataAssetProvider`
- lifecycle hooks should only touch kernel contracts, not private registries

Legacy validator-class mutation patterns are no longer the recommended public path.

## Design Rationale

This platform mirrors the broader 3.0 architecture:

- stable contracts in the center
- adapters at the edge
- optional services instead of manager duplication

The formal decisions are captured in [ADR 002](../adr/002-plugin-platform.md) and the compatibility implications are captured in [ADR 004](../adr/004-migration-compatibility.md).
