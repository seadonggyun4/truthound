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

## Approved resident MCP administration

Development Core profiles can use a pinned owner with explicit plugin class, custom
hook, provider, reporter, check-factory and capability aliases. The owner calls the
same `PluginManager` lifecycle and extension ports; it does not accept Python or
rewrite other workers' binding files. All reads and effects require separate operator
approval. See the [MCP contract](../guides/mcp.md#bound-core-security-and-plugin-administration).


## Manager-local native execution (development source)

`PluginManager.validate(check_factories, data=None, asset_provider=None, name="plugin-suite", result_format="summary", parallel=False, max_workers=None, pushdown=None)` compiles registered `CheckSpecFactory` objects (or legacy native validator registrations) into a canonical kernel suite and executes it against data or one registered native `DataAssetProvider`. It returns a `PluginValidationResult` with `.run`, detailed `.hooks` and `.hooks_failed`. This creates no implicit context/workspace persistence. `validate_async` awaits native lifecycle hooks, including decorated async handlers, while retaining the calling thread for native provider and kernel work. The synchronous method rejects a running event loop before effects.

`render(reporter_name, run)` and `render_async` consume a canonical `ValidationRunResult` through the selected registered v3 reporter class/constructor/configured object and return `PluginRenderResult.content` (text or bytes) with detailed hook outcomes. Structural v3 reporters are supported; inheritance from `ValidationReporter` is optional. Configured reporter instances remain local to the manager, while constructors may still register with the global reporter factory.

`HookManager.trigger_detailed` and `trigger_detailed_async` return `HookExecutionResult` records with hook type, source, success, result and optional error type. A successful None is distinct from a failure; exception bodies are not logged or put in this record. Legacy trigger methods keep result-list compatibility. Detailed sync calls reject awaitable handlers rather than leaking coroutines; async calls await actual returned awaitables. An optional trusted `set_execution_guard` checks before each hook. Manager workflow guards also check before provider/factory/report operations. Denied unload hooks cannot veto owned resource teardown/removal.

Validation hooks receive immutable `CheckSpec` intent as `checks` and `validators`; issue hooks receive their corresponding check intent. Hook failures are visible and do not veto a validation result. Providers own asset connections and final shutdown. Explicit [Core MCP plugin contracts](../guides/mcp.md#bound-core-security-and-plugin-administration) run these native workflows in one approved owner, with canonical result export, bounded artifacts and input authorization lifetimes. This adds no propagation into other worker registries, arbitrary imports or remote service verification.


`unregister_check_factory`, `unregister_data_asset_provider`, `unregister_reporter` and `unregister_capability` remove this manager's stable port lookups, without guessing resource ownership or closing shared providers. Reporter constructors are removed from the global factory only if their identity still matches this manager's registration; `truthound.reporters.unregister_reporter(name, expected_constructor=...)` offers the same atomic ownership check. A configured reporter instance stays manager-local. Invalid reporter contract metadata is rejected before replacing an existing binding.

`unload_plugin(name,detailed=True)` returns a `PluginCleanupResult` containing deactivation, teardown completion, registry removal, unload-hook failure and exception types. `.success` means those native callbacks completed, not that an external provider acknowledged cleanup. The default retains None compatibility; guard denial still propagates after cleanup. Unregister/teardown exception bodies are not logged, and teardown is attempted even after an unload hook is denied. [Core MCP](../guides/mcp.md#bound-core-security-and-plugin-administration) preserves failed cleanup receipts rather than treating registry removal as full cleanup success.
