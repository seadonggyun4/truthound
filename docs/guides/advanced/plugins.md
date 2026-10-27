# Plugin Architecture

Truthound provides an enterprise-grade plugin system with security sandbox, code signing, version management, and hot reload capabilities.

---

## Table of Contents

1. [Overview](#overview)
2. [Security Sandbox](#security-sandbox)
3. [Plugin Signing](#plugin-signing)
4. [Version Constraints](#version-constraints)
5. [Dependency Management](#dependency-management)
6. [Plugin Lifecycle](#plugin-lifecycle)
7. [Hot Reload](#hot-reload)
8. [Enterprise Plugin Manager](#enterprise-plugin-manager)
9. [Security Policies](#security-policies)
10. [Configuration Reference](#configuration-reference)

---

## Overview

The plugin module (`truthound.plugins`) provides:

- **Security Sandbox**: NoOp, Process, Container isolation levels
- **Plugin Signing**: HMAC, RSA, Ed25519 algorithms
- **Version Constraints**: Semver support (^, ~, >=, <, ranges)
- **Dependency Management**: Topological sort, cycle detection
- **Plugin Lifecycle**: State machine with 11 states
- **Hot Reload**: File watching with graceful reload and rollback

**Location**: `src/truthound/plugins/`

```
plugins/
├── __init__.py
├── enterprise_manager.py     # Unified facade
├── security/
│   ├── protocols.py          # Core protocols
│   ├── policies.py           # Security policy presets
│   ├── exceptions.py         # Exception hierarchy
│   ├── sandbox/
│   │   ├── factory.py        # SandboxFactory
│   │   ├── context.py        # SandboxContext
│   │   └── engines/
│   │       ├── noop.py       # NoOp engine
│   │       ├── process.py    # Process isolation
│   │       └── container.py  # Container isolation
│   └── signing/
│       ├── service.py        # SigningService
│       ├── trust_store.py    # TrustStore
│       └── verifier.py       # Verification chain
├── versioning/
│   └── constraints.py        # Version constraints
├── dependencies/
│   ├── graph.py              # Dependency graph
│   └── resolver.py           # Dependency resolver
├── lifecycle/
│   ├── manager.py            # Lifecycle manager
│   └── hot_reload.py         # Hot reload manager
└── docs/                     # Plugin documentation
```

---

## Security Sandbox

### Isolation Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| `NONE` | No isolation, only timeout | Trusted plugins, development |
| `PROCESS` | Subprocess with resource limits | Standard plugins |
| `CONTAINER` | Docker/Podman container | Maximum security |

### SandboxFactory

```python
from truthound.plugins.security.sandbox import (
    SandboxFactory,
    IsolationLevel,
)

# Create engine by isolation level (class method)
engine = SandboxFactory.create(IsolationLevel.PROCESS, singleton=True)

# Get best available engine (with fallback)
engine = SandboxFactory.get_best_available(
    preferred=IsolationLevel.CONTAINER,
    # Falls back: CONTAINER → PROCESS → NONE
)

# Check if isolation level is available
if SandboxFactory.is_available(IsolationLevel.CONTAINER):
    print("Container isolation available")

# List all available isolation levels
available = SandboxFactory.list_available()
print(f"Available levels: {[l.name for l in available]}")

# Register custom engine
SandboxFactory.register(IsolationLevel.WASM, MyCustomWasmEngine)

# Unregister an engine
SandboxFactory.unregister(IsolationLevel.WASM)

# Cleanup all cached engine instances
await SandboxFactory.cleanup_all()

# Reset factory to initial state (for testing)
SandboxFactory.reset()
```

### NoOp Sandbox Engine

No process isolation, only timeout enforcement.

```python
from truthound.plugins.security.sandbox.engines import NoopSandboxEngine
from truthound.plugins.security.protocols import SecurityPolicy

engine = NoopSandboxEngine()

# Create sandbox context
policy = SecurityPolicy(isolation_level=IsolationLevel.NONE)
context = engine.create_sandbox("my-plugin", policy)

# Execute function in sandbox
async def my_func(x, y):
    return x + y

result = await engine.execute(context, my_func, 1, 2)
print(result)  # 3

# Cleanup
await engine.cleanup()
```

### Process Sandbox Engine

Subprocess isolation with resource limits.

```python
from truthound.plugins.security.sandbox.engines import ProcessSandboxEngine
from truthound.plugins.security.protocols import SecurityPolicy, ResourceLimits

engine = ProcessSandboxEngine()

# Create policy with resource limits
policy = SecurityPolicy(
    isolation_level=IsolationLevel.PROCESS,
    resource_limits=ResourceLimits(
        max_memory_mb=512,
        max_cpu_percent=50.0,
        max_execution_time_sec=30.0,
        max_file_descriptors=100,
    ),
    blocked_modules=("os", "subprocess", "socket", "pickle"),
)

context = engine.create_sandbox("my-plugin", policy)

# Execute in isolated subprocess
result = await engine.execute(context, my_func, args)

# Terminate if needed
engine.terminate(context)
```

**Process Isolation Features:**
- Supported Unix limits plus parent-observed RSS and wall deadlines
- Module blocking via `ImportBlocker` meta path hook
- Trusted local argument transport; bounded data-only child results
- Automatic cleanup of temp directories

### Container Sandbox Engine

Docker/Podman container isolation.

```python
from truthound.plugins.security.sandbox.engines import ContainerSandboxEngine

engine = ContainerSandboxEngine()

# Auto-detects Docker/Podman
print(engine._runtime)  # "docker" or "podman"

policy = SecurityPolicy(
    isolation_level=IsolationLevel.CONTAINER,
    resource_limits=ResourceLimits(
        max_memory_mb=256,
        max_cpu_percent=50.0,
    ),
    allow_network=False,
    allow_file_write=False,
)

context = engine.create_sandbox("my-plugin", policy)
result = await engine.execute(context, my_func, args)
```

**Container Security Features:**
- Owned container ID: awaited removal after execution/cancellation
- `--memory`: Memory limit
- `--cpus`: CPU limit
- `--network=none`: Network isolation
- `--read-only`: Read-only filesystem
- `--security-opt no-new-privileges`
- `--cap-drop ALL`: Drop all capabilities

---

## Plugin Signing

### SigningService

```python
from truthound.plugins.security.signing import (
    SigningServiceImpl,
    SignatureAlgorithm,
)

# Create signing service
service = SigningServiceImpl(
    algorithm=SignatureAlgorithm.RSA_SHA256,
    signer_id="my-org",
    validity_days=365,
    verification_keys={"my-org": public_key_bytes},
)

# Sign a plugin
signature = service.sign(
    plugin_path=Path("./my-plugin"),
    private_key=private_key_bytes,
    certificate=cert_bytes,        # Optional
    metadata={"version": "1.0.0"},
)

print(f"Signature: {signature.signature.hex()[:32]}...")
print(f"Expires: {signature.expires_at}")
print(f"Plugin hash: {signature.metadata['plugin_hash']}")

# Verify signature
result = service.verify(
    plugin_path=Path("./my-plugin"),
    signature=signature,
)

print(f"Valid: {result.is_valid}")
print(f"Trust level: {result.trust_level}")
print(f"Warnings: {result.warnings}")
```

**Supported Algorithms:**

| Algorithm | Type | Description |
|-----------|------|-------------|
| `SHA256` | Hash | Simple integrity check |
| `SHA512` | Hash | Stronger integrity |
| `HMAC_SHA256` | Secret-key | Symmetric authentication |
| `HMAC_SHA512` | Secret-key | Stronger HMAC |
| `RSA_SHA256` | Asymmetric | RSA signing |
| `ED25519` | Asymmetric | Modern elliptic curve |

### Authenticated signatures and lifecycle admission

`SigningServiceImpl(verification_keys={signer_id: public_pem},
verification_secrets={signer_id: hmac_secret})` checks the actual signature bytes
against the signed content hash, algorithm, signer identity and timestamps.
Missing keys, changed content or invalid signatures fail verification. SHA256
and SHA512 are unkeyed integrity checks, not signer authentication; their
successful result includes this warning. `plugin_hash` is reserved metadata.
Verification keys come from the operator, never from the submitted signature.
Only Python files are covered when signing a directory; separately loaded data
and native dependencies are outside this digest.

`create_verification_chain(..., require_trusted_signer=True)` requires an
operator-key-authenticated signer whose trust-store level is `TRUSTED`.
A hash-only signature, unknown signer or revoked signer fails. Certificate
metadata does not replace the configured verification key or this trust check.

For lifecycle admission, configure `EnterprisePluginManagerConfig` with
`require_signature=True` and `verification_keys` or `verification_secrets`.
Use `manager.signature_trust_store.set_signer_trust(signer_id, TrustLevel.TRUSTED)`
and `manager.register_signed_plugin(name, plugin_class, source_path, signature)`.
The manager verifies before load/setup and activation, including synchronous
`load_plugin`, `load_from_class` and asynchronous `load`/`load_all` paths.
Explicit `load(..., verify_signature=True)` requires the binding even when the
global option is false. Bound signatures are always checked. A changed source,
replaced class binding or revoked signer prevents subsequent admission;
unloading remains available. Bulk loading raises an `ExceptionGroup` on failures
and retains successful loads for inspection and cleanup.

Classes are already imported when registered: this is a lifecycle admission
check, not an import sandbox or proof that previously imported bytecode matches
the current file. Discovery and imports remain operator-trusted operations.

Trust-store format 2 atomically persists public certificate bytes, trust levels
and revocation metadata. Reopening validates the whole document before adopting
any trust. Malformed or oversized documents fail; legacy signer-only files remain
readable, while legacy certificate entries without bytes require operator reimport.
The file is single-writer; publication failures are raised and retain prior in-memory
state. If directory synchronization fails after replacement, reopen to inspect the
possibly published state. Keys and HMAC secrets are never written to this file.

The shared manager uses synchronous lifecycle ports for automatic loading and
shutdown, including the enterprise facade's synchronous context manager. Bulk
load/unload attempts every selected plugin and raises grouped failures; successful
loads remain registered and cleanup failures are not reported as success.

### Trust Store

```python
from truthound.plugins.security.signing import (
    TrustStoreImpl,
    TrustLevel,
)

# Create trust store
store = TrustStoreImpl(store_path=Path("./trust_store.json"))

# Add trusted certificate
store.add_trusted_certificate(
    cert=certificate_bytes,
    trust_level=TrustLevel.TRUSTED,
    metadata={"org": "my-org"},
)

# Check if certificate is trusted
is_trusted, trust_level = store.is_trusted(certificate_bytes)
print(f"Trusted: {is_trusted}, Level: {trust_level}")

# Revoke certificate
store.revoke_certificate(
    cert_id="cert-hash",
    reason="Key compromised",
)

# Get trust level for signer
level = store.get_trust_level("my-org")
```

**Trust Levels:**

| Level | Description |
|-------|-------------|
| `TRUSTED` | Fully trusted |
| `VERIFIED` | Verified but not fully trusted |
| `UNKNOWN` | Unknown signer |
| `REVOKED` | Certificate revoked |

### Verification Chain

Chain of Responsibility pattern for multi-step verification.

```python
from truthound.plugins.security.signing.verifier import (
    VerificationChainBuilder,
)

# Build verification chain
chain = (
    VerificationChainBuilder()
    .with_integrity_check()           # Hash comparison
    .with_expiration_check(max_age_days=90)
    .with_signature_check()           # Cryptographic verification
    .with_trust_check(trust_store)    # Signer trust level
    .with_chain_check(trust_store)    # Certificate chain
    .build()
)

# Verify plugin
result = chain.verify(
    plugin_path=Path("./my-plugin"),
    signature=signature,
    context={},
)

print(f"Valid: {result.is_valid}")
print(f"Errors: {result.errors}")
```

**Verification Steps:**

1. **IntegrityVerifier**: Hash comparison (tampering detection)
2. **ExpirationVerifier**: Signature/max age check
3. **SignatureVerifier**: Cryptographic structure check
4. **TrustVerifier**: Signer trust level lookup
5. **ChainVerifier**: Certificate chain root validation

---

## Version Constraints

### VersionConstraint

```python
from truthound.plugins.versioning import (
    VersionConstraint,
    parse_constraint,
)

# Factory methods
any_version = VersionConstraint.any_version()
exact = VersionConstraint.exact("1.2.3")
at_least = VersionConstraint.at_least("1.0.0")
compatible = VersionConstraint.compatible_with("1.2.3")  # ^1.2.3

# Parse string constraints
constraint = parse_constraint("^1.2.3")
constraint = parse_constraint(">=1.0.0,<2.0.0")
constraint = parse_constraint("~1.2.3")

# Check satisfaction
print(constraint.is_satisfied_by("1.2.5"))  # True
print(constraint.is_satisfied_by("2.0.0"))  # False
```

**Supported Constraint Formats:**

| Format | Description |
|--------|-------------|
| `*` | Any version |
| `1.2.3` | Exact match |
| `>=1.2.3` | At least |
| `>1.2.3` | Greater than |
| `<=1.2.3` | At most |
| `<1.2.3` | Less than |
| `^1.2.3` | Compatible (same major) |
| `~1.2.3` | Patch updates only |
| `>=1.0.0,<2.0.0` | Range |

---

## Dependency Management

### Dependency Graph

```python
from truthound.plugins.dependencies import (
    DependencyGraph,
    DependencyNode,
    DependencyType,
)

graph = DependencyGraph()

# Add nodes
graph.add_node(
    plugin_id="plugin-a",
    version="1.0.0",
    dependencies={
        "plugin-b": DependencyType.REQUIRED,
        "plugin-c": DependencyType.OPTIONAL,
    },
    metadata={"description": "Plugin A"},
)

graph.add_node(
    plugin_id="plugin-b",
    version="2.0.0",
    dependencies={},
)

# Detect cycles
cycles = graph.detect_cycles()
if cycles:
    print(f"Cycles detected: {cycles}")

# Get load order (topological sort)
load_order = graph.get_load_order()
print(f"Load order: {load_order}")  # ['plugin-b', 'plugin-a']

# Get unload order (reverse)
unload_order = graph.get_unload_order()

# Get dependencies
deps = graph.get_dependencies("plugin-a", recursive=True, include_optional=True)

# Get dependents (reverse dependencies)
dependents = graph.get_dependents("plugin-b", recursive=True)

# Validate graph
errors = graph.validate()
```

**Dependency Types:**

| Type | Description |
|------|-------------|
| `REQUIRED` | Must be installed |
| `OPTIONAL` | Can be missing |
| `DEV` | Development only |

### Dependency Resolver

```python
from truthound.plugins.dependencies import (
    DependencyResolver,
    ResolutionResult,
)

resolver = DependencyResolver(
    strict=True,                  # Fail on missing required
    allow_missing_optional=True,  # Allow missing optional
)

# Resolve dependencies
result = resolver.resolve(plugin_infos)

print(f"Success: {result.success}")
print(f"Load order: {result.load_order}")
print(f"Conflicts: {result.conflicts}")
print(f"Missing: {result.missing}")
print(f"Warnings: {result.warnings}")

# Check if plugin can be loaded
can_load, missing = resolver.can_load("plugin-a", graph, loaded_plugins)

# Get install order (with dependencies)
install_order = resolver.get_install_order(["plugin-a"], graph)

# Get uninstall order (with dependents)
uninstall_order = resolver.get_uninstall_order(["plugin-b"], graph, force=False)
```

---

## Plugin Lifecycle

### Lifecycle States

```
DISCOVERED → LOADING → LOADED → ACTIVATING → ACTIVE
                              ↘ UNLOADING → UNLOADED
                   DEACTIVATING ↓ INACTIVE ↙
                   ERROR ↔ (any state for recovery)
```

### LifecycleManager

```python
from truthound.plugins.lifecycle import (
    LifecycleManager,
    LifecycleState,
    LifecycleEvent,
)

manager = LifecycleManager()

# Register event handlers
async def on_load(plugin, event, data):
    print(f"Plugin {plugin.id} loaded")

manager.on(LifecycleEvent.AFTER_LOAD, on_load)

# Transition plugin state
success = await manager.transition(
    plugin,
    to_state=LifecycleState.ACTIVE,
    metadata={"reason": "Manual activation"},
)

# Get current state
state = manager.get_state("plugin-id")

# Get transition history
history = manager.get_history("plugin-id", limit=10)
for transition in history:
    print(f"{transition.from_state} → {transition.to_state} at {transition.timestamp}")
```

### Lifecycle Events

| Event | Description |
|-------|-------------|
| `BEFORE_LOAD` | Before plugin loads |
| `AFTER_LOAD` | After plugin loads |
| `BEFORE_ACTIVATE` | Before activation |
| `AFTER_ACTIVATE` | After activation |
| `BEFORE_DEACTIVATE` | Before deactivation |
| `AFTER_DEACTIVATE` | After deactivation |
| `BEFORE_UNLOAD` | Before unload |
| `AFTER_UNLOAD` | After unload |
| `ON_ERROR` | On error |
| `ON_RELOAD` | On hot reload |

---

## Hot Reload

`HotReloadManager` uses the shared `PluginManager`; the old separate lifecycle
state manager is not a reload runtime. Its operator-provided loader returns a
fresh `Plugin` class (or an awaitable of that class). Loading new Python code is
the factory's responsibility, not an implicit `importlib.reload` of shared modules.

```python
from truthound.plugins.manager import PluginManager
from truthound.plugins.lifecycle import HotReloadManager, ReloadStrategy

plugins = PluginManager()
plugins.load_from_class(OriginalPlugin, activate=False)
reloads = HotReloadManager(plugins, plugin_loader=lambda name: ReplacementPlugin)
async with reloads.operation("my-plugin") as plugin:
    result = plugin.run()  # Operator-defined plugin work
outcome = await reloads.reload("my-plugin", ReloadStrategy.GRACEFUL)
print(outcome.success, outcome.duration_ms, outcome.rolled_back, outcome.error)
await reloads.aclose()
plugins.shutdown()
```

`GRACEFUL` waits for work wrapped in `operation(name)` up to `timeout` (30 seconds
by default), rejects new leases during reload, then observes old teardown and new
setup/activation. `IMMEDIATE` skips lease draining; references already held by
callers are not interrupted. Synchronous factories and callbacks cannot be forcibly
preempted. A single registry does not implement multi-instance `ROLLING`; requesting
it returns a failed result before mutation.

Failed candidate setup invokes teardown. A clean failure can restore a fresh
instance of the previous class with its configuration and activation state;
`rolled_back=True` means that reload actually succeeded. It does not restore prior
in-memory state or undo external effects. Unconfirmed cleanup prevents automatic
restoration. Errors are reported by type in `ReloadResult.error`; failed automatic
reloads stop the watcher and are also raised on `aclose()`.

File watching hashes contents, observes creation/change/deletion and awaits sync
or async callbacks in order. Duplicate watched paths/plugins are rejected. Use
`await aclose()` to cancel and join owned watchers; `cancel()`/`stop_all_watches()`
request cancellation only. Python directory watches include `*.py` files.

With `EnterprisePluginManagerConfig(enable_hot_reload=True)`, call
`await manager.configure_hot_reload(name, loader, path=source_path)`; automatic
watching starts only when `watch_for_changes=True` and a path is explicitly given.
Then use `await manager.reload(name)` and `await manager.unload_all()` for joined
cleanup. Synchronous shutdown with active watchers raises rather than abandoning
them. After joined shutdown, create a new manager for further reloads. Signed
class bindings require explicit unload, fresh signature registration and load;
automatic reload does not bypass signature renewal.

---

## Enterprise Plugin Manager

### Unified Facade

```python
from truthound.plugins import (
    EnterprisePluginManager,
    EnterprisePluginManagerConfig,
)
from truthound.plugins.security import SecurityPolicy

config = EnterprisePluginManagerConfig(
    # Plugin discovery
    plugin_dirs=[Path("./plugins")],
    scan_entrypoints=True,
    auto_load=False,
    auto_activate=True,

    # Security
    default_security_policy=SecurityPolicy.standard(),
    require_signature=True,
    trust_store_path=Path("./trust_store.json"),
    verification_keys={"my-org": public_key_bytes},

    # Hot Reload
    enable_hot_reload=True,
    watch_for_changes=True,

    # Versioning & Dependencies
    check_compatibility=True,
    strict_dependencies=True,
    allow_missing_optional=True,
)

manager = EnterprisePluginManager(config)

# Discover plugins (synchronous)
discovered = manager.discover_plugins()

# List available plugins
plugins = manager.list_plugins()
for plugin in plugins:
    print(f"{plugin.id}: {plugin.version} ({plugin.state})")

# Register operator-authenticated source before a required-signature load
manager.register_signed_plugin("my-plugin", MyPlugin, source_path, signature)
manager.signature_trust_store.set_signer_trust("my-org", TrustLevel.TRUSTED)
await manager.load("my-plugin")

# Activate is included in load() by default, or use:
await manager.activate("my-plugin")

# Get loaded plugin
plugin = manager.get_plugin("my-plugin")

# Deactivate and unload
await manager.deactivate("my-plugin")
await manager.unload("my-plugin")
```

---

## Security Policies

### Built-in Presets

| Preset | Isolation | Memory | CPU | Time | Network | Signatures |
|--------|-----------|--------|-----|------|---------|-----------|
| `DEVELOPMENT` | NONE | 4GB | 100% | 600s | Yes | 0 |
| `TESTING` | NONE | 2GB | 100% | 120s | Yes | 0 |
| `STANDARD` | PROCESS | 512MB | 50% | 30s | No | 1 |
| `ENTERPRISE` | PROCESS | 1GB | 80% | 60s | No | 1 |
| `STRICT` | CONTAINER | 128MB | 25% | 10s | No | 2 |
| `AIRGAPPED` | CONTAINER | 256MB | 50% | 30s | No | 2 |

```python
from truthound.plugins.security.policies import (
    DEVELOPMENT,
    TESTING,
    STANDARD,
    ENTERPRISE,
    STRICT,
    AIRGAPPED,
    create_policy,
)

# Use preset
policy = STANDARD

# Create custom policy
policy = create_policy(
    base=STANDARD,
    max_memory_mb=1024,
    allow_network=True,
)
```

### Custom Security Policy

```python
from truthound.plugins.security.protocols import (
    SecurityPolicy,
    ResourceLimits,
    IsolationLevel,
)

policy = SecurityPolicy(
    isolation_level=IsolationLevel.PROCESS,
    resource_limits=ResourceLimits(
        max_memory_mb=512,
        max_cpu_percent=50.0,
        max_execution_time_sec=30.0,
        max_file_descriptors=100,
        allowed_paths=("/data/input",),
        writable_paths=("/data/output",),
        denied_syscalls=("fork", "vfork", "clone"),
    ),
    allow_network=False,
    allow_subprocess=False,
    allow_file_write=False,
    allowed_modules=("polars", "numpy", "pandas"),
    blocked_modules=("os", "subprocess", "socket", "pickle", "ctypes"),
    required_signatures=1,
    require_trusted_signer=True,
    signature_max_age_days=365,
)
```

---

## Configuration Reference

### EnterprisePluginManagerConfig

```python
from truthound.plugins import EnterprisePluginManagerConfig

config = EnterprisePluginManagerConfig(
    # Plugin discovery
    plugin_dirs=[Path("./plugins")],   # Plugin directories
    scan_entrypoints=True,             # Scan Python entry points
    auto_load=False,                   # Auto-load on discovery
    auto_activate=True,                # Auto-activate on load

    # Security
    default_security_policy=SecurityPolicy.standard(),
    require_signature=True,            # Require signed plugins
    trust_store_path=None,             # Trust store file

    # Hot Reload
    enable_hot_reload=True,            # Enable hot reload
    watch_for_changes=True,            # Watch file changes

    # Versioning & Dependencies
    check_compatibility=True,         # Strict version checking
    strict_dependencies=True,          # Fail on missing required
    allow_missing_optional=True,       # Allow missing optional
)
```

### ResourceLimits

```python
from truthound.plugins.security.protocols import ResourceLimits

limits = ResourceLimits(
    max_memory_mb=512,                 # Memory limit (MB)
    max_cpu_percent=50.0,              # CPU limit (%)
    max_execution_time_sec=30.0,       # Execution timeout (s)
    max_file_descriptors=100,          # File descriptor limit
    allowed_paths=(),                  # Allowed read paths
    writable_paths=(),                 # Allowed write paths
    denied_syscalls=("fork", "vfork"), # Blocked syscalls
)

# Preset limits
minimal = ResourceLimits.minimal()     # 128MB, 25%, 10s
standard = ResourceLimits.standard()   # 512MB, 50%, 30s
generous = ResourceLimits.generous()   # 2048MB, 100%, 300s
```

---

## Exception Hierarchy

```python
from truthound.plugins.security.exceptions import (
    SecurityError,
    SandboxError,
    SandboxTimeoutError,
    SandboxResourceError,
    SandboxSecurityViolation,
    SignatureError,
    SignatureExpiredError,
    SignatureTamperError,
    UntrustedSignerError,
    InvalidSignatureError,
    CertificateError,
    CertificateExpiredError,
    CertificateRevokedError,
    CertificateNotFoundError,
    InvalidCertificateError,
)

try:
    result = await engine.execute(context, func, args)
except SandboxTimeoutError as e:
    print(f"Timeout: {e.timeout_seconds}s, ran for {e.execution_time}s")
except SandboxResourceError as e:
    print(f"Resource limit: {e.resource_type}, limit={e.limit}, actual={e.actual}")
except SandboxSecurityViolation as e:
    print(f"Security violation: {e.violation_type}, action={e.attempted_action}")
```

---

## Design Patterns

| Pattern | Usage | Location |
|---------|-------|----------|
| **Strategy** | Sandbox engines, reload strategies | Sandbox factory |
| **Chain of Responsibility** | Signature verification | Verifier chain |
| **Factory** | Sandbox engine creation | SandboxFactory |
| **Fluent Builder** | Verification chain | VerificationChainBuilder |
| **Protocol-First** | All abstractions | protocols.py |
| **Singleton** | Engine instances | SandboxFactory |
| **State Machine** | Plugin lifecycle | LifecycleManager |

---

## Thread Safety

- `LifecycleManager` uses `asyncio.Lock()` for state changes
- `EnterprisePluginManager` uses `asyncio.Lock()` for plugin operations
- All dataclasses are frozen (`@dataclass(frozen=True)`)
- Process/Container engines provide complete isolation

---

## See Also

- [Custom Validators](../validators/index.md) - Creating custom validators
- [Security](../ci-cd/index.md) - CI/CD security integration
- [CLI Extensions](../configuration/index.md) - CLI plugin system

### Dependency admission and shutdown

Plugin dependencies must be loaded before a dependent plugin can run setup; discovery alone is insufficient. `load_all()` resolves a deterministic dependency order and rejects cycles or missing required plugins. `PluginInfo.dependencies` and `optional_dependencies` accept installed plugin names with PEP 440 constraints; URLs and distribution extras are not plugin names. Optional missing plugins are permitted only when `allow_missing_optional=True`. Present optional plugins must satisfy version constraints. `python_dependencies` uses installed distribution metadata and PEP 508 requirements, including markers and selected extras’ immediate dependencies, without importing those packages. It does not install packages or prove they are importable.

`strict_version_check=True` rejects unmet versions and unparseable host compatibility. Setting it to false explicitly relaxes version checks, while dependency existence still applies. `check_compatibility=False` disables host compatibility admission; `strict_dependencies=False` disables dependency admission. Manager configuration is copied at construction.

The selected MCP plugin owner supports an approved `shutdown` command. Success is acknowledged only after native plugin teardown, reload-owner closure and factory exit. Failed cleanup receipts remain failures, other owned plugins are still attempted, and uncertain shutdown is not automatically replayed. Expiry/revocation cleanup remains bounded and cannot be used as evidence of acknowledged native shutdown.

### Discovery ownership and failures

Discovery executes operator-selected module code and metadata constructors. It does not sandbox imports. File discovery uses an owned namespace, supports package initializers and relative Python imports, retains module/source identity for inspection, and reads source rather than stale timestamp-based bytecode on explicit rediscovery. A failed package initializer is not retried for each child during the same discovery. Conflicting plugin names, missing directories, import failures and invalid metadata produce an `ExceptionGroup` by default; completed results remain in `discovered_plugins`, and `failures` contains source identifiers and error types without exception bodies.

`PluginDiscovery(strict=False)` (or manager `strict_discovery=False`) explicitly permits partial results while retaining diagnostics. Use a `PluginDiscovery` context manager or call `close()` after all discovered classes finish use; closing removes its owned import finder and modules. Manager shutdown closes its discovery owner after plugin cleanup. Explicit rediscovery creates a separate module generation; it does not replace already loaded plugin instances. Symlink sources and relative imports escaping the selected root are rejected.

### Process worker boundaries

Process workers accept trusted local callables and arguments; input pickle is never accepted from a wire or untrusted file. Child results use Core's closed, data-only scalar/container codec, with a default 16 MiB encoded-result budget (`max_result_bytes`). Bytes, dates and tuples retain their types; arbitrary objects fail. The parent never unpickles a child result. Exception bodies, stdout and stderr are not returned. A context belongs to one engine and one execution. Cancellation and cleanup kill and reap the owned worker; cleanup also joins pending process creation and memory-observation tasks.

Wall deadlines are finite and positive. Unix CPU-time, descriptor and output-file limits apply where supported. Address-space limits apply where available; a parent RSS observer also enforces the selected memory budget, including platforms that reject `RLIMIT_AS`. RSS checks are sampled and do not promise a hard peak-memory ceiling or descendant accounting. Zero memory/descriptor limits disable those limits; negative values fail. CPU percentage, filesystem allowlists, network/subprocess restrictions and syscall policies are not enforced by this process worker. Python import filtering is advisory and cannot isolate hostile code. Enterprise lifecycle registration remains in the shared manager and does not automatically run inside a process/container worker.

Container workers also use the bounded data-only result codec. Callable and argument pickle remains a trusted local input; selected images must contain their imports. Images must be installed before execution (`--pull=never`). The engine creates a container, retains its immutable ID, and removes that owned container after execution or cancellation. `terminate()` requests cancellation; await execution or `cleanup()` to observe removal. Removal failures remain visible and owned IDs are retained for retry. Container policy currently applies memory/CPU cgroups, network mode, read-only root and capability restrictions; `allowed_paths` and `writable_paths` grant explicit host mounts. Linux x86-64/AArch64 workers install a seccomp deny set before loading the selected callable. Supported names are `fork`, `vfork`, `clone`, `clone3`, `execve`, `execveat`, and `getpid`; unknown names or architectures fail closed. Denying `clone` also denies `clone3`; AArch64 fork/vfork map to clone. `allow_subprocess=False` adds all process-creation/exec denials, also blocking thread creation. To permit subprocesses, explicitly set `allow_subprocess=True` and remove the corresponding entries from `denied_syscalls`. The IPC workspace is writable even with a read-only root.

The no-isolation engine accepts only trusted in-process work. Its deadline stops waiting for a synchronous result but cannot kill a Python thread. Timed-out or cancelled threads remain owned until completion, and `cleanup()` waits for them; a permanently blocked function can therefore block cleanup. Async work is cancelled and joined. Failed operations finish their execution context, and another engine cannot execute a foreign context.

Sandbox factory registration preserves other built-in engine registrations. Replacing, unregistering or resetting a cached engine retains its cleanup ownership. `cleanup_all()` attempts all current and retired owners, reports aggregated failures, and retains failed owners for retry. Non-singleton engines remain the caller’s cleanup responsibility. Factory availability means an engine class is registered; it does not prove that Docker or another external runtime is running.

A discovered key, including an entry-point name, must match the instantiated plugin’s declared name. The manager rejects mismatches before setup or registry mutation so dependencies, activation and cleanup use one identity.

If container creation loses its acknowledgement, the engine reconciles only its generated name and ownership label before removing the immutable container ID. A runtime lookup failure remains an unknown outcome; pending ownership is retained for cleanup retry. This reconciliation lasts for the engine lifetime and is not a durable recovery journal after parent-process loss.

Host grants must be existing canonical absolute regular-file/directory paths; symbolic-link aliases, reserved `/workspace`/`proc`/`sys`/`dev` destinations, the host root, and ambiguous mount delimiters are rejected. Paths appear at the same absolute location in the container. Read grants are mounted read-only; explicit writable grants require `allow_file_write=True` and override a read grant at the same path. No host directory is mounted implicitly beyond the engine-owned IPC workspace. Image files remain visible: these grants are not a deny-all allowlist for the image filesystem. Host mutation racing mount creation is outside this trusted-local-callable contract; use a local, operator-selected runtime and immutable input snapshots when required.
