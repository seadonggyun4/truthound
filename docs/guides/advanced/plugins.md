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
- Resource limits via Unix `resource` module (RLIMIT_AS, RLIMIT_CPU)
- Module blocking via `ImportBlocker` meta path hook
- Communication via pickle files (safe IPC)
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
- `--rm`: Auto-remove container
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

### Trust Store

```python
from truthound.plugins.security.signing import (
    TrustStoreImpl,
    TrustLevel,
)

# Create trust store
store = TrustStoreImpl(persist_path=Path("./trust_store.json"))

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

### HotReloadManager

```python
from truthound.plugins.lifecycle import (
    HotReloadManager,
    ReloadStrategy,
)

manager = HotReloadManager(lifecycle_manager)

# Reload with strategy
result = await manager.reload(
    "plugin-id",
    strategy=ReloadStrategy.GRACEFUL,
)

print(f"Success: {result.success}")
print(f"Reload time: {result.reload_time_ms:.0f}ms")
print(f"Previous version: {result.previous_version}")
print(f"New version: {result.new_version}")

# Watch for file changes
handle = await manager.watch(
    plugin_id="my-plugin",
    plugin_path=Path("./my-plugin"),
    auto_reload=True,
)

# Stop watching
handle.cancel()
```

**Reload Strategies:**

| Strategy | Description |
|----------|-------------|
| `GRACEFUL` | Wait for in-flight operations |
| `IMMEDIATE` | Stop and reload immediately |
| `ROLLING` | Incremental for multi-instance |

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
    default_security_policy=SecurityPolicy.STANDARD,
    require_signature=True,
    trust_store_path=Path("./trust_store.json"),

    # Hot Reload
    enable_hot_reload=True,
    watch_for_changes=True,

    # Versioning & Dependencies
    strict_version_check=True,
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

# Load a plugin
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
    default_security_policy=SecurityPolicy.STANDARD,
    require_signature=True,            # Require signed plugins
    trust_store_path=None,             # Trust store file

    # Hot Reload
    enable_hot_reload=True,            # Enable hot reload
    watch_for_changes=True,            # Watch file changes

    # Versioning & Dependencies
    strict_version_check=True,         # Strict version checking
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
