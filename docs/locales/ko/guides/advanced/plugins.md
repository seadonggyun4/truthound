# Plugin 아키텍처

실무 운영 가이드에서 Truthound을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 테이블 of Contents

1. [개요](#overview)
2. [보안 Sandbox](#security-sandbox)
3. 실무 운영 가이드에서 Plugin, Signing을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
4. 실무 운영 가이드에서 Version, Constraints을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
5. 실무 운영 가이드에서 Dependency, Management을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
6. 실무 운영 가이드에서 Plugin, Lifecycle을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
7. 실무 운영 가이드에서 Hot, Reload을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
8. 실무 운영 가이드에서 Enterprise, Plugin, Manager을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
9. [보안 Policies](#security-policies)
10. [설정 레퍼런스](#configuration-reference)

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 개요

실무 운영 가이드에서 `truthound.plugins`을(를) 다루는 항목입니다:

- 실무 운영 가이드에서 Security, Sandbox, NoOp, Process, Container을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Plugin, Signing, HMAC, RSA, Ed25519을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Version, Constraints, Semver을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Dependency, Management, Topological을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Plugin, Lifecycle, State을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Hot, Reload, File을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 `src/truthound/plugins/`, Location을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 보안 Sandbox

### Isolation Levels

| 실무 운영 가이드에서 Level을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Case을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-------|-------------|----------|
| 실무 운영 가이드에서 `NONE`, NONE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Trusted을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `PROCESS`, PROCESS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Subprocess을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Standard을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `CONTAINER`, CONTAINER을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Docker/Podman을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Maximum 보안 |

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

### NoOp Sandbox 엔진

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

### Process Sandbox 엔진

실무 운영 가이드에서 Subprocess을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 Process, Isolation, Features을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `resource`, Resource, Unix, RLIMIT_AS, RLIMIT_CPU을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `ImportBlocker`, Module, ImportBlocker을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Communication, IPC을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Automatic을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Container Sandbox 엔진

실무 운영 가이드에서 Docker/Podman을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

**Container 보안 Features:**
- 실무 운영 가이드에서 `--rm`, Auto-remove을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `--memory`, Memory을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `--cpus`, CPU을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `--network=none`, Network을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `--read-only`, Read-only을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `--security-opt no-new-privileges`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `--cap-drop ALL`, ALL, Drop을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 Supported, Algorithms을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| 실무 운영 가이드에서 Algorithm을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|------|-------------|
| 실무 운영 가이드에서 `SHA256`, SHA256을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Hash을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Simple을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `SHA512`, SHA512을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Hash을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Stronger을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `HMAC_SHA256`, HMAC_SHA256을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 시크릿-key | 실무 운영 가이드에서 Symmetric을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `HMAC_SHA512`, HMAC_SHA512을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 시크릿-key | 실무 운영 가이드에서 Stronger, HMAC을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `RSA_SHA256`, RSA_SHA256을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Asymmetric을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 RSA을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ED25519`, ED25519을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Asymmetric을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Modern을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

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

실무 운영 가이드에서 Trust, Levels을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| 실무 운영 가이드에서 Level을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-------|-------------|
| 실무 운영 가이드에서 `TRUSTED`, TRUSTED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Fully을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `VERIFIED`, VERIFIED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Verified을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `UNKNOWN`, UNKNOWN을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Unknown을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `REVOKED`, REVOKED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Certificate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Verification Chain

실무 운영 가이드에서 Chain, Responsibility을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 Verification, Steps을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

1. 실무 운영 가이드에서 IntegrityVerifier, Hash을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. 실무 운영 가이드에서 ExpirationVerifier, Signature/max을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
3. 실무 운영 가이드에서 SignatureVerifier, Cryptographic을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
4. 실무 운영 가이드에서 TrustVerifier, Signer을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
5. 실무 운영 가이드에서 ChainVerifier, Certificate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 Supported, Constraint, Formats을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| 실무 운영 가이드에서 Format을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------------|
| 실무 운영 가이드에서 `*`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Any을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `1.2.3`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Exact을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `>=1.2.3`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `>1.2.3`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Greater을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `<=1.2.3`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `<1.2.3`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Less을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `^1.2.3`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Compatible을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `~1.2.3`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Patch을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `>=1.0.0,<2.0.0`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Range을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 Dependency, Types을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------|-------------|
| 실무 운영 가이드에서 `REQUIRED`, REQUIRED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Must을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `OPTIONAL`, OPTIONAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Can을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `DEV`, DEV을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Development을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

| 실무 운영 가이드에서 Event을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-------|-------------|
| 실무 운영 가이드에서 `BEFORE_LOAD`, BEFORE_LOAD을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `AFTER_LOAD`, AFTER_LOAD을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `BEFORE_ACTIVATE`, BEFORE_ACTIVATE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `AFTER_ACTIVATE`, AFTER_ACTIVATE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `BEFORE_DEACTIVATE`, BEFORE_DEACTIVATE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `AFTER_DEACTIVATE`, AFTER_DEACTIVATE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `BEFORE_UNLOAD`, BEFORE_UNLOAD을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `AFTER_UNLOAD`, AFTER_UNLOAD을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ON_ERROR`, ON_ERROR을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ON_RELOAD`, ON_RELOAD을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 Reload, Strategies을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| 실무 운영 가이드에서 Strategy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-------------|
| 실무 운영 가이드에서 `GRACEFUL`, GRACEFUL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Wait을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `IMMEDIATE`, IMMEDIATE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Stop을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ROLLING`, ROLLING을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Incremental을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 보안 Policies

### Built-in Presets

| 실무 운영 가이드에서 Preset을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Isolation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Memory을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 CPU을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Time을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Network을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Signatures을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-----------|--------|-----|------|---------|-----------|
| 실무 운영 가이드에서 `DEVELOPMENT`, DEVELOPMENT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 NONE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `TESTING`, TESTING을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 NONE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `STANDARD`, STANDARD을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 PROCESS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ENTERPRISE`, ENTERPRISE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 PROCESS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `STRICT`, STRICT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 CONTAINER을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `AIRGAPPED`, AIRGAPPED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 CONTAINER을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

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

### Custom 보안 Policy

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 설정 레퍼런스

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Design Patterns

| 실무 운영 가이드에서 Pattern을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Usage을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Location을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|---------|-------|----------|
| 실무 운영 가이드에서 Strategy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Sandbox을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Sandbox을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Chain, Responsibility을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Signature을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Verifier을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Factory을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Sandbox 엔진 creation | 실무 운영 가이드에서 SandboxFactory을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Fluent, Builder을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Verification을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 VerificationChainBuilder을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Protocol-First을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Singleton을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 엔진 instances | 실무 운영 가이드에서 SandboxFactory을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 State, Machine을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Plugin을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 LifecycleManager을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Thread Safety

- 실무 운영 가이드에서 `LifecycleManager`, `asyncio.Lock()`, LifecycleManager, Lock을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `EnterprisePluginManager`, `asyncio.Lock()`, EnterprisePluginManager, Lock을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `@dataclass(frozen=True)`, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Process/Container을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 함께 보기

- 실무 운영 가이드에서 Custom, Validators, Creating을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [보안](../ci-cd/index.md) - CI/CD 보안 통합
- 실무 운영 가이드에서 CLI, Extensions을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
