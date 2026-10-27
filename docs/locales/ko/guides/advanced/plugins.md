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

실무 운영 가이드에서 Supported, Algorithms을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| 실무 운영 가이드에서 Algorithm을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|------|-------------|
| 실무 운영 가이드에서 `SHA256`, SHA256을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Hash을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Simple을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `SHA512`, SHA512을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Hash을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Stronger을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `HMAC_SHA256`, HMAC_SHA256을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 시크릿-key | 실무 운영 가이드에서 Symmetric을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `HMAC_SHA512`, HMAC_SHA512을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 시크릿-key | 실무 운영 가이드에서 Stronger, HMAC을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `RSA_SHA256`, RSA_SHA256을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Asymmetric을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 RSA을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ED25519`, ED25519을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Asymmetric을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Modern을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### 인증된 서명과 생명주기 진입 검사

`SigningServiceImpl(verification_keys={signer_id: public_pem},
verification_secrets={signer_id: hmac_secret})`은 콘텐츠 해시, 알고리즘,
서명자 ID와 시각을 포함한 서명 바이트를 실제 검증합니다. 키 누락, 콘텐츠
변경 또는 잘못된 서명은 실패합니다. SHA256·SHA512는 서명자 인증이 아닌
키 없는 무결성 검사이며 성공 결과에도 이 한계를 경고합니다. `plugin_hash`는
예약 메타데이터입니다. 검증 키는 제출된 서명이 아닌 운영자가 제공합니다.
디렉터리 서명은 Python 파일만 포함하며 별도 데이터와 네이티브 의존성은 제외합니다.

`create_verification_chain(..., require_trusted_signer=True)`는 운영자 키로
인증되고 신뢰 저장소 등급이 `TRUSTED`인 서명자를 요구합니다. 해시만 있는 서명,
알 수 없는 서명자, 폐기된 서명자는 실패합니다. 인증서 메타데이터는 설정한
검증 키와 이 신뢰 검사를 대체하지 않습니다.

생명주기 진입 검사를 사용하려면 `EnterprisePluginManagerConfig`에
`require_signature=True`와 `verification_keys` 또는 `verification_secrets`를
설정합니다. `manager.signature_trust_store.set_signer_trust(signer_id, TrustLevel.TRUSTED)`와
`manager.register_signed_plugin(name, plugin_class, source_path, signature)`로 등록합니다.
동기 `load_plugin`·`load_from_class`와 비동기 `load`·`load_all` 경로에서
로드/setup 및 활성화 전에 검사합니다. `load(..., verify_signature=True)`는
전역 옵션이 꺼져 있어도 서명 바인딩을 요구하며, 등록된 서명은 항상 검사합니다.
소스 변경, 클래스 교체, 서명자 폐기는 후속 진입을 막지만 언로드는 허용합니다.
일괄 로드 실패는 `ExceptionGroup`으로 보고하며 성공한 로드는 확인·정리를 위해 유지합니다.

등록할 때 클래스는 이미 import되어 있습니다. 이는 생명주기 진입 검사이며
import 샌드박스나 기존 바이트코드와 현재 파일의 일치 증명이 아닙니다.
탐색과 import는 운영자가 신뢰하는 작업으로 남습니다.

신뢰 저장소 형식 2는 공개 인증서 바이트, 신뢰 등급과 폐기 정보를 원자적으로
저장합니다. 다시 열 때 전체 문서를 검증한 뒤 신뢰 정보를 적용합니다. 손상되거나
크기를 초과한 문서는 실패하며, 기존 서명자 전용 파일은 읽을 수 있지만 바이트가
없는 기존 인증서 항목은 운영자가 재등록해야 합니다. 파일은 단일 작성자용입니다.
게시 실패를 숨기지 않고 기존 메모리 상태를 유지합니다. 파일 교체 이후 디렉터리
동기화가 실패했다면 다시 열어 실제 게시 여부를 확인해야 합니다.
키와 HMAC 비밀은 이 파일에 저장하지 않습니다.

공유 매니저는 자동 로드와 종료에 동기 생명주기 포트를 사용하며 Enterprise의
동기 컨텍스트 매니저에도 적용됩니다. 일괄 로드·언로드는 선택한 모든 플러그인을
시도한 뒤 실패를 묶어 보고합니다. 성공한 로드는 등록 상태로 남고 정리 실패를
성공으로 보고하지 않습니다.

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

`HotReloadManager`는 공유 `PluginManager`를 사용합니다. 별도 생명주기 상태
매니저는 리로드 실행기가 아닙니다. 운영자 로더는 새 `Plugin` 클래스 또는 그
클래스의 awaitable을 반환해야 합니다. 새 Python 코드 로딩은 팩터리 책임이며
공유 모듈에 `importlib.reload`를 암묵적으로 수행하지 않습니다.

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

`GRACEFUL`은 `operation(name)`으로 감싼 작업을 `timeout`(기본 30초)까지
기다리고 리로드 중 새 작업 진입을 막은 뒤 기존 teardown과 새 setup/활성화를
확인합니다. `IMMEDIATE`는 작업 대기를 생략하지만 호출자가 보유한 기존 참조를
중단하지 않습니다. 동기 팩터리·콜백의 강제 선점은 제공하지 않습니다.
단일 레지스트리는 다중 인스턴스 `ROLLING`을 구현하지 않으며 요청하면 변경 전에
실패 결과를 반환합니다.

후보 setup 실패 시 teardown을 호출합니다. 정리가 확인되면 이전 클래스의 새
인스턴스를 기존 설정·활성화 상태로 복구할 수 있고, 실제 재로드가 성공한 경우에만
`rolled_back=True`입니다. 이전 메모리 상태나 외부 효과를 되돌리지는 않습니다.
정리가 불확실하면 자동 복구하지 않습니다. `ReloadResult.error`는 오류 유형을
보고하며 자동 리로드 실패는 감시를 중단하고 `aclose()`에서도 보고합니다.

파일 감시는 콘텐츠 해시로 생성·변경·삭제를 감지하고 동기·비동기 콜백을 순서대로
기다립니다. 중복 경로·플러그인 등록은 거부합니다. `await aclose()`로 소유 감시를
취소하고 종료까지 기다립니다. `cancel()`·`stop_all_watches()`는 취소 요청만 합니다.
디렉터리 감시는 `*.py` 파일을 대상으로 합니다.

`EnterprisePluginManagerConfig(enable_hot_reload=True)`에서는
`await manager.configure_hot_reload(name, loader, path=source_path)`로 설정합니다.
`watch_for_changes=True`이고 경로를 명시한 경우에만 자동 감시를 시작합니다.
`await manager.reload(name)`으로 리로드하고 `await manager.unload_all()`로
감시 종료를 기다리며 정리합니다. 감시 중 동기 종료는 작업을 버리지 않고 오류를
보고합니다. 종료 후 추가 리로드에는 새 매니저를 사용합니다. 서명 클래스는 명시적
언로드·새 서명 등록·로드로 갱신하며 자동 리로드가 서명 갱신을 우회하지 않습니다.

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

### 의존성 진입 검사와 종료

의존 플러그인의 setup 전에 필요한 플러그인이 실제 로드되어야 하며 발견만으로는 충족되지 않습니다. `load_all()`은 결정적인 의존 순서를 계산하고 순환·필수 플러그인 누락을 거부합니다. `PluginInfo.dependencies`, `optional_dependencies`는 플러그인 이름과 PEP 440 제약을 받으며 URL·배포판 extras는 플러그인 이름으로 받지 않습니다. `allow_missing_optional=True`일 때만 선택 의존성 누락을 허용하고, 존재하는 선택 플러그인은 버전 제약을 만족해야 합니다. `python_dependencies`는 설치 배포판 메타데이터와 PEP 508 요구사항(마커·선택 extra의 직접 의존성 포함)을 확인하며 패키지를 import하지 않습니다. 설치를 수행하거나 실제 import 가능성을 보장하지 않습니다.

`strict_version_check=True`는 버전 불일치와 해석할 수 없는 호스트 호환성을 거부합니다. false는 명시적으로 버전 검사를 완화하지만 의존성 존재는 확인합니다. `check_compatibility=False`는 호스트 호환성 검사를, `strict_dependencies=False`는 의존성 진입 검사를 비활성화합니다. 매니저는 생성 시 설정을 복사합니다.

선택된 MCP 플러그인 소유자는 승인된 `shutdown` 명령을 지원합니다. 실제 플러그인 teardown, reload 소유자 종료와 factory 종료가 끝난 뒤에만 성공을 확인합니다. 정리 실패 영수증은 실패로 남기고 다른 소유 플러그인의 정리를 계속 시도하며 결과가 불확실한 종료를 자동 재실행하지 않습니다. 만료·철회 시 제한된 시간의 정리는 명시적으로 확인된 네이티브 종료 증거가 아닙니다.

### 탐색 소유권과 실패

탐색은 운영자가 선택한 모듈 코드와 메타데이터 생성자를 실행하며 import를 샌드박스화하지 않습니다. 파일 탐색은 소유한 네임스페이스에서 패키지 초기화·상대 Python import를 지원하고 검사에 필요한 모듈·소스 식별을 유지합니다. 명시적으로 다시 탐색하면 오래된 타임스탬프 기반 바이트코드 대신 소스를 읽습니다. 패키지 초기화가 실패하면 같은 탐색에서 자식 파일마다 재시도하지 않습니다. 이름 충돌·디렉터리 누락·import 실패·잘못된 메타데이터는 기본적으로 `ExceptionGroup`을 발생시킵니다. 완료한 결과는 `discovered_plugins`, 예외 본문을 제외한 소스 식별자·오류 형식은 `failures`에 남습니다.

`PluginDiscovery(strict=False)` 또는 매니저 `strict_discovery=False`는 진단을 보존하면서 부분 결과를 명시적으로 허용합니다. 발견된 클래스 사용이 모두 끝나면 컨텍스트 매니저나 `close()`로 소유한 import finder·모듈을 해제하세요. 매니저 종료도 플러그인 정리 후 탐색 소유자를 닫습니다. 명시적 재탐색은 별도 모듈 세대를 만들며 이미 로드한 인스턴스를 교체하지 않습니다. 심볼릭 링크 소스와 선택한 루트 밖으로 나가는 상대 import를 거부합니다.

### 프로세스 워커의 경계

프로세스 워커는 신뢰한 로컬 callable·인자를 받으며 wire나 신뢰하지 않는 파일에서 입력 pickle을 받지 않습니다. 자식 반환값은 Core의 닫힌 데이터 전용 스칼라·컨테이너 코덱을 사용하고 인코딩 결과 예산은 기본 16 MiB(`max_result_bytes`)입니다. bytes·날짜·tuple 형식을 유지하며 임의 객체는 실패합니다. 부모는 자식 결과를 unpickle하지 않습니다. 예외 본문·stdout·stderr를 반환하지 않습니다. 컨텍스트는 한 엔진과 한 실행에 속합니다. 취소·정리는 소유 워커를 종료하고 회수하며, 생성 중인 프로세스와 메모리 관측 작업도 기다립니다.

실행 시간은 유한한 양수여야 합니다. 지원되는 Unix CPU 시간·파일 디스크립터·출력 파일 제한을 적용합니다. 주소 공간 제한은 지원 플랫폼에서 적용하고 부모 RSS 관측으로도 메모리 예산을 제한하므로 `RLIMIT_AS`를 거부하는 플랫폼에서도 동작합니다. RSS는 주기적 관측이므로 엄격한 순간 최대 메모리나 후손 프로세스 합산을 보장하지 않습니다. 메모리·디스크립터 제한 0은 해당 제한을 비활성화하고 음수는 거부합니다. 이 프로세스 워커는 CPU 사용률·파일 경로 목록·네트워크·하위 프로세스·시스템 호출 정책을 OS 수준에서 강제하지 않습니다. Python import 필터는 보조 수단이며 악성 코드를 격리하지 않습니다. Enterprise 생명주기 등록은 공유 매니저에서 실행되며 자동으로 프로세스·컨테이너 워커 안으로 이동하지 않습니다.

컨테이너 작업자도 크기가 제한된 데이터 전용 반환 코덱을 사용합니다. 호출 함수와 인수의 pickle은 신뢰하는 로컬 입력이며, 선택한 이미지에는 필요한 모듈이 설치되어 있어야 합니다. 이미지를 실행 전에 설치해야 합니다(`--pull=never`). 엔진은 생성한 컨테이너의 불변 ID를 보관하고 실행 완료나 취소 후 해당 컨테이너를 제거합니다. `terminate()`는 취소를 요청하므로 실행 또는 `cleanup()`을 await하여 제거 결과를 확인하세요. 제거 실패는 숨기지 않으며 소유 ID를 재시도용으로 보관합니다. 현재 메모리·CPU cgroup, 네트워크 모드, 읽기 전용 루트 및 capability 제한을 적용하지만 `allowed_paths`와 `writable_paths`로 명시적 호스트 마운트를 허용합니다. Linux x86-64/AArch64 작업자는 호출 함수를 읽기 전에 seccomp 차단 집합을 설치합니다. 지원 이름은 `fork`, `vfork`, `clone`, `clone3`, `execve`, `execveat`, `getpid`이며 알 수 없는 이름이나 아키텍처는 실행을 거부합니다. `clone` 차단은 `clone3`도 차단하며 AArch64의 fork/vfork는 clone에 대응합니다. `allow_subprocess=False`는 모든 프로세스 생성·exec 차단을 추가하고 스레드 생성도 차단합니다. subprocess 허용에는 `allow_subprocess=True`와 함께 `denied_syscalls`의 해당 항목 제거가 필요합니다. 읽기 전용 루트에서도 IPC 작업 공간은 쓰기 가능합니다.

격리 없는 엔진은 신뢰하는 프로세스 내부 작업만 받습니다. 제한 시간이 지나면 동기 결과 대기를 중단하지만 Python 스레드를 강제 종료하지는 못합니다. 타임아웃 또는 취소된 스레드는 완료될 때까지 소유 작업으로 보관하며 `cleanup()`이 기다립니다. 따라서 영구적으로 멈춘 함수는 정리도 막을 수 있습니다. 비동기 작업은 취소 후 종료를 기다립니다. 작업 실패 시 실행 컨텍스트를 종료하고 다른 엔진의 컨텍스트 실행은 거부합니다.

샌드박스 팩토리에 사용자 실행기를 등록해도 다른 기본 실행기 등록을 유지합니다. 캐시 실행기를 교체·등록 해제·초기화해도 정리 책임을 보관합니다. `cleanup_all()`은 현재 및 이전 소유자의 정리를 모두 시도하고 실패를 모아 보고하며 실패한 소유자는 재시도할 수 있도록 유지합니다. singleton이 아닌 실행기는 호출자가 정리해야 합니다. 팩토리의 사용 가능 여부는 클래스 등록 여부이며 Docker 등 외부 런타임이 실행 중임을 보증하지 않습니다.

엔트리 포인트 이름을 포함한 발견 키는 생성한 플러그인이 선언한 이름과 일치해야 합니다. 관리자는 setup이나 레지스트리 변경 전에 불일치를 거부하므로 의존성·활성화·정리가 같은 식별자를 사용합니다.

컨테이너 생성 응답이 유실되면 엔진이 생성한 고유 이름과 소유 레이블만 조회하여 불변 컨테이너 ID를 확인한 뒤 제거합니다. 런타임 조회 실패는 결과 불명 상태로 남기고 정리 재시도를 위해 소유 정보를 유지합니다. 이 복구 정보는 엔진 수명 동안만 유지되며 부모 프로세스 종료 후 복구하는 영속 저널은 아닙니다.

호스트 경로는 존재하는 일반 파일·디렉터리의 canonical 절대 경로여야 합니다. 심볼릭 링크 별칭, `/workspace`·`/proc`·`/sys`·`/dev` 예약 목적지, 호스트 루트 및 모호한 마운트 구분자는 거부합니다. 컨테이너 안에서도 같은 절대 경로에 나타납니다. 읽기 경로는 읽기 전용으로 연결하며 명시적 쓰기 경로는 `allow_file_write=True`가 필요하고 같은 경로의 읽기 전용 설정보다 우선합니다. 엔진 소유 IPC 공간 외에 호스트 디렉터리를 암묵적으로 연결하지 않습니다. 이미지 파일은 계속 보이므로 이미지 파일시스템 전체를 차단하는 허용 목록은 아닙니다. 호스트 변경과 마운트 생성의 경합은 신뢰하는 로컬 호출 계약의 보장 범위 밖입니다. 운영자가 선택한 로컬 런타임을 사용하고 필요하면 변경 불가능한 입력 스냅샷을 제공하세요.
