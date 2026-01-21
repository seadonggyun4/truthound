# 엔터프라이즈 SDK

엔터프라이즈 SDK는 프로덕션 환경에서 커스텀 검증기를 안전하게 실행하기 위한 고급 기능을 제공합니다.

## 개요

엔터프라이즈 SDK 아키텍처:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Enterprise SDK Manager                            │
└─────────────────────────────────────────────────────────────────────┘
                                │
┌───────────────┬───────────────┼───────────────┬─────────────────────┐
│               │               │               │                     │
▼               ▼               ▼               ▼                     ▼
┌─────────┐   ┌─────────┐    ┌──────────┐   ┌──────────┐    ┌────────────┐
│ Sandbox │   │ Resource│    │ Signing  │   │ Version  │    │  License   │
│ Manager │   │ Limiter │    │ Manager  │   │ Checker  │    │  Manager   │
└─────────┘   └─────────┘    └──────────┘   └──────────┘    └────────────┘
```

---

## 1. 샌드박스 실행

신뢰할 수 없는 검증기를 격리된 환경에서 실행합니다.

### SandboxBackend

| 백엔드 | 격리 수준 | 설명 |
|--------|-----------|------|
| `IN_PROCESS` | 낮음 | 프로세스 내 실행, import 제한만 적용 |
| `SUBPROCESS` | 중간 | 별도 프로세스, OS 리소스 제한 |
| `DOCKER` | 높음 | Docker 컨테이너, 완전한 격리 |

### SandboxConfig

```python
from truthound.validators.sdk.enterprise import (
    SandboxConfig,
    SandboxBackend,
    create_sandbox,
)

# 커스텀 설정
config = SandboxConfig(
    backend=SandboxBackend.SUBPROCESS,
    timeout_seconds=60.0,
    max_memory_mb=512,
    max_cpu_percent=100,
    allowed_paths=("/data", "/tmp"),
    allowed_modules=("polars", "numpy", "pandas", "truthound"),
    blocked_modules=(
        "os", "subprocess", "shutil", "socket", "urllib",
        "requests", "http", "ftplib", "smtplib", "telnetlib",
        "ctypes", "multiprocessing",
    ),
    network_enabled=False,
    env_vars={},
    docker_image="python:3.11-slim",
    working_dir="/workspace",
)

# 프리셋 설정
strict_config = SandboxConfig.strict()    # Docker, 256MB, 30초
standard_config = SandboxConfig.standard() # Subprocess, 512MB, 60초
permissive_config = SandboxConfig.permissive() # In-process, 2GB, 120초
```

### 사용 예시

```python
from truthound.validators.sdk.enterprise import (
    SandboxConfig,
    SandboxBackend,
    create_sandbox,
)

config = SandboxConfig(
    backend=SandboxBackend.SUBPROCESS,
    timeout_seconds=30,
)

executor = create_sandbox(config)
result = await executor.execute(
    validator_class=MyValidator,
    data=my_dataframe,
    config={"columns": ("col1", "col2")},
)

if result.success:
    issues = result.result
    print(f"Execution time: {result.execution_time_seconds:.2f}s")
else:
    print(f"Error: {result.error}")
```

### SandboxResult

```python
@dataclass
class SandboxResult:
    success: bool
    result: Any = None              # 검증 결과 (성공 시)
    error: str | None = None        # 에러 메시지 (실패 시)
    execution_time_seconds: float = 0.0
    memory_used_mb: float = 0.0
    cpu_time_seconds: float = 0.0
    sandbox_id: str = ""            # 고유 실행 ID
    started_at: datetime = ...
    finished_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]: ...
```

### 예외 클래스

| 예외 | 설명 |
|------|------|
| `SandboxError` | 샌드박스 기본 예외 |
| `SandboxTimeoutError` | 실행 시간 초과 |
| `SandboxResourceError` | 리소스 제한 초과 |
| `SandboxSecurityError` | 보안 위반 감지 |

---

## 2. 리소스 제한

검증기 실행 시 CPU, 메모리, 시간 제한을 적용합니다.

### ResourceLimits

```python
from truthound.validators.sdk.enterprise import (
    ResourceLimits,
    CombinedResourceLimiter,
)

# 커스텀 설정
limits = ResourceLimits(
    max_memory_mb=512,           # 최대 메모리 (MB)
    max_cpu_seconds=60.0,        # 최대 CPU 시간 (초)
    max_wall_time_seconds=120.0, # 최대 실제 시간 (초)
    max_file_descriptors=256,    # 최대 파일 디스크립터
    max_processes=4,             # 최대 자식 프로세스
    soft_memory_threshold=0.8,   # 경고 임계값 (0.0-1.0)
    check_interval_seconds=0.5,  # 모니터링 주기
    graceful_degradation=True,   # 우아한 저하 허용
)

# 프리셋
strict_limits = ResourceLimits.strict()     # 256MB, 30초
standard_limits = ResourceLimits.standard() # 512MB, 60초
generous_limits = ResourceLimits.generous() # 4GB, 300초
```

### 리소스 모니터링

```python
from truthound.validators.sdk.enterprise import (
    ResourceMonitor,
    ResourceLimits,
)

limits = ResourceLimits(max_memory_mb=512)
monitor = ResourceMonitor(
    limits=limits,
    on_threshold=lambda usage: print(f"Warning: {usage.memory_percent}% memory"),
    on_exceeded=lambda res_type, limit, actual: print(f"Exceeded: {res_type}"),
)

monitor.start()
try:
    # 검증 실행
    result = validator.validate(data)
finally:
    monitor.stop()

# 사용량 확인
usage = monitor.get_usage()
print(f"Memory: {usage.memory_mb:.1f}MB ({usage.memory_percent:.1f}%)")
print(f"CPU: {usage.cpu_seconds:.2f}s ({usage.cpu_percent:.1f}%)")

# 피크 사용량
peak = monitor.get_peak_usage()
print(f"Peak memory: {peak.memory_mb:.1f}MB")
```

### 컨텍스트 매니저

```python
from truthound.validators.sdk.enterprise import (
    CombinedResourceLimiter,
    MemoryLimiter,
    CPULimiter,
)

# 통합 리미터
limiter = CombinedResourceLimiter(limits)
with limiter.enforce() as monitor:
    result = validator.validate(data)
    print(f"Used: {monitor.get_usage().memory_mb:.1f}MB")

# 개별 리미터
with MemoryLimiter(max_memory_mb=256).enforce() as monitor:
    result = validator.validate(data)

with CPULimiter(max_cpu_seconds=30).enforce() as monitor:
    result = validator.validate(data)
```

### 데코레이터

```python
from truthound.validators.sdk.enterprise.resources import with_resource_limits

@with_resource_limits(max_memory_mb=256, max_cpu_seconds=30)
def expensive_validation(data):
    validator = MyValidator()
    return validator.validate(data)
```

### ResourceUsage

```python
@dataclass
class ResourceUsage:
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    cpu_seconds: float = 0.0
    cpu_percent: float = 0.0
    wall_seconds: float = 0.0
    wall_percent: float = 0.0
    file_descriptors: int = 0
    timestamp: datetime = ...

    def is_within_limits(self) -> bool: ...
    def is_near_limits(self, threshold: float = 0.8) -> bool: ...
    def to_dict(self) -> dict[str, Any]: ...
```

---

## 3. 코드 서명

검증기의 무결성을 보장하기 위한 암호화 서명 시스템입니다.

### SignatureAlgorithm

| 알고리즘 | 설명 | 용도 |
|----------|------|------|
| `SHA256` | SHA256 해시 | 개발/테스트 |
| `SHA512` | SHA512 해시 | 개발/테스트 |
| `HMAC_SHA256` | HMAC-SHA256 | 프로덕션 |
| `HMAC_SHA512` | HMAC-SHA512 | 프로덕션 |
| `RSA_SHA256` | RSA + SHA256 | 엔터프라이즈 (cryptography 필요) |
| `ED25519` | Ed25519 | 엔터프라이즈 (cryptography 필요) |

### SignatureConfig

```python
from truthound.validators.sdk.enterprise import (
    SignatureConfig,
    SignatureAlgorithm,
    SignatureManager,
)

# 개발용 (약한 보안)
dev_config = SignatureConfig.development()

# 프로덕션용
prod_config = SignatureConfig.production(secret_key="your-secret-key")

# 커스텀 설정
config = SignatureConfig(
    algorithm=SignatureAlgorithm.HMAC_SHA256,
    secret_key="your-secret-key",
    private_key_path=Path("/path/to/private.pem"),  # RSA용
    public_key_path=Path("/path/to/public.pem"),    # RSA용
    validity_days=365,                               # 서명 유효 기간
    require_timestamp=True,                          # 타임스탬프 필수
    trusted_signers=("admin@company.com",),         # 신뢰 서명자
    revocation_list_url="https://...",              # 폐기 목록 URL
)
```

### 서명 및 검증

```python
from truthound.validators.sdk.enterprise import (
    SignatureManager,
    SignatureConfig,
    sign_validator,
    verify_validator,
)

# 매니저 사용
config = SignatureConfig.production(secret_key="secret")
manager = SignatureManager(config)

# 서명 생성
signature = manager.sign_validator(
    MyValidator,
    signer_id="admin@company.com",
    metadata={"team": "data-quality"},
)

# 서명 저장/로드
manager.save_signature(signature, Path("my_validator.sig"))
loaded_sig = manager.load_signature(Path("my_validator.sig"))

# 서명 검증
try:
    is_valid = manager.verify_validator(
        MyValidator,
        signature,
        check_expiry=True,
        check_signer=True,
    )
except SignatureExpiredError:
    print("Signature has expired")
except SignatureTamperError:
    print("Code has been modified!")
except SignatureVerificationError as e:
    print(f"Verification failed: {e.reason}")

# 간편 함수
signature = sign_validator(
    MyValidator,
    secret_key="secret",
    algorithm=SignatureAlgorithm.HMAC_SHA256,
    signer_id="admin",
)

is_valid = verify_validator(
    MyValidator,
    signature,
    secret_key="secret",
)
```

### ValidatorSignature

```python
@dataclass
class ValidatorSignature:
    validator_name: str
    validator_version: str
    code_hash: str                    # 소스 코드 해시
    signature: str                    # Base64 인코딩된 서명
    algorithm: SignatureAlgorithm
    signer_id: str = ""
    signed_at: datetime = ...
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool: ...
    def to_dict(self) -> dict[str, Any]: ...
    def to_json(self) -> str: ...
    @classmethod
    def from_dict(cls, data: dict) -> "ValidatorSignature": ...
    @classmethod
    def from_json(cls, json_str: str) -> "ValidatorSignature": ...
```

---

## 4. 버전 호환성

검증기와 Truthound 버전 간의 호환성을 검사합니다.

### SemanticVersion

```python
from truthound.validators.sdk.enterprise import (
    SemanticVersion,
    VersionConstraint,
    VersionSpec,
)

# 버전 파싱
version = SemanticVersion.parse("2.1.0")
version_pre = SemanticVersion.parse("2.0.0-alpha.1+build.123")

# 버전 비교
v1 = SemanticVersion.parse("1.0.0")
v2 = SemanticVersion.parse("2.0.0")
print(v1 < v2)  # True

# 버전 범프
version = SemanticVersion(1, 2, 3)
print(version.bump_major())  # 2.0.0
print(version.bump_minor())  # 1.3.0
print(version.bump_patch())  # 1.2.4

# 호환성 확인
compatibility = v1.is_compatible_with(v2)
# VersionCompatibility.INCOMPATIBLE (메이저 버전 다름)
```

### VersionConstraint

```python
from truthound.validators.sdk.enterprise import VersionConstraint

# 제약 조건 파싱
constraint = VersionConstraint.parse(">=1.0.0")
constraint = VersionConstraint.parse("<2.0.0")
constraint = VersionConstraint.parse("~1.2.0")  # >=1.2.0, <1.3.0
constraint = VersionConstraint.parse("^1.2.0")  # >=1.2.0, <2.0.0

# 매칭 확인
version = SemanticVersion.parse("1.5.0")
print(constraint.matches(version))  # True
```

#### 지원 연산자

| 연산자 | 예시 | 의미 |
|--------|------|------|
| `=` | `=1.0.0` | 정확히 1.0.0 |
| `!=` | `!=1.0.0` | 1.0.0 제외 |
| `>` | `>1.0.0` | 1.0.0 초과 |
| `>=` | `>=1.0.0` | 1.0.0 이상 |
| `<` | `<2.0.0` | 2.0.0 미만 |
| `<=` | `<=2.0.0` | 2.0.0 이하 |
| `~` | `~1.2.0` | >=1.2.0, <1.3.0 (패치 변경 허용) |
| `^` | `^1.2.0` | >=1.2.0, <2.0.0 (마이너 변경 허용) |

### VersionSpec

복합 버전 조건을 지원합니다.

```python
from truthound.validators.sdk.enterprise import VersionSpec

# AND 조합 (쉼표)
spec = VersionSpec.parse(">=1.0.0,<2.0.0")

# OR 조합 (||)
spec = VersionSpec.parse(">=1.0.0,<2.0.0 || >=3.0.0")

# 매칭 확인
version = SemanticVersion.parse("1.5.0")
print(spec.matches(version))  # True

# 와일드카드 (모든 버전 허용)
spec = VersionSpec.parse("*")
```

### VersionChecker

```python
from truthound.validators.sdk.enterprise import (
    VersionChecker,
    VersionCompatibility,
)

checker = VersionChecker(
    truthound_version="1.0.0",
    python_version=None,  # 자동 감지
)

# 단일 검증기 호환성 확인
try:
    compatibility = checker.check_compatibility(
        MyValidator,
        raise_on_incompatible=True,
    )
except VersionConflictError as e:
    print(f"Incompatible: {e.required} required, {e.actual} installed")

# 여러 검증기 확인
results = checker.check_all(
    [Validator1, Validator2, Validator3],
    raise_on_first=False,
)
for name, compat in results.items():
    print(f"{name}: {compat.name}")
```

### 검증기 버전 정보

검증기 클래스에 버전 정보를 선언합니다:

```python
class MyValidator(Validator):
    name = "my_validator"
    version = "1.2.0"

    # Truthound 버전 요구사항
    min_truthound_version = "1.0.0"
    max_truthound_version = "2.0.0"

    # Python 버전 요구사항
    python_version = ">=3.11"

    # 의존성 (패키지명: 버전 스펙)
    dependencies = {
        "polars": ">=0.20.0",
        "numpy": ">=1.24.0,<2.0.0",
    }
```

---

## 5. 라이선스 관리

검증기의 라이선스를 추적하고 검증합니다.

### LicenseType

```python
from truthound.validators.sdk.enterprise import LicenseType

# 오픈 소스 라이선스
LicenseType.MIT
LicenseType.APACHE_2
LicenseType.BSD_3
LicenseType.GPL_3
LicenseType.LGPL_3

# 상용 라이선스
LicenseType.COMMERCIAL
LicenseType.ENTERPRISE
LicenseType.TRIAL

# 특수 라이선스
LicenseType.PROPRIETARY
LicenseType.CUSTOM
```

### LicenseInfo

```python
from truthound.validators.sdk.enterprise import LicenseInfo, LicenseType

# 프리셋 라이선스
mit_license = LicenseInfo.mit("my_validator")
apache_license = LicenseInfo.apache2("my_validator")
trial_license = LicenseInfo.trial("my_validator", days=30)

# 커스텀 라이선스
license_info = LicenseInfo(
    license_type=LicenseType.COMMERCIAL,
    license_key="...",
    licensee="Company Inc.",
    issued_at=datetime.now(timezone.utc),
    expires_at=datetime.now(timezone.utc) + timedelta(days=365),
    max_users=10,           # 0 = 무제한
    max_rows=1_000_000,     # 0 = 무제한
    features=("advanced", "ml"),  # 허용 기능
    restrictions=("no_export",),  # 제한 사항
    validator_name="my_validator",
    validator_version="1.0.0",
)

# 라이선스 확인
print(license_info.is_expired())      # False
print(license_info.is_open_source())  # False
print(license_info.is_commercial())   # True
print(license_info.days_until_expiry())  # 365
print(license_info.has_feature("advanced"))  # True
```

### LicenseValidator

라이선스 정책을 검증합니다.

```python
from truthound.validators.sdk.enterprise import (
    LicenseValidator,
    LicenseInfo,
)

validator = LicenseValidator(
    allow_expired=False,        # 만료 라이선스 허용
    allow_trial=True,           # 평가판 허용
    require_commercial=False,   # 상용 필수
    required_features=["ml"],   # 필수 기능
)

try:
    is_valid = validator.validate(
        license_info,
        raise_on_invalid=True,
    )
except LicenseExpiredError:
    print("License expired")
except LicenseViolationError as e:
    print(f"Violation: {e.violation_type}")
```

### LicenseManager

```python
from truthound.validators.sdk.enterprise import LicenseManager

manager = LicenseManager(
    secret_key="license-signing-key",
    license_dir=Path("/licenses"),
    validator=LicenseValidator(),
)

# 라이선스 조회
license_info = manager.get_license(MyValidator)

# 라이선스 검증
is_valid = manager.validate_license(MyValidator)

# 사용량 추적
manager.track_usage(
    MyValidator,
    rows_processed=10000,
    user_id="user@company.com",
    session_id="session-123",
)

# 사용량 리포트
report = manager.get_usage_report(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
)
```

### 검증기에 라이선스 선언

```python
class MyCommercialValidator(Validator):
    name = "my_commercial_validator"
    license_type = "COMMERCIAL"  # 또는 LicenseType.COMMERCIAL
    license_key = "..."  # 라이선스 키 (선택)
```

---

## 6. 퍼징 테스트

검증기의 안정성을 테스트하기 위한 퍼징 프레임워크입니다.

### FuzzStrategy

| 전략 | 설명 |
|------|------|
| `RANDOM` | 순수 무작위 데이터 |
| `BOUNDARY` | 경계값 테스트 |
| `MUTATION` | 유효 데이터 변형 |
| `DICTIONARY` | 알려진 문제 값 사전 |
| `STRUCTURE_AWARE` | 스키마 인식 퍼징 |

### FuzzConfig

```python
from truthound.validators.sdk.enterprise import FuzzConfig, FuzzStrategy

# 커스텀 설정
config = FuzzConfig(
    strategy=FuzzStrategy.RANDOM,
    iterations=100,
    seed=42,                    # 재현성을 위한 시드
    max_rows=1000,
    max_columns=20,
    timeout_seconds=10.0,
    include_nulls=True,
    include_edge_cases=True,
    include_unicode=True,
    mutation_rate=0.1,
)

# 프리셋
quick_config = FuzzConfig.quick()       # 10회, 5초
thorough_config = FuzzConfig.thorough() # 1000회, 30초
```

### FuzzRunner

```python
from truthound.validators.sdk.enterprise import FuzzRunner, run_fuzz_tests

# 기본 퍼징
report = run_fuzz_tests(
    MyValidator,
    iterations=100,
    seed=42,
)

print(f"Passed: {report.passed}/{report.total_iterations}")
print(f"Success rate: {report.success_rate:.1%}")
print(f"Duration: {report.total_duration_seconds:.2f}s")

# 실패 사례 확인
for error in report.errors:
    print(f"Iteration {error.iteration}:")
    print(f"  Seed: {error.seed_used}")
    print(f"  Data shape: {error.data_shape}")
    print(f"  Error: {error.error}")
```

### 속성 기반 테스트

```python
from truthound.validators.sdk.enterprise import FuzzRunner

runner = FuzzRunner(FuzzConfig.thorough())
reports = runner.fuzz_with_properties(MyValidator)

for prop_name, report in reports.items():
    print(f"{prop_name}: {report.success_rate:.1%}")
```

테스트되는 속성:

| 속성 | 설명 |
|------|------|
| `no_crash` | 어떤 입력에도 크래시 없음 |
| `returns_list` | 항상 리스트 반환 |
| `issues_have_fields` | 이슈에 필수 필드 존재 |

### PropertyBasedTester

```python
from truthound.validators.sdk.enterprise import PropertyBasedTester

tester = PropertyBasedTester(MyValidator)

# 개별 속성 테스트
print(tester.test_no_crash(data))
print(tester.test_returns_list(data))
print(tester.test_issues_have_fields(data))

# 모든 속성 테스트
results = tester.run_all(data)
```

### 엣지 케이스 값

퍼저가 생성하는 엣지 케이스 값:

**숫자:**
- `0`, `-0`, `1`, `-1`
- `float("inf")`, `float("-inf")`, `float("nan")`
- `2**31 - 1`, `-(2**31)`, `2**63 - 1`, `-(2**63)`
- `1e-300`, `1e300`, `-1e-300`, `-1e300`

**문자열:**
- `""` (빈 문자열)
- `" "`, `"\t"`, `"\n"`, `"\r\n"` (공백)
- `"null"`, `"NULL"`, `"None"`, `"undefined"`, `"NaN"`, `"inf"`
- XSS/SQL 인젝션 페이로드
- 경로 순회 패턴
- 널 바이트, 긴 문자열

**유니코드:**
- `"Hello 世界"`, `"مرحبا"`, `"שלום"`, `"🎉🚀💻"`
- 제로 폭 공백, BOM

---

## 7. EnterpriseSDKManager

모든 엔터프라이즈 기능을 통합하는 매니저 클래스입니다.

### EnterpriseConfig

```python
from truthound.validators.sdk.enterprise import (
    EnterpriseSDKManager,
    EnterpriseConfig,
)

# 프리셋 설정
dev_config = EnterpriseConfig.development()  # 최소 보안
prod_config = EnterpriseConfig.production(license_key="...")  # 표준 보안
secure_config = EnterpriseConfig.secure(license_key="...")  # 최대 보안

# 커스텀 설정
config = EnterpriseConfig(
    # 샌드박스
    sandbox_enabled=True,
    sandbox_backend=SandboxBackend.SUBPROCESS,
    sandbox_timeout_seconds=60.0,

    # 리소스 제한
    resource_limits=ResourceLimits.standard(),

    # 서명
    signing_enabled=True,
    signing_config=SignatureConfig.production("secret"),

    # 버전 검사
    version_check_enabled=True,
    truthound_version="1.0.0",

    # 라이선스
    license_check_enabled=True,
    license_secret_key="license-key",
    license_dir=Path("/licenses"),
)
```

### 통합 실행

```python
async with EnterpriseSDKManager(config) as manager:
    # 모든 보호 기능이 적용된 실행
    result = await manager.execute_validator(
        validator_class=MyValidator,
        data=my_dataframe,
        config={"columns": ("col1",)},
        signature=signature,  # 선택
    )

    if result.success:
        issues = result.validation_result
        print(f"Found {len(issues)} issues")
        print(f"Execution time: {result.execution_time_seconds:.2f}s")
    else:
        print(f"Failed: {result.error}")

    # 검사 결과 확인
    print(f"Version compatible: {result.version_compatible}")
    print(f"Signature valid: {result.signature_valid}")
    print(f"License valid: {result.license_valid}")
```

### 동기 실행

```python
manager = EnterpriseSDKManager(config)
result = manager.execute_validator_sync(
    MyValidator,
    data,
)
```

### 개별 기능 사용

```python
manager = EnterpriseSDKManager(config)

# 서명
signature = manager.sign_validator(MyValidator, signer_id="admin")
is_valid = manager.verify_validator(MyValidator, signature)

# 버전 호환성
compatibility = manager.check_compatibility(MyValidator)

# 라이선스
license_info = manager.get_license(MyValidator)

# 문서 생성
docs = manager.generate_docs(MyValidator, format=DocFormat.MARKDOWN)

# 퍼징
report = manager.fuzz_validator(MyValidator, FuzzConfig.quick())
```

### ExecutionResult

```python
@dataclass
class ExecutionResult:
    success: bool
    validation_result: Any = None     # 검증 결과
    error: str | None = None
    sandbox_result: SandboxResult | None = None
    resource_usage: ResourceUsage | None = None
    signature_valid: bool | None = None
    version_compatible: bool | None = None
    license_valid: bool | None = None
    execution_time_seconds: float = 0.0
    started_at: datetime = ...
    finished_at: datetime | None = None
```

---

## 다음 단계

- [보안 가이드](security.md) - ReDoS 보호, SQL 인젝션 방지
- [커스텀 검증기](custom-validators.md) - SDK 기본 사용법
- [내장 검증기](built-in.md) - 289개 내장 검증기 참조
