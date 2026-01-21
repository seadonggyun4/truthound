# Enterprise SDK

The Enterprise SDK provides advanced features for safely executing custom validators in production environments.

## Overview

Enterprise SDK Architecture:

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

## 1. Sandbox Execution

Execute untrusted validators in isolated environments.

### SandboxBackend

| Backend | Isolation Level | Description |
|---------|-----------------|-------------|
| `IN_PROCESS` | Low | In-process execution, import restrictions only |
| `SUBPROCESS` | Medium | Separate process, OS resource limits |
| `DOCKER` | High | Docker container, complete isolation |

### SandboxConfig

```python
from truthound.validators.sdk.enterprise import (
    SandboxConfig,
    SandboxBackend,
    create_sandbox,
)

# Custom configuration
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

# Preset configurations
strict_config = SandboxConfig.strict()    # Docker, 256MB, 30 seconds
standard_config = SandboxConfig.standard() # Subprocess, 512MB, 60 seconds
permissive_config = SandboxConfig.permissive() # In-process, 2GB, 120 seconds
```

### Usage Example

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
    result: Any = None              # Validation result (on success)
    error: str | None = None        # Error message (on failure)
    execution_time_seconds: float = 0.0
    memory_used_mb: float = 0.0
    cpu_time_seconds: float = 0.0
    sandbox_id: str = ""            # Unique execution ID
    started_at: datetime = ...
    finished_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]: ...
```

### Exception Classes

| Exception | Description |
|-----------|-------------|
| `SandboxError` | Base sandbox exception |
| `SandboxTimeoutError` | Execution timeout |
| `SandboxResourceError` | Resource limit exceeded |
| `SandboxSecurityError` | Security violation detected |

---

## 2. Resource Limits

Apply CPU, memory, and time limits during validator execution.

### ResourceLimits

```python
from truthound.validators.sdk.enterprise import (
    ResourceLimits,
    CombinedResourceLimiter,
)

# Custom configuration
limits = ResourceLimits(
    max_memory_mb=512,           # Maximum memory (MB)
    max_cpu_seconds=60.0,        # Maximum CPU time (seconds)
    max_wall_time_seconds=120.0, # Maximum wall time (seconds)
    max_file_descriptors=256,    # Maximum file descriptors
    max_processes=4,             # Maximum child processes
    soft_memory_threshold=0.8,   # Warning threshold (0.0-1.0)
    check_interval_seconds=0.5,  # Monitoring interval
    graceful_degradation=True,   # Allow graceful degradation
)

# Presets
strict_limits = ResourceLimits.strict()     # 256MB, 30 seconds
standard_limits = ResourceLimits.standard() # 512MB, 60 seconds
generous_limits = ResourceLimits.generous() # 4GB, 300 seconds
```

### Resource Monitoring

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
    # Execute validation
    result = validator.validate(data)
finally:
    monitor.stop()

# Check usage
usage = monitor.get_usage()
print(f"Memory: {usage.memory_mb:.1f}MB ({usage.memory_percent:.1f}%)")
print(f"CPU: {usage.cpu_seconds:.2f}s ({usage.cpu_percent:.1f}%)")

# Peak usage
peak = monitor.get_peak_usage()
print(f"Peak memory: {peak.memory_mb:.1f}MB")
```

### Context Manager

```python
from truthound.validators.sdk.enterprise import (
    CombinedResourceLimiter,
    MemoryLimiter,
    CPULimiter,
)

# Combined limiter
limiter = CombinedResourceLimiter(limits)
with limiter.enforce() as monitor:
    result = validator.validate(data)
    print(f"Used: {monitor.get_usage().memory_mb:.1f}MB")

# Individual limiters
with MemoryLimiter(max_memory_mb=256).enforce() as monitor:
    result = validator.validate(data)

with CPULimiter(max_cpu_seconds=30).enforce() as monitor:
    result = validator.validate(data)
```

### Decorator

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

## 3. Code Signing

Cryptographic signing system to ensure validator integrity.

### SignatureAlgorithm

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| `SHA256` | SHA256 hash | Development/testing |
| `SHA512` | SHA512 hash | Development/testing |
| `HMAC_SHA256` | HMAC-SHA256 | Production |
| `HMAC_SHA512` | HMAC-SHA512 | Production |
| `RSA_SHA256` | RSA + SHA256 | Enterprise (requires cryptography) |
| `ED25519` | Ed25519 | Enterprise (requires cryptography) |

### SignatureConfig

```python
from truthound.validators.sdk.enterprise import (
    SignatureConfig,
    SignatureAlgorithm,
    SignatureManager,
)

# Development (weak security)
dev_config = SignatureConfig.development()

# Production
prod_config = SignatureConfig.production(secret_key="your-secret-key")

# Custom configuration
config = SignatureConfig(
    algorithm=SignatureAlgorithm.HMAC_SHA256,
    secret_key="your-secret-key",
    private_key_path=Path("/path/to/private.pem"),  # For RSA
    public_key_path=Path("/path/to/public.pem"),    # For RSA
    validity_days=365,                               # Signature validity period
    require_timestamp=True,                          # Timestamp required
    trusted_signers=("admin@company.com",),         # Trusted signers
    revocation_list_url="https://...",              # Revocation list URL
)
```

### Signing and Verification

```python
from truthound.validators.sdk.enterprise import (
    SignatureManager,
    SignatureConfig,
    sign_validator,
    verify_validator,
)

# Using manager
config = SignatureConfig.production(secret_key="secret")
manager = SignatureManager(config)

# Create signature
signature = manager.sign_validator(
    MyValidator,
    signer_id="admin@company.com",
    metadata={"team": "data-quality"},
)

# Save/load signature
manager.save_signature(signature, Path("my_validator.sig"))
loaded_sig = manager.load_signature(Path("my_validator.sig"))

# Verify signature
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

# Convenience functions
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
    code_hash: str                    # Source code hash
    signature: str                    # Base64-encoded signature
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

## 4. Version Compatibility

Check compatibility between validators and Truthound versions.

### SemanticVersion

```python
from truthound.validators.sdk.enterprise import (
    SemanticVersion,
    VersionConstraint,
    VersionSpec,
)

# Parse version
version = SemanticVersion.parse("2.1.0")
version_pre = SemanticVersion.parse("2.0.0-alpha.1+build.123")

# Compare versions
v1 = SemanticVersion.parse("1.0.0")
v2 = SemanticVersion.parse("2.0.0")
print(v1 < v2)  # True

# Version bumping
version = SemanticVersion(1, 2, 3)
print(version.bump_major())  # 2.0.0
print(version.bump_minor())  # 1.3.0
print(version.bump_patch())  # 1.2.4

# Compatibility check
compatibility = v1.is_compatible_with(v2)
# VersionCompatibility.INCOMPATIBLE (different major version)
```

### VersionConstraint

```python
from truthound.validators.sdk.enterprise import VersionConstraint

# Parse constraints
constraint = VersionConstraint.parse(">=1.0.0")
constraint = VersionConstraint.parse("<2.0.0")
constraint = VersionConstraint.parse("~1.2.0")  # >=1.2.0, <1.3.0
constraint = VersionConstraint.parse("^1.2.0")  # >=1.2.0, <2.0.0

# Check matching
version = SemanticVersion.parse("1.5.0")
print(constraint.matches(version))  # True
```

#### Supported Operators

| Operator | Example | Meaning |
|----------|---------|---------|
| `=` | `=1.0.0` | Exactly 1.0.0 |
| `!=` | `!=1.0.0` | Excludes 1.0.0 |
| `>` | `>1.0.0` | Greater than 1.0.0 |
| `>=` | `>=1.0.0` | 1.0.0 or greater |
| `<` | `<2.0.0` | Less than 2.0.0 |
| `<=` | `<=2.0.0` | 2.0.0 or less |
| `~` | `~1.2.0` | >=1.2.0, <1.3.0 (patch changes allowed) |
| `^` | `^1.2.0` | >=1.2.0, <2.0.0 (minor changes allowed) |

### VersionSpec

Supports compound version conditions.

```python
from truthound.validators.sdk.enterprise import VersionSpec

# AND combination (comma)
spec = VersionSpec.parse(">=1.0.0,<2.0.0")

# OR combination (||)
spec = VersionSpec.parse(">=1.0.0,<2.0.0 || >=3.0.0")

# Check matching
version = SemanticVersion.parse("1.5.0")
print(spec.matches(version))  # True

# Wildcard (allows all versions)
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
    python_version=None,  # Auto-detect
)

# Single validator compatibility check
try:
    compatibility = checker.check_compatibility(
        MyValidator,
        raise_on_incompatible=True,
    )
except VersionConflictError as e:
    print(f"Incompatible: {e.required} required, {e.actual} installed")

# Check multiple validators
results = checker.check_all(
    [Validator1, Validator2, Validator3],
    raise_on_first=False,
)
for name, compat in results.items():
    print(f"{name}: {compat.name}")
```

### Declaring Validator Version Information

Declare version information in validator classes:

```python
class MyValidator(Validator):
    name = "my_validator"
    version = "1.2.0"

    # Truthound version requirements
    min_truthound_version = "1.0.0"
    max_truthound_version = "2.0.0"

    # Python version requirements
    python_version = ">=3.11"

    # Dependencies (package name: version spec)
    dependencies = {
        "polars": ">=0.20.0",
        "numpy": ">=1.24.0,<2.0.0",
    }
```

---

## 5. License Management

Track and verify validator licenses.

### LicenseType

```python
from truthound.validators.sdk.enterprise import LicenseType

# Open source licenses
LicenseType.MIT
LicenseType.APACHE_2
LicenseType.BSD_3
LicenseType.GPL_3
LicenseType.LGPL_3

# Commercial licenses
LicenseType.COMMERCIAL
LicenseType.ENTERPRISE
LicenseType.TRIAL

# Special licenses
LicenseType.PROPRIETARY
LicenseType.CUSTOM
```

### LicenseInfo

```python
from truthound.validators.sdk.enterprise import LicenseInfo, LicenseType

# Preset licenses
mit_license = LicenseInfo.mit("my_validator")
apache_license = LicenseInfo.apache2("my_validator")
trial_license = LicenseInfo.trial("my_validator", days=30)

# Custom license
license_info = LicenseInfo(
    license_type=LicenseType.COMMERCIAL,
    license_key="...",
    licensee="Company Inc.",
    issued_at=datetime.now(timezone.utc),
    expires_at=datetime.now(timezone.utc) + timedelta(days=365),
    max_users=10,           # 0 = unlimited
    max_rows=1_000_000,     # 0 = unlimited
    features=("advanced", "ml"),  # Allowed features
    restrictions=("no_export",),  # Restrictions
    validator_name="my_validator",
    validator_version="1.0.0",
)

# License checks
print(license_info.is_expired())      # False
print(license_info.is_open_source())  # False
print(license_info.is_commercial())   # True
print(license_info.days_until_expiry())  # 365
print(license_info.has_feature("advanced"))  # True
```

### LicenseValidator

Validates license policies.

```python
from truthound.validators.sdk.enterprise import (
    LicenseValidator,
    LicenseInfo,
)

validator = LicenseValidator(
    allow_expired=False,        # Allow expired licenses
    allow_trial=True,           # Allow trial licenses
    require_commercial=False,   # Require commercial license
    required_features=["ml"],   # Required features
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

# Retrieve license
license_info = manager.get_license(MyValidator)

# Validate license
is_valid = manager.validate_license(MyValidator)

# Track usage
manager.track_usage(
    MyValidator,
    rows_processed=10000,
    user_id="user@company.com",
    session_id="session-123",
)

# Usage report
report = manager.get_usage_report(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
)
```

### Declaring Licenses in Validators

```python
class MyCommercialValidator(Validator):
    name = "my_commercial_validator"
    license_type = "COMMERCIAL"  # Or LicenseType.COMMERCIAL
    license_key = "..."  # License key (optional)
```

---

## 6. Fuzz Testing

Fuzzing framework for testing validator stability.

### FuzzStrategy

| Strategy | Description |
|----------|-------------|
| `RANDOM` | Pure random data |
| `BOUNDARY` | Boundary value testing |
| `MUTATION` | Mutate valid data |
| `DICTIONARY` | Known problematic values dictionary |
| `STRUCTURE_AWARE` | Schema-aware fuzzing |

### FuzzConfig

```python
from truthound.validators.sdk.enterprise import FuzzConfig, FuzzStrategy

# Custom configuration
config = FuzzConfig(
    strategy=FuzzStrategy.RANDOM,
    iterations=100,
    seed=42,                    # Seed for reproducibility
    max_rows=1000,
    max_columns=20,
    timeout_seconds=10.0,
    include_nulls=True,
    include_edge_cases=True,
    include_unicode=True,
    mutation_rate=0.1,
)

# Presets
quick_config = FuzzConfig.quick()       # 10 iterations, 5 seconds
thorough_config = FuzzConfig.thorough() # 1000 iterations, 30 seconds
```

### FuzzRunner

```python
from truthound.validators.sdk.enterprise import FuzzRunner, run_fuzz_tests

# Basic fuzzing
report = run_fuzz_tests(
    MyValidator,
    iterations=100,
    seed=42,
)

print(f"Passed: {report.passed}/{report.total_iterations}")
print(f"Success rate: {report.success_rate:.1%}")
print(f"Duration: {report.total_duration_seconds:.2f}s")

# Check failures
for error in report.errors:
    print(f"Iteration {error.iteration}:")
    print(f"  Seed: {error.seed_used}")
    print(f"  Data shape: {error.data_shape}")
    print(f"  Error: {error.error}")
```

### Property-Based Testing

```python
from truthound.validators.sdk.enterprise import FuzzRunner

runner = FuzzRunner(FuzzConfig.thorough())
reports = runner.fuzz_with_properties(MyValidator)

for prop_name, report in reports.items():
    print(f"{prop_name}: {report.success_rate:.1%}")
```

Tested properties:

| Property | Description |
|----------|-------------|
| `no_crash` | No crash on any input |
| `returns_list` | Always returns a list |
| `issues_have_fields` | Issues have required fields |

### PropertyBasedTester

```python
from truthound.validators.sdk.enterprise import PropertyBasedTester

tester = PropertyBasedTester(MyValidator)

# Individual property tests
print(tester.test_no_crash(data))
print(tester.test_returns_list(data))
print(tester.test_issues_have_fields(data))

# All property tests
results = tester.run_all(data)
```

### Edge Case Values

Edge case values generated by the fuzzer:

**Numeric:**
- `0`, `-0`, `1`, `-1`
- `float("inf")`, `float("-inf")`, `float("nan")`
- `2**31 - 1`, `-(2**31)`, `2**63 - 1`, `-(2**63)`
- `1e-300`, `1e300`, `-1e-300`, `-1e300`

**String:**
- `""` (empty string)
- `" "`, `"\t"`, `"\n"`, `"\r\n"` (whitespace)
- `"null"`, `"NULL"`, `"None"`, `"undefined"`, `"NaN"`, `"inf"`
- XSS/SQL injection payloads
- Path traversal patterns
- Null bytes, long strings

**Unicode:**
- `"Hello 世界"`, `"مرحبا"`, `"שלום"`, `"🎉🚀💻"`
- Zero-width spaces, BOM

---

## 7. EnterpriseSDKManager

Manager class integrating all enterprise features.

### EnterpriseConfig

```python
from truthound.validators.sdk.enterprise import (
    EnterpriseSDKManager,
    EnterpriseConfig,
)

# Preset configurations
dev_config = EnterpriseConfig.development()  # Minimal security
prod_config = EnterpriseConfig.production(license_key="...")  # Standard security
secure_config = EnterpriseConfig.secure(license_key="...")  # Maximum security

# Custom configuration
config = EnterpriseConfig(
    # Sandbox
    sandbox_enabled=True,
    sandbox_backend=SandboxBackend.SUBPROCESS,
    sandbox_timeout_seconds=60.0,

    # Resource limits
    resource_limits=ResourceLimits.standard(),

    # Signing
    signing_enabled=True,
    signing_config=SignatureConfig.production("secret"),

    # Version checking
    version_check_enabled=True,
    truthound_version="1.0.0",

    # License
    license_check_enabled=True,
    license_secret_key="license-key",
    license_dir=Path("/licenses"),
)
```

### Integrated Execution

```python
async with EnterpriseSDKManager(config) as manager:
    # Execute with all protection features applied
    result = await manager.execute_validator(
        validator_class=MyValidator,
        data=my_dataframe,
        config={"columns": ("col1",)},
        signature=signature,  # Optional
    )

    if result.success:
        issues = result.validation_result
        print(f"Found {len(issues)} issues")
        print(f"Execution time: {result.execution_time_seconds:.2f}s")
    else:
        print(f"Failed: {result.error}")

    # Check results
    print(f"Version compatible: {result.version_compatible}")
    print(f"Signature valid: {result.signature_valid}")
    print(f"License valid: {result.license_valid}")
```

### Synchronous Execution

```python
manager = EnterpriseSDKManager(config)
result = manager.execute_validator_sync(
    MyValidator,
    data,
)
```

### Using Individual Features

```python
manager = EnterpriseSDKManager(config)

# Signing
signature = manager.sign_validator(MyValidator, signer_id="admin")
is_valid = manager.verify_validator(MyValidator, signature)

# Version compatibility
compatibility = manager.check_compatibility(MyValidator)

# License
license_info = manager.get_license(MyValidator)

# Documentation generation
docs = manager.generate_docs(MyValidator, format=DocFormat.MARKDOWN)

# Fuzzing
report = manager.fuzz_validator(MyValidator, FuzzConfig.quick())
```

### ExecutionResult

```python
@dataclass
class ExecutionResult:
    success: bool
    validation_result: Any = None     # Validation result
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

## Next Steps

- [Security Guide](security.md) - ReDoS protection, SQL injection prevention
- [Custom Validators](custom-validators.md) - SDK basic usage
- [Built-in Validators](built-in.md) - 289 built-in validators reference
