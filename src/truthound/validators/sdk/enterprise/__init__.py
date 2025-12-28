"""Enterprise SDK Extensions for Custom Validators.

This module provides enterprise-grade features for custom validators:
- Sandbox execution (subprocess/Docker isolation)
- Runtime resource limits (Memory/CPU)
- Code signing and verification
- Version compatibility checking
- License management
- Auto documentation generation
- Validator template CLI
- Fuzz testing support

Architecture:
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
      │               │               │               │                │
      └───────────────┴───────────────┴───────────────┴────────────────┘
                                    │
                            ┌───────────────┐
                            │   Validator   │
                            │   Executor    │
                            └───────────────┘

Usage:
    from truthound.validators.sdk.enterprise import (
        EnterpriseSDKManager,
        SandboxConfig,
        ResourceLimits,
        SignatureManager,
    )

    # Create enterprise manager
    manager = EnterpriseSDKManager(
        sandbox_config=SandboxConfig(enabled=True, backend="subprocess"),
        resource_limits=ResourceLimits(max_memory_mb=512, max_cpu_seconds=30),
    )

    # Execute validator in sandbox
    result = await manager.execute_validator(
        validator_class=MyValidator,
        data=my_dataframe,
    )
"""

from truthound.validators.sdk.enterprise.sandbox import (
    SandboxBackend,
    SandboxConfig,
    SandboxExecutor,
    SubprocessSandbox,
    DockerSandbox,
    InProcessSandbox,
    SandboxResult,
    SandboxError,
    SandboxTimeoutError,
    SandboxResourceError,
    create_sandbox,
)

from truthound.validators.sdk.enterprise.resources import (
    ResourceLimits,
    ResourceMonitor,
    ResourceLimiter,
    MemoryLimiter,
    CPULimiter,
    CombinedResourceLimiter,
    ResourceExceededError,
    ResourceUsage,
)

from truthound.validators.sdk.enterprise.signing import (
    SignatureAlgorithm,
    SignatureConfig,
    SignatureManager,
    ValidatorSignature,
    SignatureVerificationError,
    sign_validator,
    verify_validator,
)

from truthound.validators.sdk.enterprise.versioning import (
    VersionConstraint,
    VersionCompatibility,
    VersionChecker,
    SemanticVersion,
    VersionSpec,
    VersionConflictError,
    check_compatibility,
)

from truthound.validators.sdk.enterprise.licensing import (
    LicenseType,
    LicenseInfo,
    LicenseManager,
    LicenseValidator,
    LicenseExpiredError,
    LicenseViolationError,
)

from truthound.validators.sdk.enterprise.docs import (
    DocFormat,
    DocConfig,
    DocGenerator,
    ValidatorDocumentation,
    generate_docs,
)

from truthound.validators.sdk.enterprise.templates import (
    TemplateType,
    TemplateConfig,
    TemplateCLI,
    TemplateGenerator,
    ValidatorTemplate,
    create_validator_template,
)

from truthound.validators.sdk.enterprise.fuzzing import (
    FuzzConfig,
    FuzzStrategy,
    FuzzRunner,
    FuzzResult,
    FuzzReport,
    PropertyBasedTester,
    run_fuzz_tests,
)

from truthound.validators.sdk.enterprise.manager import (
    EnterpriseSDKManager,
    EnterpriseConfig,
    ExecutionResult,
)

__all__ = [
    # Sandbox
    "SandboxBackend",
    "SandboxConfig",
    "SandboxExecutor",
    "SubprocessSandbox",
    "DockerSandbox",
    "InProcessSandbox",
    "SandboxResult",
    "SandboxError",
    "SandboxTimeoutError",
    "SandboxResourceError",
    "create_sandbox",
    # Resources
    "ResourceLimits",
    "ResourceMonitor",
    "ResourceLimiter",
    "MemoryLimiter",
    "CPULimiter",
    "CombinedResourceLimiter",
    "ResourceExceededError",
    "ResourceUsage",
    # Signing
    "SignatureAlgorithm",
    "SignatureConfig",
    "SignatureManager",
    "ValidatorSignature",
    "SignatureVerificationError",
    "sign_validator",
    "verify_validator",
    # Versioning
    "VersionConstraint",
    "VersionCompatibility",
    "VersionChecker",
    "SemanticVersion",
    "VersionSpec",
    "VersionConflictError",
    "check_compatibility",
    # Licensing
    "LicenseType",
    "LicenseInfo",
    "LicenseManager",
    "LicenseValidator",
    "LicenseExpiredError",
    "LicenseViolationError",
    # Docs
    "DocFormat",
    "DocConfig",
    "DocGenerator",
    "ValidatorDocumentation",
    "generate_docs",
    # Templates
    "TemplateType",
    "TemplateConfig",
    "TemplateCLI",
    "TemplateGenerator",
    "ValidatorTemplate",
    "create_validator_template",
    # Fuzzing
    "FuzzConfig",
    "FuzzStrategy",
    "FuzzRunner",
    "FuzzResult",
    "FuzzReport",
    "PropertyBasedTester",
    "run_fuzz_tests",
    # Manager
    "EnterpriseSDKManager",
    "EnterpriseConfig",
    "ExecutionResult",
]
