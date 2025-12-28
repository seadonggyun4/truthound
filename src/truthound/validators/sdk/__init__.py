"""Truthound Custom Validator SDK.

Enterprise-grade SDK for building custom validators with full type safety,
automatic registration, and comprehensive testing utilities.

Quick Start:
    from truthound.validators.sdk import (
        custom_validator,
        Validator,
        ValidationIssue,
        ValidatorConfig,
        Severity,
    )

    @custom_validator(name="my_validator", category="business")
    class MyValidator(Validator):
        def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
            # Your validation logic here
            return []

Features:
    - Type-safe validator development
    - Automatic registration and discovery
    - Built-in mixins for common patterns
    - Testing utilities and fixtures
    - Performance profiling integration
    - i18n support for error messages

See Also:
    - ValidatorBuilder: Fluent API for simple validators
    - ValidatorTestCase: Base class for validator testing
    - ValidatorBenchmark: Performance testing utilities
"""

from truthound.validators.base import (
    # Core classes
    Validator,
    ColumnValidator,
    AggregateValidator,
    ValidationIssue,
    ValidatorConfig,
    ValidatorExecutionResult,
    ValidationResult,
    # Mixins
    NumericValidatorMixin,
    StringValidatorMixin,
    DatetimeValidatorMixin,
    FloatValidatorMixin,
    RegexValidatorMixin,
    StreamingValidatorMixin,
    EnterpriseScaleSamplingMixin,
    SamplingInfo,
    # Utilities
    SchemaValidator,
    SafeSampler,
    TimeoutHandler,
    with_timeout,
    # Type filters
    NUMERIC_TYPES,
    STRING_TYPES,
    DATETIME_TYPES,
    FLOAT_TYPES,
    # Errors
    RegexValidationError,
    ValidationTimeoutError,
    ColumnNotFoundError,
)

from truthound.types import Severity

from truthound.validators.sdk.decorators import (
    custom_validator,
    register_validator,
    validator_metadata,
    deprecated_validator,
)

from truthound.validators.sdk.builder import (
    ValidatorBuilder,
    ColumnCheckBuilder,
    AggregateCheckBuilder,
)

from truthound.validators.sdk.testing import (
    ValidatorTestCase,
    ValidatorTestResult,
    create_test_dataframe,
    assert_no_issues,
    assert_has_issue,
    assert_issue_count,
)

from truthound.validators.sdk.templates import (
    SimpleColumnValidator,
    SimplePatternValidator,
    SimpleRangeValidator,
    SimpleComparisonValidator,
    CompositeValidator,
)

# Enterprise SDK features
from truthound.validators.sdk.enterprise import (
    # Manager
    EnterpriseSDKManager,
    EnterpriseConfig,
    ExecutionResult,
    # Sandbox
    SandboxExecutor,
    SandboxConfig,
    SandboxResult,
    SandboxBackend,
    create_sandbox,
    # Resources
    ResourceLimits,
    ResourceMonitor,
    ResourceUsage,
    CombinedResourceLimiter,
    # Signing
    SignatureManager,
    SignatureConfig,
    SignatureAlgorithm,
    ValidatorSignature,
    # Versioning
    SemanticVersion,
    VersionChecker,
    VersionCompatibility,
    VersionConstraint,
    VersionSpec,
    # Licensing
    LicenseManager,
    LicenseInfo,
    LicenseType,
    # Docs
    DocGenerator,
    DocConfig,
    DocFormat,
    ValidatorDocumentation,
    # Templates
    TemplateGenerator,
    TemplateCLI,
    TemplateConfig,
    TemplateType,
    # Fuzzing
    FuzzRunner,
    FuzzConfig,
    FuzzReport,
    FuzzStrategy,
)

__all__ = [
    # Core
    "Validator",
    "ColumnValidator",
    "AggregateValidator",
    "ValidationIssue",
    "ValidatorConfig",
    "ValidatorExecutionResult",
    "ValidationResult",
    "Severity",
    # Mixins
    "NumericValidatorMixin",
    "StringValidatorMixin",
    "DatetimeValidatorMixin",
    "FloatValidatorMixin",
    "RegexValidatorMixin",
    "StreamingValidatorMixin",
    "EnterpriseScaleSamplingMixin",
    "SamplingInfo",
    # Utilities
    "SchemaValidator",
    "SafeSampler",
    "TimeoutHandler",
    "with_timeout",
    # Type filters
    "NUMERIC_TYPES",
    "STRING_TYPES",
    "DATETIME_TYPES",
    "FLOAT_TYPES",
    # Errors
    "RegexValidationError",
    "ValidationTimeoutError",
    "ColumnNotFoundError",
    # Decorators
    "custom_validator",
    "register_validator",
    "validator_metadata",
    "deprecated_validator",
    # Builder
    "ValidatorBuilder",
    "ColumnCheckBuilder",
    "AggregateCheckBuilder",
    # Testing
    "ValidatorTestCase",
    "ValidatorTestResult",
    "create_test_dataframe",
    "assert_no_issues",
    "assert_has_issue",
    "assert_issue_count",
    # Templates
    "SimpleColumnValidator",
    "SimplePatternValidator",
    "SimpleRangeValidator",
    "SimpleComparisonValidator",
    "CompositeValidator",
    # Enterprise SDK - Manager
    "EnterpriseSDKManager",
    "EnterpriseConfig",
    "ExecutionResult",
    # Enterprise SDK - Sandbox
    "SandboxExecutor",
    "SandboxConfig",
    "SandboxResult",
    "SandboxBackend",
    "create_sandbox",
    # Enterprise SDK - Resources
    "ResourceLimits",
    "ResourceMonitor",
    "ResourceUsage",
    "CombinedResourceLimiter",
    # Enterprise SDK - Signing
    "SignatureManager",
    "SignatureConfig",
    "SignatureAlgorithm",
    "ValidatorSignature",
    # Enterprise SDK - Versioning
    "SemanticVersion",
    "VersionChecker",
    "VersionCompatibility",
    "VersionConstraint",
    "VersionSpec",
    # Enterprise SDK - Licensing
    "LicenseManager",
    "LicenseInfo",
    "LicenseType",
    # Enterprise SDK - Docs
    "DocGenerator",
    "DocConfig",
    "DocFormat",
    "ValidatorDocumentation",
    # Enterprise SDK - Templates
    "TemplateGenerator",
    "TemplateCLI",
    "TemplateConfig",
    "TemplateType",
    # Enterprise SDK - Fuzzing
    "FuzzRunner",
    "FuzzConfig",
    "FuzzReport",
    "FuzzStrategy",
]
