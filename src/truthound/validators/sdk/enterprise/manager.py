"""Enterprise SDK Manager - Unified interface for all enterprise features.

This module provides a single entry point for all enterprise SDK features:
- Sandbox execution
- Resource limiting
- Code signing
- Version checking
- License management
- Documentation generation
- Template creation
- Fuzz testing

Example:
    from truthound.validators.sdk.enterprise import EnterpriseSDKManager

    # Create manager with all features
    manager = EnterpriseSDKManager()

    # Execute validator with full protection
    result = await manager.execute_validator(
        validator_class=MyValidator,
        data=my_dataframe,
    )
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from truthound.validators.sdk.enterprise.sandbox import (
    SandboxBackend,
    SandboxConfig,
    SandboxExecutor,
    SandboxResult,
    create_sandbox,
)
from truthound.validators.sdk.enterprise.resources import (
    ResourceLimits,
    ResourceMonitor,
    CombinedResourceLimiter,
    ResourceUsage,
)
from truthound.validators.sdk.enterprise.signing import (
    SignatureAlgorithm,
    SignatureConfig,
    SignatureManager,
    ValidatorSignature,
)
from truthound.validators.sdk.enterprise.versioning import (
    VersionChecker,
    VersionCompatibility,
)
from truthound.validators.sdk.enterprise.licensing import (
    LicenseManager,
    LicenseInfo,
)
from truthound.validators.sdk.enterprise.docs import (
    DocConfig,
    DocFormat,
    DocGenerator,
    ValidatorDocumentation,
)
from truthound.validators.sdk.enterprise.fuzzing import (
    FuzzConfig,
    FuzzRunner,
    FuzzReport,
)

logger = logging.getLogger(__name__)


@dataclass
class EnterpriseConfig:
    """Configuration for Enterprise SDK Manager.

    Attributes:
        sandbox_enabled: Enable sandbox execution
        sandbox_backend: Sandbox backend type
        resource_limits: Resource limits configuration
        signing_enabled: Enable signature verification
        signing_config: Signature configuration
        version_check_enabled: Enable version compatibility checking
        truthound_version: Current Truthound version
        license_check_enabled: Enable license validation
        license_secret_key: Secret key for license validation
    """

    # Sandbox settings
    sandbox_enabled: bool = True
    sandbox_backend: SandboxBackend = SandboxBackend.SUBPROCESS
    sandbox_timeout_seconds: float = 60.0

    # Resource limits
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)

    # Signing settings
    signing_enabled: bool = False
    signing_config: SignatureConfig = field(default_factory=SignatureConfig)

    # Version settings
    version_check_enabled: bool = True
    truthound_version: str = "0.2.0"

    # License settings
    license_check_enabled: bool = False
    license_secret_key: str = ""
    license_dir: Path | None = None

    @classmethod
    def development(cls) -> "EnterpriseConfig":
        """Create development configuration (less restrictive)."""
        return cls(
            sandbox_enabled=False,
            resource_limits=ResourceLimits.generous(),
            signing_enabled=False,
            version_check_enabled=False,
            license_check_enabled=False,
        )

    @classmethod
    def production(cls, license_key: str = "") -> "EnterpriseConfig":
        """Create production configuration (full protection)."""
        return cls(
            sandbox_enabled=True,
            sandbox_backend=SandboxBackend.SUBPROCESS,
            resource_limits=ResourceLimits.standard(),
            signing_enabled=True,
            version_check_enabled=True,
            license_check_enabled=bool(license_key),
            license_secret_key=license_key,
        )

    @classmethod
    def secure(cls, license_key: str = "") -> "EnterpriseConfig":
        """Create maximum security configuration."""
        return cls(
            sandbox_enabled=True,
            sandbox_backend=SandboxBackend.DOCKER,
            sandbox_timeout_seconds=30.0,
            resource_limits=ResourceLimits.strict(),
            signing_enabled=True,
            signing_config=SignatureConfig(
                algorithm=SignatureAlgorithm.HMAC_SHA256,
                secret_key=license_key,
            ),
            version_check_enabled=True,
            license_check_enabled=True,
            license_secret_key=license_key,
        )


@dataclass
class ExecutionResult:
    """Result of enterprise validator execution.

    Attributes:
        success: Whether execution succeeded
        validation_result: Validator result (if successful)
        error: Error message (if failed)
        sandbox_result: Sandbox execution result
        resource_usage: Resource usage statistics
        signature_valid: Whether signature was valid
        version_compatible: Whether version is compatible
        license_valid: Whether license is valid
        execution_time_seconds: Total execution time
        started_at: When execution started
        finished_at: When execution finished
    """

    success: bool
    validation_result: Any = None
    error: str | None = None
    sandbox_result: SandboxResult | None = None
    resource_usage: ResourceUsage | None = None
    signature_valid: bool | None = None
    version_compatible: bool | None = None
    license_valid: bool | None = None
    execution_time_seconds: float = 0.0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "error": self.error,
            "sandbox_result": self.sandbox_result.to_dict() if self.sandbox_result else None,
            "resource_usage": self.resource_usage.to_dict() if self.resource_usage else None,
            "signature_valid": self.signature_valid,
            "version_compatible": self.version_compatible,
            "license_valid": self.license_valid,
            "execution_time_seconds": self.execution_time_seconds,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


class EnterpriseSDKManager:
    """Unified manager for all enterprise SDK features.

    Provides a single interface for:
    - Sandbox execution of validators
    - Resource monitoring and limiting
    - Signature verification
    - Version compatibility checking
    - License validation
    - Documentation generation
    - Template creation
    - Fuzz testing

    Example:
        manager = EnterpriseSDKManager()

        # Execute with all protections
        result = await manager.execute_validator(MyValidator, data)

        # Generate documentation
        docs = manager.generate_docs(MyValidator)

        # Run fuzz tests
        fuzz_report = manager.fuzz_validator(MyValidator)
    """

    def __init__(self, config: EnterpriseConfig | None = None):
        """Initialize manager.

        Args:
            config: Enterprise configuration
        """
        self.config = config or EnterpriseConfig()

        # Initialize components lazily
        self._sandbox: SandboxExecutor | None = None
        self._signature_manager: SignatureManager | None = None
        self._version_checker: VersionChecker | None = None
        self._license_manager: LicenseManager | None = None
        self._doc_generator: DocGenerator | None = None
        self._fuzz_runner: FuzzRunner | None = None

    @property
    def sandbox(self) -> SandboxExecutor:
        """Get sandbox executor (lazy initialization)."""
        if self._sandbox is None:
            sandbox_config = SandboxConfig(
                backend=self.config.sandbox_backend,
                timeout_seconds=self.config.sandbox_timeout_seconds,
                max_memory_mb=self.config.resource_limits.max_memory_mb,
            )
            self._sandbox = create_sandbox(sandbox_config)
        return self._sandbox

    @property
    def signature_manager(self) -> SignatureManager:
        """Get signature manager (lazy initialization)."""
        if self._signature_manager is None:
            self._signature_manager = SignatureManager(self.config.signing_config)
        return self._signature_manager

    @property
    def version_checker(self) -> VersionChecker:
        """Get version checker (lazy initialization)."""
        if self._version_checker is None:
            self._version_checker = VersionChecker(
                truthound_version=self.config.truthound_version
            )
        return self._version_checker

    @property
    def license_manager(self) -> LicenseManager:
        """Get license manager (lazy initialization)."""
        if self._license_manager is None:
            self._license_manager = LicenseManager(
                secret_key=self.config.license_secret_key,
                license_dir=self.config.license_dir,
            )
        return self._license_manager

    @property
    def doc_generator(self) -> DocGenerator:
        """Get documentation generator (lazy initialization)."""
        if self._doc_generator is None:
            self._doc_generator = DocGenerator()
        return self._doc_generator

    @property
    def fuzz_runner(self) -> FuzzRunner:
        """Get fuzz test runner (lazy initialization)."""
        if self._fuzz_runner is None:
            self._fuzz_runner = FuzzRunner()
        return self._fuzz_runner

    async def execute_validator(
        self,
        validator_class: type,
        data: Any,
        config: dict[str, Any] | None = None,
        signature: ValidatorSignature | None = None,
    ) -> ExecutionResult:
        """Execute a validator with full enterprise protection.

        This method:
        1. Checks version compatibility
        2. Verifies signature (if enabled)
        3. Validates license (if enabled)
        4. Executes in sandbox with resource limits
        5. Reports all results

        Args:
            validator_class: Validator class to execute
            data: Data to validate
            config: Validator configuration
            signature: Signature to verify (optional)

        Returns:
            ExecutionResult with all details
        """
        import time

        started_at = datetime.now(timezone.utc)
        start_time = time.perf_counter()

        result = ExecutionResult(
            success=False,
            started_at=started_at,
        )

        try:
            # 1. Check version compatibility
            if self.config.version_check_enabled:
                try:
                    compatibility = self.version_checker.check_compatibility(
                        validator_class,
                        raise_on_incompatible=True,
                    )
                    result.version_compatible = compatibility == VersionCompatibility.COMPATIBLE
                except Exception as e:
                    result.version_compatible = False
                    result.error = f"Version check failed: {e}"
                    return result

            # 2. Verify signature
            if self.config.signing_enabled and signature:
                try:
                    self.signature_manager.verify_validator(validator_class, signature)
                    result.signature_valid = True
                except Exception as e:
                    result.signature_valid = False
                    result.error = f"Signature verification failed: {e}"
                    return result

            # 3. Validate license
            if self.config.license_check_enabled:
                try:
                    self.license_manager.validate_license(
                        validator_class,
                        raise_on_invalid=True,
                    )
                    result.license_valid = True
                except Exception as e:
                    result.license_valid = False
                    result.error = f"License validation failed: {e}"
                    return result

            # 4. Execute in sandbox with resource limits
            if self.config.sandbox_enabled:
                sandbox_result = await self.sandbox.execute(
                    validator_class,
                    data,
                    config,
                )
                result.sandbox_result = sandbox_result

                if sandbox_result.success:
                    result.success = True
                    result.validation_result = sandbox_result.result
                else:
                    result.error = sandbox_result.error
            else:
                # Direct execution with resource limits
                limiter = CombinedResourceLimiter(self.config.resource_limits)

                with limiter.enforce() as monitor:
                    validator = validator_class(**(config or {}))
                    validation_result = validator.validate(data)

                    result.success = True
                    result.validation_result = validation_result
                    result.resource_usage = monitor.get_peak_usage()

        except Exception as e:
            result.error = str(e)

        finally:
            result.execution_time_seconds = time.perf_counter() - start_time
            result.finished_at = datetime.now(timezone.utc)

        return result

    def execute_validator_sync(
        self,
        validator_class: type,
        data: Any,
        config: dict[str, Any] | None = None,
        signature: ValidatorSignature | None = None,
    ) -> ExecutionResult:
        """Synchronous version of execute_validator.

        Args:
            validator_class: Validator class to execute
            data: Data to validate
            config: Validator configuration
            signature: Signature to verify

        Returns:
            ExecutionResult
        """
        return asyncio.run(
            self.execute_validator(validator_class, data, config, signature)
        )

    def sign_validator(
        self,
        validator_class: type,
        signer_id: str = "",
    ) -> ValidatorSignature:
        """Sign a validator.

        Args:
            validator_class: Validator to sign
            signer_id: Signer identifier

        Returns:
            ValidatorSignature
        """
        return self.signature_manager.sign_validator(validator_class, signer_id)

    def verify_validator(
        self,
        validator_class: type,
        signature: ValidatorSignature,
    ) -> bool:
        """Verify validator signature.

        Args:
            validator_class: Validator to verify
            signature: Signature to check

        Returns:
            True if valid
        """
        return self.signature_manager.verify_validator(validator_class, signature)

    def check_compatibility(
        self,
        validator_class: type,
    ) -> VersionCompatibility:
        """Check validator version compatibility.

        Args:
            validator_class: Validator to check

        Returns:
            VersionCompatibility level
        """
        return self.version_checker.check_compatibility(
            validator_class,
            raise_on_incompatible=False,
        )

    def get_license(
        self,
        validator_class: type,
    ) -> LicenseInfo:
        """Get validator license information.

        Args:
            validator_class: Validator to check

        Returns:
            LicenseInfo
        """
        return self.license_manager.get_license(validator_class)

    def generate_docs(
        self,
        validator_class: type,
        format: DocFormat = DocFormat.MARKDOWN,
    ) -> ValidatorDocumentation:
        """Generate documentation for a validator.

        Args:
            validator_class: Validator to document
            format: Output format

        Returns:
            ValidatorDocumentation
        """
        self._doc_generator = DocGenerator(DocConfig(format=format))
        return self.doc_generator.generate(validator_class)

    def fuzz_validator(
        self,
        validator_class: type,
        config: FuzzConfig | None = None,
    ) -> FuzzReport:
        """Run fuzz tests on a validator.

        Args:
            validator_class: Validator to test
            config: Fuzz configuration

        Returns:
            FuzzReport
        """
        if config:
            self._fuzz_runner = FuzzRunner(config)
        return self.fuzz_runner.fuzz(validator_class)

    async def cleanup(self) -> None:
        """Clean up all resources."""
        if self._sandbox:
            await self._sandbox.cleanup()

    def __enter__(self) -> "EnterpriseSDKManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        asyncio.run(self.cleanup())

    async def __aenter__(self) -> "EnterpriseSDKManager":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.cleanup()
