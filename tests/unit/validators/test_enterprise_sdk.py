"""Tests for Enterprise SDK Features.

This test suite covers all 8 enterprise SDK features:
1. Sandbox execution (subprocess/Docker)
2. Runtime resource limits (Memory/CPU)
3. Code signing/verification
4. Version compatibility checking
5. License management
6. Auto documentation generation
7. Validator template CLI
8. Fuzz testing support
"""

import asyncio
import pytest
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from truthound.validators.base import Validator, ValidationIssue
from truthound.types import Severity


# =============================================================================
# Test Fixtures
# =============================================================================


class SimpleTestValidator(Validator):
    """Simple validator for testing."""

    name = "simple_test"
    category = "test"
    version = "1.0.0"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Return empty list."""
        return []


class NullCheckValidator(Validator):
    """Validator that checks for null values."""

    name = "null_check"
    category = "completeness"
    version = "1.0.0"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Check for null values."""
        issues = []
        df = lf.collect()

        for col in df.columns:
            null_count = df[col].null_count()
            if null_count > 0:
                issues.append(ValidationIssue(
                    column=col,
                    issue_type="null_values",
                    count=null_count,
                    severity=Severity.MEDIUM,
                    details=f"Found {null_count} null values in {col}",
                ))
        return issues


class SlowValidator(Validator):
    """Validator that runs slowly (for timeout tests)."""

    name = "slow_validator"
    category = "test"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Simulate slow operation."""
        import time
        time.sleep(5)
        return []


class MemoryHungryValidator(Validator):
    """Validator that uses lots of memory."""

    name = "memory_hungry"
    category = "test"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Allocate large amount of memory."""
        # Allocate ~100MB
        data = [0] * (25 * 1024 * 1024)
        return []


@pytest.fixture
def sample_lf():
    """Create sample LazyFrame for testing."""
    return pl.LazyFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", None, "David", "Eve"],
        "value": [100.0, 200.0, 300.0, None, 500.0],
    })


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# 1. Sandbox Tests
# =============================================================================


class TestSandbox:
    """Tests for sandbox execution."""

    def test_sandbox_config_creation(self):
        """Test sandbox config creation."""
        from truthound.validators.sdk.enterprise.sandbox import SandboxConfig, SandboxBackend

        config = SandboxConfig()
        assert config.backend == SandboxBackend.SUBPROCESS
        assert config.timeout_seconds == 60.0
        assert config.max_memory_mb == 512

    def test_sandbox_config_strict(self):
        """Test strict sandbox config."""
        from truthound.validators.sdk.enterprise.sandbox import SandboxConfig, SandboxBackend

        config = SandboxConfig.strict()
        assert config.backend == SandboxBackend.DOCKER
        assert config.timeout_seconds == 30.0
        assert config.max_memory_mb == 256

    def test_sandbox_config_permissive(self):
        """Test permissive sandbox config."""
        from truthound.validators.sdk.enterprise.sandbox import SandboxConfig, SandboxBackend

        config = SandboxConfig.permissive()
        assert config.backend == SandboxBackend.IN_PROCESS
        assert config.network_enabled == True

    def test_in_process_sandbox(self, sample_lf):
        """Test in-process sandbox execution."""
        from truthound.validators.sdk.enterprise.sandbox import (
            InProcessSandbox,
            SandboxConfig,
            SandboxBackend,
        )

        config = SandboxConfig(backend=SandboxBackend.IN_PROCESS, timeout_seconds=10.0)
        sandbox = InProcessSandbox(config)

        result = asyncio.run(sandbox.execute(SimpleTestValidator, sample_lf))

        assert result.success
        assert result.sandbox_id
        assert result.execution_time_seconds < 10.0

    def test_sandbox_result_structure(self):
        """Test sandbox result structure."""
        from truthound.validators.sdk.enterprise.sandbox import SandboxResult

        result = SandboxResult(
            success=True,
            result=[],
            execution_time_seconds=0.5,
            sandbox_id="test123",
        )

        result_dict = result.to_dict()
        assert result_dict["success"] == True
        assert result_dict["sandbox_id"] == "test123"
        assert "started_at" in result_dict


# =============================================================================
# 2. Resource Limits Tests
# =============================================================================


class TestResourceLimits:
    """Tests for resource limiting."""

    def test_resource_limits_creation(self):
        """Test resource limits creation."""
        from truthound.validators.sdk.enterprise.resources import ResourceLimits

        limits = ResourceLimits()
        assert limits.max_memory_mb == 512
        assert limits.max_cpu_seconds == 60.0

    def test_resource_limits_strict(self):
        """Test strict resource limits."""
        from truthound.validators.sdk.enterprise.resources import ResourceLimits

        limits = ResourceLimits.strict()
        assert limits.max_memory_mb == 256
        assert limits.max_cpu_seconds == 30.0
        assert limits.graceful_degradation == False

    def test_resource_limits_generous(self):
        """Test generous resource limits."""
        from truthound.validators.sdk.enterprise.resources import ResourceLimits

        limits = ResourceLimits.generous()
        assert limits.max_memory_mb == 4096
        assert limits.max_cpu_seconds == 300.0

    def test_resource_usage_within_limits(self):
        """Test resource usage checking."""
        from truthound.validators.sdk.enterprise.resources import ResourceUsage

        usage = ResourceUsage(
            memory_mb=256,
            memory_percent=50.0,
            cpu_seconds=30.0,
            cpu_percent=50.0,
            wall_seconds=60.0,
            wall_percent=50.0,
        )

        assert usage.is_within_limits()
        assert not usage.is_near_limits(threshold=0.6)

    def test_resource_usage_near_limits(self):
        """Test near limits detection."""
        from truthound.validators.sdk.enterprise.resources import ResourceUsage

        usage = ResourceUsage(
            memory_percent=85.0,
            cpu_percent=50.0,
            wall_percent=50.0,
        )

        assert usage.is_near_limits(threshold=0.8)

    def test_resource_monitor_creation(self):
        """Test resource monitor creation."""
        from truthound.validators.sdk.enterprise.resources import (
            ResourceLimits,
            ResourceMonitor,
        )

        limits = ResourceLimits()
        monitor = ResourceMonitor(limits)

        assert monitor.limits == limits


# =============================================================================
# 3. Code Signing Tests
# =============================================================================


class TestSigning:
    """Tests for code signing and verification."""

    def test_signature_algorithm_enum(self):
        """Test signature algorithm enum."""
        from truthound.validators.sdk.enterprise.signing import SignatureAlgorithm

        assert SignatureAlgorithm.SHA256.name == "SHA256"
        assert SignatureAlgorithm.HMAC_SHA256.name == "HMAC_SHA256"

    def test_signature_config_development(self):
        """Test development signature config."""
        from truthound.validators.sdk.enterprise.signing import SignatureConfig

        config = SignatureConfig.development()
        assert config.validity_days == 30
        assert config.require_timestamp == False

    def test_signature_config_production(self):
        """Test production signature config."""
        from truthound.validators.sdk.enterprise.signing import (
            SignatureConfig,
            SignatureAlgorithm,
        )

        config = SignatureConfig.production(secret_key="test-secret")
        assert config.algorithm == SignatureAlgorithm.HMAC_SHA256
        assert config.secret_key == "test-secret"

    def test_sign_validator(self):
        """Test validator signing."""
        from truthound.validators.sdk.enterprise.signing import (
            SignatureManager,
            SignatureConfig,
        )

        config = SignatureConfig()
        manager = SignatureManager(config)

        signature = manager.sign_validator(SimpleTestValidator, signer_id="test")

        assert signature.validator_name == "simple_test"
        assert signature.validator_version == "1.0.0"
        assert signature.signer_id == "test"
        assert signature.signature

    def test_verify_validator_success(self):
        """Test successful validator verification."""
        from truthound.validators.sdk.enterprise.signing import (
            SignatureManager,
            SignatureConfig,
        )

        config = SignatureConfig()
        manager = SignatureManager(config)

        signature = manager.sign_validator(SimpleTestValidator)
        is_valid = manager.verify_validator(SimpleTestValidator, signature)

        assert is_valid == True

    def test_signature_serialization(self):
        """Test signature serialization."""
        from truthound.validators.sdk.enterprise.signing import (
            SignatureManager,
            SignatureConfig,
            ValidatorSignature,
        )

        config = SignatureConfig()
        manager = SignatureManager(config)

        signature = manager.sign_validator(SimpleTestValidator)

        # Serialize and deserialize
        json_str = signature.to_json()
        restored = ValidatorSignature.from_json(json_str)

        assert restored.validator_name == signature.validator_name
        assert restored.signature == signature.signature


# =============================================================================
# 4. Version Compatibility Tests
# =============================================================================


class TestVersioning:
    """Tests for version compatibility checking."""

    def test_semantic_version_parse(self):
        """Test semantic version parsing."""
        from truthound.validators.sdk.enterprise.versioning import SemanticVersion

        version = SemanticVersion.parse("1.2.3")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

    def test_semantic_version_with_prerelease(self):
        """Test version with prerelease."""
        from truthound.validators.sdk.enterprise.versioning import SemanticVersion

        version = SemanticVersion.parse("1.2.3-alpha.1")
        assert version.prerelease == "alpha.1"
        assert version.is_prerelease()

    def test_version_comparison(self):
        """Test version comparison."""
        from truthound.validators.sdk.enterprise.versioning import SemanticVersion

        v1 = SemanticVersion.parse("1.0.0")
        v2 = SemanticVersion.parse("2.0.0")
        v3 = SemanticVersion.parse("1.1.0")

        assert v1 < v2
        assert v1 < v3
        assert v2 > v3

    def test_version_constraint_parse(self):
        """Test version constraint parsing."""
        from truthound.validators.sdk.enterprise.versioning import (
            VersionConstraint,
            ConstraintOperator,
        )

        constraint = VersionConstraint.parse(">=1.0.0")
        assert constraint.operator == ConstraintOperator.GE
        assert constraint.version.major == 1

    def test_version_constraint_matches(self):
        """Test version constraint matching."""
        from truthound.validators.sdk.enterprise.versioning import (
            VersionConstraint,
            SemanticVersion,
        )

        constraint = VersionConstraint.parse(">=1.0.0")
        v1 = SemanticVersion.parse("1.0.0")
        v2 = SemanticVersion.parse("2.0.0")
        v3 = SemanticVersion.parse("0.9.0")

        assert constraint.matches(v1)
        assert constraint.matches(v2)
        assert not constraint.matches(v3)

    def test_version_spec_with_multiple_constraints(self):
        """Test version spec with multiple constraints."""
        from truthound.validators.sdk.enterprise.versioning import (
            VersionSpec,
            SemanticVersion,
        )

        spec = VersionSpec.parse(">=1.0.0,<2.0.0")
        v1 = SemanticVersion.parse("1.5.0")
        v2 = SemanticVersion.parse("2.0.0")

        assert spec.matches(v1)
        assert not spec.matches(v2)

    def test_version_checker(self):
        """Test version checker."""
        from truthound.validators.sdk.enterprise.versioning import (
            VersionChecker,
            VersionCompatibility,
        )

        checker = VersionChecker(truthound_version="0.2.0")
        result = checker.check_compatibility(SimpleTestValidator, raise_on_incompatible=False)

        assert result == VersionCompatibility.COMPATIBLE


# =============================================================================
# 5. License Management Tests
# =============================================================================


class TestLicensing:
    """Tests for license management."""

    def test_license_type_enum(self):
        """Test license type enum."""
        from truthound.validators.sdk.enterprise.licensing import LicenseType

        assert LicenseType.MIT.name == "MIT"
        assert LicenseType.COMMERCIAL.name == "COMMERCIAL"

    def test_license_info_creation(self):
        """Test license info creation."""
        from truthound.validators.sdk.enterprise.licensing import LicenseInfo, LicenseType

        license_info = LicenseInfo(
            license_type=LicenseType.MIT,
            validator_name="test",
        )

        assert license_info.is_open_source()
        assert not license_info.is_commercial()
        assert not license_info.is_expired()

    def test_license_info_mit(self):
        """Test MIT license shortcut."""
        from truthound.validators.sdk.enterprise.licensing import LicenseInfo

        license_info = LicenseInfo.mit("test_validator")
        assert license_info.validator_name == "test_validator"
        assert license_info.is_open_source()

    def test_license_info_trial(self):
        """Test trial license."""
        from truthound.validators.sdk.enterprise.licensing import LicenseInfo, LicenseType

        license_info = LicenseInfo.trial("test", days=30)
        assert license_info.license_type == LicenseType.TRIAL
        assert license_info.max_rows == 100000
        # Allow for slight timing differences (29-30 days)
        assert 29 <= license_info.days_until_expiry() <= 30

    def test_license_validator(self):
        """Test license validation."""
        from truthound.validators.sdk.enterprise.licensing import (
            LicenseInfo,
            LicenseValidator,
        )

        license_info = LicenseInfo.mit("test")
        validator = LicenseValidator()

        assert validator.validate(license_info)

    def test_license_manager(self):
        """Test license manager."""
        from truthound.validators.sdk.enterprise.licensing import LicenseManager

        manager = LicenseManager()
        license_info = manager.get_license(SimpleTestValidator)

        # Default should be MIT
        assert license_info.is_open_source()


# =============================================================================
# 6. Documentation Generation Tests
# =============================================================================


class TestDocGeneration:
    """Tests for documentation generation."""

    def test_doc_format_enum(self):
        """Test doc format enum."""
        from truthound.validators.sdk.enterprise.docs import DocFormat

        assert DocFormat.MARKDOWN.name == "MARKDOWN"
        assert DocFormat.RST.name == "RST"

    def test_doc_config_sphinx(self, temp_dir):
        """Test Sphinx doc config."""
        from truthound.validators.sdk.enterprise.docs import DocConfig, DocFormat

        config = DocConfig.sphinx(temp_dir)
        assert config.format == DocFormat.RST
        assert config.include_source == True

    def test_doc_config_mkdocs(self, temp_dir):
        """Test MkDocs doc config."""
        from truthound.validators.sdk.enterprise.docs import DocConfig, DocFormat

        config = DocConfig.mkdocs(temp_dir)
        assert config.format == DocFormat.MARKDOWN
        assert config.include_examples == True

    def test_doc_generator(self):
        """Test documentation generation."""
        from truthound.validators.sdk.enterprise.docs import DocGenerator

        generator = DocGenerator()
        docs = generator.generate(SimpleTestValidator)

        assert docs.name == "simple_test"
        assert docs.category == "test"
        assert docs.version == "1.0.0"

    def test_markdown_formatter(self):
        """Test markdown formatting."""
        from truthound.validators.sdk.enterprise.docs import DocGenerator, DocConfig, DocFormat

        config = DocConfig(format=DocFormat.MARKDOWN)
        generator = DocGenerator(config)
        docs = generator.generate(SimpleTestValidator)
        markdown = generator.format(docs)

        assert "# simple_test" in markdown
        assert "test validator" in markdown

    def test_doc_write(self, temp_dir):
        """Test documentation file writing."""
        from truthound.validators.sdk.enterprise.docs import DocGenerator

        generator = DocGenerator()
        docs = generator.generate(SimpleTestValidator)
        path = generator.write(docs, temp_dir)

        assert path.exists()
        assert path.suffix == ".md"


# =============================================================================
# 7. Template CLI Tests
# =============================================================================


class TestTemplateCLI:
    """Tests for validator template CLI."""

    def test_template_type_enum(self):
        """Test template type enum."""
        from truthound.validators.sdk.enterprise.templates import TemplateType

        assert TemplateType.BASIC.name == "BASIC"
        assert TemplateType.COLUMN.name == "COLUMN"
        assert TemplateType.PATTERN.name == "PATTERN"

    def test_template_config(self):
        """Test template config creation."""
        from truthound.validators.sdk.enterprise.templates import TemplateConfig, TemplateType

        config = TemplateConfig(
            name="my_validator",
            category="custom",
            template_type=TemplateType.COLUMN,
        )

        assert config.name == "my_validator"
        assert config.include_tests == True

    def test_template_generator(self):
        """Test template generation."""
        from truthound.validators.sdk.enterprise.templates import (
            TemplateConfig,
            TemplateGenerator,
            TemplateType,
        )

        config = TemplateConfig(
            name="test_validator",
            category="test",
            template_type=TemplateType.BASIC,
            author="Test Author",
        )

        generator = TemplateGenerator(config)
        template = generator.generate()

        assert template.name == "test_validator"
        assert "class TestValidatorValidator" in template.source_code
        assert "@custom_validator" in template.source_code

    def test_template_cli_create(self, temp_dir):
        """Test CLI validator creation."""
        from truthound.validators.sdk.enterprise.templates import TemplateCLI, TemplateType

        cli = TemplateCLI(temp_dir)
        files = cli.create_validator(
            name="my_test_validator",
            category="custom",
            template_type=TemplateType.BASIC,
        )

        assert "source" in files
        assert files["source"].exists()
        assert "init" in files

    def test_template_cli_list_templates(self):
        """Test listing available templates."""
        from truthound.validators.sdk.enterprise.templates import TemplateCLI

        cli = TemplateCLI()
        templates = cli.list_templates()

        assert len(templates) > 0
        assert any(t["type"] == "BASIC" for t in templates)
        assert any(t["type"] == "COLUMN" for t in templates)


# =============================================================================
# 8. Fuzz Testing Tests
# =============================================================================


class TestFuzzing:
    """Tests for fuzz testing support."""

    def test_fuzz_strategy_enum(self):
        """Test fuzz strategy enum."""
        from truthound.validators.sdk.enterprise.fuzzing import FuzzStrategy

        assert FuzzStrategy.RANDOM.name == "RANDOM"
        assert FuzzStrategy.BOUNDARY.name == "BOUNDARY"

    def test_fuzz_config_quick(self):
        """Test quick fuzz config."""
        from truthound.validators.sdk.enterprise.fuzzing import FuzzConfig

        config = FuzzConfig.quick()
        assert config.iterations == 10
        assert config.max_rows == 100

    def test_fuzz_config_thorough(self):
        """Test thorough fuzz config."""
        from truthound.validators.sdk.enterprise.fuzzing import FuzzConfig

        config = FuzzConfig.thorough()
        assert config.iterations == 1000
        assert config.max_rows == 10000

    def test_fuzz_result(self):
        """Test fuzz result structure."""
        from truthound.validators.sdk.enterprise.fuzzing import FuzzResult

        result = FuzzResult(
            iteration=0,
            success=True,
            duration_seconds=0.5,
            data_shape=(100, 5),
            seed_used=12345,
        )

        result_dict = result.to_dict()
        assert result_dict["success"] == True
        assert result_dict["data_shape"] == (100, 5)

    def test_fuzz_report(self):
        """Test fuzz report."""
        from truthound.validators.sdk.enterprise.fuzzing import FuzzReport, FuzzResult

        report = FuzzReport(
            total_iterations=10,
            passed=8,
            failed=2,
        )

        assert report.success_rate == 0.8

    def test_random_data_generator(self):
        """Test random data generation."""
        from truthound.validators.sdk.enterprise.fuzzing import (
            RandomDataGenerator,
            FuzzConfig,
        )

        config = FuzzConfig(max_rows=10, max_columns=3)
        generator = RandomDataGenerator(config)

        data = generator.generate(rows=10, columns=3, seed=42)

        df = data.collect()
        assert df.height == 10
        assert len(df.columns) == 3

    def test_fuzz_runner(self):
        """Test fuzz runner."""
        from truthound.validators.sdk.enterprise.fuzzing import FuzzRunner, FuzzConfig

        config = FuzzConfig(iterations=5, max_rows=10, max_columns=3)
        runner = FuzzRunner(config)

        report = runner.fuzz(SimpleTestValidator)

        assert report.total_iterations == 5
        assert report.success_rate >= 0.0

    def test_property_based_tester(self, sample_lf):
        """Test property-based testing."""
        from truthound.validators.sdk.enterprise.fuzzing import PropertyBasedTester

        tester = PropertyBasedTester(SimpleTestValidator)
        results = tester.run_all(sample_lf)

        assert results["no_crash"] == True
        assert results["returns_list"] == True
        assert results["issues_have_fields"] == True


# =============================================================================
# Enterprise Manager Integration Tests
# =============================================================================


class TestEnterpriseManager:
    """Integration tests for Enterprise SDK Manager."""

    def test_enterprise_config_development(self):
        """Test development config."""
        from truthound.validators.sdk.enterprise.manager import EnterpriseConfig

        config = EnterpriseConfig.development()
        assert config.sandbox_enabled == False
        assert config.signing_enabled == False

    def test_enterprise_config_production(self):
        """Test production config."""
        from truthound.validators.sdk.enterprise.manager import EnterpriseConfig

        config = EnterpriseConfig.production(license_key="test-key")
        assert config.sandbox_enabled == True
        assert config.signing_enabled == True

    def test_manager_creation(self):
        """Test manager creation."""
        from truthound.validators.sdk.enterprise.manager import EnterpriseSDKManager

        manager = EnterpriseSDKManager()
        assert manager.config is not None

    def test_manager_sign_validator(self):
        """Test signing through manager."""
        from truthound.validators.sdk.enterprise.manager import EnterpriseSDKManager

        manager = EnterpriseSDKManager()
        signature = manager.sign_validator(SimpleTestValidator, signer_id="test")

        assert signature.validator_name == "simple_test"

    def test_manager_check_compatibility(self):
        """Test compatibility check through manager."""
        from truthound.validators.sdk.enterprise.manager import EnterpriseSDKManager
        from truthound.validators.sdk.enterprise.versioning import VersionCompatibility

        manager = EnterpriseSDKManager()
        result = manager.check_compatibility(SimpleTestValidator)

        assert result == VersionCompatibility.COMPATIBLE

    def test_manager_generate_docs(self):
        """Test doc generation through manager."""
        from truthound.validators.sdk.enterprise.manager import EnterpriseSDKManager

        manager = EnterpriseSDKManager()
        docs = manager.generate_docs(SimpleTestValidator)

        assert docs.name == "simple_test"

    def test_manager_fuzz_validator(self):
        """Test fuzz testing through manager."""
        from truthound.validators.sdk.enterprise.manager import EnterpriseSDKManager
        from truthound.validators.sdk.enterprise.fuzzing import FuzzConfig

        manager = EnterpriseSDKManager()
        config = FuzzConfig(iterations=3, max_rows=10)
        report = manager.fuzz_validator(SimpleTestValidator, config)

        assert report.total_iterations == 3

    def test_manager_execute_validator_sync(self, sample_lf):
        """Test synchronous validator execution."""
        from truthound.validators.sdk.enterprise.manager import (
            EnterpriseSDKManager,
            EnterpriseConfig,
        )

        config = EnterpriseConfig.development()
        manager = EnterpriseSDKManager(config)

        result = manager.execute_validator_sync(
            SimpleTestValidator,
            sample_lf,
        )

        assert result.success
        assert result.validation_result == []

    def test_execution_result_structure(self):
        """Test execution result structure."""
        from truthound.validators.sdk.enterprise.manager import ExecutionResult

        result = ExecutionResult(
            success=True,
            validation_result=[],
            execution_time_seconds=0.5,
        )

        result_dict = result.to_dict()
        assert result_dict["success"] == True
        assert "started_at" in result_dict
