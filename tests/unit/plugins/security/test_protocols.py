"""Tests for security protocols and data classes."""

import pytest
from datetime import datetime, timedelta, timezone

from truthound.plugins.security.protocols import (
    IsolationLevel,
    TrustLevel,
    ResourceLimits,
    SecurityPolicy,
    SignatureInfo,
    VerificationResult,
)


class TestIsolationLevel:
    """Tests for IsolationLevel enum."""

    def test_isolation_levels_exist(self):
        """Test all isolation levels are defined."""
        assert IsolationLevel.NONE is not None
        assert IsolationLevel.PROCESS is not None
        assert IsolationLevel.CONTAINER is not None
        assert IsolationLevel.WASM is not None

    def test_isolation_level_ordering(self):
        """Test isolation levels can be compared by value."""
        # Values should be auto-assigned in definition order
        assert IsolationLevel.NONE.value < IsolationLevel.PROCESS.value
        assert IsolationLevel.PROCESS.value < IsolationLevel.CONTAINER.value


class TestResourceLimits:
    """Tests for ResourceLimits dataclass."""

    def test_default_limits(self):
        """Test default resource limits."""
        limits = ResourceLimits()
        assert limits.max_memory_mb == 512
        assert limits.max_cpu_percent == 50.0
        assert limits.max_execution_time_sec == 30.0
        assert limits.max_file_descriptors == 100

    def test_minimal_preset(self):
        """Test minimal resource limits."""
        limits = ResourceLimits.minimal()
        assert limits.max_memory_mb == 128
        assert limits.max_cpu_percent == 25.0
        assert limits.max_execution_time_sec == 10.0

    def test_standard_preset(self):
        """Test standard resource limits."""
        limits = ResourceLimits.standard()
        assert limits.max_memory_mb == 512

    def test_generous_preset(self):
        """Test generous resource limits."""
        limits = ResourceLimits.generous()
        assert limits.max_memory_mb == 2048
        assert limits.max_execution_time_sec == 300.0

    def test_immutability(self):
        """Test ResourceLimits is immutable."""
        limits = ResourceLimits()
        with pytest.raises(AttributeError):
            limits.max_memory_mb = 1024  # type: ignore


class TestSecurityPolicy:
    """Tests for SecurityPolicy dataclass."""

    def test_default_policy(self):
        """Test default security policy values."""
        policy = SecurityPolicy()
        assert policy.isolation_level == IsolationLevel.PROCESS
        assert policy.allow_network is False
        assert policy.allow_subprocess is False
        assert policy.required_signatures == 1

    def test_strict_policy(self):
        """Test strict security policy."""
        policy = SecurityPolicy.strict()
        assert policy.isolation_level == IsolationLevel.CONTAINER
        assert policy.required_signatures == 2
        assert policy.require_trusted_signer is True
        assert policy.resource_limits.max_memory_mb == 128

    def test_permissive_policy(self):
        """Test permissive security policy."""
        policy = SecurityPolicy.permissive()
        assert policy.isolation_level == IsolationLevel.NONE
        assert policy.allow_network is True
        assert policy.allow_file_write is True
        assert policy.required_signatures == 0

    def test_development_policy(self):
        """Test development security policy."""
        policy = SecurityPolicy.development()
        assert policy.isolation_level == IsolationLevel.NONE
        assert policy.allow_subprocess is True
        assert len(policy.blocked_modules) == 0

    def test_blocked_modules_default(self):
        """Test default blocked modules."""
        policy = SecurityPolicy.standard()
        assert "os" in policy.blocked_modules
        assert "subprocess" in policy.blocked_modules
        assert "socket" in policy.blocked_modules

    def test_allowed_modules_default(self):
        """Test default allowed modules."""
        policy = SecurityPolicy.standard()
        assert "polars" in policy.allowed_modules
        assert "numpy" in policy.allowed_modules


class TestSignatureInfo:
    """Tests for SignatureInfo dataclass."""

    def test_create_signature_info(self):
        """Test creating SignatureInfo."""
        sig = SignatureInfo(
            signer_id="test-signer",
            algorithm="RSA-SHA256",
            signature=b"test-signature",
        )
        assert sig.signer_id == "test-signer"
        assert sig.algorithm == "RSA-SHA256"
        assert sig.signature == b"test-signature"

    def test_signature_not_expired(self):
        """Test non-expired signature."""
        future = datetime.now(timezone.utc) + timedelta(days=30)
        sig = SignatureInfo(
            signer_id="test",
            algorithm="SHA256",
            signature=b"sig",
            expires_at=future,
        )
        assert sig.is_expired() is False

    def test_signature_expired(self):
        """Test expired signature."""
        past = datetime.now(timezone.utc) - timedelta(days=1)
        sig = SignatureInfo(
            signer_id="test",
            algorithm="SHA256",
            signature=b"sig",
            expires_at=past,
        )
        assert sig.is_expired() is True

    def test_signature_no_expiry(self):
        """Test signature without expiry."""
        sig = SignatureInfo(
            signer_id="test",
            algorithm="SHA256",
            signature=b"sig",
            expires_at=None,
        )
        assert sig.is_expired() is False

    def test_signature_age_days(self):
        """Test signature age calculation."""
        past = datetime.now(timezone.utc) - timedelta(days=10)
        sig = SignatureInfo(
            signer_id="test",
            algorithm="SHA256",
            signature=b"sig",
            timestamp=past,
        )
        assert sig.age_days >= 10


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_success_result(self):
        """Test creating successful verification result."""
        result = VerificationResult.success(
            signer_id="trusted-signer",
            trust_level=TrustLevel.TRUSTED,
        )
        assert result.is_valid is True
        assert result.signer_id == "trusted-signer"
        assert result.trust_level == TrustLevel.TRUSTED
        assert len(result.errors) == 0

    def test_failure_result(self):
        """Test creating failed verification result."""
        result = VerificationResult.failure(
            "Signature invalid",
            "Hash mismatch",
        )
        assert result.is_valid is False
        assert len(result.errors) == 2
        assert "Signature invalid" in result.errors

    def test_success_with_warnings(self):
        """Test successful result with warnings."""
        result = VerificationResult.success(
            signer_id="signer",
            warnings=("Signature near expiry",),
        )
        assert result.is_valid is True
        assert len(result.warnings) == 1
