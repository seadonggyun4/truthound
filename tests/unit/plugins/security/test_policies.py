"""Tests for security policies and presets."""

import pytest

from truthound.plugins.security.policies import (
    SecurityPolicyPresets,
    create_policy,
    get_preset,
    list_presets,
)
from truthound.plugins.security.protocols import IsolationLevel, SecurityPolicy


class TestSecurityPolicyPresets:
    """Tests for SecurityPolicyPresets enum."""

    def test_all_presets_defined(self):
        """Test all presets are defined."""
        presets = [
            SecurityPolicyPresets.DEVELOPMENT,
            SecurityPolicyPresets.TESTING,
            SecurityPolicyPresets.STANDARD,
            SecurityPolicyPresets.ENTERPRISE,
            SecurityPolicyPresets.STRICT,
            SecurityPolicyPresets.AIRGAPPED,
        ]
        assert len(presets) == 6

    def test_preset_to_policy(self):
        """Test converting preset to policy."""
        policy = SecurityPolicyPresets.STANDARD.to_policy()
        assert isinstance(policy, SecurityPolicy)
        assert policy.isolation_level == IsolationLevel.PROCESS

    def test_development_preset(self):
        """Test development preset has no restrictions."""
        policy = SecurityPolicyPresets.DEVELOPMENT.to_policy()
        assert policy.isolation_level == IsolationLevel.NONE
        assert policy.allow_network is True
        assert policy.allow_subprocess is True
        assert policy.required_signatures == 0

    def test_strict_preset(self):
        """Test strict preset has maximum restrictions."""
        policy = SecurityPolicyPresets.STRICT.to_policy()
        assert policy.isolation_level == IsolationLevel.CONTAINER
        assert policy.allow_network is False
        assert policy.required_signatures == 2
        assert policy.require_trusted_signer is True

    def test_airgapped_preset(self):
        """Test airgapped preset blocks all network."""
        policy = SecurityPolicyPresets.AIRGAPPED.to_policy()
        assert policy.allow_network is False
        # Check syscalls are blocked
        assert "socket" in policy.resource_limits.denied_syscalls
        assert "connect" in policy.resource_limits.denied_syscalls


class TestCreatePolicy:
    """Tests for create_policy factory function."""

    def test_create_default_policy(self):
        """Test creating policy with no arguments."""
        policy = create_policy()
        # Should use STANDARD preset
        assert policy.isolation_level == IsolationLevel.PROCESS

    def test_create_from_preset_string(self):
        """Test creating policy from preset name string."""
        policy = create_policy(preset="enterprise")
        assert policy.require_trusted_signer is True

    def test_create_from_preset_enum(self):
        """Test creating policy from preset enum."""
        policy = create_policy(preset=SecurityPolicyPresets.STRICT)
        assert policy.isolation_level == IsolationLevel.CONTAINER

    def test_override_isolation_level(self):
        """Test overriding isolation level."""
        policy = create_policy(
            preset="standard",
            isolation_level="container",
        )
        assert policy.isolation_level == IsolationLevel.CONTAINER

    def test_override_memory_limit(self):
        """Test overriding memory limit."""
        policy = create_policy(max_memory_mb=1024)
        assert policy.resource_limits.max_memory_mb == 1024

    def test_override_network_access(self):
        """Test overriding network access."""
        policy = create_policy(
            preset="standard",
            allow_network=True,
        )
        assert policy.allow_network is True

    def test_override_signature_requirements(self):
        """Test overriding signature requirements."""
        policy = create_policy(
            required_signatures=3,
            require_trusted_signer=True,
        )
        assert policy.required_signatures == 3
        assert policy.require_trusted_signer is True

    def test_override_modules(self):
        """Test overriding module lists."""
        policy = create_policy(
            allowed_modules=["custom_module"],
            blocked_modules=["dangerous_module"],
        )
        assert "custom_module" in policy.allowed_modules
        assert "dangerous_module" in policy.blocked_modules

    def test_list_to_tuple_conversion(self):
        """Test list arguments are converted to tuples."""
        policy = create_policy(
            allowed_modules=["mod1", "mod2"],
        )
        assert isinstance(policy.allowed_modules, tuple)

    def test_invalid_preset_raises(self):
        """Test invalid preset name raises error."""
        with pytest.raises(ValueError, match="Unknown preset"):
            create_policy(preset="nonexistent")

    def test_invalid_isolation_level_raises(self):
        """Test invalid isolation level raises error."""
        with pytest.raises(ValueError, match="Unknown isolation level"):
            create_policy(isolation_level="invalid")


class TestGetPreset:
    """Tests for get_preset function."""

    def test_get_preset_by_name(self):
        """Test getting preset by name."""
        policy = get_preset("standard")
        assert isinstance(policy, SecurityPolicy)

    def test_case_insensitive(self):
        """Test preset names are case-insensitive."""
        policy1 = get_preset("STANDARD")
        policy2 = get_preset("standard")
        assert policy1.isolation_level == policy2.isolation_level

    def test_invalid_name_raises(self):
        """Test invalid name raises error."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("invalid")


class TestListPresets:
    """Tests for list_presets function."""

    def test_list_returns_all_presets(self):
        """Test list_presets returns all preset names."""
        presets = list_presets()
        assert "development" in presets
        assert "standard" in presets
        assert "enterprise" in presets
        assert "strict" in presets
        assert len(presets) >= 6
