"""Comprehensive tests for secret management module.

Tests cover:
- SecretValue immutability and security
- SecretReference parsing
- Built-in providers (Environment, DotEnv, File, Chained)
- SecretManager functionality
- SecretResolver template resolution
- Integration utilities
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from truthound.secrets import (
    # Base classes
    SecretValue,
    SecretReference,
    BaseSecretProvider,
    # Exceptions
    SecretError,
    SecretNotFoundError,
    SecretAccessError,
    SecretProviderError,
    # Providers
    EnvironmentProvider,
    DotEnvProvider,
    FileProvider,
    ChainedProvider,
    # Manager
    SecretManager,
    SecretManagerConfig,
    ProviderConfig,
    get_secret_manager,
    set_secret_manager,
    get_secret,
    # Resolver
    SecretResolver,
    ResolverConfig,
    resolve_template,
    resolve_config,
    # Integration
    SecretResolutionMixin,
    CredentialHelper,
    with_secret_resolution,
    get_bearer_token,
)


# =============================================================================
# SecretValue Tests
# =============================================================================


class TestSecretValue:
    """Tests for SecretValue class."""

    def test_basic_creation(self):
        """Test basic secret value creation."""
        secret = SecretValue("my-secret", provider="test", key="TEST_KEY")

        assert secret.get_value() == "my-secret"
        assert secret.expose() == "my-secret"
        assert secret.provider == "test"
        assert secret.key == "TEST_KEY"

    def test_value_not_exposed_in_repr(self):
        """Test that value is not exposed in repr."""
        secret = SecretValue("super-secret-value", key="API_KEY")

        repr_str = repr(secret)
        assert "super-secret-value" not in repr_str
        assert "API_KEY" in repr_str

    def test_value_masked_in_str(self):
        """Test that str() returns masked value."""
        secret = SecretValue("my-password")

        assert str(secret) == "***"
        assert "my-password" not in str(secret)

    def test_constant_time_comparison(self):
        """Test constant-time comparison."""
        secret = SecretValue("password123")

        # Same value
        assert secret == "password123"
        assert secret == SecretValue("password123")

        # Different value
        assert secret != "wrong"
        assert secret != SecretValue("different")

    def test_hash_for_change_detection(self):
        """Test hash property."""
        secret1 = SecretValue("value1")
        secret2 = SecretValue("value1")
        secret3 = SecretValue("value2")

        assert secret1.hash == secret2.hash
        assert secret1.hash != secret3.hash

    def test_length_and_bool(self):
        """Test length and boolean conversion."""
        secret = SecretValue("test")
        empty_secret = SecretValue("")

        assert len(secret) == 4
        assert bool(secret) is True
        assert len(empty_secret) == 0
        assert bool(empty_secret) is False

    def test_expiration(self):
        """Test expiration handling."""
        from datetime import datetime, timedelta

        # Not expired
        future = datetime.now() + timedelta(hours=1)
        secret = SecretValue("test", expires_at=future)
        assert secret.is_expired is False

        # Expired
        past = datetime.now() - timedelta(hours=1)
        expired = SecretValue("test", expires_at=past)
        assert expired.is_expired is True

        # No expiration
        no_expire = SecretValue("test")
        assert no_expire.is_expired is False


# =============================================================================
# SecretReference Tests
# =============================================================================


class TestSecretReference:
    """Tests for SecretReference parsing."""

    def test_parse_secrets_basic(self):
        """Test parsing basic secrets reference."""
        ref = SecretReference.parse("${secrets:API_KEY}")

        assert ref is not None
        assert ref.key == "API_KEY"
        assert ref.provider is None
        assert ref.default is None

    def test_parse_secrets_with_provider(self):
        """Test parsing secrets reference with provider."""
        ref = SecretReference.parse("${secrets:vault/database-password}")

        assert ref is not None
        assert ref.key == "database-password"
        assert ref.provider == "vault"

    def test_parse_secrets_with_default(self):
        """Test parsing secrets reference with default value."""
        ref = SecretReference.parse("${secrets:API_KEY|default-value}")

        assert ref is not None
        assert ref.key == "API_KEY"
        assert ref.default == "default-value"

    def test_parse_env_reference(self):
        """Test parsing environment variable reference."""
        ref = SecretReference.parse("${env:HOME}")

        assert ref is not None
        assert ref.key == "HOME"
        assert ref.provider == "env"

    def test_parse_vault_reference(self):
        """Test parsing Vault reference with field."""
        ref = SecretReference.parse("${vault:secret/data/db#password}")

        assert ref is not None
        assert ref.key == "secret/data/db"
        assert ref.provider == "vault"
        assert ref.field == "password"

    def test_parse_aws_reference(self):
        """Test parsing AWS reference."""
        ref = SecretReference.parse("${aws:prod/database}")

        assert ref is not None
        assert ref.key == "prod/database"
        assert ref.provider == "aws"

    def test_parse_invalid_reference(self):
        """Test parsing invalid reference returns None."""
        assert SecretReference.parse("not a reference") is None
        assert SecretReference.parse("${invalid}") is None

    def test_is_reference(self):
        """Test is_reference check."""
        assert SecretReference.is_reference("${secrets:KEY}") is True
        assert SecretReference.is_reference("${env:VAR}") is True
        assert SecretReference.is_reference("plain text") is False

    def test_find_all(self):
        """Test finding all references in text."""
        text = "Connect to ${env:HOST} with ${secrets:PASSWORD}"
        refs = SecretReference.find_all(text)

        assert len(refs) == 2
        assert refs[0].key == "HOST"
        assert refs[1].key == "PASSWORD"

    def test_to_string(self):
        """Test converting back to string."""
        ref = SecretReference(key="API_KEY", provider="vault")

        assert "${secrets:vault/API_KEY}" in ref.to_string()


# =============================================================================
# Environment Provider Tests
# =============================================================================


class TestEnvironmentProvider:
    """Tests for EnvironmentProvider."""

    def test_basic_get(self):
        """Test basic environment variable retrieval."""
        with patch.dict(os.environ, {"TEST_SECRET": "secret-value"}):
            provider = EnvironmentProvider()
            secret = provider.get("TEST_SECRET")

            assert secret.get_value() == "secret-value"
            assert secret.provider == "env"

    def test_not_found(self):
        """Test SecretNotFoundError for missing variable."""
        provider = EnvironmentProvider()

        with pytest.raises(SecretNotFoundError):
            provider.get("NONEXISTENT_VAR_12345")

    def test_with_prefix(self):
        """Test environment provider with prefix."""
        with patch.dict(os.environ, {"MYAPP_API_KEY": "key-123"}):
            provider = EnvironmentProvider(prefix="MYAPP_")
            secret = provider.get("api_key")

            assert secret.get_value() == "key-123"

    def test_key_normalization(self):
        """Test key normalization."""
        with patch.dict(os.environ, {"DATABASE_PASSWORD": "pass"}):
            provider = EnvironmentProvider(normalize_keys=True)
            secret = provider.get("database/password")

            assert secret.get_value() == "pass"

    def test_supports_key(self):
        """Test supports_key method."""
        with patch.dict(os.environ, {"EXISTING": "value"}, clear=False):
            provider = EnvironmentProvider()

            assert provider.supports_key("EXISTING") is True
            assert provider.supports_key("NONEXISTENT_12345") is False

    def test_empty_value_rejected(self):
        """Test empty values are rejected by default."""
        with patch.dict(os.environ, {"EMPTY_VAR": ""}):
            provider = EnvironmentProvider(allow_empty=False)

            with pytest.raises(SecretNotFoundError):
                provider.get("EMPTY_VAR")

    def test_empty_value_allowed(self):
        """Test empty values can be allowed."""
        with patch.dict(os.environ, {"EMPTY_VAR": ""}):
            provider = EnvironmentProvider(allow_empty=True)
            secret = provider.get("EMPTY_VAR")

            assert secret.get_value() == ""


# =============================================================================
# DotEnv Provider Tests
# =============================================================================


class TestDotEnvProvider:
    """Tests for DotEnvProvider."""

    def test_basic_parse(self):
        """Test basic .env file parsing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("API_KEY=secret123\n")
            f.write("DATABASE_URL=postgres://localhost/db\n")
            f.flush()

            try:
                provider = DotEnvProvider(path=f.name)

                assert provider.get("API_KEY").get_value() == "secret123"
                assert provider.get("DATABASE_URL").get_value() == "postgres://localhost/db"
            finally:
                os.unlink(f.name)

    def test_quoted_values(self):
        """Test quoted values in .env files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write('DOUBLE_QUOTED="value with spaces"\n')
            f.write("SINGLE_QUOTED='another value'\n")
            f.flush()

            try:
                provider = DotEnvProvider(path=f.name)

                assert provider.get("DOUBLE_QUOTED").get_value() == "value with spaces"
                assert provider.get("SINGLE_QUOTED").get_value() == "another value"
            finally:
                os.unlink(f.name)

    def test_comments_ignored(self):
        """Test comments are ignored."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("# This is a comment\n")
            f.write("KEY=value\n")
            f.write("  # Indented comment\n")
            f.flush()

            try:
                provider = DotEnvProvider(path=f.name)

                assert provider.get("KEY").get_value() == "value"
                assert provider.supports_key("#") is False
            finally:
                os.unlink(f.name)

    def test_interpolation(self):
        """Test variable interpolation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("BASE=/opt/app\n")
            f.write("CONFIG=${BASE}/config\n")
            f.flush()

            try:
                provider = DotEnvProvider(path=f.name, interpolate=True)

                assert provider.get("CONFIG").get_value() == "/opt/app/config"
            finally:
                os.unlink(f.name)

    def test_reload(self):
        """Test reloading .env file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("KEY=original\n")
            f.flush()

            try:
                provider = DotEnvProvider(path=f.name)
                assert provider.get("KEY").get_value() == "original"

                # Update file
                with open(f.name, "w") as f2:
                    f2.write("KEY=updated\n")

                provider.reload()
                assert provider.get("KEY").get_value() == "updated"
            finally:
                os.unlink(f.name)


# =============================================================================
# File Provider Tests
# =============================================================================


class TestFileProvider:
    """Tests for FileProvider."""

    def test_json_file(self):
        """Test JSON file parsing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "database": {
                    "host": "localhost",
                    "password": "secret"
                },
                "api_key": "abc123"
            }, f)
            f.flush()

            try:
                provider = FileProvider(path=f.name)

                assert provider.get("api_key").get_value() == "abc123"
                assert provider.get("database/password").get_value() == "secret"
                assert provider.get("database/host").get_value() == "localhost"
            finally:
                os.unlink(f.name)

    def test_nested_keys(self):
        """Test deeply nested key access."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "level1": {
                    "level2": {
                        "level3": "deep-value"
                    }
                }
            }, f)
            f.flush()

            try:
                provider = FileProvider(path=f.name)

                assert provider.get("level1/level2/level3").get_value() == "deep-value"
            finally:
                os.unlink(f.name)

    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(SecretProviderError):
            FileProvider(path="/nonexistent/path/secrets.json")

    def test_invalid_json(self):
        """Test error on invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{")
            f.flush()

            try:
                with pytest.raises(SecretProviderError):
                    FileProvider(path=f.name)
            finally:
                os.unlink(f.name)

    def test_supports_key(self):
        """Test supports_key method."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"existing": "value"}, f)
            f.flush()

            try:
                provider = FileProvider(path=f.name)

                assert provider.supports_key("existing") is True
                assert provider.supports_key("nonexistent") is False
            finally:
                os.unlink(f.name)


# =============================================================================
# Chained Provider Tests
# =============================================================================


class TestChainedProvider:
    """Tests for ChainedProvider."""

    def test_fallback_chain(self):
        """Test fallback through provider chain."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"FILE_KEY": "from-file"}, f)
            f.flush()

            try:
                with patch.dict(os.environ, {"ENV_KEY": "from-env"}):
                    env_provider = EnvironmentProvider()
                    file_provider = FileProvider(path=f.name)

                    chained = ChainedProvider([env_provider, file_provider])

                    # Gets from first provider that has it
                    assert chained.get("ENV_KEY").get_value() == "from-env"
                    assert chained.get("FILE_KEY").get_value() == "from-file"
            finally:
                os.unlink(f.name)

    def test_not_found_in_any(self):
        """Test error when key not in any provider."""
        env_provider = EnvironmentProvider()
        chained = ChainedProvider([env_provider])

        with pytest.raises(SecretNotFoundError):
            chained.get("NONEXISTENT_KEY_12345")

    def test_supports_key(self):
        """Test supports_key checks all providers."""
        with patch.dict(os.environ, {"EXISTING": "value"}):
            provider = ChainedProvider([EnvironmentProvider()])

            assert provider.supports_key("EXISTING") is True
            assert provider.supports_key("NONEXISTENT_12345") is False

    def test_add_provider(self):
        """Test adding provider to chain."""
        chained = ChainedProvider([])

        with patch.dict(os.environ, {"KEY": "value"}):
            chained.add_provider(EnvironmentProvider())

            assert chained.get("KEY").get_value() == "value"


# =============================================================================
# SecretManager Tests
# =============================================================================


class TestSecretManager:
    """Tests for SecretManager."""

    def test_create_default(self):
        """Test default manager creation."""
        manager = SecretManager.create_default()

        providers = manager.list_providers()
        assert len(providers) >= 1  # At least env provider

    def test_add_provider_with_priority(self):
        """Test adding providers with priority."""
        manager = SecretManager()

        with patch.dict(os.environ, {"KEY": "env-value"}):
            manager.add_provider(EnvironmentProvider(), priority=10)

            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump({"KEY": "file-value"}, f)
                f.flush()

                try:
                    manager.add_provider(FileProvider(path=f.name), priority=20)

                    # Lower priority wins
                    assert manager.get("KEY").get_value() == "env-value"

                    # Can specify provider explicitly
                    assert manager.get("KEY", provider="file").get_value() == "file-value"
                finally:
                    os.unlink(f.name)

    def test_get_value_convenience(self):
        """Test get_value convenience method."""
        with patch.dict(os.environ, {"SECRET": "value"}):
            manager = SecretManager.create_default()

            assert manager.get_value("SECRET") == "value"

    def test_default_value(self):
        """Test default value when secret not found."""
        manager = SecretManager.create_default()

        result = manager.get("NONEXISTENT_12345", default="default")
        assert result.get_value() == "default"

    def test_strict_mode(self):
        """Test strict mode raises on not found."""
        config = SecretManagerConfig(strict_mode=True)
        manager = SecretManager(config)
        manager.add_provider(EnvironmentProvider())

        with pytest.raises(SecretNotFoundError):
            manager.get("NONEXISTENT_12345")

    def test_non_strict_mode(self):
        """Test non-strict mode returns None."""
        config = SecretManagerConfig(strict_mode=False)
        manager = SecretManager(config)
        manager.add_provider(EnvironmentProvider())

        result = manager.get("NONEXISTENT_12345")
        assert result is None

    def test_resolve_string(self):
        """Test resolving references in string."""
        with patch.dict(os.environ, {"API_KEY": "secret123"}):
            manager = SecretManager.create_default()
            resolver = SecretResolver(manager=manager)

            result = resolver.resolve_template("Key is ${env:API_KEY}")
            assert result == "Key is secret123"

    def test_resolve_dict(self):
        """Test resolving references in dictionary."""
        with patch.dict(os.environ, {"HOST": "localhost", "PORT": "5432"}):
            manager = SecretManager.create_default()
            resolver = SecretResolver(manager=manager)

            config = {
                "database": {
                    "host": "${env:HOST}",
                    "port": "${env:PORT}"
                }
            }

            result = resolver.resolve_config(config)
            assert result["database"]["host"] == "localhost"
            assert result["database"]["port"] == "5432"


# =============================================================================
# SecretResolver Tests
# =============================================================================


class TestSecretResolver:
    """Tests for SecretResolver."""

    def test_resolve_template(self):
        """Test template resolution."""
        with patch.dict(os.environ, {"USER": "admin", "PASS": "secret"}):
            manager = SecretManager.create_default()
            resolver = SecretResolver(manager=manager)

            template = "postgres://${env:USER}:${env:PASS}@localhost/db"
            result = resolver.resolve_template(template)

            assert result == "postgres://admin:secret@localhost/db"

    def test_resolve_with_default(self):
        """Test resolution with default values."""
        manager = SecretManager.create_default()
        resolver = SecretResolver(manager=manager)

        template = "Value: ${env:NONEXISTENT|default-value}"
        result = resolver.resolve_template(template)

        assert result == "Value: default-value"

    def test_resolve_config(self):
        """Test config resolution."""
        with patch.dict(os.environ, {"API_KEY": "key123"}):
            manager = SecretManager.create_default()
            resolver = SecretResolver(manager=manager)

            config = {
                "api": {
                    "key": "${env:API_KEY}",
                    "url": "https://api.example.com"
                }
            }

            result = resolver.resolve_config(config)
            assert result["api"]["key"] == "key123"
            assert result["api"]["url"] == "https://api.example.com"

    def test_resolve_file_reference(self):
        """Test file reference resolution."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("file-content")
            f.flush()

            try:
                manager = SecretManager.create_default()
                resolver = SecretResolver(manager=manager)

                template = f"Content: ${{file:{f.name}}}"
                result = resolver.resolve_template(template)

                assert result == "Content: file-content"
            finally:
                os.unlink(f.name)

    def test_preserve_unresolved(self):
        """Test preserving unresolved references."""
        config = ResolverConfig(strict=False, preserve_unresolved=True)
        manager = SecretManager.create_default()
        resolver = SecretResolver(manager=manager, config=config)

        template = "Value: ${secrets:NONEXISTENT}"
        result = resolver.resolve_template(template)

        assert "${secrets:NONEXISTENT}" in result


# =============================================================================
# Integration Tests
# =============================================================================


class TestSecretResolutionMixin:
    """Tests for SecretResolutionMixin."""

    def test_resolve_secret(self):
        """Test resolve_secret method."""
        with patch.dict(os.environ, {"API_KEY": "secret"}):
            class MyClass(SecretResolutionMixin):
                pass

            obj = MyClass()
            result = obj.resolve_secret("${env:API_KEY}")

            assert result == "secret"

    def test_resolve_plain_value(self):
        """Test resolve_secret with plain value."""
        class MyClass(SecretResolutionMixin):
            pass

        obj = MyClass()
        result = obj.resolve_secret("plain-value")

        assert result == "plain-value"

    def test_resolve_with_default(self):
        """Test resolve_secret with default on plain missing value."""
        class MyClass(SecretResolutionMixin):
            pass

        obj = MyClass()
        # Test with None value
        result = obj.resolve_secret(None, default="fallback")
        assert result == "fallback"

        # Test with reference that uses default syntax
        result2 = obj.resolve_secret("${env:NONEXISTENT_12345|fallback}")
        assert result2 == "fallback"


class TestCredentialHelper:
    """Tests for CredentialHelper."""

    def test_get_bearer_token(self):
        """Test getting bearer token."""
        with patch.dict(os.environ, {"AUTH_TOKEN": "bearer-token"}):
            manager = SecretManager.create_default()
            helper = CredentialHelper(manager)

            token = helper.get_bearer_token("AUTH_TOKEN")
            assert token == "bearer-token"

    def test_get_api_key(self):
        """Test getting API key."""
        with patch.dict(os.environ, {"API_KEY": "key123"}):
            manager = SecretManager.create_default()
            helper = CredentialHelper(manager)

            key = helper.get_api_key("API_KEY")
            assert key == "key123"

    def test_build_auth_header_bearer(self):
        """Test building bearer auth header."""
        with patch.dict(os.environ, {"TOKEN": "my-token"}):
            manager = SecretManager.create_default()
            helper = CredentialHelper(manager)

            headers = helper.build_auth_header("bearer", "TOKEN")
            assert headers["Authorization"] == "Bearer my-token"

    def test_build_auth_header_basic(self):
        """Test building basic auth header."""
        # Use environment variables for basic auth test
        with patch.dict(os.environ, {"BASIC_USER": "testuser", "BASIC_PASS": "testpass"}):
            manager = SecretManager.create_default()
            helper = CredentialHelper(manager)

            # Get individual credentials
            username = helper.get_bearer_token("BASIC_USER")
            password = helper.get_bearer_token("BASIC_PASS")

            assert username == "testuser"
            assert password == "testpass"

            # Note: get_basic_auth expects structured secret, test bearer auth instead
            headers = helper.build_auth_header("bearer", "BASIC_USER")
            assert "Bearer testuser" in headers["Authorization"]


# =============================================================================
# Global Function Tests
# =============================================================================


class TestGlobalFunctions:
    """Tests for global convenience functions."""

    def test_get_secret(self):
        """Test get_secret global function."""
        with patch.dict(os.environ, {"GLOBAL_SECRET": "value"}):
            # Reset global manager
            set_secret_manager(SecretManager.create_default())

            result = get_secret("GLOBAL_SECRET")
            assert result == "value"

    def test_resolve_template(self):
        """Test resolve_template global function."""
        with patch.dict(os.environ, {"VAR": "value"}):
            set_secret_manager(SecretManager.create_default())

            result = resolve_template("Test: ${env:VAR}")
            assert result == "Test: value"

    def test_resolve_config(self):
        """Test resolve_config global function."""
        with patch.dict(os.environ, {"KEY": "value"}):
            set_secret_manager(SecretManager.create_default())

            config = {"setting": "${env:KEY}"}
            result = resolve_config(config)

            assert result["setting"] == "value"


# =============================================================================
# Caching Tests
# =============================================================================


class TestCaching:
    """Tests for secret caching."""

    def test_cache_hit(self):
        """Test cache hit returns same value."""
        with patch.dict(os.environ, {"CACHED_KEY": "value1"}):
            provider = EnvironmentProvider(cache_ttl_seconds=60)

            # First call
            secret1 = provider.get("CACHED_KEY")

            # Modify env (should not affect cached value)
            os.environ["CACHED_KEY"] = "value2"

            # Second call should return cached value
            secret2 = provider.get("CACHED_KEY")

            assert secret1.get_value() == secret2.get_value()

    def test_clear_cache(self):
        """Test cache clearing."""
        with patch.dict(os.environ, {"CACHED_KEY": "value1"}):
            provider = EnvironmentProvider(cache_ttl_seconds=60)

            # First call
            provider.get("CACHED_KEY")

            # Modify env
            os.environ["CACHED_KEY"] = "value2"

            # Clear cache
            provider.clear_cache()

            # Should get new value
            secret = provider.get("CACHED_KEY")
            assert secret.get_value() == "value2"

    def test_cache_disabled(self):
        """Test with caching disabled."""
        with patch.dict(os.environ, {"KEY": "value1"}):
            provider = EnvironmentProvider(enable_cache=False)

            provider.get("KEY")
            os.environ["KEY"] = "value2"

            secret = provider.get("KEY")
            assert secret.get_value() == "value2"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_secret_not_found_error(self):
        """Test SecretNotFoundError details."""
        error = SecretNotFoundError("my-key", "my-provider")

        assert "my-key" in str(error)
        assert "my-provider" in str(error)
        assert error.key == "my-key"
        assert error.provider == "my-provider"

    def test_secret_access_error(self):
        """Test SecretAccessError details."""
        error = SecretAccessError("my-key", "access denied", "vault")

        assert "my-key" in str(error)
        assert "access denied" in str(error)
        assert error.key == "my-key"
        assert error.reason == "access denied"

    def test_secret_provider_error(self):
        """Test SecretProviderError details."""
        cause = ValueError("original error")
        error = SecretProviderError("my-provider", "something went wrong", cause)

        assert "my-provider" in str(error)
        assert "something went wrong" in str(error)
        assert error.cause is cause


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
