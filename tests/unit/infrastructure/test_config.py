"""Tests for enterprise configuration system."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from truthound.infrastructure.config import (
    # Core
    ConfigManager,
    ConfigProfile,
    Environment,
    # Sources
    ConfigSource,
    EnvConfigSource,
    FileConfigSource,
    VaultConfigSource,
    AwsSecretsSource,
    # Validation
    ConfigSchema,
    ConfigField,
    ConfigValidator,
    ConfigValidationError,
    # Factory
    get_config,
    load_config,
    reload_config,
)


class TestEnvironment:
    """Tests for Environment enum."""

    def test_from_string(self):
        """Test parsing environment from string."""
        assert Environment.from_string("dev") == Environment.DEVELOPMENT
        assert Environment.from_string("development") == Environment.DEVELOPMENT
        assert Environment.from_string("test") == Environment.TESTING
        assert Environment.from_string("testing") == Environment.TESTING
        assert Environment.from_string("stage") == Environment.STAGING
        assert Environment.from_string("staging") == Environment.STAGING
        assert Environment.from_string("prod") == Environment.PRODUCTION
        assert Environment.from_string("production") == Environment.PRODUCTION
        assert Environment.from_string("local") == Environment.LOCAL

    def test_from_string_case_insensitive(self):
        """Test case insensitivity."""
        assert Environment.from_string("PRODUCTION") == Environment.PRODUCTION
        assert Environment.from_string("Production") == Environment.PRODUCTION
        assert Environment.from_string("PROD") == Environment.PRODUCTION

    def test_unknown_defaults_to_development(self):
        """Test unknown environment defaults to development."""
        assert Environment.from_string("unknown") == Environment.DEVELOPMENT
        assert Environment.from_string("") == Environment.DEVELOPMENT

    def test_is_production(self):
        """Test is_production property."""
        assert Environment.PRODUCTION.is_production is True
        assert Environment.STAGING.is_production is True
        assert Environment.DEVELOPMENT.is_production is False
        assert Environment.TESTING.is_production is False
        assert Environment.LOCAL.is_production is False

    def test_is_development(self):
        """Test is_development property."""
        assert Environment.DEVELOPMENT.is_development is True
        assert Environment.TESTING.is_development is True
        assert Environment.LOCAL.is_development is True
        assert Environment.PRODUCTION.is_development is False
        assert Environment.STAGING.is_development is False

    def test_current_from_env(self):
        """Test getting current environment from env vars."""
        with patch.dict("os.environ", {"TRUTHOUND_ENV": "production"}):
            assert Environment.current() == Environment.PRODUCTION

        with patch.dict("os.environ", {"ENVIRONMENT": "staging"}, clear=True):
            # Clear TRUTHOUND_ENV first
            os.environ.pop("TRUTHOUND_ENV", None)
            assert Environment.current() == Environment.STAGING


class TestEnvConfigSource:
    """Tests for EnvConfigSource."""

    def test_load_simple_values(self):
        """Test loading simple values."""
        with patch.dict("os.environ", {
            "TRUTHOUND_LOG_LEVEL": "DEBUG",
            "TRUTHOUND_PORT": "8080",
            "TRUTHOUND_DEBUG": "true",
        }):
            source = EnvConfigSource(prefix="TRUTHOUND")
            config = source.load()

        assert config["log"]["level"] == "DEBUG"
        assert config["port"] == 8080
        assert config["debug"] is True

    def test_nested_values(self):
        """Test loading nested values."""
        with patch.dict("os.environ", {
            "TRUTHOUND_DATABASE_HOST": "localhost",
            "TRUTHOUND_DATABASE_PORT": "5432",
            "TRUTHOUND_DATABASE_USER": "admin",
        }):
            source = EnvConfigSource(prefix="TRUTHOUND")
            config = source.load()

        assert config["database"]["host"] == "localhost"
        assert config["database"]["port"] == 5432
        assert config["database"]["user"] == "admin"

    def test_boolean_parsing(self):
        """Test boolean value parsing."""
        with patch.dict("os.environ", {
            "TRUTHOUND_ENABLED": "true",
            "TRUTHOUND_DEBUG": "yes",
            "TRUTHOUND_ACTIVE": "1",
            "TRUTHOUND_DISABLED": "false",
            "TRUTHOUND_OFF": "no",
            "TRUTHOUND_INACTIVE": "0",
        }):
            source = EnvConfigSource(prefix="TRUTHOUND")
            config = source.load()

        assert config["enabled"] is True
        assert config["debug"] is True
        assert config["active"] is True
        assert config["disabled"] is False
        assert config["off"] is False
        assert config["inactive"] is False

    def test_json_values(self):
        """Test JSON value parsing."""
        with patch.dict("os.environ", {
            "TRUTHOUND_HOSTS": '["host1", "host2"]',
            "TRUTHOUND_OPTIONS": '{"key": "value"}',
        }):
            source = EnvConfigSource(prefix="TRUTHOUND")
            config = source.load()

        assert config["hosts"] == ["host1", "host2"]
        assert config["options"] == {"key": "value"}

    def test_custom_prefix(self):
        """Test custom prefix."""
        with patch.dict("os.environ", {
            "MYAPP_VALUE": "test",
        }):
            source = EnvConfigSource(prefix="MYAPP")
            config = source.load()

        assert config["value"] == "test"


class TestFileConfigSource:
    """Tests for FileConfigSource."""

    def test_load_json_file(self, tmp_path):
        """Test loading JSON file."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "database": {"host": "localhost", "port": 5432},
            "debug": True,
        }))

        source = FileConfigSource(config_file)
        config = source.load()

        assert config["database"]["host"] == "localhost"
        assert config["database"]["port"] == 5432
        assert config["debug"] is True

    def test_load_yaml_file(self, tmp_path):
        """Test loading YAML file."""
        pytest.importorskip("yaml")

        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
database:
  host: localhost
  port: 5432
debug: true
""")

        source = FileConfigSource(config_file)
        config = source.load()

        assert config["database"]["host"] == "localhost"
        assert config["database"]["port"] == 5432
        assert config["debug"] is True

    def test_missing_optional_file(self, tmp_path):
        """Test missing optional file returns empty config."""
        config_file = tmp_path / "missing.json"

        source = FileConfigSource(config_file, required=False)
        config = source.load()

        assert config == {}

    def test_missing_required_file(self, tmp_path):
        """Test missing required file raises error."""
        from truthound.infrastructure.config import ConfigSourceError

        config_file = tmp_path / "missing.json"

        source = FileConfigSource(config_file, required=True)
        with pytest.raises(ConfigSourceError):
            source.load()

    def test_reload_on_change(self, tmp_path):
        """Test reloading when file changes."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"value": 1}))

        source = FileConfigSource(config_file)
        config1 = source.load()
        assert config1["value"] == 1

        # Modify file
        import time
        time.sleep(0.01)  # Ensure mtime changes
        config_file.write_text(json.dumps({"value": 2}))

        config2 = source.reload()
        assert config2["value"] == 2


class TestConfigSchema:
    """Tests for ConfigSchema."""

    def test_create_schema(self):
        """Test creating a schema."""
        schema = ConfigSchema()
        schema.add_field("name", str, required=True)
        schema.add_field("port", int, min_value=1, max_value=65535)
        schema.add_field("level", str, choices=["DEBUG", "INFO", "ERROR"])

        assert len(schema.fields) == 3

    def test_field_with_all_options(self):
        """Test field with all options."""
        field = ConfigField(
            name="email",
            type=str,
            required=True,
            pattern=r"^[\w.-]+@[\w.-]+\.\w+$",
            description="Email address",
        )

        assert field.name == "email"
        assert field.type == str
        assert field.required is True


class TestConfigValidator:
    """Tests for ConfigValidator."""

    def test_validate_required_field(self):
        """Test validation of required field."""
        schema = ConfigSchema(fields=[
            ConfigField(name="required_field", type=str, required=True),
        ])
        validator = ConfigValidator(schema)

        errors = validator.validate({"required_field": "value"})
        assert len(errors) == 0

        errors = validator.validate({})
        assert len(errors) == 1
        assert "required" in errors[0].lower()

    def test_validate_type(self):
        """Test type validation."""
        schema = ConfigSchema(fields=[
            ConfigField(name="port", type=int),
        ])
        validator = ConfigValidator(schema)

        errors = validator.validate({"port": 8080})
        assert len(errors) == 0

        errors = validator.validate({"port": "8080"})
        assert len(errors) == 1
        assert "int" in errors[0].lower()

    def test_validate_range(self):
        """Test range validation."""
        schema = ConfigSchema(fields=[
            ConfigField(name="port", type=int, min_value=1, max_value=65535),
        ])
        validator = ConfigValidator(schema)

        errors = validator.validate({"port": 8080})
        assert len(errors) == 0

        errors = validator.validate({"port": 0})
        assert len(errors) == 1
        assert ">=" in errors[0]

        errors = validator.validate({"port": 70000})
        assert len(errors) == 1
        assert "<=" in errors[0]

    def test_validate_pattern(self):
        """Test pattern validation."""
        schema = ConfigSchema(fields=[
            ConfigField(name="email", type=str, pattern=r"^[\w.-]+@[\w.-]+\.\w+$"),
        ])
        validator = ConfigValidator(schema)

        errors = validator.validate({"email": "user@example.com"})
        assert len(errors) == 0

        errors = validator.validate({"email": "invalid-email"})
        assert len(errors) == 1
        assert "pattern" in errors[0].lower()

    def test_validate_choices(self):
        """Test choices validation."""
        schema = ConfigSchema(fields=[
            ConfigField(name="level", type=str, choices=["DEBUG", "INFO", "ERROR"]),
        ])
        validator = ConfigValidator(schema)

        errors = validator.validate({"level": "INFO"})
        assert len(errors) == 0

        errors = validator.validate({"level": "TRACE"})
        assert len(errors) == 1
        assert "one of" in errors[0].lower()

    def test_validate_nested_field(self):
        """Test validation of nested field."""
        schema = ConfigSchema(fields=[
            ConfigField(name="database.host", type=str, required=True),
            ConfigField(name="database.port", type=int),
        ])
        validator = ConfigValidator(schema)

        errors = validator.validate({
            "database": {"host": "localhost", "port": 5432}
        })
        assert len(errors) == 0

        errors = validator.validate({
            "database": {"port": 5432}
        })
        assert len(errors) == 1
        assert "database.host" in errors[0]


class TestConfigProfile:
    """Tests for ConfigProfile."""

    def test_get_value(self):
        """Test getting a value."""
        profile = ConfigProfile({"key": "value"})
        assert profile.get("key") == "value"

    def test_get_nested_value(self):
        """Test getting a nested value."""
        profile = ConfigProfile({
            "database": {"host": "localhost", "port": 5432}
        })
        assert profile.get("database.host") == "localhost"
        assert profile.get("database.port") == 5432

    def test_get_with_default(self):
        """Test getting with default."""
        profile = ConfigProfile({})
        assert profile.get("missing", default="default") == "default"

    def test_get_required(self):
        """Test getting required value."""
        from truthound.infrastructure.config import ConfigError

        profile = ConfigProfile({})
        with pytest.raises(ConfigError):
            profile.get("missing", required=True)

    def test_typed_getters(self):
        """Test typed getter methods."""
        profile = ConfigProfile({
            "string": "hello",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "list": [1, 2, 3],
            "dict": {"key": "value"},
        })

        assert profile.get_str("string") == "hello"
        assert profile.get_int("int") == 42
        assert profile.get_float("float") == 3.14
        assert profile.get_bool("bool") is True
        assert profile.get_list("list") == [1, 2, 3]
        assert profile.get_dict("dict") == {"key": "value"}

    def test_type_conversion(self):
        """Test automatic type conversion."""
        profile = ConfigProfile({
            "int_str": "42",
            "float_str": "3.14",
            "bool_str": "true",
        })

        assert profile.get_int("int_str") == 42
        assert profile.get_float("float_str") == 3.14
        assert profile.get_bool("bool_str") is True

    def test_environment_properties(self):
        """Test environment properties."""
        prod_profile = ConfigProfile({}, environment=Environment.PRODUCTION)
        dev_profile = ConfigProfile({}, environment=Environment.DEVELOPMENT)

        assert prod_profile.is_production is True
        assert prod_profile.is_development is False
        assert dev_profile.is_production is False
        assert dev_profile.is_development is True

    def test_contains(self):
        """Test contains check."""
        profile = ConfigProfile({"key": "value", "nested": {"key": "value"}})

        assert "key" in profile
        assert "nested.key" in profile
        assert "missing" not in profile

    def test_getitem(self):
        """Test bracket access."""
        profile = ConfigProfile({"key": "value"})

        assert profile["key"] == "value"

        with pytest.raises(KeyError):
            _ = profile["missing"]


class TestConfigManager:
    """Tests for ConfigManager."""

    def test_create_manager(self):
        """Test creating a manager."""
        manager = ConfigManager(environment=Environment.PRODUCTION)
        assert manager.environment == Environment.PRODUCTION

    def test_add_sources(self):
        """Test adding configuration sources."""
        manager = ConfigManager()

        with patch.dict("os.environ", {"TRUTHOUND_KEY": "value"}):
            manager.add_source(EnvConfigSource())
            config = manager.load(validate=False)

        assert config.get("key") == "value"

    def test_source_priority(self, tmp_path):
        """Test source priority (higher priority overrides)."""
        # Create file with lower priority
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"key": "from_file"}))

        manager = ConfigManager()
        manager.add_source(FileConfigSource(config_file, priority=10))

        with patch.dict("os.environ", {"TRUTHOUND_KEY": "from_env"}):
            manager.add_source(EnvConfigSource(priority=100))  # Higher priority
            config = manager.load(validate=False)

        # Env source should override file
        assert config.get("key") == "from_env"

    def test_config_merging(self, tmp_path):
        """Test deep config merging."""
        base_file = tmp_path / "base.json"
        base_file.write_text(json.dumps({
            "database": {"host": "localhost", "port": 5432},
            "logging": {"level": "INFO"},
        }))

        override_file = tmp_path / "override.json"
        override_file.write_text(json.dumps({
            "database": {"port": 3306},
            "debug": True,
        }))

        manager = ConfigManager()
        manager.add_source(FileConfigSource(base_file, priority=10))
        manager.add_source(FileConfigSource(override_file, priority=20))
        config = manager.load(validate=False)

        # Merged config
        assert config.get("database.host") == "localhost"  # From base
        assert config.get("database.port") == 3306  # Overridden
        assert config.get("logging.level") == "INFO"  # From base
        assert config.get("debug") is True  # From override

    def test_validation(self):
        """Test config validation."""
        schema = ConfigSchema(fields=[
            ConfigField(name="required", type=str, required=True),
        ])

        manager = ConfigManager()
        manager.set_schema(schema)

        with patch.dict("os.environ", {"TRUTHOUND_OTHER": "value"}):
            manager.add_source(EnvConfigSource())
            with pytest.raises(ConfigValidationError):
                manager.load(validate=True)

    def test_reload_callback(self, tmp_path):
        """Test reload callback."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"value": 1}))

        callback_called = []

        def on_reload(config):
            callback_called.append(config.get("value"))

        manager = ConfigManager()
        manager.add_source(FileConfigSource(config_file))
        manager.on_reload(on_reload)
        manager.load(validate=False)

        # Modify and reload
        import time
        time.sleep(0.01)
        config_file.write_text(json.dumps({"value": 2}))
        manager.reload()

        assert callback_called == [2]


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_from_path(self, tmp_path):
        """Test loading config from path."""
        # Create config files
        base_file = tmp_path / "base.json"
        base_file.write_text(json.dumps({"key": "from_base"}))

        config = load_config(
            environment=Environment.DEVELOPMENT,
            config_path=tmp_path,
            validate=False,
        )

        assert config.get("key") == "from_base"

    def test_load_env_specific(self, tmp_path):
        """Test loading environment-specific config."""
        base_file = tmp_path / "base.json"
        base_file.write_text(json.dumps({"key": "base", "base_only": True}))

        prod_file = tmp_path / "production.json"
        prod_file.write_text(json.dumps({"key": "production"}))

        config = load_config(
            environment=Environment.PRODUCTION,
            config_path=tmp_path,
            validate=False,
        )

        assert config.get("key") == "production"  # Overridden
        assert config.get("base_only") is True  # From base

    def test_env_override(self, tmp_path):
        """Test environment variable override."""
        base_file = tmp_path / "base.json"
        base_file.write_text(json.dumps({"key": "from_file"}))

        with patch.dict("os.environ", {"TRUTHOUND_KEY": "from_env"}):
            config = load_config(
                environment=Environment.DEVELOPMENT,
                config_path=tmp_path,
                validate=False,
            )

        assert config.get("key") == "from_env"
