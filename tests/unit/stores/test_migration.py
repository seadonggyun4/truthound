"""Unit tests for the schema migration module."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from truthound.stores.migration.base import (
    FunctionalMigration,
    IncompatibleVersionError,
    MigrationConfig,
    MigrationDirection,
    MigrationFailedError,
    MigrationInfo,
    MigrationPathNotFoundError,
    MigrationResult,
    MigrationStrategy,
    SchemaMigration,
    SchemaVersion,
)
from truthound.stores.migration.registry import MigrationRegistry


class TestSchemaVersion:
    """Tests for SchemaVersion data class."""

    def test_version_creation(self) -> None:
        """Test creating a version."""
        version = SchemaVersion(1, 2, 3)

        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

    def test_version_str(self) -> None:
        """Test version string representation."""
        version = SchemaVersion(2, 0, 1)

        assert str(version) == "2.0.1"

    def test_version_parse(self) -> None:
        """Test parsing version string."""
        version = SchemaVersion.parse("1.2.3")

        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

    def test_version_parse_short(self) -> None:
        """Test parsing short version strings."""
        v1 = SchemaVersion.parse("2")
        v2 = SchemaVersion.parse("2.1")

        assert v1 == SchemaVersion(2, 0, 0)
        assert v2 == SchemaVersion(2, 1, 0)

    def test_version_parse_invalid(self) -> None:
        """Test parsing invalid version string."""
        with pytest.raises(ValueError):
            SchemaVersion.parse("invalid")

    def test_version_comparison(self) -> None:
        """Test version comparison."""
        v1 = SchemaVersion(1, 0, 0)
        v2 = SchemaVersion(2, 0, 0)
        v1_1 = SchemaVersion(1, 1, 0)

        assert v1 < v2
        assert v1 < v1_1
        assert v2 > v1
        assert v1 == SchemaVersion(1, 0, 0)

    def test_version_compatibility(self) -> None:
        """Test version compatibility check."""
        v1 = SchemaVersion(1, 0, 0)
        v1_1 = SchemaVersion(1, 1, 0)
        v2 = SchemaVersion(2, 0, 0)

        assert v1.is_compatible_with(v1_1)
        assert not v1.is_compatible_with(v2)


class TestMigrationInfo:
    """Tests for MigrationInfo data class."""

    def test_info_creation(self) -> None:
        """Test creating migration info."""
        info = MigrationInfo(
            from_version=SchemaVersion(1, 0, 0),
            to_version=SchemaVersion(2, 0, 0),
            description="Test migration",
            reversible=True,
        )

        assert info.from_version == SchemaVersion(1, 0, 0)
        assert info.to_version == SchemaVersion(2, 0, 0)
        assert info.description == "Test migration"
        assert info.reversible is True

    def test_info_to_dict(self) -> None:
        """Test serializing info."""
        info = MigrationInfo(
            from_version=SchemaVersion(1, 0, 0),
            to_version=SchemaVersion(2, 0, 0),
            deprecated_fields=["old_field"],
            new_fields=["new_field"],
        )

        data = info.to_dict()

        assert data["from_version"] == "1.0.0"
        assert data["to_version"] == "2.0.0"
        assert data["deprecated_fields"] == ["old_field"]
        assert data["new_fields"] == ["new_field"]


class TestMigrationResult:
    """Tests for MigrationResult data class."""

    def test_result_duration(self) -> None:
        """Test result duration calculation."""
        result = MigrationResult(
            start_time=datetime(2025, 1, 1, 12, 0, 0),
            end_time=datetime(2025, 1, 1, 12, 1, 30),
        )

        assert result.duration_seconds == 90.0

    def test_result_success(self) -> None:
        """Test result success check."""
        success_result = MigrationResult(
            start_time=datetime.now(),
            items_migrated=10,
        )

        failed_result = MigrationResult(
            start_time=datetime.now(),
            items_migrated=10,
            items_failed=2,
        )

        assert success_result.success is True
        assert failed_result.success is False

    def test_result_to_dict(self) -> None:
        """Test serializing result."""
        result = MigrationResult(
            start_time=datetime.now(),
            items_migrated=5,
            migrations_applied=["1.0.0 -> 2.0.0"],
        )

        data = result.to_dict()

        assert data["items_migrated"] == 5
        assert data["migrations_applied"] == ["1.0.0 -> 2.0.0"]
        assert "success" in data


class TestMigrationConfig:
    """Tests for MigrationConfig data class."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = MigrationConfig()

        assert config.current_version == "1.0.0"
        assert config.auto_migrate is True
        assert config.strategy == MigrationStrategy.LAZY

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = MigrationConfig(
            current_version="2.0.0",
            auto_migrate=False,
            strategy=MigrationStrategy.BATCH,
            batch_size=50,
        )

        assert config.current_version == "2.0.0"
        assert config.auto_migrate is False
        assert config.batch_size == 50


class TestFunctionalMigration:
    """Tests for FunctionalMigration class."""

    def test_functional_migration(self) -> None:
        """Test creating functional migration."""

        def upgrade(data: dict[str, Any]) -> dict[str, Any]:
            data["new_field"] = "default"
            return data

        migration = FunctionalMigration(
            from_version="1.0.0",
            to_version="2.0.0",
            migrate_func=upgrade,
            description="Add new field",
        )

        assert migration.from_version == SchemaVersion(1, 0, 0)
        assert migration.to_version == SchemaVersion(2, 0, 0)
        assert migration.description == "Add new field"

    def test_functional_migration_execute(self) -> None:
        """Test executing functional migration."""

        def upgrade(data: dict[str, Any]) -> dict[str, Any]:
            data["version"] = "2.0.0"
            return data

        migration = FunctionalMigration("1.0.0", "2.0.0", upgrade)

        result = migration.migrate({"field": "value"})

        assert result["field"] == "value"
        assert result["version"] == "2.0.0"

    def test_functional_migration_with_rollback(self) -> None:
        """Test functional migration with rollback."""

        def upgrade(data: dict[str, Any]) -> dict[str, Any]:
            data["new"] = data.pop("old", None)
            return data

        def downgrade(data: dict[str, Any]) -> dict[str, Any]:
            data["old"] = data.pop("new", None)
            return data

        migration = FunctionalMigration(
            "1.0.0",
            "2.0.0",
            upgrade,
            rollback_func=downgrade,
        )

        upgraded = migration.migrate({"old": "value"})
        assert upgraded == {"new": "value"}

        downgraded = migration.rollback({"new": "value"})
        assert downgraded == {"old": "value"}

        assert migration.reversible is True

    def test_functional_migration_no_rollback(self) -> None:
        """Test functional migration without rollback."""

        def upgrade(data: dict[str, Any]) -> dict[str, Any]:
            return data

        migration = FunctionalMigration("1.0.0", "2.0.0", upgrade)

        assert migration.reversible is False

        with pytest.raises(NotImplementedError):
            migration.rollback({})


class TestMigrationRegistry:
    """Tests for MigrationRegistry class."""

    def test_registry_add(self) -> None:
        """Test adding migration to registry."""
        registry = MigrationRegistry()

        migration = FunctionalMigration(
            "1.0.0",
            "2.0.0",
            lambda d: d,
        )

        registry.add(migration)

        assert len(registry) == 1
        assert ("1.0.0", "2.0.0") in registry

    def test_registry_decorator(self) -> None:
        """Test decorator-based registration."""
        registry = MigrationRegistry()

        @registry.register("1.0.0", "2.0.0")
        def upgrade(data: dict[str, Any]) -> dict[str, Any]:
            return data

        assert len(registry) == 1
        assert ("1.0.0", "2.0.0") in registry

    def test_registry_find_path_direct(self) -> None:
        """Test finding direct migration path."""
        registry = MigrationRegistry()

        registry.add(FunctionalMigration("1.0.0", "2.0.0", lambda d: d))

        path = registry.find_path("1.0.0", "2.0.0")

        assert len(path) == 1
        assert path[0].from_version == SchemaVersion(1, 0, 0)
        assert path[0].to_version == SchemaVersion(2, 0, 0)

    def test_registry_find_path_multi_step(self) -> None:
        """Test finding multi-step migration path."""
        registry = MigrationRegistry()

        registry.add(FunctionalMigration("1.0.0", "2.0.0", lambda d: d))
        registry.add(FunctionalMigration("2.0.0", "3.0.0", lambda d: d))
        registry.add(FunctionalMigration("3.0.0", "4.0.0", lambda d: d))

        path = registry.find_path("1.0.0", "4.0.0")

        assert len(path) == 3

    def test_registry_find_path_same_version(self) -> None:
        """Test finding path for same version."""
        registry = MigrationRegistry()

        path = registry.find_path("1.0.0", "1.0.0")

        assert len(path) == 0

    def test_registry_find_path_not_found(self) -> None:
        """Test path not found error."""
        registry = MigrationRegistry()

        registry.add(FunctionalMigration("1.0.0", "2.0.0", lambda d: d))

        with pytest.raises(MigrationPathNotFoundError):
            registry.find_path("1.0.0", "5.0.0")

    def test_registry_list_versions(self) -> None:
        """Test listing versions."""
        registry = MigrationRegistry()

        registry.add(FunctionalMigration("1.0.0", "2.0.0", lambda d: d))
        registry.add(FunctionalMigration("2.0.0", "3.0.0", lambda d: d))

        versions = registry.list_versions()

        assert "1.0.0" in versions
        assert "2.0.0" in versions
        assert "3.0.0" in versions

    def test_registry_get_latest_version(self) -> None:
        """Test getting latest version."""
        registry = MigrationRegistry()

        registry.add(FunctionalMigration("1.0.0", "2.0.0", lambda d: d))
        registry.add(FunctionalMigration("2.0.0", "3.0.0", lambda d: d))

        latest = registry.get_latest_version()

        assert latest == "3.0.0"

    def test_registry_validate_path(self) -> None:
        """Test validating migration path."""
        registry = MigrationRegistry()

        def add_field(data: dict[str, Any]) -> dict[str, Any]:
            data["new_field"] = "default"
            return data

        registry.add(FunctionalMigration("1.0.0", "2.0.0", add_field))

        success, errors = registry.validate_path(
            "1.0.0",
            "2.0.0",
            {"existing": "value"},
        )

        assert success is True
        assert len(errors) == 0

    def test_registry_clear(self) -> None:
        """Test clearing registry."""
        registry = MigrationRegistry()

        registry.add(FunctionalMigration("1.0.0", "2.0.0", lambda d: d))
        registry.clear()

        assert len(registry) == 0


class TestMigrationExceptions:
    """Tests for migration exceptions."""

    def test_incompatible_version_error(self) -> None:
        """Test IncompatibleVersionError."""
        error = IncompatibleVersionError("1.0.0", "3.0.0")

        assert error.data_version == "1.0.0"
        assert error.current_version == "3.0.0"
        assert "1.0.0" in str(error)
        assert "3.0.0" in str(error)

    def test_migration_path_not_found_error(self) -> None:
        """Test MigrationPathNotFoundError."""
        error = MigrationPathNotFoundError("1.0.0", "5.0.0")

        assert error.from_version == "1.0.0"
        assert error.to_version == "5.0.0"
        assert "no migration path" in str(error).lower()

    def test_migration_failed_error(self) -> None:
        """Test MigrationFailedError."""
        error = MigrationFailedError("1.0.0", "2.0.0", "Missing field")

        assert error.from_version == "1.0.0"
        assert error.to_version == "2.0.0"
        assert "Missing field" in str(error)


class TestCustomMigration:
    """Tests for custom migration class."""

    def test_custom_migration_class(self) -> None:
        """Test creating a custom migration class."""

        class AddVersionField(SchemaMigration):
            """Add version field to data."""

            @property
            def from_version(self) -> SchemaVersion:
                return SchemaVersion(1, 0, 0)

            @property
            def to_version(self) -> SchemaVersion:
                return SchemaVersion(2, 0, 0)

            def migrate(self, data: dict[str, Any]) -> dict[str, Any]:
                data["schema_version"] = "2.0.0"
                return data

            def rollback(self, data: dict[str, Any]) -> dict[str, Any]:
                data.pop("schema_version", None)
                return data

            @property
            def description(self) -> str:
                return "Add schema_version field"

        migration = AddVersionField()

        assert migration.from_version == SchemaVersion(1, 0, 0)
        assert migration.description == "Add schema_version field"
        assert migration.reversible is True

        upgraded = migration.migrate({"field": "value"})
        assert upgraded["schema_version"] == "2.0.0"

        downgraded = migration.rollback({"field": "value", "schema_version": "2.0.0"})
        assert "schema_version" not in downgraded


class TestMigrationDirection:
    """Tests for MigrationDirection enum."""

    def test_upgrade_direction(self) -> None:
        """Test upgrade direction."""
        assert MigrationDirection.UPGRADE.value == "upgrade"

    def test_downgrade_direction(self) -> None:
        """Test downgrade direction."""
        assert MigrationDirection.DOWNGRADE.value == "downgrade"


class TestMigrationStrategy:
    """Tests for MigrationStrategy enum."""

    def test_eager_strategy(self) -> None:
        """Test eager strategy."""
        assert MigrationStrategy.EAGER.value == "eager"

    def test_lazy_strategy(self) -> None:
        """Test lazy strategy."""
        assert MigrationStrategy.LAZY.value == "lazy"

    def test_batch_strategy(self) -> None:
        """Test batch strategy."""
        assert MigrationStrategy.BATCH.value == "batch"
