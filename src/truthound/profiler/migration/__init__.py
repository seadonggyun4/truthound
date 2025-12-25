"""Schema Migration Framework.

This package provides schema migration capabilities for profile data:
- Version management
- Automatic migration detection
- Forward and backward migrations
- Migration history tracking

Example:
    from truthound.profiler.migration import (
        MigrationManager,
        migrate_profile,
        get_schema_version,
    )

    # Check current version
    version = get_schema_version(profile_data)

    # Migrate to latest
    migrated = migrate_profile(profile_data)

    # Or use manager for more control
    manager = MigrationManager()
    migrated = manager.migrate(profile_data, target_version="1.1")
"""

from truthound.profiler.migration.base import (
    SchemaVersion,
    MigrationDirection,
    MigrationResult,
    MigrationError,
    Migration,
    MigrationRegistry,
    migration_registry,
)
from truthound.profiler.migration.manager import (
    MigrationManager,
    MigrationConfig,
    get_schema_version,
    migrate_profile,
)
from truthound.profiler.migration.v1_0_to_v1_1 import V1_0_to_V1_1_Migration

__all__ = [
    # Base types
    "SchemaVersion",
    "MigrationDirection",
    "MigrationResult",
    "MigrationError",
    "Migration",
    "MigrationRegistry",
    "migration_registry",
    # Manager
    "MigrationManager",
    "MigrationConfig",
    "get_schema_version",
    "migrate_profile",
    # Migrations
    "V1_0_to_V1_1_Migration",
]
