"""Native schema migration must persist versions and fail before unsafe rewrites."""

from datetime import datetime

import pytest

from truthound.stores.backends.memory import MemoryStore
from truthound.stores.migration.base import (
    FunctionalMigration,
    MigrationConfig,
    MigrationFailedError,
)
from truthound.stores.migration.registry import MigrationRegistry
from truthound.stores.migration.store import MigratableStore
from truthound.stores.results import ResultStatus, ValidationResult

pytestmark = pytest.mark.fault


def migration(base=None, *, validate=None, **config):
    base = base if base is not None else MemoryStore()
    item = ValidationResult(
        "private-run",
        datetime(2026, 1, 1),
        "private-asset",
        ResultStatus.SUCCESS,
        metadata={"_schema_version": "1.0.0"},
    )
    base.save(item)
    registry = MigrationRegistry()

    def increment(data):
        data.setdefault("metadata", {})["applied"] = data["metadata"].get("applied", 0) + 1
        return data

    step = FunctionalMigration("1.0.0", "2.0.0", increment)
    if validate is not None:
        step.validate_output = lambda data: validate
    registry.add(step)
    return (
        base,
        MigratableStore(base, MigrationConfig(current_version="2.0.0", **config), registry),
        item,
    )


@pytest.mark.parametrize("batch", [False, True])
def test_migrated_version_survives_native_serialization_and_reopen(batch):
    base, store, item = migration()
    if batch:
        assert store.migrate_all().success
    first = store.get(item.id)
    assert first.metadata["_schema_version"] == "2.0.0"
    reopened = MigratableStore(base, store.config, store.registry)
    again = reopened.get(item.id)
    assert again.metadata["applied"] == 1
    assert reopened.get_migration_status()["needs_migration_estimated"] == 0


def test_failed_native_validation_prevents_rewrite():
    base, store, item = migration(validate=False)
    with pytest.raises(MigrationFailedError):
        store.get(item.id)
    assert base.get(item.id).metadata == {"_schema_version": "1.0.0"}


def test_failed_required_backup_prevents_migration_and_private_logs(caplog):
    class FailingBackup(MemoryStore):
        def save(self, item):
            if item.id.startswith("_backup_"):
                raise RuntimeError("synthetic-private-backup-location")
            return super().save(item)

    base, store, item = migration(FailingBackup(), backup_before_migrate=True)
    result = store.migrate_all()
    assert result.items_failed == 1 and not result.success
    assert base.get(item.id).metadata == {"_schema_version": "1.0.0"}
    assert "synthetic-private-backup-location" not in caplog.text


def test_lazy_write_failure_is_not_a_successful_read_or_silent_query(caplog):
    class FailedWrite(MemoryStore):
        fail = False

        def save(self, item):
            if self.fail:
                raise RuntimeError("synthetic-private-save-location")
            return super().save(item)

    base, store, item = migration(FailedWrite())
    base.fail = True
    with pytest.raises(RuntimeError):
        store.get(item.id)
    from truthound.stores.base import StoreQuery

    with pytest.raises(RuntimeError):
        store.query(StoreQuery())
    assert "synthetic-private-save-location" not in caplog.text


def test_saved_version_does_not_mutate_input_and_backup_is_not_remigrated():
    base, store, item = migration()
    original = dict(item.metadata)
    store.save(item)
    assert item.metadata == original
    assert base.get(item.id).metadata["_schema_version"] == "2.0.0"
    # Exercise required backup and then a second migration pass.
    base, store, item = migration()
    assert store.migrate_all().items_migrated == 1
    second = store.migrate_all()
    assert second.success and second.items_migrated == 0
    backup = store.get("_backup_" + item.id)
    assert backup.metadata["_schema_version"] == "1.0.0"
    assert "applied" not in backup.metadata


def test_guard_revocation_inside_native_callback_prevents_later_store_write():
    base, store, item = migration(backup_before_migrate=False)
    revoked = False

    def guard():
        if revoked:
            raise PermissionError("synthetic-local-revocation")

    def revoke(data):
        nonlocal revoked
        revoked = True
        return data

    registry = MigrationRegistry()
    registry.add(FunctionalMigration("1.0.0", "2.0.0", revoke))
    store = MigratableStore(base, store.config, registry)
    store.set_io_guard(guard)
    with pytest.raises(PermissionError):
        store.get(item.id)
    assert base.get(item.id).metadata == {"_schema_version": "1.0.0"}
