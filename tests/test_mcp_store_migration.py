"""Installed store migrations retain native versions and expose partial failure safely."""

import json

import pytest

from tests.test_mcp_brokers import approved
from tests.test_mcp_execution_stores import command, plugins, prepare

runtime = plugins.runtime
pytestmark = pytest.mark.contract


def test_native_migration_preview_apply_reopen_and_metadata_privacy(runtime, monkeypatch):
    group = "store_services"
    identity = prepare(runtime, monkeypatch, group, mode="migration")

    def call(name, **args):
        return command(runtime, group, identity, name, **args)

    status = call("migration_status")
    assert status["status"]["needs_migration_estimated"] == 1
    check = call("migration_validate", record_id="seed")
    assert check["validation"]["can_migrate"] and check["validation"]["needs_migration"]
    preview = call("migration_run")
    assert preview["migration"]["dry_run"] and preview["migration"]["items_migrated"] == 1
    assert "migration-callback" not in runtime[2].read_text()
    result = call("migration_run", dry_run=False, key="migration-apply")
    assert result["migration"]["items_migrated"] == 1 and not result["native_failed"]
    path = (
        runtime[2].with_name("native-store") / "default" / "validations" / "private-native-run.json"
    )
    stored = json.loads(path.read_text())
    assert stored["metadata"]["_schema_version"] == "2.0.0" and stored["metadata"]["applied"] == 1
    assert "private-native-run" not in str(result) and "synthetic-private-record" not in str(
        call("get", record_id="seed")
    )
    plugins.invoke(runtime[0], "resources_release", resource_id=identity)
    identity = approved(
        runtime, group + "_open", binding_id="local", idempotency_key="migration-reopen"
    )["data"]["resource_id"]
    assert call("migration_status", key="migration-status-reopen")["status"]["needs_migration_estimated"] == 0
    assert not call("migration_validate", record_id="seed", key="migration-check-reopen")["validation"]["needs_migration"]
    assert (
        call("migration_run", dry_run=False, key="already-migrated")["migration"]["items_migrated"]
        == 0
    )
    assert runtime[2].read_text().splitlines().count("migration-callback") == 1


@pytest.mark.fault
def test_native_failed_migration_keeps_original_and_reports_failure_without_bodies(
    runtime, monkeypatch
):
    identity = prepare(runtime, monkeypatch, "store_services", mode="migration-failure")
    result = command(runtime, "store_services", identity, "migration_run", dry_run=False)
    assert (
        result["native_failed"]
        and not result["migration"]["success"]
        and result["migration"]["items_failed"] == 1
    )
    assert "private-native-run" not in str(result) and "synthetic-private-migration" not in str(
        result
    )
    status = command(runtime, "store_services", identity, "migration_status")
    assert status["status"]["needs_migration_estimated"] > 0
    path = (
        runtime[2].with_name("native-store") / "default" / "validations" / "private-native-run.json"
    )
    assert "_schema_version" not in json.loads(path.read_text())["metadata"]
