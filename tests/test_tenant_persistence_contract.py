"""Actual native tenant status/quota behavior after reopening SQLite state."""

from dataclasses import replace

import pytest

pytestmark = pytest.mark.contract


def test_tenant_suspend_metadata_and_quota_definition_survive_sqlite_reopen(tmp_path):
    from truthound.multitenancy.core import ResourceType
    from truthound.multitenancy.manager import TenantManager
    from truthound.multitenancy.storage import SQLiteTenantStore

    path = tmp_path / "tenant.db"
    store = SQLiteTenantStore(path)
    try:
        manager = TenantManager(store=store)
        tenant = manager.create("Local", tenant_id="local")
        manager.update(
            tenant.id, quota=replace(tenant.quota, api_calls_per_minute=7, api_calls_per_day=2)
        )
        manager.suspend(tenant.id, reason="operator review")
    finally:
        store.close()
    store = SQLiteTenantStore(path)
    try:
        manager = TenantManager(store=store)
        tenant = manager.require("local")
        assert tenant.metadata.custom_metadata["suspension_reason"] == "operator review"
        assert tenant.quota.api_calls_per_minute == 7
        tenant = manager.activate(tenant.id)
        assert "suspension_reason" not in tenant.metadata.custom_metadata
        assert manager.check_quota(ResourceType.API_CALLS, 2, tenant=tenant)
        assert not manager.check_quota(ResourceType.API_CALLS, 3, tenant=tenant)
        with manager.context(tenant.id):
            assert manager.require_current().id == tenant.id
        assert manager.current() is None
    finally:
        store.close()


def test_unlimited_tenant_quota_does_not_report_an_exceeded_summary():
    from truthound.multitenancy.core import ResourceType
    from truthound.multitenancy.manager import TenantManager

    manager = TenantManager()
    tenant = manager.create("Unlimited", tenant_id="local")
    tenant.quota.api_calls_per_day = 0
    manager.track_usage(ResourceType.API_CALLS, 3, tenant=tenant)
    assert manager.check_quota(ResourceType.API_CALLS, 1, tenant=tenant)
    summary = manager.get_usage_summary(ResourceType.API_CALLS, tenant=tenant)
    assert summary.total_usage == 3
    assert not summary.is_exceeded


def test_resolve_or_create_fences_selected_identity_before_persistence(tmp_path):
    from truthound.multitenancy.core import TenantAccessDeniedError
    from truthound.multitenancy.manager import TenantManager
    from truthound.multitenancy.storage import SQLiteTenantStore

    store = SQLiteTenantStore(tmp_path / "resolved.db")
    try:
        manager = TenantManager(store=store)
        with pytest.raises(TenantAccessDeniedError):
            manager.resolve_or_create(
                {"headers": {"X-Tenant-ID": "outside"}}, expected_tenant_id="selected"
            )
        assert manager.list() == []
        with pytest.raises(TenantAccessDeniedError):
            manager.resolve_or_create({}, expected_tenant_id="selected")
        assert manager.list() == []
        created = manager.resolve_or_create(
            {"headers": {"X-Tenant-ID": "selected"}}, "Selected", expected_tenant_id="selected"
        )
        assert created.id == "selected"
        assert (
            manager.resolve_or_create(
                {"headers": {"X-Tenant-ID": "selected"}}, expected_tenant_id="selected"
            ).id
            == created.id
        )
        assert len(manager.list()) == 1
    finally:
        store.close()


def test_required_native_header_and_complete_format_are_enforced():
    from truthound.multitenancy.core import TenantAccessDeniedError
    from truthound.multitenancy.resolvers import HeaderResolver, HeaderResolverConfig

    resolver = HeaderResolver(HeaderResolverConfig(required=True))
    with pytest.raises(TenantAccessDeniedError):
        resolver.resolve({"headers": {}})
    with pytest.raises(TenantAccessDeniedError):
        resolver.resolve({"headers": {"X-Tenant-ID": "selected\n"}})
    assert resolver.resolve({"headers": {"x-tenant-id": "selected"}}) == "selected"
