"""Comprehensive tests for the multi-tenancy module.

This test suite covers:
- Core types and tenant management
- Storage backends (memory, file, SQLite)
- Isolation strategies (shared, row-level, schema, database)
- Tenant resolvers (header, subdomain, path, JWT, API key)
- Quota tracking and enforcement
- Middleware and decorators
- Integration with Truthound core
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Skip if polars not available
pytest.importorskip("polars")
import polars as pl


# =============================================================================
# Core Types Tests
# =============================================================================


class TestTenantEnums:
    """Tests for tenant enums."""

    def test_tenant_status_values(self) -> None:
        """Test TenantStatus enum values."""
        from truthound.multitenancy import TenantStatus

        assert TenantStatus.ACTIVE.value == "active"
        assert TenantStatus.SUSPENDED.value == "suspended"
        assert TenantStatus.DELETED.value == "deleted"

    def test_isolation_level_values(self) -> None:
        """Test IsolationLevel enum values."""
        from truthound.multitenancy import IsolationLevel

        assert IsolationLevel.SHARED.value == "shared"
        assert IsolationLevel.ROW_LEVEL.value == "row_level"
        assert IsolationLevel.SCHEMA.value == "schema"
        assert IsolationLevel.DATABASE.value == "database"

    def test_tenant_tier_values(self) -> None:
        """Test TenantTier enum values."""
        from truthound.multitenancy import TenantTier

        assert TenantTier.FREE.value == "free"
        assert TenantTier.PROFESSIONAL.value == "professional"
        assert TenantTier.ENTERPRISE.value == "enterprise"

    def test_resource_type_values(self) -> None:
        """Test ResourceType enum values."""
        from truthound.multitenancy import ResourceType

        assert ResourceType.VALIDATIONS.value == "validations"
        assert ResourceType.ROWS_PROCESSED.value == "rows_processed"
        assert ResourceType.API_CALLS.value == "api_calls"


class TestTenantQuota:
    """Tests for TenantQuota."""

    def test_default_quota(self) -> None:
        """Test default quota values."""
        from truthound.multitenancy import TenantQuota

        quota = TenantQuota()
        assert quota.validations_per_day == 100
        assert quota.rows_per_validation == 100_000
        assert quota.max_users == 5

    def test_tier_based_quota(self) -> None:
        """Test tier-based quota creation."""
        from truthound.multitenancy import TenantQuota, TenantTier

        free_quota = TenantQuota.for_tier(TenantTier.FREE)
        assert free_quota.validations_per_day == 10
        assert free_quota.max_users == 1

        enterprise_quota = TenantQuota.for_tier(TenantTier.ENTERPRISE)
        assert enterprise_quota.validations_per_day == 10_000
        assert enterprise_quota.max_users == 100

    def test_quota_get_limit(self) -> None:
        """Test get_limit method."""
        from truthound.multitenancy import TenantQuota, ResourceType

        quota = TenantQuota(validations_per_day=500)
        assert quota.get_limit(ResourceType.VALIDATIONS) == 500

    def test_quota_to_dict(self) -> None:
        """Test quota serialization."""
        from truthound.multitenancy import TenantQuota

        quota = TenantQuota(validations_per_day=100)
        data = quota.to_dict()
        assert data["validations_per_day"] == 100
        assert "custom_limits" in data


class TestTenant:
    """Tests for Tenant dataclass."""

    def test_tenant_creation(self) -> None:
        """Test basic tenant creation."""
        from truthound.multitenancy import Tenant, TenantTier

        tenant = Tenant(
            id="test_tenant",
            name="Test Tenant",
            tier=TenantTier.PROFESSIONAL,
        )
        assert tenant.id == "test_tenant"
        assert tenant.name == "Test Tenant"
        assert tenant.tier == TenantTier.PROFESSIONAL
        assert tenant.slug == "test_tenant"

    def test_tenant_is_active(self) -> None:
        """Test is_active property."""
        from truthound.multitenancy import Tenant, TenantStatus

        tenant = Tenant(id="test", name="Test")
        assert tenant.is_active is True

        tenant.status = TenantStatus.SUSPENDED
        assert tenant.is_active is False

    def test_tenant_serialization(self) -> None:
        """Test tenant to_dict and from_dict."""
        from truthound.multitenancy import Tenant, TenantTier, IsolationLevel

        original = Tenant(
            id="test_tenant",
            name="Test Tenant",
            tier=TenantTier.PROFESSIONAL,
            isolation_level=IsolationLevel.SCHEMA,
        )
        data = original.to_dict()
        restored = Tenant.from_dict(data)

        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.tier == original.tier
        assert restored.isolation_level == original.isolation_level

    def test_tenant_get_namespace(self) -> None:
        """Test get_namespace for different isolation levels."""
        from truthound.multitenancy import Tenant, IsolationLevel

        # Schema isolation
        tenant = Tenant(
            id="test",
            name="Test",
            isolation_level=IsolationLevel.SCHEMA,
            schema_name="custom_schema",
        )
        assert tenant.get_namespace() == "custom_schema"

        # Database isolation
        tenant = Tenant(
            id="test",
            name="Test",
            isolation_level=IsolationLevel.DATABASE,
            database_name="custom_db",
        )
        assert tenant.get_namespace() == "custom_db"


class TestTenantContext:
    """Tests for TenantContext."""

    def test_set_and_get_current(self) -> None:
        """Test setting and getting current tenant."""
        from truthound.multitenancy import Tenant, TenantContext

        tenant = Tenant(id="test", name="Test")

        assert TenantContext.get_current_tenant() is None

        with TenantContext.set_current(tenant):
            current = TenantContext.get_current_tenant()
            assert current is not None
            assert current.id == "test"

        assert TenantContext.get_current_tenant() is None

    def test_context_nesting(self) -> None:
        """Test nested tenant contexts."""
        from truthound.multitenancy import Tenant, TenantContext

        tenant1 = Tenant(id="tenant1", name="Tenant 1")
        tenant2 = Tenant(id="tenant2", name="Tenant 2")

        with TenantContext.set_current(tenant1):
            assert TenantContext.get_current_tenant_id() == "tenant1"

            with TenantContext.set_current(tenant2):
                assert TenantContext.get_current_tenant_id() == "tenant2"

            assert TenantContext.get_current_tenant_id() == "tenant1"

        assert TenantContext.get_current_tenant_id() is None

    def test_require_current_tenant(self) -> None:
        """Test require_current_tenant raises when not set."""
        from truthound.multitenancy import TenantContext, TenantError

        with pytest.raises(TenantError, match="No tenant context"):
            TenantContext.require_current_tenant()

    def test_thread_safety(self) -> None:
        """Test that context is thread-local."""
        from truthound.multitenancy import Tenant, TenantContext

        results: dict[str, str | None] = {}

        def set_tenant(name: str) -> None:
            tenant = Tenant(id=name, name=name)
            with TenantContext.set_current(tenant):
                time.sleep(0.01)  # Small delay to test isolation
                current = TenantContext.get_current_tenant()
                results[name] = current.id if current else None

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(set_tenant, f"tenant_{i}")
                for i in range(3)
            ]
            for f in futures:
                f.result()

        # Each thread should have seen its own tenant
        assert results["tenant_0"] == "tenant_0"
        assert results["tenant_1"] == "tenant_1"
        assert results["tenant_2"] == "tenant_2"


# =============================================================================
# Storage Backend Tests
# =============================================================================


class TestMemoryTenantStore:
    """Tests for MemoryTenantStore."""

    def test_crud_operations(self) -> None:
        """Test basic CRUD operations."""
        from truthound.multitenancy import MemoryTenantStore, Tenant

        store = MemoryTenantStore()
        tenant = Tenant(id="test", name="Test")

        # Create
        store.save(tenant)
        assert store.exists("test")

        # Read
        retrieved = store.get("test")
        assert retrieved is not None
        assert retrieved.name == "Test"

        # Update
        tenant.name = "Updated"
        store.save(tenant)
        retrieved = store.get("test")
        assert retrieved is not None
        assert retrieved.name == "Updated"

        # Delete
        deleted = store.delete("test")
        assert deleted is True
        assert store.exists("test") is False

    def test_get_by_slug(self) -> None:
        """Test getting tenant by slug."""
        from truthound.multitenancy import MemoryTenantStore, Tenant

        store = MemoryTenantStore()
        tenant = Tenant(id="test_123", name="Test Tenant")
        store.save(tenant)

        # Find by slug
        retrieved = store.get_by_slug("test_123")
        assert retrieved is not None
        assert retrieved.id == "test_123"

    def test_list_with_filters(self) -> None:
        """Test listing tenants with filters."""
        from truthound.multitenancy import (
            MemoryTenantStore, Tenant, TenantStatus, TenantTier
        )

        store = MemoryTenantStore()

        store.save(Tenant(id="t1", name="T1", tier=TenantTier.FREE))
        store.save(Tenant(id="t2", name="T2", tier=TenantTier.PROFESSIONAL))
        store.save(Tenant(
            id="t3", name="T3",
            status=TenantStatus.SUSPENDED,
            tier=TenantTier.FREE,
        ))

        # All tenants
        all_tenants = store.list()
        assert len(all_tenants) == 3

        # Filter by tier
        free_tenants = store.list(tier=TenantTier.FREE)
        assert len(free_tenants) == 2

        # Filter by status
        active_tenants = store.list(status=TenantStatus.ACTIVE)
        assert len(active_tenants) == 2

    def test_pagination(self) -> None:
        """Test list pagination."""
        from truthound.multitenancy import MemoryTenantStore, Tenant

        store = MemoryTenantStore()
        for i in range(10):
            store.save(Tenant(id=f"t{i}", name=f"Tenant {i}"))

        page1 = store.list(limit=3, offset=0)
        page2 = store.list(limit=3, offset=3)

        assert len(page1) == 3
        assert len(page2) == 3
        assert page1[0].id != page2[0].id


class TestFileTenantStore:
    """Tests for FileTenantStore."""

    def test_file_storage(self) -> None:
        """Test file-based storage."""
        from truthound.multitenancy import FileTenantStore, FileStorageConfig, Tenant

        with tempfile.TemporaryDirectory() as tmpdir:
            config = FileStorageConfig(base_path=tmpdir)
            store = FileTenantStore(config=config)

            tenant = Tenant(id="test", name="Test Tenant")
            store.save(tenant)

            # Check file exists
            file_path = Path(tmpdir) / "test.json"
            assert file_path.exists()

            # Retrieve
            retrieved = store.get("test")
            assert retrieved is not None
            assert retrieved.name == "Test Tenant"

    def test_persistence(self) -> None:
        """Test that data persists across instances."""
        from truthound.multitenancy import FileTenantStore, FileStorageConfig, Tenant

        with tempfile.TemporaryDirectory() as tmpdir:
            config = FileStorageConfig(base_path=tmpdir)

            # Create and save
            store1 = FileTenantStore(config=config)
            store1.save(Tenant(id="test", name="Test Tenant"))

            # New instance should see the data
            store2 = FileTenantStore(config=config)
            retrieved = store2.get("test")
            assert retrieved is not None
            assert retrieved.name == "Test Tenant"


class TestSQLiteTenantStore:
    """Tests for SQLiteTenantStore."""

    def test_sqlite_storage(self) -> None:
        """Test SQLite-based storage."""
        from truthound.multitenancy import SQLiteTenantStore, Tenant

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "tenants.db"
            store = SQLiteTenantStore(db_path=db_path)

            tenant = Tenant(id="test", name="Test Tenant")
            store.save(tenant)

            retrieved = store.get("test")
            assert retrieved is not None
            assert retrieved.name == "Test Tenant"

            store.close()

    def test_count(self) -> None:
        """Test count method."""
        from truthound.multitenancy import SQLiteTenantStore, Tenant, TenantTier

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteTenantStore(db_path=Path(tmpdir) / "tenants.db")

            store.save(Tenant(id="t1", name="T1", tier=TenantTier.FREE))
            store.save(Tenant(id="t2", name="T2", tier=TenantTier.PROFESSIONAL))
            store.save(Tenant(id="t3", name="T3", tier=TenantTier.FREE))

            assert store.count() == 3
            assert store.count(tier=TenantTier.FREE) == 2

            store.close()


class TestCachedTenantStore:
    """Tests for CachedTenantStore."""

    def test_caching(self) -> None:
        """Test that caching works."""
        from truthound.multitenancy import (
            MemoryTenantStore, CachedTenantStore, CacheConfig, Tenant
        )

        backend = MemoryTenantStore()
        cache_config = CacheConfig(ttl_seconds=60)
        store = CachedTenantStore(backend, cache_config=cache_config)

        tenant = Tenant(id="test", name="Test")
        store.save(tenant)

        # First get should cache
        retrieved1 = store.get("test")

        # Delete from backend directly
        backend.delete("test")

        # Should still get from cache
        retrieved2 = store.get("test")
        assert retrieved2 is not None

    def test_cache_invalidation(self) -> None:
        """Test cache invalidation on update."""
        from truthound.multitenancy import (
            MemoryTenantStore, CachedTenantStore, Tenant
        )

        backend = MemoryTenantStore()
        store = CachedTenantStore(backend)

        tenant = Tenant(id="test", name="Original")
        store.save(tenant)
        store.get("test")  # Cache it

        # Update through cached store
        tenant.name = "Updated"
        store.save(tenant)

        # Should see updated value
        retrieved = store.get("test")
        assert retrieved is not None
        assert retrieved.name == "Updated"


# =============================================================================
# Isolation Strategy Tests
# =============================================================================


class TestSharedIsolation:
    """Tests for SharedIsolation strategy."""

    def test_apply_filter(self) -> None:
        """Test applying tenant filter."""
        from truthound.multitenancy import SharedIsolation, Tenant

        strategy = SharedIsolation(tenant_column="tenant_id")
        tenant = Tenant(id="tenant_1", name="Tenant 1")

        df = pl.LazyFrame({
            "tenant_id": ["tenant_1", "tenant_2", "tenant_1"],
            "value": [1, 2, 3],
        })

        filtered = strategy.apply_filter(df, tenant).collect()
        assert len(filtered) == 2
        assert filtered["tenant_id"].to_list() == ["tenant_1", "tenant_1"]

    def test_add_tenant_column(self) -> None:
        """Test adding tenant column."""
        from truthound.multitenancy import SharedIsolation, Tenant

        strategy = SharedIsolation(tenant_column="tenant_id")
        tenant = Tenant(id="tenant_1", name="Tenant 1")

        df = pl.LazyFrame({"value": [1, 2, 3]})
        result = strategy.add_tenant_column(df, tenant).collect()

        assert "tenant_id" in result.columns
        assert result["tenant_id"].to_list() == ["tenant_1", "tenant_1", "tenant_1"]


class TestRowLevelIsolation:
    """Tests for RowLevelIsolation strategy."""

    def test_basic_filtering(self) -> None:
        """Test basic row-level filtering."""
        from truthound.multitenancy import RowLevelIsolation, Tenant

        strategy = RowLevelIsolation(tenant_column="tenant_id")
        tenant = Tenant(id="tenant_1", name="Tenant 1")

        df = pl.LazyFrame({
            "tenant_id": ["tenant_1", "tenant_2"],
            "value": [1, 2],
        })

        filtered = strategy.apply_filter(df, tenant).collect()
        assert len(filtered) == 1
        assert filtered["value"][0] == 1


class TestSchemaIsolation:
    """Tests for SchemaIsolation strategy."""

    def test_get_namespace(self) -> None:
        """Test namespace generation."""
        from truthound.multitenancy import (
            SchemaIsolation, SchemaConfig, Tenant
        )

        config = SchemaConfig(schema_prefix="t_")
        strategy = SchemaIsolation(config=config)

        tenant = Tenant(id="test_tenant", name="Test", slug="test_tenant")
        namespace = strategy.get_namespace(tenant)

        assert namespace == "t_test_tenant"

    def test_custom_schema_name(self) -> None:
        """Test using custom schema name from tenant."""
        from truthound.multitenancy import SchemaIsolation, Tenant

        strategy = SchemaIsolation()
        tenant = Tenant(
            id="test",
            name="Test",
            schema_name="custom_schema",
        )

        assert strategy.get_namespace(tenant) == "custom_schema"


class TestDatabaseIsolation:
    """Tests for DatabaseIsolation strategy."""

    def test_get_namespace(self) -> None:
        """Test database namespace generation."""
        from truthound.multitenancy import (
            DatabaseIsolation, DatabaseConfig, Tenant
        )

        config = DatabaseConfig(database_prefix="th_")
        strategy = DatabaseIsolation(config=config)

        tenant = Tenant(id="test", name="Test", slug="acme")
        namespace = strategy.get_namespace(tenant)

        assert namespace == "th_acme"


class TestIsolationManager:
    """Tests for IsolationManager."""

    def test_strategy_selection(self) -> None:
        """Test selecting strategy based on tenant isolation level."""
        from truthound.multitenancy import (
            IsolationManager, IsolationLevel, Tenant
        )

        manager = IsolationManager()

        # Row-level tenant
        tenant = Tenant(
            id="test",
            name="Test",
            isolation_level=IsolationLevel.ROW_LEVEL,
        )
        strategy = manager.get_strategy(tenant)
        assert strategy.isolation_level == IsolationLevel.ROW_LEVEL

        # Schema tenant
        tenant.isolation_level = IsolationLevel.SCHEMA
        strategy = manager.get_strategy(tenant)
        assert strategy.isolation_level == IsolationLevel.SCHEMA


# =============================================================================
# Resolver Tests
# =============================================================================


class TestHeaderResolver:
    """Tests for HeaderResolver."""

    def test_resolve_from_header(self) -> None:
        """Test resolving tenant from header."""
        from truthound.multitenancy import HeaderResolver, HeaderResolverConfig

        config = HeaderResolverConfig(header_name="X-Tenant-ID")
        resolver = HeaderResolver(config=config)

        context = {
            "headers": {"X-Tenant-ID": "tenant_123"}
        }
        tenant_id = resolver.resolve(context)
        assert tenant_id == "tenant_123"

    def test_case_insensitive(self) -> None:
        """Test case-insensitive header matching."""
        from truthound.multitenancy import HeaderResolver

        resolver = HeaderResolver()
        context = {
            "headers": {"x-tenant-id": "tenant_123"}
        }
        tenant_id = resolver.resolve(context)
        assert tenant_id == "tenant_123"

    def test_alternative_headers(self) -> None:
        """Test using alternative header names."""
        from truthound.multitenancy import HeaderResolver, HeaderResolverConfig

        config = HeaderResolverConfig(
            header_name="X-Tenant-ID",
            header_alternatives=["Tenant-ID", "X-Tenant"],
        )
        resolver = HeaderResolver(config=config)

        context = {"headers": {"Tenant-ID": "tenant_456"}}
        tenant_id = resolver.resolve(context)
        assert tenant_id == "tenant_456"


class TestSubdomainResolver:
    """Tests for SubdomainResolver."""

    def test_resolve_from_subdomain(self) -> None:
        """Test resolving tenant from subdomain."""
        from truthound.multitenancy import SubdomainResolver, SubdomainResolverConfig

        config = SubdomainResolverConfig(base_domain="truthound.io")
        resolver = SubdomainResolver(config=config)

        context = {"host": "acme.truthound.io"}
        tenant_id = resolver.resolve(context)
        assert tenant_id == "acme"

    def test_excluded_subdomains(self) -> None:
        """Test that excluded subdomains return None."""
        from truthound.multitenancy import SubdomainResolver, SubdomainResolverConfig

        config = SubdomainResolverConfig(
            base_domain="truthound.io",
            exclude_subdomains=["www", "api"],
        )
        resolver = SubdomainResolver(config=config)

        context = {"host": "www.truthound.io"}
        tenant_id = resolver.resolve(context)
        assert tenant_id is None


class TestPathResolver:
    """Tests for PathResolver."""

    def test_resolve_from_path(self) -> None:
        """Test resolving tenant from URL path."""
        from truthound.multitenancy import PathResolver, PathResolverConfig

        config = PathResolverConfig(
            path_pattern=r"^/tenants/([a-zA-Z0-9_-]+)",
        )
        resolver = PathResolver(config=config)

        context = {"path": "/tenants/acme/validate"}
        tenant_id = resolver.resolve(context)
        assert tenant_id == "acme"

    def test_no_match(self) -> None:
        """Test when path doesn't match pattern."""
        from truthound.multitenancy import PathResolver

        resolver = PathResolver()
        context = {"path": "/api/health"}
        tenant_id = resolver.resolve(context)
        assert tenant_id is None


class TestJWTResolver:
    """Tests for JWTResolver."""

    def test_resolve_from_jwt(self) -> None:
        """Test resolving tenant from JWT token."""
        import base64

        from truthound.multitenancy import JWTResolver, JWTResolverConfig

        # Create a simple JWT with tenant claim
        header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
        payload_data = {"tenant_id": "tenant_abc", "sub": "user123"}
        payload = base64.urlsafe_b64encode(
            json.dumps(payload_data).encode()
        ).decode().rstrip("=")
        token = f"{header}.{payload}.signature"

        config = JWTResolverConfig(
            tenant_claim="tenant_id",
            verify_signature=False,
        )
        resolver = JWTResolver(config=config)

        context = {"headers": {"Authorization": f"Bearer {token}"}}
        tenant_id = resolver.resolve(context)
        assert tenant_id == "tenant_abc"


class TestCompositeResolver:
    """Tests for CompositeResolver."""

    def test_fallback_resolution(self) -> None:
        """Test fallback through multiple resolvers."""
        from truthound.multitenancy import (
            CompositeResolver, HeaderResolver, PathResolver
        )

        resolver = CompositeResolver([
            HeaderResolver(),
            PathResolver(),
        ])

        # Header should be tried first
        context1 = {"headers": {"X-Tenant-ID": "from_header"}}
        assert resolver.resolve(context1) == "from_header"

        # Falls back to path
        context2 = {"path": "/tenants/from_path/data", "headers": {}}
        assert resolver.resolve(context2) == "from_path"


# =============================================================================
# Quota Tracking Tests
# =============================================================================


class TestMemoryQuotaTracker:
    """Tests for MemoryQuotaTracker."""

    def test_increment_and_get(self) -> None:
        """Test incrementing and getting usage."""
        from truthound.multitenancy import MemoryQuotaTracker, ResourceType

        tracker = MemoryQuotaTracker()

        tracker.increment("tenant_1", ResourceType.VALIDATIONS, 1)
        tracker.increment("tenant_1", ResourceType.VALIDATIONS, 1)

        usage = tracker.get_usage("tenant_1", ResourceType.VALIDATIONS)
        assert usage == 2

    def test_check_quota(self) -> None:
        """Test quota checking."""
        from truthound.multitenancy import (
            MemoryQuotaTracker, ResourceType, TenantQuota
        )

        tracker = MemoryQuotaTracker()
        quota = TenantQuota(validations_per_day=5)

        # Should allow initially
        assert tracker.check_quota("t1", ResourceType.VALIDATIONS, quota, 1) is True

        # Use up quota
        for _ in range(5):
            tracker.increment("t1", ResourceType.VALIDATIONS)

        # Should deny now
        assert tracker.check_quota("t1", ResourceType.VALIDATIONS, quota, 1) is False

    def test_reset(self) -> None:
        """Test resetting usage."""
        from truthound.multitenancy import MemoryQuotaTracker, ResourceType

        tracker = MemoryQuotaTracker()
        tracker.increment("t1", ResourceType.VALIDATIONS, 5)
        assert tracker.get_usage("t1", ResourceType.VALIDATIONS) == 5

        tracker.reset("t1", ResourceType.VALIDATIONS)
        assert tracker.get_usage("t1", ResourceType.VALIDATIONS) == 0


class TestQuotaEnforcer:
    """Tests for QuotaEnforcer."""

    def test_require_raises_on_exceeded(self) -> None:
        """Test that require raises when quota exceeded."""
        from truthound.multitenancy import (
            QuotaEnforcer, MemoryQuotaTracker, ResourceType,
            Tenant, TenantQuota, TenantQuotaExceededError,
        )

        tracker = MemoryQuotaTracker()
        enforcer = QuotaEnforcer(tracker)

        tenant = Tenant(
            id="t1",
            name="T1",
            quota=TenantQuota(validations_per_day=2),
        )

        # Use up quota
        tracker.increment("t1", ResourceType.VALIDATIONS)
        tracker.increment("t1", ResourceType.VALIDATIONS)

        # Should raise
        with pytest.raises(TenantQuotaExceededError):
            enforcer.require(tenant, ResourceType.VALIDATIONS)

    def test_acquire_context_manager(self) -> None:
        """Test acquire context manager."""
        from truthound.multitenancy import (
            QuotaEnforcer, MemoryQuotaTracker, ResourceType,
            Tenant, TenantQuota,
        )

        tracker = MemoryQuotaTracker()
        enforcer = QuotaEnforcer(tracker)

        tenant = Tenant(
            id="t1",
            name="T1",
            quota=TenantQuota(validations_per_day=10),
        )

        with enforcer.acquire(tenant, ResourceType.VALIDATIONS):
            pass  # Simulated work

        # Usage should be tracked
        assert tracker.get_usage("t1", ResourceType.VALIDATIONS) == 1


# =============================================================================
# Middleware Tests
# =============================================================================


class TestTenantMiddleware:
    """Tests for TenantMiddleware."""

    def test_resolve_tenant(self) -> None:
        """Test tenant resolution through middleware."""
        from truthound.multitenancy import (
            TenantMiddleware, MemoryTenantStore, HeaderResolver, Tenant
        )

        store = MemoryTenantStore()
        store.save(Tenant(id="test_tenant", name="Test"))

        resolver = HeaderResolver()
        middleware = TenantMiddleware(store=store, resolver=resolver)

        context = {"headers": {"X-Tenant-ID": "test_tenant"}}
        tenant = middleware.resolve_tenant(context)

        assert tenant is not None
        assert tenant.id == "test_tenant"

    def test_validate_suspended_tenant(self) -> None:
        """Test that suspended tenants are rejected."""
        from truthound.multitenancy import (
            TenantMiddleware, MemoryTenantStore, HeaderResolver,
            Tenant, TenantStatus, TenantSuspendedError,
        )

        store = MemoryTenantStore()
        store.save(Tenant(
            id="suspended_tenant",
            name="Suspended",
            status=TenantStatus.SUSPENDED,
        ))

        middleware = TenantMiddleware(
            store=store,
            resolver=HeaderResolver(),
        )

        context = {"headers": {"X-Tenant-ID": "suspended_tenant"}}
        tenant = middleware.resolve_tenant(context)

        with pytest.raises(TenantSuspendedError):
            middleware.validate_tenant(tenant)


class TestDecorators:
    """Tests for tenant decorators."""

    def test_tenant_required_decorator(self) -> None:
        """Test @tenant_required decorator."""
        from truthound.multitenancy import (
            tenant_required, Tenant, TenantContext, TenantError
        )

        @tenant_required()
        def protected_function() -> str:
            return "success"

        # Should raise without tenant context
        with pytest.raises(TenantError):
            protected_function()

        # Should succeed with tenant context
        tenant = Tenant(id="test", name="Test")
        with TenantContext.set_current(tenant):
            result = protected_function()
            assert result == "success"

    def test_with_tenant_decorator(self) -> None:
        """Test @with_tenant decorator."""
        from truthound.multitenancy import with_tenant, Tenant, TenantContext

        tenant = Tenant(id="test", name="Test")

        @with_tenant(tenant)
        def get_tenant_name() -> str:
            current = TenantContext.get_current_tenant()
            return current.name if current else ""

        result = get_tenant_name()
        assert result == "Test"


# =============================================================================
# TenantManager Tests
# =============================================================================


class TestTenantManager:
    """Tests for TenantManager."""

    def test_create_tenant(self) -> None:
        """Test creating a tenant."""
        from truthound.multitenancy import TenantManager, TenantTier

        manager = TenantManager()
        tenant = manager.create(
            name="Acme Corp",
            tier=TenantTier.PROFESSIONAL,
        )

        assert tenant.name == "Acme Corp"
        assert tenant.tier == TenantTier.PROFESSIONAL
        assert manager.get(tenant.id) is not None

    def test_create_with_custom_id(self) -> None:
        """Test creating tenant with custom ID."""
        from truthound.multitenancy import TenantManager

        manager = TenantManager()
        tenant = manager.create(
            name="Test",
            tenant_id="custom_id_123",
        )

        assert tenant.id == "custom_id_123"

    def test_duplicate_tenant_error(self) -> None:
        """Test that duplicate tenant ID raises error."""
        from truthound.multitenancy import TenantManager, TenantError

        manager = TenantManager()
        manager.create(name="First", tenant_id="dup_id")

        with pytest.raises(TenantError, match="already exists"):
            manager.create(name="Second", tenant_id="dup_id")

    def test_update_tenant(self) -> None:
        """Test updating a tenant."""
        from truthound.multitenancy import TenantManager, TenantTier

        manager = TenantManager()
        tenant = manager.create(name="Original", tier=TenantTier.FREE)

        updated = manager.update(
            tenant.id,
            name="Updated",
            tier=TenantTier.PROFESSIONAL,
        )

        assert updated.name == "Updated"
        assert updated.tier == TenantTier.PROFESSIONAL

    def test_suspend_and_activate(self) -> None:
        """Test suspending and activating a tenant."""
        from truthound.multitenancy import TenantManager, TenantStatus

        manager = TenantManager()
        tenant = manager.create(name="Test")

        # Suspend
        suspended = manager.suspend(tenant.id, reason="Non-payment")
        assert suspended.status == TenantStatus.SUSPENDED
        assert suspended.suspended_at is not None

        # Activate
        activated = manager.activate(tenant.id)
        assert activated.status == TenantStatus.ACTIVE
        assert activated.suspended_at is None

    def test_context_manager(self) -> None:
        """Test using manager's context method."""
        from truthound.multitenancy import TenantManager, TenantContext

        manager = TenantManager()
        tenant = manager.create(name="Test")

        with manager.context(tenant):
            current = TenantContext.get_current_tenant()
            assert current is not None
            assert current.id == tenant.id

        assert TenantContext.get_current_tenant() is None

    def test_quota_management(self) -> None:
        """Test quota checking through manager."""
        from truthound.multitenancy import (
            TenantManager, TenantQuota, ResourceType
        )

        manager = TenantManager()
        tenant = manager.create(
            name="Test",
            quota=TenantQuota(validations_per_day=5),
        )

        with manager.context(tenant):
            # Should allow
            assert manager.check_quota(ResourceType.VALIDATIONS) is True

            # Track usage
            for _ in range(5):
                manager.track_usage(ResourceType.VALIDATIONS)

            # Should deny now
            assert manager.check_quota(ResourceType.VALIDATIONS) is False

    def test_apply_isolation(self) -> None:
        """Test applying isolation through manager."""
        from truthound.multitenancy import TenantManager, IsolationLevel

        manager = TenantManager()
        tenant = manager.create(
            name="Test",
            isolation_level=IsolationLevel.ROW_LEVEL,
        )

        df = pl.LazyFrame({
            "tenant_id": [tenant.id, "other_tenant", tenant.id],
            "value": [1, 2, 3],
        })

        filtered = manager.apply_isolation(df, tenant).collect()
        assert len(filtered) == 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestTruthoundIntegration:
    """Tests for Truthound integration."""

    def test_tenant_aware_validation_mixin(self) -> None:
        """Test TenantAwareValidation mixin."""
        from truthound.multitenancy import (
            TenantAwareValidation, TenantManager, TenantContext
        )

        manager = TenantManager()
        tenant = manager.create(name="Test")
        validation = TenantAwareValidation(manager)

        with validation.tenant_scope(tenant) as ctx:
            assert TenantContext.get_current_tenant() is not None

        assert TenantContext.get_current_tenant() is None

    def test_tenant_aware_checkpoint(self) -> None:
        """Test TenantAwareCheckpoint."""
        from truthound.multitenancy import (
            TenantAwareCheckpoint, TenantContext, Tenant
        )

        checkpoint = TenantAwareCheckpoint()
        tenant = Tenant(id="test", name="Test")

        with TenantContext.set_current(tenant):
            name = checkpoint.get_checkpoint_name("daily_check")
            assert "test" in name
            assert "daily_check" in name


# =============================================================================
# Concurrent Access Tests
# =============================================================================


class TestConcurrentAccess:
    """Tests for concurrent access scenarios."""

    def test_concurrent_tenant_creation(self) -> None:
        """Test creating tenants concurrently."""
        from truthound.multitenancy import TenantManager

        manager = TenantManager()
        results: list[str] = []

        def create_tenant(i: int) -> None:
            tenant = manager.create(name=f"Tenant {i}")
            results.append(tenant.id)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_tenant, i) for i in range(10)]
            for f in futures:
                f.result()

        assert len(results) == 10
        assert len(set(results)) == 10  # All unique

    def test_concurrent_quota_tracking(self) -> None:
        """Test tracking quotas concurrently."""
        from truthound.multitenancy import (
            MemoryQuotaTracker, ResourceType
        )

        tracker = MemoryQuotaTracker()

        def increment_usage() -> None:
            for _ in range(10):
                tracker.increment("tenant_1", ResourceType.VALIDATIONS)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(increment_usage) for _ in range(5)]
            for f in futures:
                f.result()

        # Should have tracked all increments
        usage = tracker.get_usage("tenant_1", ResourceType.VALIDATIONS)
        assert usage == 50


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_tenant_store(self) -> None:
        """Test create_tenant_store factory."""
        from truthound.multitenancy import create_tenant_store

        # Memory store
        memory_store = create_tenant_store("memory")
        assert memory_store is not None

        # File store
        with tempfile.TemporaryDirectory() as tmpdir:
            file_store = create_tenant_store("file", base_path=tmpdir)
            assert file_store is not None

    def test_create_resolver(self) -> None:
        """Test create_resolver factory."""
        from truthound.multitenancy import create_resolver

        header_resolver = create_resolver("header")
        assert header_resolver.name == "header"

        subdomain_resolver = create_resolver(
            "subdomain",
            base_domain="example.com",
        )
        assert subdomain_resolver.name == "subdomain"

    def test_create_isolation_strategy(self) -> None:
        """Test create_isolation_strategy factory."""
        from truthound.multitenancy import (
            create_isolation_strategy, IsolationLevel
        )

        row_level = create_isolation_strategy(IsolationLevel.ROW_LEVEL)
        assert row_level.isolation_level == IsolationLevel.ROW_LEVEL

        schema = create_isolation_strategy(
            IsolationLevel.SCHEMA,
            schema_prefix="t_",
        )
        assert schema.isolation_level == IsolationLevel.SCHEMA

    def test_create_quota_tracker(self) -> None:
        """Test create_quota_tracker factory."""
        from truthound.multitenancy import create_quota_tracker

        memory_tracker = create_quota_tracker("memory")
        assert memory_tracker is not None


# =============================================================================
# Global Manager Tests
# =============================================================================


class TestGlobalManager:
    """Tests for global manager functions."""

    def test_get_and_set_manager(self) -> None:
        """Test getting and setting global manager."""
        from truthound.multitenancy import (
            TenantManager, get_tenant_manager, set_tenant_manager
        )

        original = get_tenant_manager()

        new_manager = TenantManager()
        set_tenant_manager(new_manager)

        assert get_tenant_manager() is new_manager

        # Restore
        set_tenant_manager(original)

    def test_convenience_functions(self) -> None:
        """Test convenience functions."""
        from truthound.multitenancy import (
            create_tenant, get_tenant, tenant_context,
            TenantContext, set_tenant_manager, TenantManager,
        )

        # Use fresh manager
        manager = TenantManager()
        set_tenant_manager(manager)

        tenant = create_tenant(name="Test Tenant")
        assert tenant is not None

        retrieved = get_tenant(tenant.id)
        assert retrieved is not None
        assert retrieved.name == "Test Tenant"

        with tenant_context(tenant):
            assert TenantContext.get_current_tenant() is not None
