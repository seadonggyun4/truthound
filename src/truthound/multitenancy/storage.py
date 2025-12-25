"""Tenant storage backends for multi-tenancy.

This module provides various storage backends for persisting tenant data,
from simple in-memory stores for testing to production-ready database stores.
"""

from __future__ import annotations

import json
import os
import threading
import time
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from truthound.multitenancy.core import (
    IsolationLevel,
    Tenant,
    TenantError,
    TenantNotFoundError,
    TenantQuota,
    TenantSettings,
    TenantStatus,
    TenantStore,
    TenantTier,
)


# =============================================================================
# In-Memory Storage (Testing & Development)
# =============================================================================


class MemoryTenantStore(TenantStore):
    """In-memory tenant storage for testing and development.

    Thread-safe storage using a dictionary backend.

    Example:
        >>> store = MemoryTenantStore()
        >>> tenant = Tenant(id="tenant_1", name="Test Tenant")
        >>> store.save(tenant)
        >>> retrieved = store.get("tenant_1")
        >>> assert retrieved.name == "Test Tenant"
    """

    def __init__(self) -> None:
        self._tenants: dict[str, Tenant] = {}
        self._slug_index: dict[str, str] = {}  # slug -> tenant_id
        self._lock = threading.RLock()

    def get(self, tenant_id: str) -> Tenant | None:
        """Get a tenant by ID."""
        with self._lock:
            return self._tenants.get(tenant_id)

    def get_by_slug(self, slug: str) -> Tenant | None:
        """Get a tenant by slug."""
        with self._lock:
            tenant_id = self._slug_index.get(slug)
            if tenant_id:
                return self._tenants.get(tenant_id)
            return None

    def list(
        self,
        status: TenantStatus | None = None,
        tier: TenantTier | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Tenant]:
        """List tenants with optional filters."""
        with self._lock:
            tenants = list(self._tenants.values())

            # Apply filters
            if status is not None:
                tenants = [t for t in tenants if t.status == status]
            if tier is not None:
                tenants = [t for t in tenants if t.tier == tier]

            # Sort by created_at descending
            tenants.sort(key=lambda t: t.created_at, reverse=True)

            # Apply pagination
            return tenants[offset : offset + limit]

    def save(self, tenant: Tenant) -> None:
        """Save a tenant (create or update)."""
        with self._lock:
            # Update timestamp
            tenant.updated_at = datetime.now(timezone.utc)

            # Remove old slug index if updating
            if tenant.id in self._tenants:
                old_tenant = self._tenants[tenant.id]
                if old_tenant.slug in self._slug_index:
                    del self._slug_index[old_tenant.slug]

            # Save tenant
            self._tenants[tenant.id] = tenant
            self._slug_index[tenant.slug] = tenant.id

    def delete(self, tenant_id: str) -> bool:
        """Delete a tenant. Returns True if deleted."""
        with self._lock:
            if tenant_id in self._tenants:
                tenant = self._tenants[tenant_id]
                if tenant.slug in self._slug_index:
                    del self._slug_index[tenant.slug]
                del self._tenants[tenant_id]
                return True
            return False

    def exists(self, tenant_id: str) -> bool:
        """Check if a tenant exists."""
        with self._lock:
            return tenant_id in self._tenants

    def count(
        self,
        status: TenantStatus | None = None,
        tier: TenantTier | None = None,
    ) -> int:
        """Count tenants matching the filters."""
        with self._lock:
            count = 0
            for tenant in self._tenants.values():
                if status is not None and tenant.status != status:
                    continue
                if tier is not None and tenant.tier != tier:
                    continue
                count += 1
            return count

    def clear(self) -> None:
        """Clear all tenants (for testing)."""
        with self._lock:
            self._tenants.clear()
            self._slug_index.clear()


# =============================================================================
# File-Based Storage (Development & Small Deployments)
# =============================================================================


@dataclass
class FileStorageConfig:
    """Configuration for file-based tenant storage."""

    base_path: str | Path = ".truthound/tenants"
    file_extension: str = ".json"
    create_dirs: bool = True
    pretty_print: bool = True
    backup_on_update: bool = False


class FileTenantStore(TenantStore):
    """File-based tenant storage.

    Stores each tenant as a JSON file, suitable for development
    and small deployments.

    Example:
        >>> store = FileTenantStore(config=FileStorageConfig(base_path="/tmp/tenants"))
        >>> tenant = Tenant(id="tenant_1", name="Test Tenant")
        >>> store.save(tenant)
    """

    def __init__(self, config: FileStorageConfig | None = None) -> None:
        self.config = config or FileStorageConfig()
        self._base_path = Path(self.config.base_path)
        self._index_file = self._base_path / "_index.json"
        self._lock = threading.RLock()

        if self.config.create_dirs:
            self._base_path.mkdir(parents=True, exist_ok=True)

        self._index: dict[str, dict[str, str]] = {}  # tenant_id -> {slug, file}
        self._load_index()

    def _load_index(self) -> None:
        """Load the index file."""
        if self._index_file.exists():
            try:
                with open(self._index_file) as f:
                    self._index = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._index = {}
                self._rebuild_index()

    def _save_index(self) -> None:
        """Save the index file."""
        with open(self._index_file, "w") as f:
            json.dump(self._index, f, indent=2 if self.config.pretty_print else None)

    def _rebuild_index(self) -> None:
        """Rebuild index from files."""
        self._index = {}
        for file_path in self._base_path.glob(f"*{self.config.file_extension}"):
            if file_path.name.startswith("_"):
                continue
            try:
                with open(file_path) as f:
                    data = json.load(f)
                    tenant_id = data.get("id")
                    slug = data.get("slug", "")
                    if tenant_id:
                        self._index[tenant_id] = {
                            "slug": slug,
                            "file": file_path.name,
                        }
            except (json.JSONDecodeError, OSError):
                continue
        self._save_index()

    def _tenant_file(self, tenant_id: str) -> Path:
        """Get the file path for a tenant."""
        return self._base_path / f"{tenant_id}{self.config.file_extension}"

    def get(self, tenant_id: str) -> Tenant | None:
        """Get a tenant by ID."""
        with self._lock:
            file_path = self._tenant_file(tenant_id)
            if not file_path.exists():
                return None
            try:
                with open(file_path) as f:
                    data = json.load(f)
                    return Tenant.from_dict(data)
            except (json.JSONDecodeError, OSError):
                return None

    def get_by_slug(self, slug: str) -> Tenant | None:
        """Get a tenant by slug."""
        with self._lock:
            for tenant_id, info in self._index.items():
                if info.get("slug") == slug:
                    return self.get(tenant_id)
            return None

    def list(
        self,
        status: TenantStatus | None = None,
        tier: TenantTier | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Tenant]:
        """List tenants with optional filters."""
        with self._lock:
            tenants = []
            for tenant_id in self._index:
                tenant = self.get(tenant_id)
                if tenant is None:
                    continue
                if status is not None and tenant.status != status:
                    continue
                if tier is not None and tenant.tier != tier:
                    continue
                tenants.append(tenant)

            # Sort by created_at descending
            tenants.sort(key=lambda t: t.created_at, reverse=True)

            return tenants[offset : offset + limit]

    def save(self, tenant: Tenant) -> None:
        """Save a tenant (create or update)."""
        with self._lock:
            tenant.updated_at = datetime.now(timezone.utc)
            file_path = self._tenant_file(tenant.id)

            # Backup if updating and backup enabled
            if self.config.backup_on_update and file_path.exists():
                backup_path = file_path.with_suffix(f".backup_{int(time.time())}.json")
                file_path.rename(backup_path)

            # Write tenant
            with open(file_path, "w") as f:
                json.dump(
                    tenant.to_dict(),
                    f,
                    indent=2 if self.config.pretty_print else None,
                    default=str,
                )

            # Update index
            self._index[tenant.id] = {
                "slug": tenant.slug,
                "file": file_path.name,
            }
            self._save_index()

    def delete(self, tenant_id: str) -> bool:
        """Delete a tenant. Returns True if deleted."""
        with self._lock:
            file_path = self._tenant_file(tenant_id)
            if file_path.exists():
                file_path.unlink()
                if tenant_id in self._index:
                    del self._index[tenant_id]
                    self._save_index()
                return True
            return False

    def exists(self, tenant_id: str) -> bool:
        """Check if a tenant exists."""
        with self._lock:
            return self._tenant_file(tenant_id).exists()


# =============================================================================
# SQLite Storage (Small-Medium Deployments)
# =============================================================================


class SQLiteTenantStore(TenantStore):
    """SQLite-based tenant storage.

    Suitable for small to medium deployments with ACID guarantees.

    Example:
        >>> store = SQLiteTenantStore(db_path="/tmp/tenants.db")
        >>> tenant = Tenant(id="tenant_1", name="Test Tenant")
        >>> store.save(tenant)
    """

    def __init__(
        self,
        db_path: str | Path = ".truthound/tenants.db",
        create_tables: bool = True,
    ) -> None:
        import sqlite3

        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()

        if create_tables:
            self._create_tables()

    def _create_tables(self) -> None:
        """Create the tenants table."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tenants (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    slug TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    tier TEXT NOT NULL DEFAULT 'free',
                    isolation_level TEXT NOT NULL DEFAULT 'row_level',
                    database_name TEXT DEFAULT '',
                    schema_name TEXT DEFAULT '',
                    tenant_column TEXT DEFAULT 'tenant_id',
                    owner_id TEXT DEFAULT '',
                    owner_email TEXT DEFAULT '',
                    quota_json TEXT DEFAULT '{}',
                    settings_json TEXT DEFAULT '{}',
                    metadata_json TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    suspended_at TEXT
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_tenants_slug ON tenants(slug)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_tenants_status ON tenants(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_tenants_tier ON tenants(tier)
            """)
            self._conn.commit()

    def _row_to_tenant(self, row: Any) -> Tenant:
        """Convert a database row to a Tenant object."""
        quota_data = json.loads(row["quota_json"]) if row["quota_json"] else {}
        settings_data = json.loads(row["settings_json"]) if row["settings_json"] else {}
        metadata_data = json.loads(row["metadata_json"]) if row["metadata_json"] else {}

        return Tenant(
            id=row["id"],
            name=row["name"],
            slug=row["slug"],
            status=TenantStatus(row["status"]),
            tier=TenantTier(row["tier"]),
            isolation_level=IsolationLevel(row["isolation_level"]),
            database_name=row["database_name"] or "",
            schema_name=row["schema_name"] or "",
            tenant_column=row["tenant_column"] or "tenant_id",
            owner_id=row["owner_id"] or "",
            owner_email=row["owner_email"] or "",
            quota=TenantQuota(**quota_data) if quota_data else TenantQuota(),
            settings=TenantSettings(**settings_data) if settings_data else TenantSettings(),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            suspended_at=datetime.fromisoformat(row["suspended_at"]) if row["suspended_at"] else None,
        )

    def get(self, tenant_id: str) -> Tenant | None:
        """Get a tenant by ID."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("SELECT * FROM tenants WHERE id = ?", (tenant_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_tenant(row)
            return None

    def get_by_slug(self, slug: str) -> Tenant | None:
        """Get a tenant by slug."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("SELECT * FROM tenants WHERE slug = ?", (slug,))
            row = cursor.fetchone()
            if row:
                return self._row_to_tenant(row)
            return None

    def list(
        self,
        status: TenantStatus | None = None,
        tier: TenantTier | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Tenant]:
        """List tenants with optional filters."""
        with self._lock:
            query = "SELECT * FROM tenants WHERE 1=1"
            params: list[Any] = []

            if status is not None:
                query += " AND status = ?"
                params.append(status.value)
            if tier is not None:
                query += " AND tier = ?"
                params.append(tier.value)

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor = self._conn.cursor()
            cursor.execute(query, params)
            return [self._row_to_tenant(row) for row in cursor.fetchall()]

    def save(self, tenant: Tenant) -> None:
        """Save a tenant (create or update)."""
        with self._lock:
            tenant.updated_at = datetime.now(timezone.utc)
            cursor = self._conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO tenants (
                    id, name, slug, status, tier, isolation_level,
                    database_name, schema_name, tenant_column,
                    owner_id, owner_email,
                    quota_json, settings_json, metadata_json,
                    created_at, updated_at, suspended_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant.id,
                    tenant.name,
                    tenant.slug,
                    tenant.status.value,
                    tenant.tier.value,
                    tenant.isolation_level.value,
                    tenant.database_name,
                    tenant.schema_name,
                    tenant.tenant_column,
                    tenant.owner_id,
                    tenant.owner_email,
                    json.dumps(tenant.quota.to_dict()),
                    json.dumps(tenant.settings.to_dict()),
                    json.dumps(tenant.metadata.to_dict()),
                    tenant.created_at.isoformat(),
                    tenant.updated_at.isoformat(),
                    tenant.suspended_at.isoformat() if tenant.suspended_at else None,
                ),
            )
            self._conn.commit()

    def delete(self, tenant_id: str) -> bool:
        """Delete a tenant. Returns True if deleted."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM tenants WHERE id = ?", (tenant_id,))
            self._conn.commit()
            return cursor.rowcount > 0

    def exists(self, tenant_id: str) -> bool:
        """Check if a tenant exists."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("SELECT 1 FROM tenants WHERE id = ?", (tenant_id,))
            return cursor.fetchone() is not None

    def count(
        self,
        status: TenantStatus | None = None,
        tier: TenantTier | None = None,
    ) -> int:
        """Count tenants matching the filters."""
        with self._lock:
            query = "SELECT COUNT(*) FROM tenants WHERE 1=1"
            params: list[Any] = []

            if status is not None:
                query += " AND status = ?"
                params.append(status.value)
            if tier is not None:
                query += " AND tier = ?"
                params.append(tier.value)

            cursor = self._conn.cursor()
            cursor.execute(query, params)
            result = cursor.fetchone()
            return result[0] if result else 0

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


# =============================================================================
# Cached Tenant Store (Wrapper)
# =============================================================================


@dataclass
class CacheConfig:
    """Configuration for tenant caching."""

    ttl_seconds: int = 300  # 5 minutes
    max_size: int = 1000
    refresh_on_access: bool = True


class CachedTenantStore(TenantStore):
    """Caching wrapper for any TenantStore.

    Provides in-memory caching to reduce backend load.

    Example:
        >>> backend = SQLiteTenantStore(db_path="/tmp/tenants.db")
        >>> store = CachedTenantStore(backend, cache_config=CacheConfig(ttl_seconds=60))
    """

    def __init__(
        self,
        backend: TenantStore,
        cache_config: CacheConfig | None = None,
    ) -> None:
        self._backend = backend
        self._config = cache_config or CacheConfig()
        self._cache: dict[str, tuple[Tenant, float]] = {}
        self._slug_cache: dict[str, str] = {}  # slug -> tenant_id
        self._lock = threading.RLock()

    def _is_expired(self, cached_at: float) -> bool:
        """Check if a cached entry is expired."""
        return time.time() - cached_at > self._config.ttl_seconds

    def _cache_tenant(self, tenant: Tenant) -> None:
        """Add tenant to cache."""
        with self._lock:
            # Evict if cache is full
            if len(self._cache) >= self._config.max_size:
                # Remove oldest entries
                sorted_entries = sorted(self._cache.items(), key=lambda x: x[1][1])
                for key, _ in sorted_entries[: len(sorted_entries) // 4]:
                    del self._cache[key]
                    # Clean slug cache
                    self._slug_cache = {
                        s: t for s, t in self._slug_cache.items() if t in self._cache
                    }

            self._cache[tenant.id] = (tenant, time.time())
            self._slug_cache[tenant.slug] = tenant.id

    def _invalidate(self, tenant_id: str) -> None:
        """Invalidate cached tenant."""
        with self._lock:
            if tenant_id in self._cache:
                tenant, _ = self._cache[tenant_id]
                if tenant.slug in self._slug_cache:
                    del self._slug_cache[tenant.slug]
                del self._cache[tenant_id]

    def get(self, tenant_id: str) -> Tenant | None:
        """Get a tenant by ID."""
        with self._lock:
            if tenant_id in self._cache:
                tenant, cached_at = self._cache[tenant_id]
                if not self._is_expired(cached_at):
                    if self._config.refresh_on_access:
                        self._cache[tenant_id] = (tenant, time.time())
                    return tenant
                else:
                    self._invalidate(tenant_id)

        # Cache miss - fetch from backend
        tenant = self._backend.get(tenant_id)
        if tenant:
            self._cache_tenant(tenant)
        return tenant

    def get_by_slug(self, slug: str) -> Tenant | None:
        """Get a tenant by slug."""
        with self._lock:
            if slug in self._slug_cache:
                return self.get(self._slug_cache[slug])

        # Cache miss - fetch from backend
        tenant = self._backend.get_by_slug(slug)
        if tenant:
            self._cache_tenant(tenant)
        return tenant

    def list(
        self,
        status: TenantStatus | None = None,
        tier: TenantTier | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Tenant]:
        """List tenants with optional filters."""
        # List always goes to backend (could be cached with more complex logic)
        tenants = self._backend.list(status=status, tier=tier, limit=limit, offset=offset)
        for tenant in tenants:
            self._cache_tenant(tenant)
        return tenants

    def save(self, tenant: Tenant) -> None:
        """Save a tenant (create or update)."""
        self._invalidate(tenant.id)
        self._backend.save(tenant)
        self._cache_tenant(tenant)

    def delete(self, tenant_id: str) -> bool:
        """Delete a tenant."""
        self._invalidate(tenant_id)
        return self._backend.delete(tenant_id)

    def exists(self, tenant_id: str) -> bool:
        """Check if a tenant exists."""
        with self._lock:
            if tenant_id in self._cache:
                _, cached_at = self._cache[tenant_id]
                if not self._is_expired(cached_at):
                    return True

        return self._backend.exists(tenant_id)

    def invalidate_all(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._slug_cache.clear()


# =============================================================================
# Factory Function
# =============================================================================


def create_tenant_store(
    backend: str = "memory",
    **kwargs: Any,
) -> TenantStore:
    """Create a tenant store.

    Args:
        backend: Storage backend type ("memory", "file", "sqlite")
        **kwargs: Backend-specific configuration

    Returns:
        Configured TenantStore instance.

    Example:
        >>> store = create_tenant_store("sqlite", db_path="/tmp/tenants.db")
        >>> store = create_tenant_store("file", base_path="/tmp/tenants")
        >>> store = create_tenant_store("memory")
    """
    if backend == "memory":
        return MemoryTenantStore()
    elif backend == "file":
        config = FileStorageConfig(**kwargs)
        return FileTenantStore(config=config)
    elif backend == "sqlite":
        return SQLiteTenantStore(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
