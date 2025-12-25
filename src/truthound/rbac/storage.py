"""Storage backends for RBAC.

This module provides various storage backends for persisting roles,
principals, and permissions.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from truthound.rbac.core import (
    Permission,
    Principal,
    PrincipalStore,
    PrincipalType,
    Role,
    RoleStore,
    RoleType,
    CircularRoleError,
)


# =============================================================================
# In-Memory Storage
# =============================================================================


class MemoryRoleStore(RoleStore):
    """In-memory role storage for testing and development.

    Thread-safe implementation using a dictionary backend.

    Example:
        >>> store = MemoryRoleStore()
        >>> role = Role(id="admin", name="Administrator")
        >>> store.save(role)
        >>> retrieved = store.get("admin")
    """

    def __init__(self) -> None:
        self._roles: dict[str, Role] = {}
        self._lock = threading.RLock()

    def get(self, role_id: str) -> Role | None:
        """Get a role by ID."""
        with self._lock:
            return self._roles.get(role_id)

    def list(
        self,
        tenant_id: str | None = None,
        role_type: RoleType | None = None,
        enabled: bool | None = None,
    ) -> list[Role]:
        """List roles with optional filters."""
        with self._lock:
            roles = list(self._roles.values())

            if tenant_id is not None:
                roles = [r for r in roles if r.tenant_id == tenant_id]
            if role_type is not None:
                roles = [r for r in roles if r.role_type == role_type]
            if enabled is not None:
                roles = [r for r in roles if r.enabled == enabled]

            return sorted(roles, key=lambda r: r.name)

    def save(self, role: Role) -> None:
        """Save a role (create or update)."""
        with self._lock:
            role.updated_at = datetime.now(timezone.utc)
            self._roles[role.id] = role

    def delete(self, role_id: str) -> bool:
        """Delete a role. Returns True if deleted."""
        with self._lock:
            if role_id in self._roles:
                del self._roles[role_id]
                return True
            return False

    def exists(self, role_id: str) -> bool:
        """Check if a role exists."""
        with self._lock:
            return role_id in self._roles

    def clear(self) -> None:
        """Clear all roles (for testing)."""
        with self._lock:
            self._roles.clear()


class MemoryPrincipalStore(PrincipalStore):
    """In-memory principal storage for testing and development.

    Example:
        >>> store = MemoryPrincipalStore()
        >>> principal = Principal(id="user:123", name="John Doe")
        >>> store.save(principal)
        >>> retrieved = store.get("user:123")
    """

    def __init__(self) -> None:
        self._principals: dict[str, Principal] = {}
        self._email_index: dict[str, str] = {}  # email -> principal_id
        self._lock = threading.RLock()

    def get(self, principal_id: str) -> Principal | None:
        """Get a principal by ID."""
        with self._lock:
            return self._principals.get(principal_id)

    def get_by_email(self, email: str) -> Principal | None:
        """Get a principal by email."""
        with self._lock:
            principal_id = self._email_index.get(email.lower())
            if principal_id:
                return self._principals.get(principal_id)
            return None

    def list(
        self,
        tenant_id: str | None = None,
        principal_type: PrincipalType | None = None,
        role_id: str | None = None,
    ) -> list[Principal]:
        """List principals with optional filters."""
        with self._lock:
            principals = list(self._principals.values())

            if tenant_id is not None:
                principals = [p for p in principals if p.tenant_id == tenant_id]
            if principal_type is not None:
                principals = [p for p in principals if p.type == principal_type]
            if role_id is not None:
                principals = [p for p in principals if role_id in p.roles]

            return sorted(principals, key=lambda p: p.name or p.id)

    def save(self, principal: Principal) -> None:
        """Save a principal (create or update)."""
        with self._lock:
            # Remove old email index
            if principal.id in self._principals:
                old = self._principals[principal.id]
                if old.email and old.email.lower() in self._email_index:
                    del self._email_index[old.email.lower()]

            # Save principal
            self._principals[principal.id] = principal

            # Update email index
            if principal.email:
                self._email_index[principal.email.lower()] = principal.id

    def delete(self, principal_id: str) -> bool:
        """Delete a principal. Returns True if deleted."""
        with self._lock:
            if principal_id in self._principals:
                principal = self._principals[principal_id]
                if principal.email and principal.email.lower() in self._email_index:
                    del self._email_index[principal.email.lower()]
                del self._principals[principal_id]
                return True
            return False

    def clear(self) -> None:
        """Clear all principals (for testing)."""
        with self._lock:
            self._principals.clear()
            self._email_index.clear()


# =============================================================================
# File-Based Storage
# =============================================================================


@dataclass
class FileStorageConfig:
    """Configuration for file-based RBAC storage."""

    base_path: str | Path = ".truthound/rbac"
    roles_file: str = "roles.json"
    principals_file: str = "principals.json"
    create_dirs: bool = True
    pretty_print: bool = True


class FileRoleStore(RoleStore):
    """File-based role storage.

    Stores roles in a JSON file.

    Example:
        >>> store = FileRoleStore(config=FileStorageConfig(base_path="/tmp/rbac"))
        >>> store.save(Role(id="admin", name="Administrator"))
    """

    def __init__(self, config: FileStorageConfig | None = None) -> None:
        self._config = config or FileStorageConfig()
        self._base_path = Path(self._config.base_path)
        self._file_path = self._base_path / self._config.roles_file
        self._lock = threading.RLock()
        self._cache: dict[str, Role] = {}
        self._cache_loaded = False

        if self._config.create_dirs:
            self._base_path.mkdir(parents=True, exist_ok=True)

    def _load_cache(self) -> None:
        """Load roles from file into cache."""
        if self._cache_loaded:
            return

        if self._file_path.exists():
            try:
                with open(self._file_path) as f:
                    data = json.load(f)
                    self._cache = {
                        role_id: Role.from_dict(role_data)
                        for role_id, role_data in data.items()
                    }
            except (json.JSONDecodeError, OSError):
                self._cache = {}

        self._cache_loaded = True

    def _save_cache(self) -> None:
        """Save cache to file."""
        data = {role_id: role.to_dict() for role_id, role in self._cache.items()}
        with open(self._file_path, "w") as f:
            json.dump(data, f, indent=2 if self._config.pretty_print else None, default=str)

    def get(self, role_id: str) -> Role | None:
        """Get a role by ID."""
        with self._lock:
            self._load_cache()
            return self._cache.get(role_id)

    def list(
        self,
        tenant_id: str | None = None,
        role_type: RoleType | None = None,
        enabled: bool | None = None,
    ) -> list[Role]:
        """List roles with optional filters."""
        with self._lock:
            self._load_cache()
            roles = list(self._cache.values())

            if tenant_id is not None:
                roles = [r for r in roles if r.tenant_id == tenant_id]
            if role_type is not None:
                roles = [r for r in roles if r.role_type == role_type]
            if enabled is not None:
                roles = [r for r in roles if r.enabled == enabled]

            return sorted(roles, key=lambda r: r.name)

    def save(self, role: Role) -> None:
        """Save a role (create or update)."""
        with self._lock:
            self._load_cache()
            role.updated_at = datetime.now(timezone.utc)
            self._cache[role.id] = role
            self._save_cache()

    def delete(self, role_id: str) -> bool:
        """Delete a role. Returns True if deleted."""
        with self._lock:
            self._load_cache()
            if role_id in self._cache:
                del self._cache[role_id]
                self._save_cache()
                return True
            return False

    def exists(self, role_id: str) -> bool:
        """Check if a role exists."""
        with self._lock:
            self._load_cache()
            return role_id in self._cache


class FilePrincipalStore(PrincipalStore):
    """File-based principal storage.

    Stores principals in a JSON file.
    """

    def __init__(self, config: FileStorageConfig | None = None) -> None:
        self._config = config or FileStorageConfig()
        self._base_path = Path(self._config.base_path)
        self._file_path = self._base_path / self._config.principals_file
        self._lock = threading.RLock()
        self._cache: dict[str, Principal] = {}
        self._email_index: dict[str, str] = {}
        self._cache_loaded = False

        if self._config.create_dirs:
            self._base_path.mkdir(parents=True, exist_ok=True)

    def _load_cache(self) -> None:
        """Load principals from file into cache."""
        if self._cache_loaded:
            return

        if self._file_path.exists():
            try:
                with open(self._file_path) as f:
                    data = json.load(f)
                    self._cache = {
                        principal_id: Principal.from_dict(principal_data)
                        for principal_id, principal_data in data.items()
                    }
                    # Rebuild email index
                    self._email_index = {
                        p.email.lower(): p.id
                        for p in self._cache.values()
                        if p.email
                    }
            except (json.JSONDecodeError, OSError):
                self._cache = {}

        self._cache_loaded = True

    def _save_cache(self) -> None:
        """Save cache to file."""
        data = {
            principal_id: principal.to_dict()
            for principal_id, principal in self._cache.items()
        }
        with open(self._file_path, "w") as f:
            json.dump(data, f, indent=2 if self._config.pretty_print else None, default=str)

    def get(self, principal_id: str) -> Principal | None:
        """Get a principal by ID."""
        with self._lock:
            self._load_cache()
            return self._cache.get(principal_id)

    def get_by_email(self, email: str) -> Principal | None:
        """Get a principal by email."""
        with self._lock:
            self._load_cache()
            principal_id = self._email_index.get(email.lower())
            if principal_id:
                return self._cache.get(principal_id)
            return None

    def list(
        self,
        tenant_id: str | None = None,
        principal_type: PrincipalType | None = None,
        role_id: str | None = None,
    ) -> list[Principal]:
        """List principals with optional filters."""
        with self._lock:
            self._load_cache()
            principals = list(self._cache.values())

            if tenant_id is not None:
                principals = [p for p in principals if p.tenant_id == tenant_id]
            if principal_type is not None:
                principals = [p for p in principals if p.type == principal_type]
            if role_id is not None:
                principals = [p for p in principals if role_id in p.roles]

            return sorted(principals, key=lambda p: p.name or p.id)

    def save(self, principal: Principal) -> None:
        """Save a principal (create or update)."""
        with self._lock:
            self._load_cache()

            # Update email index
            if principal.id in self._cache:
                old = self._cache[principal.id]
                if old.email and old.email.lower() in self._email_index:
                    del self._email_index[old.email.lower()]

            self._cache[principal.id] = principal
            if principal.email:
                self._email_index[principal.email.lower()] = principal.id

            self._save_cache()

    def delete(self, principal_id: str) -> bool:
        """Delete a principal. Returns True if deleted."""
        with self._lock:
            self._load_cache()
            if principal_id in self._cache:
                principal = self._cache[principal_id]
                if principal.email and principal.email.lower() in self._email_index:
                    del self._email_index[principal.email.lower()]
                del self._cache[principal_id]
                self._save_cache()
                return True
            return False


# =============================================================================
# SQLite Storage
# =============================================================================


class SQLiteRoleStore(RoleStore):
    """SQLite-based role storage.

    Example:
        >>> store = SQLiteRoleStore(db_path="/tmp/rbac.db")
        >>> store.save(Role(id="admin", name="Administrator"))
    """

    def __init__(
        self,
        db_path: str | Path = ".truthound/rbac.db",
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
        """Create the roles table."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS roles (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    role_type TEXT NOT NULL DEFAULT 'custom',
                    permissions_json TEXT DEFAULT '[]',
                    parent_roles_json TEXT DEFAULT '[]',
                    conditions_json TEXT DEFAULT '[]',
                    tenant_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    created_by TEXT DEFAULT '',
                    metadata_json TEXT DEFAULT '{}',
                    enabled INTEGER DEFAULT 1
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_roles_tenant ON roles(tenant_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_roles_type ON roles(role_type)")
            self._conn.commit()

    def _row_to_role(self, row: Any) -> Role:
        """Convert a database row to a Role object."""
        permissions = json.loads(row["permissions_json"]) if row["permissions_json"] else []
        parent_roles = json.loads(row["parent_roles_json"]) if row["parent_roles_json"] else []
        metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}

        return Role(
            id=row["id"],
            name=row["name"],
            description=row["description"] or "",
            role_type=RoleType(row["role_type"]),
            permissions={Permission.parse(p) for p in permissions},
            parent_roles=set(parent_roles),
            tenant_id=row["tenant_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            created_by=row["created_by"] or "",
            metadata=metadata,
            enabled=bool(row["enabled"]),
        )

    def get(self, role_id: str) -> Role | None:
        """Get a role by ID."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("SELECT * FROM roles WHERE id = ?", (role_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_role(row)
            return None

    def list(
        self,
        tenant_id: str | None = None,
        role_type: RoleType | None = None,
        enabled: bool | None = None,
    ) -> list[Role]:
        """List roles with optional filters."""
        with self._lock:
            query = "SELECT * FROM roles WHERE 1=1"
            params: list[Any] = []

            if tenant_id is not None:
                query += " AND tenant_id = ?"
                params.append(tenant_id)
            if role_type is not None:
                query += " AND role_type = ?"
                params.append(role_type.value)
            if enabled is not None:
                query += " AND enabled = ?"
                params.append(1 if enabled else 0)

            query += " ORDER BY name"

            cursor = self._conn.cursor()
            cursor.execute(query, params)
            return [self._row_to_role(row) for row in cursor.fetchall()]

    def save(self, role: Role) -> None:
        """Save a role (create or update)."""
        with self._lock:
            role.updated_at = datetime.now(timezone.utc)
            cursor = self._conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO roles (
                    id, name, description, role_type,
                    permissions_json, parent_roles_json, conditions_json,
                    tenant_id, created_at, updated_at, created_by,
                    metadata_json, enabled
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    role.id,
                    role.name,
                    role.description,
                    role.role_type.value,
                    json.dumps([p.to_string() for p in role.permissions]),
                    json.dumps(list(role.parent_roles)),
                    json.dumps([c.to_dict() for c in role.conditions]),
                    role.tenant_id,
                    role.created_at.isoformat(),
                    role.updated_at.isoformat(),
                    role.created_by,
                    json.dumps(role.metadata),
                    1 if role.enabled else 0,
                ),
            )
            self._conn.commit()

    def delete(self, role_id: str) -> bool:
        """Delete a role. Returns True if deleted."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM roles WHERE id = ?", (role_id,))
            self._conn.commit()
            return cursor.rowcount > 0

    def exists(self, role_id: str) -> bool:
        """Check if a role exists."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("SELECT 1 FROM roles WHERE id = ?", (role_id,))
            return cursor.fetchone() is not None

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


class SQLitePrincipalStore(PrincipalStore):
    """SQLite-based principal storage."""

    def __init__(
        self,
        db_path: str | Path = ".truthound/rbac.db",
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
        """Create the principals table."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS principals (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL DEFAULT 'user',
                    name TEXT DEFAULT '',
                    email TEXT DEFAULT '',
                    roles_json TEXT DEFAULT '[]',
                    direct_permissions_json TEXT DEFAULT '[]',
                    attributes_json TEXT DEFAULT '{}',
                    tenant_id TEXT,
                    enabled INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    last_active_at TEXT,
                    metadata_json TEXT DEFAULT '{}'
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_principals_email ON principals(email)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_principals_tenant ON principals(tenant_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_principals_type ON principals(type)")
            self._conn.commit()

    def _row_to_principal(self, row: Any) -> Principal:
        """Convert a database row to a Principal object."""
        roles = json.loads(row["roles_json"]) if row["roles_json"] else []
        direct_permissions = json.loads(row["direct_permissions_json"]) if row["direct_permissions_json"] else []
        attributes = json.loads(row["attributes_json"]) if row["attributes_json"] else {}
        metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}

        return Principal(
            id=row["id"],
            type=PrincipalType(row["type"]),
            name=row["name"] or "",
            email=row["email"] or "",
            roles=set(roles),
            direct_permissions={Permission.parse(p) for p in direct_permissions},
            attributes=attributes,
            tenant_id=row["tenant_id"],
            enabled=bool(row["enabled"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            last_active_at=datetime.fromisoformat(row["last_active_at"]) if row["last_active_at"] else None,
            metadata=metadata,
        )

    def get(self, principal_id: str) -> Principal | None:
        """Get a principal by ID."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("SELECT * FROM principals WHERE id = ?", (principal_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_principal(row)
            return None

    def get_by_email(self, email: str) -> Principal | None:
        """Get a principal by email."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT * FROM principals WHERE LOWER(email) = LOWER(?)",
                (email,),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_principal(row)
            return None

    def list(
        self,
        tenant_id: str | None = None,
        principal_type: PrincipalType | None = None,
        role_id: str | None = None,
    ) -> list[Principal]:
        """List principals with optional filters."""
        with self._lock:
            query = "SELECT * FROM principals WHERE 1=1"
            params: list[Any] = []

            if tenant_id is not None:
                query += " AND tenant_id = ?"
                params.append(tenant_id)
            if principal_type is not None:
                query += " AND type = ?"
                params.append(principal_type.value)

            query += " ORDER BY name"

            cursor = self._conn.cursor()
            cursor.execute(query, params)
            principals = [self._row_to_principal(row) for row in cursor.fetchall()]

            # Filter by role (needs to check JSON)
            if role_id is not None:
                principals = [p for p in principals if role_id in p.roles]

            return principals

    def save(self, principal: Principal) -> None:
        """Save a principal (create or update)."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO principals (
                    id, type, name, email, roles_json, direct_permissions_json,
                    attributes_json, tenant_id, enabled, created_at, last_active_at,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    principal.id,
                    principal.type.value,
                    principal.name,
                    principal.email,
                    json.dumps(list(principal.roles)),
                    json.dumps([p.to_string() for p in principal.direct_permissions]),
                    json.dumps(principal.attributes),
                    principal.tenant_id,
                    1 if principal.enabled else 0,
                    principal.created_at.isoformat(),
                    principal.last_active_at.isoformat() if principal.last_active_at else None,
                    json.dumps(principal.metadata),
                ),
            )
            self._conn.commit()

    def delete(self, principal_id: str) -> bool:
        """Delete a principal. Returns True if deleted."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM principals WHERE id = ?", (principal_id,))
            self._conn.commit()
            return cursor.rowcount > 0

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


# =============================================================================
# Cached Storage Wrapper
# =============================================================================


@dataclass
class CacheConfig:
    """Configuration for caching."""

    ttl_seconds: int = 300  # 5 minutes
    max_size: int = 1000
    refresh_on_access: bool = True


class CachedRoleStore(RoleStore):
    """Caching wrapper for any RoleStore."""

    def __init__(
        self,
        backend: RoleStore,
        cache_config: CacheConfig | None = None,
    ) -> None:
        self._backend = backend
        self._config = cache_config or CacheConfig()
        self._cache: dict[str, tuple[Role, float]] = {}
        self._lock = threading.RLock()

    def _is_expired(self, cached_at: float) -> bool:
        return time.time() - cached_at > self._config.ttl_seconds

    def get(self, role_id: str) -> Role | None:
        with self._lock:
            if role_id in self._cache:
                role, cached_at = self._cache[role_id]
                if not self._is_expired(cached_at):
                    if self._config.refresh_on_access:
                        self._cache[role_id] = (role, time.time())
                    return role
                else:
                    del self._cache[role_id]

        role = self._backend.get(role_id)
        if role:
            with self._lock:
                self._cache[role_id] = (role, time.time())
        return role

    def list(
        self,
        tenant_id: str | None = None,
        role_type: RoleType | None = None,
        enabled: bool | None = None,
    ) -> list[Role]:
        # List always goes to backend
        roles = self._backend.list(
            tenant_id=tenant_id,
            role_type=role_type,
            enabled=enabled,
        )
        # Update cache
        with self._lock:
            for role in roles:
                self._cache[role.id] = (role, time.time())
        return roles

    def save(self, role: Role) -> None:
        with self._lock:
            if role.id in self._cache:
                del self._cache[role.id]
        self._backend.save(role)
        with self._lock:
            self._cache[role.id] = (role, time.time())

    def delete(self, role_id: str) -> bool:
        with self._lock:
            if role_id in self._cache:
                del self._cache[role_id]
        return self._backend.delete(role_id)

    def exists(self, role_id: str) -> bool:
        return self.get(role_id) is not None

    def invalidate(self, role_id: str | None = None) -> None:
        """Invalidate cache entries."""
        with self._lock:
            if role_id:
                if role_id in self._cache:
                    del self._cache[role_id]
            else:
                self._cache.clear()


# =============================================================================
# Factory Functions
# =============================================================================


def create_role_store(
    backend: str = "memory",
    **kwargs: Any,
) -> RoleStore:
    """Create a role store.

    Args:
        backend: Storage backend ("memory", "file", "sqlite")
        **kwargs: Backend-specific configuration

    Returns:
        Configured RoleStore instance.
    """
    if backend == "memory":
        return MemoryRoleStore()
    elif backend == "file":
        config = FileStorageConfig(**kwargs)
        return FileRoleStore(config=config)
    elif backend == "sqlite":
        return SQLiteRoleStore(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def create_principal_store(
    backend: str = "memory",
    **kwargs: Any,
) -> PrincipalStore:
    """Create a principal store.

    Args:
        backend: Storage backend ("memory", "file", "sqlite")
        **kwargs: Backend-specific configuration

    Returns:
        Configured PrincipalStore instance.
    """
    if backend == "memory":
        return MemoryPrincipalStore()
    elif backend == "file":
        config = FileStorageConfig(**kwargs)
        return FilePrincipalStore(config=config)
    elif backend == "sqlite":
        return SQLitePrincipalStore(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
