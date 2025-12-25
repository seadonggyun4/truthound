"""Core types, configuration, and interfaces for multi-tenancy.

This module provides the foundational types and interfaces for multi-tenant
support in Truthound, enabling data isolation, tenant-specific configuration,
and resource management across multiple tenants.

Design Principles:
    - Isolation: Strong data isolation between tenants
    - Flexibility: Multiple isolation strategies (schema, row-level, database)
    - Performance: Minimal overhead for tenant resolution
    - Security: Defense in depth with multiple isolation layers
    - Extensibility: Easy to add new isolation strategies
"""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterator,
    Mapping,
    TypeVar,
)

if TYPE_CHECKING:
    import polars as pl


# =============================================================================
# Enums
# =============================================================================


class TenantStatus(Enum):
    """Status of a tenant."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"
    DELETED = "deleted"


class IsolationLevel(Enum):
    """Data isolation strategy for tenants.

    Levels from lowest to highest isolation:
        - SHARED: All tenants share the same data space (column-based filtering)
        - ROW_LEVEL: Tenants share tables but rows are isolated
        - SCHEMA: Each tenant has its own schema within the same database
        - DATABASE: Each tenant has its own database
        - INSTANCE: Each tenant has its own application instance
    """

    SHARED = "shared"
    ROW_LEVEL = "row_level"
    SCHEMA = "schema"
    DATABASE = "database"
    INSTANCE = "instance"


class TenantTier(Enum):
    """Pricing/feature tier for tenants."""

    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class ResourceType(Enum):
    """Types of resources that can be quota-limited."""

    VALIDATIONS = "validations"
    ROWS_PROCESSED = "rows_processed"
    STORAGE_BYTES = "storage_bytes"
    API_CALLS = "api_calls"
    CHECKPOINTS = "checkpoints"
    CONCURRENT_JOBS = "concurrent_jobs"
    RETENTION_DAYS = "retention_days"
    USERS = "users"


# =============================================================================
# Exceptions
# =============================================================================


class TenantError(Exception):
    """Base exception for tenant-related errors."""

    def __init__(self, message: str, tenant_id: str | None = None) -> None:
        self.tenant_id = tenant_id
        super().__init__(message)


class TenantNotFoundError(TenantError):
    """Raised when a tenant is not found."""
    pass


class TenantAccessDeniedError(TenantError):
    """Raised when access to a tenant is denied."""
    pass


class TenantSuspendedError(TenantError):
    """Raised when trying to access a suspended tenant."""
    pass


class TenantQuotaExceededError(TenantError):
    """Raised when a tenant exceeds their quota."""

    def __init__(
        self,
        message: str,
        tenant_id: str | None = None,
        resource_type: ResourceType | None = None,
        limit: int | None = None,
        current: int | None = None,
    ) -> None:
        super().__init__(message, tenant_id)
        self.resource_type = resource_type
        self.limit = limit
        self.current = current


class TenantConfigError(TenantError):
    """Raised when tenant configuration is invalid."""
    pass


class TenantIsolationError(TenantError):
    """Raised when tenant isolation is violated."""
    pass


# =============================================================================
# Core Data Types
# =============================================================================


@dataclass
class TenantQuota:
    """Resource quotas for a tenant.

    Example:
        >>> quota = TenantQuota(
        ...     validations_per_day=1000,
        ...     rows_per_validation=1_000_000,
        ...     storage_bytes=10 * 1024 * 1024 * 1024,  # 10 GB
        ... )
    """

    # Validation limits
    validations_per_day: int = 100
    validations_per_hour: int = 20
    rows_per_validation: int = 100_000

    # Storage limits
    storage_bytes: int = 1024 * 1024 * 1024  # 1 GB default

    # API limits
    api_calls_per_minute: int = 60
    api_calls_per_day: int = 10_000

    # Checkpoint limits
    checkpoints_per_day: int = 50
    max_retention_days: int = 30

    # Concurrent limits
    max_concurrent_jobs: int = 2

    # User limits
    max_users: int = 5

    # Custom limits
    custom_limits: dict[str, int] = field(default_factory=dict)

    def get_limit(self, resource_type: ResourceType) -> int:
        """Get limit for a resource type."""
        mapping = {
            ResourceType.VALIDATIONS: self.validations_per_day,
            ResourceType.ROWS_PROCESSED: self.rows_per_validation,
            ResourceType.STORAGE_BYTES: self.storage_bytes,
            ResourceType.API_CALLS: self.api_calls_per_day,
            ResourceType.CHECKPOINTS: self.checkpoints_per_day,
            ResourceType.CONCURRENT_JOBS: self.max_concurrent_jobs,
            ResourceType.RETENTION_DAYS: self.max_retention_days,
            ResourceType.USERS: self.max_users,
        }
        return mapping.get(resource_type, 0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "validations_per_day": self.validations_per_day,
            "validations_per_hour": self.validations_per_hour,
            "rows_per_validation": self.rows_per_validation,
            "storage_bytes": self.storage_bytes,
            "api_calls_per_minute": self.api_calls_per_minute,
            "api_calls_per_day": self.api_calls_per_day,
            "checkpoints_per_day": self.checkpoints_per_day,
            "max_retention_days": self.max_retention_days,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "max_users": self.max_users,
            "custom_limits": self.custom_limits,
        }

    @classmethod
    def for_tier(cls, tier: TenantTier) -> "TenantQuota":
        """Create quota based on tier."""
        tier_configs = {
            TenantTier.FREE: cls(
                validations_per_day=10,
                validations_per_hour=5,
                rows_per_validation=10_000,
                storage_bytes=100 * 1024 * 1024,  # 100 MB
                api_calls_per_day=100,
                checkpoints_per_day=5,
                max_retention_days=7,
                max_concurrent_jobs=1,
                max_users=1,
            ),
            TenantTier.STARTER: cls(
                validations_per_day=100,
                validations_per_hour=20,
                rows_per_validation=100_000,
                storage_bytes=1024 * 1024 * 1024,  # 1 GB
                api_calls_per_day=1_000,
                checkpoints_per_day=20,
                max_retention_days=30,
                max_concurrent_jobs=2,
                max_users=5,
            ),
            TenantTier.PROFESSIONAL: cls(
                validations_per_day=1_000,
                validations_per_hour=100,
                rows_per_validation=1_000_000,
                storage_bytes=10 * 1024 * 1024 * 1024,  # 10 GB
                api_calls_per_day=10_000,
                checkpoints_per_day=100,
                max_retention_days=90,
                max_concurrent_jobs=5,
                max_users=25,
            ),
            TenantTier.ENTERPRISE: cls(
                validations_per_day=10_000,
                validations_per_hour=1_000,
                rows_per_validation=100_000_000,
                storage_bytes=100 * 1024 * 1024 * 1024,  # 100 GB
                api_calls_per_day=100_000,
                checkpoints_per_day=1_000,
                max_retention_days=365,
                max_concurrent_jobs=20,
                max_users=100,
            ),
            TenantTier.CUSTOM: cls(),  # Default values
        }
        return tier_configs.get(tier, cls())


@dataclass
class TenantSettings:
    """Tenant-specific settings and configuration.

    Example:
        >>> settings = TenantSettings(
        ...     default_schema="my_schema.yaml",
        ...     timezone="Asia/Seoul",
        ...     locale="ko_KR",
        ... )
    """

    # Validation settings
    default_schema: str = ""
    auto_schema_learning: bool = True
    default_validators: list[str] = field(default_factory=list)
    severity_threshold: str = "low"

    # Localization
    timezone: str = "UTC"
    locale: str = "en_US"
    date_format: str = "%Y-%m-%d"

    # Notification settings
    notification_email: str = ""
    webhook_url: str = ""
    slack_channel: str = ""

    # Feature flags
    features: dict[str, bool] = field(default_factory=dict)

    # Custom settings
    custom_settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_schema": self.default_schema,
            "auto_schema_learning": self.auto_schema_learning,
            "default_validators": self.default_validators,
            "severity_threshold": self.severity_threshold,
            "timezone": self.timezone,
            "locale": self.locale,
            "date_format": self.date_format,
            "notification_email": self.notification_email,
            "webhook_url": self.webhook_url,
            "slack_channel": self.slack_channel,
            "features": self.features,
            "custom_settings": self.custom_settings,
        }

    def has_feature(self, feature: str) -> bool:
        """Check if a feature is enabled."""
        return self.features.get(feature, False)


@dataclass
class TenantMetadata:
    """Metadata about a tenant.

    Example:
        >>> metadata = TenantMetadata(
        ...     industry="finance",
        ...     company_size="enterprise",
        ...     region="asia-pacific",
        ... )
    """

    industry: str = ""
    company_size: str = ""
    region: str = ""
    tags: list[str] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)
    custom_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "industry": self.industry,
            "company_size": self.company_size,
            "region": self.region,
            "tags": self.tags,
            "labels": self.labels,
            "custom_metadata": self.custom_metadata,
        }


@dataclass
class Tenant:
    """Represents a tenant in the multi-tenant system.

    Example:
        >>> tenant = Tenant(
        ...     id="tenant_acme_corp",
        ...     name="Acme Corporation",
        ...     tier=TenantTier.PROFESSIONAL,
        ...     isolation_level=IsolationLevel.SCHEMA,
        ... )
    """

    # Identity
    id: str
    name: str
    slug: str = ""

    # Status
    status: TenantStatus = TenantStatus.ACTIVE

    # Tier and isolation
    tier: TenantTier = TenantTier.FREE
    isolation_level: IsolationLevel = IsolationLevel.ROW_LEVEL

    # Configuration
    quota: TenantQuota = field(default_factory=TenantQuota)
    settings: TenantSettings = field(default_factory=TenantSettings)
    metadata: TenantMetadata = field(default_factory=TenantMetadata)

    # Isolation configuration
    database_name: str = ""  # For DATABASE isolation
    schema_name: str = ""    # For SCHEMA isolation
    tenant_column: str = "tenant_id"  # For ROW_LEVEL isolation

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    suspended_at: datetime | None = None

    # Owner
    owner_id: str = ""
    owner_email: str = ""

    def __post_init__(self) -> None:
        """Initialize slug if not provided."""
        if not self.slug:
            self.slug = self.id.lower().replace(" ", "_").replace("-", "_")

    @property
    def is_active(self) -> bool:
        """Check if tenant is active."""
        return self.status == TenantStatus.ACTIVE

    @property
    def is_suspended(self) -> bool:
        """Check if tenant is suspended."""
        return self.status == TenantStatus.SUSPENDED

    def get_namespace(self) -> str:
        """Get the namespace for this tenant based on isolation level."""
        if self.isolation_level == IsolationLevel.DATABASE:
            return self.database_name or f"db_{self.slug}"
        elif self.isolation_level == IsolationLevel.SCHEMA:
            return self.schema_name or f"schema_{self.slug}"
        else:
            return self.slug

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "slug": self.slug,
            "status": self.status.value,
            "tier": self.tier.value,
            "isolation_level": self.isolation_level.value,
            "quota": self.quota.to_dict(),
            "settings": self.settings.to_dict(),
            "metadata": self.metadata.to_dict(),
            "database_name": self.database_name,
            "schema_name": self.schema_name,
            "tenant_column": self.tenant_column,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "suspended_at": self.suspended_at.isoformat() if self.suspended_at else None,
            "owner_id": self.owner_id,
            "owner_email": self.owner_email,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Tenant":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            slug=data.get("slug", ""),
            status=TenantStatus(data.get("status", "active")),
            tier=TenantTier(data.get("tier", "free")),
            isolation_level=IsolationLevel(data.get("isolation_level", "row_level")),
            quota=TenantQuota(**data.get("quota", {})) if data.get("quota") else TenantQuota(),
            settings=TenantSettings(**data.get("settings", {})) if data.get("settings") else TenantSettings(),
            metadata=TenantMetadata(**data.get("metadata", {})) if data.get("metadata") else TenantMetadata(),
            database_name=data.get("database_name", ""),
            schema_name=data.get("schema_name", ""),
            tenant_column=data.get("tenant_column", "tenant_id"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(timezone.utc),
            suspended_at=datetime.fromisoformat(data["suspended_at"]) if data.get("suspended_at") else None,
            owner_id=data.get("owner_id", ""),
            owner_email=data.get("owner_email", ""),
        )


# =============================================================================
# Tenant Context (Thread-Local & ContextVar)
# =============================================================================


# ContextVar for async support
_current_tenant: ContextVar[Tenant | None] = ContextVar("current_tenant", default=None)
_current_tenant_id: ContextVar[str | None] = ContextVar("current_tenant_id", default=None)


@dataclass
class TenantContext:
    """Context for the current tenant operation.

    This class provides thread-safe and async-safe context management
    for tenant operations, supporting both sync and async code paths.

    Example:
        >>> with TenantContext.set_current(tenant):
        ...     # All operations within this block are tenant-scoped
        ...     current = TenantContext.get_current()
        ...     assert current.id == tenant.id
    """

    tenant: Tenant
    user_id: str = ""
    roles: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    request_id: str = ""
    trace_id: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def get_current_tenant(cls) -> Tenant | None:
        """Get the current tenant from context."""
        return _current_tenant.get()

    @classmethod
    def get_current_tenant_id(cls) -> str | None:
        """Get the current tenant ID from context."""
        tenant = _current_tenant.get()
        if tenant:
            return tenant.id
        return _current_tenant_id.get()

    @classmethod
    def require_current_tenant(cls) -> Tenant:
        """Get the current tenant, raising if not set."""
        tenant = cls.get_current_tenant()
        if tenant is None:
            raise TenantError("No tenant context set")
        return tenant

    @classmethod
    @contextmanager
    def set_current(cls, tenant: Tenant) -> Iterator["TenantContext"]:
        """Set the current tenant for the context.

        Example:
            >>> with TenantContext.set_current(tenant) as ctx:
            ...     process_data()
        """
        token_tenant = _current_tenant.set(tenant)
        token_id = _current_tenant_id.set(tenant.id)
        context = cls(tenant=tenant)
        try:
            yield context
        finally:
            _current_tenant.reset(token_tenant)
            _current_tenant_id.reset(token_id)

    @classmethod
    @contextmanager
    def set_current_id(cls, tenant_id: str) -> Iterator[None]:
        """Set only the tenant ID for the context.

        This is useful when you have the ID but not the full tenant object.
        """
        token = _current_tenant_id.set(tenant_id)
        try:
            yield
        finally:
            _current_tenant_id.reset(token)

    @classmethod
    def clear(cls) -> None:
        """Clear the current tenant context."""
        _current_tenant.set(None)
        _current_tenant_id.set(None)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tenant_id": self.tenant.id,
            "tenant_name": self.tenant.name,
            "user_id": self.user_id,
            "roles": self.roles,
            "permissions": self.permissions,
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "started_at": self.started_at.isoformat(),
            "metadata": self.metadata,
        }


# =============================================================================
# Interfaces (Abstract Base Classes)
# =============================================================================


class TenantStore(ABC):
    """Abstract interface for tenant storage.

    Implementations can store tenants in memory, files, databases, etc.
    """

    @abstractmethod
    def get(self, tenant_id: str) -> Tenant | None:
        """Get a tenant by ID."""
        ...

    @abstractmethod
    def get_by_slug(self, slug: str) -> Tenant | None:
        """Get a tenant by slug."""
        ...

    @abstractmethod
    def list(
        self,
        status: TenantStatus | None = None,
        tier: TenantTier | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Tenant]:
        """List tenants with optional filters."""
        ...

    @abstractmethod
    def save(self, tenant: Tenant) -> None:
        """Save a tenant (create or update)."""
        ...

    @abstractmethod
    def delete(self, tenant_id: str) -> bool:
        """Delete a tenant. Returns True if deleted."""
        ...

    @abstractmethod
    def exists(self, tenant_id: str) -> bool:
        """Check if a tenant exists."""
        ...

    def count(
        self,
        status: TenantStatus | None = None,
        tier: TenantTier | None = None,
    ) -> int:
        """Count tenants matching the filters."""
        return len(self.list(status=status, tier=tier, limit=1000000))


class TenantResolver(ABC):
    """Abstract interface for resolving tenant from request context.

    Different strategies can be implemented:
        - Header-based: X-Tenant-ID header
        - Subdomain-based: acme.truthound.io
        - Path-based: /tenants/acme/...
        - API key-based: API key includes tenant info
        - JWT claim-based: Tenant ID in JWT token
    """

    @abstractmethod
    def resolve(self, context: Mapping[str, Any]) -> str | None:
        """Resolve tenant ID from context.

        Args:
            context: Request context (headers, path, etc.)

        Returns:
            Tenant ID if resolved, None otherwise.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this resolver."""
        ...


class IsolationStrategy(ABC):
    """Abstract interface for tenant data isolation.

    Different strategies provide different isolation levels:
        - SharedIsolation: Column-based filtering (tenant_id column)
        - RowLevelIsolation: Row-level security policies
        - SchemaIsolation: Separate schemas per tenant
        - DatabaseIsolation: Separate databases per tenant
    """

    @property
    @abstractmethod
    def isolation_level(self) -> IsolationLevel:
        """The isolation level this strategy provides."""
        ...

    @abstractmethod
    def apply_filter(
        self,
        df: "pl.LazyFrame",
        tenant: Tenant,
    ) -> "pl.LazyFrame":
        """Apply tenant filter to a LazyFrame.

        Args:
            df: Input LazyFrame
            tenant: Current tenant

        Returns:
            Filtered LazyFrame with only tenant's data.
        """
        ...

    @abstractmethod
    def add_tenant_column(
        self,
        df: "pl.LazyFrame",
        tenant: Tenant,
    ) -> "pl.LazyFrame":
        """Add tenant identifier to a LazyFrame.

        Args:
            df: Input LazyFrame
            tenant: Current tenant

        Returns:
            LazyFrame with tenant column added.
        """
        ...

    @abstractmethod
    def get_namespace(self, tenant: Tenant) -> str:
        """Get the namespace (schema/database) for a tenant."""
        ...

    def validate_isolation(self, tenant: Tenant) -> None:
        """Validate that isolation is properly configured for tenant."""
        if tenant.status != TenantStatus.ACTIVE:
            raise TenantSuspendedError(
                f"Tenant {tenant.id} is not active",
                tenant_id=tenant.id,
            )


class QuotaTracker(ABC):
    """Abstract interface for tracking tenant resource usage."""

    @abstractmethod
    def get_usage(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        window: str = "day",
    ) -> int:
        """Get current usage for a resource.

        Args:
            tenant_id: Tenant ID
            resource_type: Type of resource
            window: Time window (hour, day, month)

        Returns:
            Current usage count.
        """
        ...

    @abstractmethod
    def increment(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        amount: int = 1,
    ) -> int:
        """Increment usage counter.

        Args:
            tenant_id: Tenant ID
            resource_type: Type of resource
            amount: Amount to increment

        Returns:
            New usage count.
        """
        ...

    @abstractmethod
    def check_quota(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        quota: TenantQuota,
        required: int = 1,
    ) -> bool:
        """Check if quota allows the requested amount.

        Args:
            tenant_id: Tenant ID
            resource_type: Type of resource
            quota: Tenant's quota configuration
            required: Amount required

        Returns:
            True if quota allows, False otherwise.
        """
        ...

    @abstractmethod
    def reset(
        self,
        tenant_id: str,
        resource_type: ResourceType | None = None,
    ) -> None:
        """Reset usage counters.

        Args:
            tenant_id: Tenant ID
            resource_type: Specific resource to reset, or None for all
        """
        ...


# =============================================================================
# Utility Functions
# =============================================================================


def generate_tenant_id(prefix: str = "tenant") -> str:
    """Generate a unique tenant ID."""
    unique_part = uuid.uuid4().hex[:12]
    return f"{prefix}_{unique_part}"


def generate_slug(name: str) -> str:
    """Generate a URL-safe slug from a name."""
    import re
    # Convert to lowercase
    slug = name.lower()
    # Replace spaces and special chars with underscores
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    # Remove leading/trailing underscores
    slug = slug.strip("_")
    # Limit length
    return slug[:50]


def hash_tenant_id(tenant_id: str) -> str:
    """Create a hash of tenant ID for use in paths/keys."""
    return hashlib.sha256(tenant_id.encode()).hexdigest()[:16]


def current_tenant() -> Tenant | None:
    """Get the current tenant from context (convenience function)."""
    return TenantContext.get_current_tenant()


def current_tenant_id() -> str | None:
    """Get the current tenant ID from context (convenience function)."""
    return TenantContext.get_current_tenant_id()


def require_tenant() -> Tenant:
    """Get the current tenant, raising if not set (convenience function)."""
    return TenantContext.require_current_tenant()
