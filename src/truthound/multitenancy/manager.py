"""Tenant manager for multi-tenancy.

This module provides the central TenantManager class that orchestrates
all tenant-related operations including CRUD, resolution, isolation,
and quota management.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterator

from truthound.multitenancy.core import (
    IsolationLevel,
    IsolationStrategy,
    QuotaTracker,
    ResourceType,
    Tenant,
    TenantContext,
    TenantError,
    TenantNotFoundError,
    TenantQuota,
    TenantResolver,
    TenantSettings,
    TenantStatus,
    TenantStore,
    TenantTier,
    generate_slug,
    generate_tenant_id,
)
from truthound.multitenancy.isolation import IsolationManager, create_isolation_strategy
from truthound.multitenancy.quota import (
    MemoryQuotaTracker,
    QuotaEnforcer,
    UsageSummary,
)
from truthound.multitenancy.resolvers import CompositeResolver, HeaderResolver
from truthound.multitenancy.storage import MemoryTenantStore


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TenantManagerConfig:
    """Configuration for the tenant manager."""

    # Default settings for new tenants
    default_tier: TenantTier = TenantTier.FREE
    default_isolation_level: IsolationLevel = IsolationLevel.ROW_LEVEL
    default_tenant_column: str = "tenant_id"

    # Auto-create settings
    auto_create_anonymous: bool = True
    anonymous_tenant_id: str = "anonymous"

    # Quota settings
    enforce_quotas: bool = True
    quota_check_on_operation: bool = True

    # Lifecycle hooks
    on_create: list[Callable[[Tenant], None]] = field(default_factory=list)
    on_update: list[Callable[[Tenant, Tenant], None]] = field(default_factory=list)
    on_delete: list[Callable[[str], None]] = field(default_factory=list)
    on_suspend: list[Callable[[Tenant], None]] = field(default_factory=list)
    on_activate: list[Callable[[Tenant], None]] = field(default_factory=list)


# =============================================================================
# Tenant Manager
# =============================================================================


class TenantManager:
    """Central manager for all tenant operations.

    The TenantManager provides a unified interface for:
    - CRUD operations on tenants
    - Tenant resolution from request context
    - Data isolation management
    - Quota tracking and enforcement
    - Lifecycle hooks

    Example:
        >>> manager = TenantManager()
        >>> tenant = manager.create(
        ...     name="Acme Corp",
        ...     tier=TenantTier.PROFESSIONAL,
        ... )
        >>> with manager.context(tenant):
        ...     run_validation()
    """

    def __init__(
        self,
        store: TenantStore | None = None,
        resolver: TenantResolver | None = None,
        quota_tracker: QuotaTracker | None = None,
        isolation_manager: IsolationManager | None = None,
        config: TenantManagerConfig | None = None,
    ) -> None:
        self._store = store or MemoryTenantStore()
        self._resolver = resolver or HeaderResolver()
        self._quota_tracker = quota_tracker or MemoryQuotaTracker()
        self._isolation_manager = isolation_manager or IsolationManager()
        self._config = config or TenantManagerConfig()

        self._quota_enforcer = QuotaEnforcer(
            tracker=self._quota_tracker,
            raise_on_exceeded=self._config.enforce_quotas,
        )
        self._lock = threading.RLock()

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def store(self) -> TenantStore:
        """Get the tenant store."""
        return self._store

    @property
    def resolver(self) -> TenantResolver:
        """Get the tenant resolver."""
        return self._resolver

    @property
    def quota_tracker(self) -> QuotaTracker:
        """Get the quota tracker."""
        return self._quota_tracker

    @property
    def quota_enforcer(self) -> QuotaEnforcer:
        """Get the quota enforcer."""
        return self._quota_enforcer

    @property
    def isolation_manager(self) -> IsolationManager:
        """Get the isolation manager."""
        return self._isolation_manager

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    def create(
        self,
        name: str,
        tenant_id: str | None = None,
        tier: TenantTier | None = None,
        isolation_level: IsolationLevel | None = None,
        owner_id: str = "",
        owner_email: str = "",
        quota: TenantQuota | None = None,
        settings: TenantSettings | None = None,
        **kwargs: Any,
    ) -> Tenant:
        """Create a new tenant.

        Args:
            name: Human-readable tenant name
            tenant_id: Optional custom ID (auto-generated if not provided)
            tier: Pricing tier (defaults to config default)
            isolation_level: Data isolation level (defaults to config default)
            owner_id: ID of the tenant owner
            owner_email: Email of the tenant owner
            quota: Custom quota (uses tier defaults if not provided)
            settings: Tenant settings
            **kwargs: Additional tenant attributes

        Returns:
            Created Tenant object.

        Example:
            >>> tenant = manager.create(
            ...     name="Acme Corp",
            ...     tier=TenantTier.PROFESSIONAL,
            ...     owner_email="admin@acme.com",
            ... )
        """
        with self._lock:
            # Generate ID if not provided
            if not tenant_id:
                tenant_id = generate_tenant_id()

            # Check for duplicates
            if self._store.exists(tenant_id):
                raise TenantError(f"Tenant already exists: {tenant_id}")

            # Use defaults from config
            tier = tier or self._config.default_tier
            isolation_level = isolation_level or self._config.default_isolation_level

            # Create quota from tier if not provided
            if quota is None:
                quota = TenantQuota.for_tier(tier)

            # Create tenant
            tenant = Tenant(
                id=tenant_id,
                name=name,
                slug=generate_slug(name),
                tier=tier,
                isolation_level=isolation_level,
                tenant_column=self._config.default_tenant_column,
                owner_id=owner_id,
                owner_email=owner_email,
                quota=quota,
                settings=settings or TenantSettings(),
                **kwargs,
            )

            # Save
            self._store.save(tenant)

            # Call hooks
            for hook in self._config.on_create:
                try:
                    hook(tenant)
                except Exception:
                    pass  # Don't fail on hook errors

            return tenant

    def get(self, tenant_id: str) -> Tenant | None:
        """Get a tenant by ID.

        Args:
            tenant_id: Tenant ID or slug

        Returns:
            Tenant object if found, None otherwise.
        """
        tenant = self._store.get(tenant_id)
        if not tenant:
            tenant = self._store.get_by_slug(tenant_id)
        return tenant

    def require(self, tenant_id: str) -> Tenant:
        """Get a tenant by ID, raising if not found.

        Args:
            tenant_id: Tenant ID or slug

        Returns:
            Tenant object.

        Raises:
            TenantNotFoundError: If tenant not found.
        """
        tenant = self.get(tenant_id)
        if not tenant:
            raise TenantNotFoundError(
                f"Tenant not found: {tenant_id}",
                tenant_id=tenant_id,
            )
        return tenant

    def list(
        self,
        status: TenantStatus | None = None,
        tier: TenantTier | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Tenant]:
        """List tenants with optional filters.

        Args:
            status: Filter by status
            tier: Filter by tier
            limit: Maximum number of results
            offset: Skip this many results

        Returns:
            List of Tenant objects.
        """
        return self._store.list(
            status=status,
            tier=tier,
            limit=limit,
            offset=offset,
        )

    def update(
        self,
        tenant_id: str,
        **updates: Any,
    ) -> Tenant:
        """Update a tenant.

        Args:
            tenant_id: Tenant ID
            **updates: Fields to update

        Returns:
            Updated Tenant object.

        Example:
            >>> tenant = manager.update(
            ...     "tenant_123",
            ...     name="New Name",
            ...     tier=TenantTier.ENTERPRISE,
            ... )
        """
        with self._lock:
            tenant = self.require(tenant_id)
            old_tenant = Tenant.from_dict(tenant.to_dict())  # Copy

            # Apply updates
            for key, value in updates.items():
                if hasattr(tenant, key):
                    setattr(tenant, key, value)

            tenant.updated_at = datetime.now(timezone.utc)
            self._store.save(tenant)

            # Call hooks
            for hook in self._config.on_update:
                try:
                    hook(old_tenant, tenant)
                except Exception:
                    pass

            return tenant

    def delete(self, tenant_id: str) -> bool:
        """Delete a tenant.

        Args:
            tenant_id: Tenant ID

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            # Reset quotas
            self._quota_tracker.reset(tenant_id)

            # Delete from store
            deleted = self._store.delete(tenant_id)

            if deleted:
                # Call hooks
                for hook in self._config.on_delete:
                    try:
                        hook(tenant_id)
                    except Exception:
                        pass

            return deleted

    # =========================================================================
    # Status Management
    # =========================================================================

    def suspend(self, tenant_id: str, reason: str = "") -> Tenant:
        """Suspend a tenant.

        Args:
            tenant_id: Tenant ID
            reason: Suspension reason

        Returns:
            Updated Tenant object.
        """
        with self._lock:
            tenant = self.require(tenant_id)
            tenant.status = TenantStatus.SUSPENDED
            tenant.suspended_at = datetime.now(timezone.utc)
            if reason:
                tenant.metadata.custom_metadata["suspension_reason"] = reason
            self._store.save(tenant)

            for hook in self._config.on_suspend:
                try:
                    hook(tenant)
                except Exception:
                    pass

            return tenant

    def activate(self, tenant_id: str) -> Tenant:
        """Activate a tenant.

        Args:
            tenant_id: Tenant ID

        Returns:
            Updated Tenant object.
        """
        with self._lock:
            tenant = self.require(tenant_id)
            tenant.status = TenantStatus.ACTIVE
            tenant.suspended_at = None
            if "suspension_reason" in tenant.metadata.custom_metadata:
                del tenant.metadata.custom_metadata["suspension_reason"]
            self._store.save(tenant)

            for hook in self._config.on_activate:
                try:
                    hook(tenant)
                except Exception:
                    pass

            return tenant

    def set_tier(
        self,
        tenant_id: str,
        tier: TenantTier,
        update_quota: bool = True,
    ) -> Tenant:
        """Change a tenant's tier.

        Args:
            tenant_id: Tenant ID
            tier: New tier
            update_quota: Whether to update quota to tier defaults

        Returns:
            Updated Tenant object.
        """
        with self._lock:
            tenant = self.require(tenant_id)
            tenant.tier = tier
            if update_quota:
                tenant.quota = TenantQuota.for_tier(tier)
            self._store.save(tenant)
            return tenant

    # =========================================================================
    # Context Management
    # =========================================================================

    def context(self, tenant: Tenant | str) -> TenantContext:
        """Get a context manager for tenant operations.

        Args:
            tenant: Tenant object or ID

        Returns:
            Context manager that sets tenant context.

        Example:
            >>> with manager.context("tenant_123"):
            ...     # All operations use tenant_123
            ...     run_validation()
        """
        if isinstance(tenant, str):
            tenant = self.require(tenant)

        # Return the context manager directly
        return TenantContext.set_current(tenant)

    def current(self) -> Tenant | None:
        """Get the current tenant from context.

        Returns:
            Current Tenant or None.
        """
        return TenantContext.get_current_tenant()

    def require_current(self) -> Tenant:
        """Get the current tenant, raising if not set.

        Returns:
            Current Tenant.

        Raises:
            TenantError: If no tenant context is set.
        """
        return TenantContext.require_current_tenant()

    # =========================================================================
    # Resolution
    # =========================================================================

    def resolve(self, context: dict[str, Any]) -> Tenant | None:
        """Resolve tenant from request context.

        Args:
            context: Request context (headers, path, etc.)

        Returns:
            Resolved Tenant or None.
        """
        tenant_id = self._resolver.resolve(context)
        if not tenant_id:
            return None
        return self.get(tenant_id)

    def resolve_or_create(
        self,
        context: dict[str, Any],
        default_name: str = "New Tenant",
    ) -> Tenant:
        """Resolve tenant or create if not exists.

        Args:
            context: Request context
            default_name: Name for newly created tenant

        Returns:
            Tenant object.
        """
        tenant_id = self._resolver.resolve(context)
        if not tenant_id:
            return self.create(name=default_name)

        tenant = self.get(tenant_id)
        if tenant:
            return tenant

        return self.create(
            name=default_name,
            tenant_id=tenant_id,
        )

    # =========================================================================
    # Isolation
    # =========================================================================

    def apply_isolation(
        self,
        df: Any,
        tenant: Tenant | None = None,
    ) -> Any:
        """Apply tenant isolation to a DataFrame.

        Args:
            df: Polars DataFrame or LazyFrame
            tenant: Tenant (uses current if not provided)

        Returns:
            Filtered DataFrame.
        """
        if tenant is None:
            tenant = self.current()
        if tenant is None:
            return df

        # Convert to LazyFrame if needed
        if hasattr(df, "lazy"):
            df = df.lazy()

        return self._isolation_manager.apply_filter(df, tenant)

    def add_tenant_column(
        self,
        df: Any,
        tenant: Tenant | None = None,
    ) -> Any:
        """Add tenant column to a DataFrame.

        Args:
            df: Polars DataFrame or LazyFrame
            tenant: Tenant (uses current if not provided)

        Returns:
            DataFrame with tenant column.
        """
        if tenant is None:
            tenant = self.current()
        if tenant is None:
            return df

        if hasattr(df, "lazy"):
            df = df.lazy()

        return self._isolation_manager.add_tenant_column(df, tenant)

    # =========================================================================
    # Quota Management
    # =========================================================================

    def check_quota(
        self,
        resource_type: ResourceType,
        amount: int = 1,
        tenant: Tenant | None = None,
    ) -> bool:
        """Check if quota allows an operation.

        Args:
            resource_type: Type of resource
            amount: Amount required
            tenant: Tenant (uses current if not provided)

        Returns:
            True if allowed.
        """
        if tenant is None:
            tenant = self.current()
        if tenant is None:
            return True

        return self._quota_enforcer.check(tenant, resource_type, amount)

    def track_usage(
        self,
        resource_type: ResourceType,
        amount: int = 1,
        tenant: Tenant | None = None,
    ) -> int:
        """Track resource usage.

        Args:
            resource_type: Type of resource
            amount: Amount used
            tenant: Tenant (uses current if not provided)

        Returns:
            New usage count.
        """
        if tenant is None:
            tenant = self.current()
        if tenant is None:
            return 0

        return self._quota_enforcer.track(tenant, resource_type, amount)

    def get_usage_summary(
        self,
        resource_type: ResourceType,
        window: str = "day",
        tenant: Tenant | None = None,
    ) -> UsageSummary:
        """Get usage summary for a resource.

        Args:
            resource_type: Type of resource
            window: Time window (hour, day, month)
            tenant: Tenant (uses current if not provided)

        Returns:
            Usage summary.
        """
        if tenant is None:
            tenant = self.require_current()

        return self._quota_enforcer.get_summary(tenant, resource_type, window)

    def get_all_usage_summaries(
        self,
        window: str = "day",
        tenant: Tenant | None = None,
    ) -> dict[ResourceType, UsageSummary]:
        """Get usage summaries for all resources.

        Args:
            window: Time window
            tenant: Tenant (uses current if not provided)

        Returns:
            Dict of resource type to usage summary.
        """
        if tenant is None:
            tenant = self.require_current()

        return self._quota_enforcer.get_all_summaries(tenant, window)


# =============================================================================
# Global Manager
# =============================================================================


_default_manager: TenantManager | None = None
_manager_lock = threading.Lock()


def get_tenant_manager() -> TenantManager:
    """Get the global tenant manager.

    Returns:
        TenantManager instance.
    """
    global _default_manager
    with _manager_lock:
        if _default_manager is None:
            _default_manager = TenantManager()
        return _default_manager


def set_tenant_manager(manager: TenantManager) -> None:
    """Set the global tenant manager.

    Args:
        manager: TenantManager instance to use globally.
    """
    global _default_manager
    with _manager_lock:
        _default_manager = manager


def configure_tenant_manager(
    store: TenantStore | None = None,
    resolver: TenantResolver | None = None,
    quota_tracker: QuotaTracker | None = None,
    config: TenantManagerConfig | None = None,
) -> TenantManager:
    """Configure and set the global tenant manager.

    Args:
        store: Tenant store backend
        resolver: Tenant resolver
        quota_tracker: Quota tracker
        config: Manager configuration

    Returns:
        Configured TenantManager.
    """
    manager = TenantManager(
        store=store,
        resolver=resolver,
        quota_tracker=quota_tracker,
        config=config,
    )
    set_tenant_manager(manager)
    return manager


# =============================================================================
# Convenience Functions
# =============================================================================


def create_tenant(name: str, **kwargs: Any) -> Tenant:
    """Create a tenant using the global manager.

    See TenantManager.create for full documentation.
    """
    return get_tenant_manager().create(name=name, **kwargs)


def get_tenant(tenant_id: str) -> Tenant | None:
    """Get a tenant using the global manager.

    See TenantManager.get for full documentation.
    """
    return get_tenant_manager().get(tenant_id)


def require_tenant(tenant_id: str) -> Tenant:
    """Require a tenant using the global manager.

    See TenantManager.require for full documentation.
    """
    return get_tenant_manager().require(tenant_id)


def current_tenant() -> Tenant | None:
    """Get the current tenant from context.

    See TenantManager.current for full documentation.
    """
    return get_tenant_manager().current()


def tenant_context(tenant: Tenant | str) -> TenantContext:
    """Get tenant context manager.

    See TenantManager.context for full documentation.
    """
    return get_tenant_manager().context(tenant)
