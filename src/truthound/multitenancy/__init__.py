"""Multi-tenancy support for Truthound.

This module provides a comprehensive multi-tenant architecture for Truthound,
enabling data isolation, tenant-specific configuration, quota management,
and seamless integration with the core validation functionality.

Features:
    - Multiple isolation strategies (shared, row-level, schema, database)
    - Flexible tenant resolution (header, subdomain, path, JWT, API key)
    - Quota tracking and enforcement
    - ASGI/WSGI middleware for web frameworks
    - Decorators for tenant-aware functions
    - Integration with Truthound validation and checkpoint systems

Architecture:
    The multi-tenancy system follows a layered design:

    Resolution Layer
        │
        ├── TenantResolver (Header, Subdomain, Path, JWT, etc.)
        │
        v
    Context Layer
        │
        ├── TenantContext (Thread-safe & async-safe context)
        │
        v
    Management Layer
        │
        ├── TenantManager (CRUD, lifecycle, configuration)
        │
        v
    Isolation Layer
        │
        ├── IsolationManager (Shared, Row-Level, Schema, Database)
        │
        v
    Quota Layer
        │
        ├── QuotaEnforcer (Usage tracking, limit enforcement)
        │
        v
    Integration Layer
        │
        ├── Middleware (ASGI, WSGI, Flask, FastAPI)
        ├── Decorators (@tenant_required, @with_tenant)
        └── API (tenant_check, tenant_compare)

Usage:
    >>> from truthound.multitenancy import (
    ...     TenantManager, Tenant, TenantTier, IsolationLevel,
    ...     TenantContext, tenant_required,
    ... )
    >>>
    >>> # Create manager
    >>> manager = TenantManager()
    >>>
    >>> # Create a tenant
    >>> tenant = manager.create(
    ...     name="Acme Corp",
    ...     tier=TenantTier.PROFESSIONAL,
    ...     isolation_level=IsolationLevel.SCHEMA,
    ... )
    >>>
    >>> # Use tenant context
    >>> with manager.context(tenant):
    ...     # All operations are now tenant-scoped
    ...     result = th.check(data)
    >>>
    >>> # Using decorators
    >>> @tenant_required()
    ... def process_data(df):
    ...     return validate(df)
    >>>
    >>> # Using middleware (FastAPI)
    >>> from fastapi import FastAPI, Depends
    >>> from truthound.multitenancy import create_fastapi_tenant_dependency
    >>> app = FastAPI()
    >>> get_tenant = create_fastapi_tenant_dependency(store, resolver)
    >>> @app.get("/validate")
    >>> async def validate(tenant: Tenant = Depends(get_tenant)):
    ...     return {"tenant": tenant.name}
"""

# Core types and configuration
from truthound.multitenancy.core import (
    # Enums
    TenantStatus,
    IsolationLevel,
    TenantTier,
    ResourceType,
    # Exceptions
    TenantError,
    TenantNotFoundError,
    TenantAccessDeniedError,
    TenantSuspendedError,
    TenantQuotaExceededError,
    TenantConfigError,
    TenantIsolationError,
    # Data types
    TenantQuota,
    TenantSettings,
    TenantMetadata,
    Tenant,
    # Context
    TenantContext,
    # Interfaces
    TenantStore,
    TenantResolver,
    IsolationStrategy,
    QuotaTracker,
    # Utilities
    generate_tenant_id,
    generate_slug,
    hash_tenant_id,
    current_tenant,
    current_tenant_id,
    require_tenant,
)

# Storage backends
from truthound.multitenancy.storage import (
    MemoryTenantStore,
    FileTenantStore,
    FileStorageConfig,
    SQLiteTenantStore,
    CachedTenantStore,
    CacheConfig,
    create_tenant_store,
)

# Isolation strategies
from truthound.multitenancy.isolation import (
    SharedIsolation,
    RowLevelIsolation,
    RowLevelPolicy,
    SchemaIsolation,
    SchemaConfig,
    DatabaseIsolation,
    DatabaseConfig,
    CompositeIsolation,
    IsolationManager,
    create_isolation_strategy,
)

# Resolvers
from truthound.multitenancy.resolvers import (
    HeaderResolver,
    HeaderResolverConfig,
    SubdomainResolver,
    SubdomainResolverConfig,
    PathResolver,
    PathResolverConfig,
    APIKeyResolver,
    APIKeyResolverConfig,
    JWTResolver,
    JWTResolverConfig,
    CompositeResolver,
    ContextResolver,
    CallableResolver,
    create_resolver,
)

# Quota management
from truthound.multitenancy.quota import (
    UsageRecord,
    UsageSummary,
    MemoryQuotaTracker,
    RedisQuotaTracker,
    RedisQuotaConfig,
    QuotaEnforcer,
    QuotaContext,
    create_quota_tracker,
)

# Middleware and decorators
from truthound.multitenancy.middleware import (
    TenantMiddlewareConfig,
    TenantMiddleware,
    ASGITenantMiddleware,
    WSGITenantMiddleware,
    tenant_required,
    tenant_required_async,
    with_tenant,
    with_tenant_async,
    tenant_isolated,
    create_flask_tenant_middleware,
    create_flask_tenant_teardown,
    create_fastapi_tenant_dependency,
)

# Manager
from truthound.multitenancy.manager import (
    TenantManagerConfig,
    TenantManager,
    get_tenant_manager,
    set_tenant_manager,
    configure_tenant_manager,
    create_tenant,
    get_tenant,
    tenant_context,
)

# Integration
from truthound.multitenancy.integration import (
    TenantValidationConfig,
    TenantAwareValidation,
    TenantCheckpointConfig,
    TenantAwareCheckpoint,
    TenantAwareStore,
    tenant_validation,
    tenant_checkpoint,
    tenant_check,
    tenant_compare,
)


__all__ = [
    # Enums
    "TenantStatus",
    "IsolationLevel",
    "TenantTier",
    "ResourceType",
    # Exceptions
    "TenantError",
    "TenantNotFoundError",
    "TenantAccessDeniedError",
    "TenantSuspendedError",
    "TenantQuotaExceededError",
    "TenantConfigError",
    "TenantIsolationError",
    # Data types
    "TenantQuota",
    "TenantSettings",
    "TenantMetadata",
    "Tenant",
    # Context
    "TenantContext",
    # Interfaces
    "TenantStore",
    "TenantResolver",
    "IsolationStrategy",
    "QuotaTracker",
    # Utilities
    "generate_tenant_id",
    "generate_slug",
    "hash_tenant_id",
    "current_tenant",
    "current_tenant_id",
    "require_tenant",
    # Storage
    "MemoryTenantStore",
    "FileTenantStore",
    "FileStorageConfig",
    "SQLiteTenantStore",
    "CachedTenantStore",
    "CacheConfig",
    "create_tenant_store",
    # Isolation
    "SharedIsolation",
    "RowLevelIsolation",
    "RowLevelPolicy",
    "SchemaIsolation",
    "SchemaConfig",
    "DatabaseIsolation",
    "DatabaseConfig",
    "CompositeIsolation",
    "IsolationManager",
    "create_isolation_strategy",
    # Resolvers
    "HeaderResolver",
    "HeaderResolverConfig",
    "SubdomainResolver",
    "SubdomainResolverConfig",
    "PathResolver",
    "PathResolverConfig",
    "APIKeyResolver",
    "APIKeyResolverConfig",
    "JWTResolver",
    "JWTResolverConfig",
    "CompositeResolver",
    "ContextResolver",
    "CallableResolver",
    "create_resolver",
    # Quota
    "UsageRecord",
    "UsageSummary",
    "MemoryQuotaTracker",
    "RedisQuotaTracker",
    "RedisQuotaConfig",
    "QuotaEnforcer",
    "QuotaContext",
    "create_quota_tracker",
    # Middleware
    "TenantMiddlewareConfig",
    "TenantMiddleware",
    "ASGITenantMiddleware",
    "WSGITenantMiddleware",
    "tenant_required",
    "tenant_required_async",
    "with_tenant",
    "with_tenant_async",
    "tenant_isolated",
    "create_flask_tenant_middleware",
    "create_flask_tenant_teardown",
    "create_fastapi_tenant_dependency",
    # Manager
    "TenantManagerConfig",
    "TenantManager",
    "get_tenant_manager",
    "set_tenant_manager",
    "configure_tenant_manager",
    "create_tenant",
    "get_tenant",
    "tenant_context",
    # Integration
    "TenantValidationConfig",
    "TenantAwareValidation",
    "TenantCheckpointConfig",
    "TenantAwareCheckpoint",
    "TenantAwareStore",
    "tenant_validation",
    "tenant_checkpoint",
    "tenant_check",
    "tenant_compare",
]
