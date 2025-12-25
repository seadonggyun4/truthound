"""Role-Based Access Control (RBAC) module for Truthound.

This module provides comprehensive RBAC capabilities with support for:
- Role management with hierarchical inheritance
- Principal (user/service) management
- Permission checking with caching
- Policy-based evaluation with ABAC extensions
- Multi-tenancy integration
- Middleware for web frameworks (ASGI/WSGI)

Quick Start
-----------

Basic usage with the global manager:

    >>> from truthound.rbac import (
    ...     create_role,
    ...     create_principal,
    ...     check_permission,
    ...     require_permission,
    ... )
    >>>
    >>> # Create a role
    >>> role = create_role(
    ...     name="Data Analyst",
    ...     permissions={"dataset:read", "validation:execute"},
    ... )
    >>>
    >>> # Create a principal
    >>> principal = create_principal(
    ...     name="john.doe@example.com",
    ...     roles={role.id},
    ... )
    >>>
    >>> # Check permission
    >>> decision = check_permission(principal, "dataset", "read")
    >>> print(decision.allowed)  # True

Using the RBACManager directly:

    >>> from truthound.rbac import RBACManager, RBACManagerConfig
    >>>
    >>> config = RBACManagerConfig(enable_caching=True)
    >>> manager = RBACManager(config=config)
    >>>
    >>> with manager.context(principal):
    ...     # All operations use this principal
    ...     manager.require(principal, "dataset", "read")

Using decorators:

    >>> from truthound.rbac import require_permission, require_role
    >>>
    >>> @require_permission("dataset", "read")
    ... def get_dataset(dataset_id: str):
    ...     return load_dataset(dataset_id)
    >>>
    >>> @require_role("admin")
    ... def admin_operation():
    ...     pass

Multi-tenancy integration:

    >>> from truthound.rbac import (
    ...     MultitenancyRBACIntegration,
    ...     require_tenant_permission,
    ... )
    >>>
    >>> @require_tenant_permission("dataset", "read")
    ... def get_tenant_dataset(dataset_id: str):
    ...     # Automatically checks tenant context
    ...     return load_dataset(dataset_id)

Architecture
------------

The RBAC module is organized into several sub-modules:

- core: Core types (Permission, Role, Principal, etc.)
- storage: Storage backends (Memory, File, SQLite)
- policy: Policy engine and evaluators
- middleware: Web framework middleware and decorators
- integration: Multi-tenancy and Truthound integration
- manager: Central RBACManager class
"""

from truthound.rbac.core import (
    # Enums
    ConditionOperator,
    PermissionAction,
    PermissionEffect,
    PrincipalType,
    ResourceType,
    RoleType,
    # Exceptions
    CircularRoleError,
    InvalidPermissionError,
    PermissionDeniedError,
    PermissionNotFoundError,
    PolicyEvaluationError,
    RBACError,
    RoleNotFoundError,
    # Core types
    AccessContext,
    AccessDecision,
    Condition,
    Permission,
    Principal,
    Role,
    SecurityContext,
    # Interfaces
    PermissionChecker,
    PolicyEvaluator,
    PrincipalStore,
    RoleStore,
    # Utility functions
    current_principal,
    generate_principal_id,
    generate_role_id,
    require_principal,
)

from truthound.rbac.storage import (
    # Role stores
    CachedRoleStore,
    FileRoleStore,
    MemoryRoleStore,
    SQLiteRoleStore,
    create_role_store,
    # Principal stores
    FilePrincipalStore,
    MemoryPrincipalStore,
    SQLitePrincipalStore,
    create_principal_store,
)

from truthound.rbac.policy import (
    # Enums
    PolicyCombination,
    # Configuration
    PolicyEngineConfig,
    # Policy types
    Policy,
    PolicyEngine,
    # Evaluators
    ABACEvaluator,
    OwnershipEvaluator,
    PolicyBasedEvaluator,
    RoleBasedEvaluator,
    SuperuserEvaluator,
    TenantIsolationEvaluator,
)

from truthound.rbac.middleware import (
    # Configuration
    RBACMiddlewareConfig,
    # Middleware classes
    ASGIRBACMiddleware,
    RBACMiddleware,
    WSGIRBACMiddleware,
    # Decorators (sync)
    require_permission,
    require_role,
    with_principal,
    # Decorators (async)
    require_permission_async,
    require_role_async,
    with_principal_async,
    # Framework integration
    create_fastapi_permission_dependency,
    create_fastapi_rbac_dependency,
    create_flask_rbac_middleware,
    create_flask_rbac_teardown,
    # Engine management
    set_default_engine,
)

from truthound.rbac.integration import (
    # Configuration
    TenantRBACConfig,
    # Evaluators
    TenantAwarePolicyEvaluator,
    # Managers
    TenantScopedPrincipalManager,
    TenantScopedRoleManager,
    # Integration
    MultitenancyRBACIntegration,
    create_tenant_rbac_hooks,
    # Decorators
    require_tenant_permission,
    require_tenant_role,
    tenant_rbac_context,
    # Default manager access
    set_default_rbac_manager,
)

from truthound.rbac.manager import (
    # Configuration
    RBACManagerConfig,
    # Manager class
    RBACManager,
    # Global manager functions
    configure_rbac_manager,
    get_rbac_manager,
    set_rbac_manager,
    # Convenience functions
    check_permission,
    create_principal,
    create_role,
    get_principal,
    get_role,
    require_permission as require_permission_global,
)


__all__ = [
    # ==========================================================================
    # Enums
    # ==========================================================================
    "ConditionOperator",
    "PermissionAction",
    "PermissionEffect",
    "PolicyCombination",
    "PrincipalType",
    "ResourceType",
    "RoleType",
    # ==========================================================================
    # Exceptions
    # ==========================================================================
    "CircularRoleError",
    "InvalidPermissionError",
    "PermissionDeniedError",
    "PermissionNotFoundError",
    "PolicyEvaluationError",
    "RBACError",
    "RoleNotFoundError",
    # ==========================================================================
    # Core Types
    # ==========================================================================
    "AccessContext",
    "AccessDecision",
    "Condition",
    "Permission",
    "Principal",
    "Role",
    "SecurityContext",
    # ==========================================================================
    # Interfaces
    # ==========================================================================
    "PermissionChecker",
    "PolicyEvaluator",
    "PrincipalStore",
    "RoleStore",
    # ==========================================================================
    # Storage Backends
    # ==========================================================================
    # Role stores
    "CachedRoleStore",
    "FileRoleStore",
    "MemoryRoleStore",
    "SQLiteRoleStore",
    "create_role_store",
    # Principal stores
    "FilePrincipalStore",
    "MemoryPrincipalStore",
    "SQLitePrincipalStore",
    "create_principal_store",
    # ==========================================================================
    # Policy Engine
    # ==========================================================================
    "Policy",
    "PolicyEngine",
    "PolicyEngineConfig",
    # Evaluators
    "ABACEvaluator",
    "OwnershipEvaluator",
    "PolicyBasedEvaluator",
    "RoleBasedEvaluator",
    "SuperuserEvaluator",
    "TenantIsolationEvaluator",
    # ==========================================================================
    # Middleware
    # ==========================================================================
    "ASGIRBACMiddleware",
    "RBACMiddleware",
    "RBACMiddlewareConfig",
    "WSGIRBACMiddleware",
    # Framework integration
    "create_fastapi_permission_dependency",
    "create_fastapi_rbac_dependency",
    "create_flask_rbac_middleware",
    "create_flask_rbac_teardown",
    # ==========================================================================
    # Decorators
    # ==========================================================================
    # Sync decorators
    "require_permission",
    "require_role",
    "with_principal",
    # Async decorators
    "require_permission_async",
    "require_role_async",
    "with_principal_async",
    # Tenant-aware decorators
    "require_tenant_permission",
    "require_tenant_role",
    "tenant_rbac_context",
    # ==========================================================================
    # Multi-tenancy Integration
    # ==========================================================================
    "MultitenancyRBACIntegration",
    "TenantAwarePolicyEvaluator",
    "TenantRBACConfig",
    "TenantScopedPrincipalManager",
    "TenantScopedRoleManager",
    "create_tenant_rbac_hooks",
    "set_default_rbac_manager",
    # ==========================================================================
    # Manager
    # ==========================================================================
    "RBACManager",
    "RBACManagerConfig",
    "configure_rbac_manager",
    "get_rbac_manager",
    "set_rbac_manager",
    "set_default_engine",
    # ==========================================================================
    # Convenience Functions
    # ==========================================================================
    "check_permission",
    "create_principal",
    "create_role",
    "current_principal",
    "generate_principal_id",
    "generate_role_id",
    "get_principal",
    "get_role",
    "require_permission_global",
    "require_principal",
]
