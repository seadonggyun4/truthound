"""Integration with multi-tenancy and Truthound core functionality.

This module provides integration points between the RBAC system
and Truthound's multi-tenancy, validation, checkpoint, and other features.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Iterator, TypeVar

from truthound.rbac.core import (
    AccessContext,
    AccessDecision,
    Permission,
    PermissionAction,
    PermissionDeniedError,
    PermissionEffect,
    Principal,
    PrincipalType,
    ResourceType,
    Role,
    RoleType,
    SecurityContext,
)
from truthound.rbac.policy import PolicyEngine

if TYPE_CHECKING:
    from truthound.multitenancy.core import Tenant
    from truthound.multitenancy.manager import TenantManager


F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Tenant-Aware RBAC Configuration
# =============================================================================


@dataclass
class TenantRBACConfig:
    """Configuration for tenant-aware RBAC."""

    # Tenant isolation
    isolate_roles_by_tenant: bool = True
    isolate_principals_by_tenant: bool = True

    # Cross-tenant access
    allow_cross_tenant_superuser: bool = True
    cross_tenant_roles: set[str] = field(
        default_factory=lambda: {"system_admin", "platform_admin"}
    )

    # Role inheritance
    inherit_platform_roles: bool = True
    platform_roles: set[str] = field(
        default_factory=lambda: {"viewer", "editor", "admin"}
    )

    # Automatic role assignment
    auto_assign_tenant_member: bool = True
    default_tenant_role: str = "tenant_member"


# =============================================================================
# Tenant-Aware Policy Evaluator
# =============================================================================


class TenantAwarePolicyEvaluator:
    """Evaluator that considers tenant context in access decisions.

    This evaluator ensures that principals can only access resources
    within their tenant unless they have cross-tenant permissions.

    Example:
        >>> evaluator = TenantAwarePolicyEvaluator(config)
        >>> engine.add_evaluator(evaluator)
    """

    def __init__(
        self,
        config: TenantRBACConfig | None = None,
        tenant_manager: "TenantManager | None" = None,
    ) -> None:
        self._config = config or TenantRBACConfig()
        self._tenant_manager = tenant_manager

    @property
    def name(self) -> str:
        return "tenant_isolation"

    @property
    def priority(self) -> int:
        return 1000  # High priority - checked early

    def evaluate(self, context: AccessContext) -> AccessDecision:
        """Evaluate tenant isolation rules."""
        if not context.principal:
            return AccessDecision.deny("No principal in context")

        principal_tenant = context.principal.tenant_id
        resource_tenant = context.resource_attributes.get("tenant_id")

        # If no tenant context, skip this evaluator
        if not principal_tenant and not resource_tenant:
            return AccessDecision(
                allowed=True,
                reason="No tenant context - skipping tenant isolation",
                effect=PermissionEffect.ALLOW,
            )

        # Check for cross-tenant access
        if principal_tenant and resource_tenant:
            if principal_tenant != resource_tenant:
                # Check if principal has cross-tenant role
                if self._has_cross_tenant_access(context.principal):
                    return AccessDecision.allow(
                        f"Cross-tenant access granted via role"
                    )

                return AccessDecision.deny(
                    f"Tenant isolation: principal tenant {principal_tenant} "
                    f"cannot access resource in tenant {resource_tenant}"
                )

        return AccessDecision(
            allowed=True,
            reason="Tenant isolation check passed",
            effect=PermissionEffect.ALLOW,
        )

    def _has_cross_tenant_access(self, principal: Principal) -> bool:
        """Check if principal has cross-tenant access roles."""
        if not self._config.allow_cross_tenant_superuser:
            return False

        return bool(principal.roles.intersection(self._config.cross_tenant_roles))


# =============================================================================
# Tenant-Scoped Role Management
# =============================================================================


class TenantScopedRoleManager:
    """Manages roles within a tenant context.

    Ensures that roles are properly scoped to tenants and handles
    role inheritance from platform to tenant level.

    Example:
        >>> manager = TenantScopedRoleManager(role_store, config)
        >>> role = manager.create_tenant_role(
        ...     tenant_id="tenant_123",
        ...     name="Data Analyst",
        ...     permissions={"dataset:read", "validation:execute"},
        ... )
    """

    def __init__(
        self,
        role_store: Any,
        config: TenantRBACConfig | None = None,
    ) -> None:
        from truthound.rbac.storage import MemoryRoleStore

        self._role_store = role_store or MemoryRoleStore()
        self._config = config or TenantRBACConfig()

    def create_tenant_role(
        self,
        tenant_id: str,
        name: str,
        permissions: set[str] | None = None,
        parent_roles: set[str] | None = None,
        description: str = "",
    ) -> Role:
        """Create a role scoped to a tenant.

        Args:
            tenant_id: Tenant ID
            name: Role name
            permissions: Permission strings
            parent_roles: Parent role IDs
            description: Role description

        Returns:
            Created Role object.
        """
        from truthound.rbac.core import generate_role_id

        role_id = f"{tenant_id}:{generate_role_id(name)}"

        role = Role(
            id=role_id,
            name=name,
            description=description,
            role_type=RoleType.CUSTOM,
            permissions={Permission.parse(p) for p in (permissions or set())},
            parent_roles=parent_roles or set(),
            tenant_id=tenant_id,
        )

        self._role_store.save(role)
        return role

    def get_tenant_roles(self, tenant_id: str) -> list[Role]:
        """Get all roles for a tenant."""
        return self._role_store.list(tenant_id=tenant_id)

    def get_inherited_roles(self, tenant_id: str) -> list[Role]:
        """Get platform roles that a tenant inherits."""
        if not self._config.inherit_platform_roles:
            return []

        all_roles = []
        for role_id in self._config.platform_roles:
            role = self._role_store.get(role_id)
            if role and role.tenant_id is None:
                all_roles.append(role)

        return all_roles

    def assign_default_role(
        self,
        principal: Principal,
        tenant_id: str,
    ) -> Principal:
        """Assign the default tenant role to a principal.

        Args:
            principal: Principal to update
            tenant_id: Tenant ID

        Returns:
            Updated Principal.
        """
        if not self._config.auto_assign_tenant_member:
            return principal

        default_role_id = f"{tenant_id}:{self._config.default_tenant_role}"

        # Check if role exists, create if not
        if not self._role_store.exists(default_role_id):
            self.create_tenant_role(
                tenant_id=tenant_id,
                name="Tenant Member",
                permissions={"dataset:read", "validation:read"},
                description="Default role for tenant members",
            )

        principal.add_role(default_role_id)
        return principal


# =============================================================================
# Tenant-Scoped Principal Management
# =============================================================================


class TenantScopedPrincipalManager:
    """Manages principals within a tenant context.

    Example:
        >>> manager = TenantScopedPrincipalManager(principal_store)
        >>> principal = manager.create_tenant_principal(
        ...     tenant_id="tenant_123",
        ...     name="john.doe@example.com",
        ... )
    """

    def __init__(
        self,
        principal_store: Any,
        role_manager: TenantScopedRoleManager | None = None,
        config: TenantRBACConfig | None = None,
    ) -> None:
        from truthound.rbac.storage import MemoryPrincipalStore

        self._principal_store = principal_store or MemoryPrincipalStore()
        self._role_manager = role_manager
        self._config = config or TenantRBACConfig()

    def create_tenant_principal(
        self,
        tenant_id: str,
        name: str,
        email: str = "",
        principal_type: PrincipalType = PrincipalType.USER,
        roles: set[str] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Principal:
        """Create a principal scoped to a tenant.

        Args:
            tenant_id: Tenant ID
            name: Principal name
            email: Principal email
            principal_type: Type of principal
            roles: Initial roles
            attributes: ABAC attributes

        Returns:
            Created Principal object.
        """
        from truthound.rbac.core import generate_principal_id

        principal_id = generate_principal_id(f"{tenant_id}_user")

        principal = Principal(
            id=principal_id,
            type=principal_type,
            name=name,
            email=email,
            roles=roles or set(),
            attributes=attributes or {},
            tenant_id=tenant_id,
        )

        # Assign default tenant role
        if self._role_manager and self._config.auto_assign_tenant_member:
            principal = self._role_manager.assign_default_role(principal, tenant_id)

        self._principal_store.save(principal)
        return principal

    def get_tenant_principals(self, tenant_id: str) -> list[Principal]:
        """Get all principals for a tenant."""
        return self._principal_store.list(tenant_id=tenant_id)

    def is_tenant_member(self, principal: Principal, tenant_id: str) -> bool:
        """Check if principal is a member of a tenant."""
        if principal.tenant_id == tenant_id:
            return True

        # Check for cross-tenant access
        if self._config.allow_cross_tenant_superuser:
            return bool(principal.roles.intersection(self._config.cross_tenant_roles))

        return False


# =============================================================================
# Integration with Truthound Multitenancy Module
# =============================================================================


class MultitenancyRBACIntegration:
    """Integration between RBAC and multitenancy modules.

    Provides utilities for creating principals from tenants,
    syncing tenant membership, and enforcing tenant-scoped access.

    Example:
        >>> integration = MultitenancyRBACIntegration(
        ...     rbac_manager=rbac_manager,
        ...     tenant_manager=tenant_manager,
        ... )
        >>> principal = integration.get_or_create_tenant_admin(tenant)
    """

    def __init__(
        self,
        rbac_manager: "RBACManager | None" = None,
        tenant_manager: "TenantManager | None" = None,
        config: TenantRBACConfig | None = None,
    ) -> None:
        self._rbac_manager = rbac_manager
        self._tenant_manager = tenant_manager
        self._config = config or TenantRBACConfig()

    def get_or_create_tenant_admin(
        self,
        tenant: "Tenant",
    ) -> Principal:
        """Get or create the admin principal for a tenant.

        Args:
            tenant: Tenant object

        Returns:
            Principal for tenant admin.
        """
        if not self._rbac_manager:
            raise RuntimeError("RBAC manager not configured")

        admin_id = f"tenant_admin:{tenant.id}"
        principal = self._rbac_manager.principal_store.get(admin_id)

        if principal:
            return principal

        # Create admin principal
        principal = Principal(
            id=admin_id,
            type=PrincipalType.USER,
            name=f"Admin for {tenant.name}",
            email=tenant.owner_email,
            roles={f"{tenant.id}:admin", "tenant_admin"},
            tenant_id=tenant.id,
        )

        self._rbac_manager.principal_store.save(principal)
        return principal

    def sync_tenant_membership(
        self,
        principal: Principal,
        tenant: "Tenant",
    ) -> Principal:
        """Sync principal membership with tenant.

        Ensures the principal has appropriate roles based on their
        relationship to the tenant.

        Args:
            principal: Principal to sync
            tenant: Tenant to sync with

        Returns:
            Updated Principal.
        """
        if principal.tenant_id != tenant.id:
            return principal

        # Ensure basic tenant membership role
        member_role = f"{tenant.id}:member"
        if member_role not in principal.roles:
            principal.add_role(member_role)

        # Check if principal is tenant owner
        if principal.email and principal.email == tenant.owner_email:
            owner_role = f"{tenant.id}:owner"
            if owner_role not in principal.roles:
                principal.add_role(owner_role)

        return principal

    def create_tenant_roles(self, tenant: "Tenant") -> list[Role]:
        """Create default roles for a new tenant.

        Args:
            tenant: Tenant to create roles for

        Returns:
            List of created roles.
        """
        if not self._rbac_manager:
            raise RuntimeError("RBAC manager not configured")

        roles = []

        # Owner role - full access within tenant
        owner_role = Role(
            id=f"{tenant.id}:owner",
            name=f"{tenant.name} Owner",
            description="Full access to all tenant resources",
            role_type=RoleType.SYSTEM,
            permissions={
                Permission("*", "*", scope="tenant"),
            },
            tenant_id=tenant.id,
        )
        self._rbac_manager.role_store.save(owner_role)
        roles.append(owner_role)

        # Admin role
        admin_role = Role(
            id=f"{tenant.id}:admin",
            name=f"{tenant.name} Admin",
            description="Administrative access to tenant resources",
            role_type=RoleType.SYSTEM,
            permissions={
                Permission("dataset", "*"),
                Permission("validation", "*"),
                Permission("checkpoint", "*"),
                Permission("user", "read"),
                Permission("user", "update"),
                Permission("role", "read"),
            },
            parent_roles={f"{tenant.id}:member"},
            tenant_id=tenant.id,
        )
        self._rbac_manager.role_store.save(admin_role)
        roles.append(admin_role)

        # Member role - basic access
        member_role = Role(
            id=f"{tenant.id}:member",
            name=f"{tenant.name} Member",
            description="Basic access to tenant resources",
            role_type=RoleType.SYSTEM,
            permissions={
                Permission("dataset", "read"),
                Permission("validation", "read"),
                Permission("validation", "execute"),
                Permission("checkpoint", "read"),
            },
            tenant_id=tenant.id,
        )
        self._rbac_manager.role_store.save(member_role)
        roles.append(member_role)

        # Viewer role - read-only
        viewer_role = Role(
            id=f"{tenant.id}:viewer",
            name=f"{tenant.name} Viewer",
            description="Read-only access to tenant resources",
            role_type=RoleType.SYSTEM,
            permissions={
                Permission("dataset", "read"),
                Permission("validation", "read"),
                Permission("checkpoint", "read"),
            },
            tenant_id=tenant.id,
        )
        self._rbac_manager.role_store.save(viewer_role)
        roles.append(viewer_role)

        return roles

    def check_tenant_access(
        self,
        principal: Principal,
        tenant: "Tenant",
        resource: str,
        action: str,
    ) -> AccessDecision:
        """Check if principal can access resource within tenant.

        Args:
            principal: Principal requesting access
            tenant: Tenant context
            resource: Resource being accessed
            action: Action being performed

        Returns:
            Access decision.
        """
        if not self._rbac_manager:
            raise RuntimeError("RBAC manager not configured")

        # Check tenant membership
        if principal.tenant_id != tenant.id:
            # Check cross-tenant access
            if not principal.roles.intersection(self._config.cross_tenant_roles):
                return AccessDecision.deny(
                    f"Principal {principal.id} is not a member of tenant {tenant.id}"
                )

        # Check permission
        return self._rbac_manager.check(
            principal=principal,
            resource=resource,
            action=action,
            resource_attributes={"tenant_id": tenant.id},
        )


# =============================================================================
# Decorators for Tenant-Aware RBAC
# =============================================================================


def require_tenant_permission(
    resource: str,
    action: str,
    use_current_tenant: bool = True,
) -> Callable[[F], F]:
    """Decorator that requires permission within the current tenant context.

    Example:
        >>> @require_tenant_permission("dataset", "read")
        ... def get_dataset(dataset_id: str):
        ...     return load_dataset(dataset_id)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            from truthound.multitenancy.core import TenantContext

            principal = SecurityContext.get_current_principal()
            if not principal:
                raise PermissionDeniedError("No authenticated principal")

            tenant = TenantContext.get_current_tenant() if use_current_tenant else None
            tenant_id = tenant.id if tenant else principal.tenant_id

            if not tenant_id:
                raise PermissionDeniedError("No tenant context")

            # Get RBAC manager
            rbac_manager = _get_default_rbac_manager()

            decision = rbac_manager.check(
                principal=principal,
                resource=resource,
                action=action,
                resource_attributes={"tenant_id": tenant_id},
            )

            if not decision.allowed:
                raise PermissionDeniedError(
                    decision.reason,
                    principal_id=principal.id,
                    resource=resource,
                    action=action,
                )

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def require_tenant_role(
    role: str | set[str],
    use_current_tenant: bool = True,
) -> Callable[[F], F]:
    """Decorator that requires a tenant-scoped role.

    The role name will be prefixed with the tenant ID.

    Example:
        >>> @require_tenant_role("admin")
        ... def admin_function():
        ...     # Requires {tenant_id}:admin role
        ...     pass
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            from truthound.multitenancy.core import TenantContext

            principal = SecurityContext.get_current_principal()
            if not principal:
                raise PermissionDeniedError("No authenticated principal")

            tenant = TenantContext.get_current_tenant() if use_current_tenant else None
            tenant_id = tenant.id if tenant else principal.tenant_id

            if not tenant_id:
                raise PermissionDeniedError("No tenant context")

            # Build tenant-scoped role names
            roles = {role} if isinstance(role, str) else role
            tenant_roles = {f"{tenant_id}:{r}" for r in roles}

            # Check if principal has any of the tenant-scoped roles
            if not tenant_roles.intersection(principal.roles):
                raise PermissionDeniedError(
                    f"Requires one of roles: {tenant_roles}",
                    principal_id=principal.id,
                )

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


@contextmanager
def tenant_rbac_context(
    tenant: "Tenant | str",
    principal: Principal,
) -> Iterator[tuple["Tenant", Principal]]:
    """Context manager for tenant-scoped RBAC operations.

    Example:
        >>> with tenant_rbac_context(tenant, principal) as (t, p):
        ...     # Operations within tenant and principal context
        ...     run_validation()
    """
    from truthound.multitenancy.core import TenantContext

    # Get tenant manager if needed
    if isinstance(tenant, str):
        from truthound.multitenancy.manager import get_tenant_manager
        tenant_mgr = get_tenant_manager()
        tenant = tenant_mgr.require(tenant)

    with TenantContext.set_current(tenant):
        with SecurityContext.set_principal(principal):
            yield tenant, principal


# =============================================================================
# Integration Hooks
# =============================================================================


def create_tenant_rbac_hooks(
    integration: MultitenancyRBACIntegration,
) -> dict[str, Callable[..., Any]]:
    """Create hooks for tenant lifecycle events.

    These hooks can be registered with TenantManager to automatically
    create roles and principals when tenants are created.

    Example:
        >>> hooks = create_tenant_rbac_hooks(integration)
        >>> config = TenantManagerConfig(
        ...     on_create=[hooks["on_create"]],
        ...     on_delete=[hooks["on_delete"]],
        ... )
    """

    def on_tenant_create(tenant: "Tenant") -> None:
        """Called when a new tenant is created."""
        # Create default roles
        integration.create_tenant_roles(tenant)

        # Create admin principal
        integration.get_or_create_tenant_admin(tenant)

    def on_tenant_delete(tenant_id: str) -> None:
        """Called when a tenant is deleted."""
        # Clean up roles and principals would be handled here
        # In production, this should remove all tenant-scoped entities
        pass

    return {
        "on_create": on_tenant_create,
        "on_delete": on_tenant_delete,
    }


# =============================================================================
# Default Manager Access
# =============================================================================


_default_rbac_manager: "RBACManager | None" = None


def set_default_rbac_manager(manager: "RBACManager") -> None:
    """Set the default RBAC manager for decorator use."""
    global _default_rbac_manager
    _default_rbac_manager = manager


def _get_default_rbac_manager() -> "RBACManager":
    """Get the default RBAC manager."""
    if _default_rbac_manager is None:
        raise RuntimeError("No default RBAC manager configured")
    return _default_rbac_manager


# Import for type hints (avoid circular import)
if TYPE_CHECKING:
    from truthound.rbac.manager import RBACManager
