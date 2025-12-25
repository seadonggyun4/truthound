"""RBAC manager for role-based access control.

This module provides the central RBACManager class that orchestrates
all RBAC-related operations including role management, principal management,
permission checking, and policy evaluation.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterator

from truthound.rbac.core import (
    AccessContext,
    AccessDecision,
    Permission,
    PermissionAction,
    PermissionDeniedError,
    Principal,
    PrincipalStore,
    PrincipalType,
    Role,
    RoleNotFoundError,
    RoleStore,
    RoleType,
    SecurityContext,
    generate_principal_id,
    generate_role_id,
)
from truthound.rbac.policy import (
    PolicyCombination,
    PolicyEngine,
    PolicyEngineConfig,
    RoleBasedEvaluator,
)
from truthound.rbac.storage import (
    MemoryPrincipalStore,
    MemoryRoleStore,
    create_principal_store,
    create_role_store,
)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RBACManagerConfig:
    """Configuration for the RBAC manager."""

    # Default settings for new principals
    default_principal_type: PrincipalType = PrincipalType.USER
    default_principal_roles: set[str] = field(default_factory=set)

    # Policy evaluation
    policy_combination: PolicyCombination = PolicyCombination.DENY_OVERRIDES
    cache_decisions: bool = True
    cache_ttl_seconds: int = 300

    # Security settings
    require_authentication: bool = True
    anonymous_principal: Principal | None = None

    # Lifecycle hooks
    on_role_create: list[Callable[[Role], None]] = field(default_factory=list)
    on_role_update: list[Callable[[Role, Role], None]] = field(default_factory=list)
    on_role_delete: list[Callable[[str], None]] = field(default_factory=list)
    on_principal_create: list[Callable[[Principal], None]] = field(default_factory=list)
    on_principal_update: list[Callable[[Principal, Principal], None]] = field(default_factory=list)
    on_principal_delete: list[Callable[[str], None]] = field(default_factory=list)
    on_access_decision: list[Callable[[AccessContext, AccessDecision], None]] = field(default_factory=list)

    # Audit
    audit_access_decisions: bool = True


# =============================================================================
# RBAC Manager
# =============================================================================


class RBACManager:
    """Central manager for all RBAC operations.

    The RBACManager provides a unified interface for:
    - CRUD operations on roles and principals
    - Permission checking and policy evaluation
    - Role inheritance management
    - Security context management

    Example:
        >>> manager = RBACManager()
        >>>
        >>> # Create a role
        >>> role = manager.create_role(
        ...     name="Data Analyst",
        ...     permissions={"dataset:read", "validation:execute"},
        ... )
        >>>
        >>> # Create a principal
        >>> principal = manager.create_principal(
        ...     name="john.doe@example.com",
        ...     roles={role.id},
        ... )
        >>>
        >>> # Check permission
        >>> decision = manager.check(principal, "dataset", "read")
        >>> print(decision.allowed)  # True
    """

    def __init__(
        self,
        role_store: RoleStore | None = None,
        principal_store: PrincipalStore | None = None,
        policy_engine: PolicyEngine | None = None,
        config: RBACManagerConfig | None = None,
    ) -> None:
        self._role_store = role_store or MemoryRoleStore()
        self._principal_store = principal_store or MemoryPrincipalStore()
        self._config = config or RBACManagerConfig()

        # Create or use provided policy engine
        if policy_engine:
            self._engine = policy_engine
        else:
            engine_config = PolicyEngineConfig(
                combination=self._config.policy_combination,
                cache_decisions=self._config.cache_decisions,
                cache_ttl_seconds=self._config.cache_ttl_seconds,
            )
            self._engine = PolicyEngine(config=engine_config)
            # Add default role-based evaluator
            self._engine.add_evaluator(RoleBasedEvaluator(self._role_store))

        self._lock = threading.RLock()

        # Initialize default roles
        self._init_default_roles()

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def role_store(self) -> RoleStore:
        """Get the role store."""
        return self._role_store

    @property
    def principal_store(self) -> PrincipalStore:
        """Get the principal store."""
        return self._principal_store

    @property
    def engine(self) -> PolicyEngine:
        """Get the policy engine."""
        return self._engine

    @property
    def config(self) -> RBACManagerConfig:
        """Get the configuration."""
        return self._config

    # =========================================================================
    # Role Management
    # =========================================================================

    def create_role(
        self,
        name: str,
        role_id: str | None = None,
        permissions: set[str | Permission] | None = None,
        parent_roles: set[str] | None = None,
        role_type: RoleType = RoleType.CUSTOM,
        description: str = "",
        tenant_id: str | None = None,
        **kwargs: Any,
    ) -> Role:
        """Create a new role.

        Args:
            name: Human-readable role name
            role_id: Optional custom ID (auto-generated if not provided)
            permissions: Permission strings or objects
            parent_roles: Parent role IDs for inheritance
            role_type: Type of role
            description: Role description
            tenant_id: Tenant ID for scoped roles
            **kwargs: Additional role attributes

        Returns:
            Created Role object.

        Example:
            >>> role = manager.create_role(
            ...     name="Editor",
            ...     permissions={"dataset:read", "dataset:update"},
            ... )
        """
        with self._lock:
            # Generate ID if not provided
            if not role_id:
                role_id = generate_role_id(name)
                if tenant_id:
                    role_id = f"{tenant_id}:{role_id}"

            # Check for duplicates
            if self._role_store.exists(role_id):
                raise ValueError(f"Role already exists: {role_id}")

            # Parse permissions
            parsed_permissions = set()
            for perm in (permissions or set()):
                if isinstance(perm, str):
                    parsed_permissions.add(Permission.parse(perm))
                else:
                    parsed_permissions.add(perm)

            # Create role
            role = Role(
                id=role_id,
                name=name,
                description=description,
                role_type=role_type,
                permissions=parsed_permissions,
                parent_roles=parent_roles or set(),
                tenant_id=tenant_id,
                **kwargs,
            )

            self._role_store.save(role)

            # Call hooks
            for hook in self._config.on_role_create:
                try:
                    hook(role)
                except Exception:
                    pass

            return role

    def get_role(self, role_id: str) -> Role | None:
        """Get a role by ID.

        Args:
            role_id: Role ID

        Returns:
            Role object if found, None otherwise.
        """
        return self._role_store.get(role_id)

    def require_role(self, role_id: str) -> Role:
        """Get a role by ID, raising if not found.

        Args:
            role_id: Role ID

        Returns:
            Role object.

        Raises:
            RoleNotFoundError: If role not found.
        """
        role = self.get_role(role_id)
        if not role:
            raise RoleNotFoundError(f"Role not found: {role_id}")
        return role

    def list_roles(
        self,
        tenant_id: str | None = None,
        role_type: RoleType | None = None,
        enabled: bool | None = None,
    ) -> list[Role]:
        """List roles with optional filters.

        Args:
            tenant_id: Filter by tenant
            role_type: Filter by role type
            enabled: Filter by enabled status

        Returns:
            List of Role objects.
        """
        return self._role_store.list(
            tenant_id=tenant_id,
            role_type=role_type,
            enabled=enabled,
        )

    def update_role(
        self,
        role_id: str,
        **updates: Any,
    ) -> Role:
        """Update a role.

        Args:
            role_id: Role ID
            **updates: Fields to update

        Returns:
            Updated Role object.

        Example:
            >>> role = manager.update_role(
            ...     "editor",
            ...     name="Senior Editor",
            ...     description="Updated description",
            ... )
        """
        with self._lock:
            role = self.require_role(role_id)
            old_role = Role.from_dict(role.to_dict())  # Copy

            # Apply updates
            for key, value in updates.items():
                if hasattr(role, key):
                    setattr(role, key, value)

            role.updated_at = datetime.now(timezone.utc)
            self._role_store.save(role)

            # Call hooks
            for hook in self._config.on_role_update:
                try:
                    hook(old_role, role)
                except Exception:
                    pass

            return role

    def delete_role(self, role_id: str) -> bool:
        """Delete a role.

        Args:
            role_id: Role ID

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            deleted = self._role_store.delete(role_id)

            if deleted:
                for hook in self._config.on_role_delete:
                    try:
                        hook(role_id)
                    except Exception:
                        pass

            return deleted

    def add_permission_to_role(
        self,
        role_id: str,
        permission: str | Permission,
    ) -> Role:
        """Add a permission to a role.

        Args:
            role_id: Role ID
            permission: Permission string or object

        Returns:
            Updated Role object.
        """
        with self._lock:
            role = self.require_role(role_id)

            if isinstance(permission, str):
                permission = Permission.parse(permission)

            role.add_permission(permission)
            role.updated_at = datetime.now(timezone.utc)
            self._role_store.save(role)

            return role

    def remove_permission_from_role(
        self,
        role_id: str,
        permission: str | Permission,
    ) -> Role:
        """Remove a permission from a role.

        Args:
            role_id: Role ID
            permission: Permission string or object

        Returns:
            Updated Role object.
        """
        with self._lock:
            role = self.require_role(role_id)

            if isinstance(permission, str):
                permission = Permission.parse(permission)

            role.remove_permission(permission)
            role.updated_at = datetime.now(timezone.utc)
            self._role_store.save(role)

            return role

    def get_all_role_permissions(self, role_id: str) -> set[Permission]:
        """Get all permissions for a role, including inherited ones.

        Args:
            role_id: Role ID

        Returns:
            Set of Permission objects.
        """
        return self._role_store.get_all_permissions(role_id)

    # =========================================================================
    # Principal Management
    # =========================================================================

    def create_principal(
        self,
        name: str,
        principal_id: str | None = None,
        principal_type: PrincipalType | None = None,
        email: str = "",
        roles: set[str] | None = None,
        permissions: set[str | Permission] | None = None,
        attributes: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        **kwargs: Any,
    ) -> Principal:
        """Create a new principal.

        Args:
            name: Principal name
            principal_id: Optional custom ID
            principal_type: Type of principal
            email: Principal email
            roles: Role IDs to assign
            permissions: Direct permissions
            attributes: ABAC attributes
            tenant_id: Tenant ID
            **kwargs: Additional attributes

        Returns:
            Created Principal object.

        Example:
            >>> principal = manager.create_principal(
            ...     name="john.doe@example.com",
            ...     email="john.doe@example.com",
            ...     roles={"editor"},
            ... )
        """
        with self._lock:
            # Generate ID if not provided
            if not principal_id:
                prefix = "user"
                if tenant_id:
                    prefix = f"{tenant_id}_{prefix}"
                principal_id = generate_principal_id(prefix)

            # Use default type from config
            if principal_type is None:
                principal_type = self._config.default_principal_type

            # Parse direct permissions
            parsed_permissions = set()
            for perm in (permissions or set()):
                if isinstance(perm, str):
                    parsed_permissions.add(Permission.parse(perm))
                else:
                    parsed_permissions.add(perm)

            # Combine with default roles
            all_roles = set(roles or set())
            all_roles.update(self._config.default_principal_roles)

            # Create principal
            principal = Principal(
                id=principal_id,
                type=principal_type,
                name=name,
                email=email,
                roles=all_roles,
                direct_permissions=parsed_permissions,
                attributes=attributes or {},
                tenant_id=tenant_id,
                **kwargs,
            )

            self._principal_store.save(principal)

            # Call hooks
            for hook in self._config.on_principal_create:
                try:
                    hook(principal)
                except Exception:
                    pass

            return principal

    def get_principal(self, principal_id: str) -> Principal | None:
        """Get a principal by ID.

        Args:
            principal_id: Principal ID

        Returns:
            Principal object if found, None otherwise.
        """
        return self._principal_store.get(principal_id)

    def get_principal_by_email(self, email: str) -> Principal | None:
        """Get a principal by email.

        Args:
            email: Principal email

        Returns:
            Principal object if found, None otherwise.
        """
        return self._principal_store.get_by_email(email)

    def require_principal(self, principal_id: str) -> Principal:
        """Get a principal by ID, raising if not found.

        Args:
            principal_id: Principal ID

        Returns:
            Principal object.

        Raises:
            ValueError: If principal not found.
        """
        principal = self.get_principal(principal_id)
        if not principal:
            raise ValueError(f"Principal not found: {principal_id}")
        return principal

    def list_principals(
        self,
        tenant_id: str | None = None,
        principal_type: PrincipalType | None = None,
        role_id: str | None = None,
    ) -> list[Principal]:
        """List principals with optional filters.

        Args:
            tenant_id: Filter by tenant
            principal_type: Filter by type
            role_id: Filter by role

        Returns:
            List of Principal objects.
        """
        return self._principal_store.list(
            tenant_id=tenant_id,
            principal_type=principal_type,
            role_id=role_id,
        )

    def update_principal(
        self,
        principal_id: str,
        **updates: Any,
    ) -> Principal:
        """Update a principal.

        Args:
            principal_id: Principal ID
            **updates: Fields to update

        Returns:
            Updated Principal object.
        """
        with self._lock:
            principal = self.require_principal(principal_id)
            old_principal = Principal.from_dict(principal.to_dict())

            for key, value in updates.items():
                if hasattr(principal, key):
                    setattr(principal, key, value)

            self._principal_store.save(principal)

            for hook in self._config.on_principal_update:
                try:
                    hook(old_principal, principal)
                except Exception:
                    pass

            return principal

    def delete_principal(self, principal_id: str) -> bool:
        """Delete a principal.

        Args:
            principal_id: Principal ID

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            deleted = self._principal_store.delete(principal_id)

            if deleted:
                for hook in self._config.on_principal_delete:
                    try:
                        hook(principal_id)
                    except Exception:
                        pass

            return deleted

    def assign_role(self, principal_id: str, role_id: str) -> Principal:
        """Assign a role to a principal.

        Args:
            principal_id: Principal ID
            role_id: Role ID

        Returns:
            Updated Principal object.
        """
        with self._lock:
            principal = self.require_principal(principal_id)
            principal.add_role(role_id)
            self._principal_store.save(principal)
            return principal

    def revoke_role(self, principal_id: str, role_id: str) -> Principal:
        """Revoke a role from a principal.

        Args:
            principal_id: Principal ID
            role_id: Role ID

        Returns:
            Updated Principal object.
        """
        with self._lock:
            principal = self.require_principal(principal_id)
            principal.remove_role(role_id)
            self._principal_store.save(principal)
            return principal

    def get_principal_permissions(self, principal_id: str) -> set[Permission]:
        """Get all effective permissions for a principal.

        Includes permissions from all roles and direct permissions.

        Args:
            principal_id: Principal ID

        Returns:
            Set of Permission objects.
        """
        principal = self.require_principal(principal_id)
        permissions = set(principal.direct_permissions)

        for role_id in principal.roles:
            role_perms = self.get_all_role_permissions(role_id)
            permissions.update(role_perms)

        return permissions

    # =========================================================================
    # Permission Checking
    # =========================================================================

    def check(
        self,
        principal: Principal | str,
        resource: str,
        action: str | PermissionAction,
        resource_attributes: dict[str, Any] | None = None,
    ) -> AccessDecision:
        """Check if a principal has permission.

        Args:
            principal: Principal or principal ID
            resource: Resource being accessed
            action: Action being performed
            resource_attributes: Resource attributes for ABAC

        Returns:
            Access decision.

        Example:
            >>> decision = manager.check(principal, "dataset", "read")
            >>> if decision.allowed:
            ...     print("Access granted")
        """
        if isinstance(principal, str):
            principal = self.require_principal(principal)

        decision = self._engine.check(
            principal=principal,
            resource=resource,
            action=action,
            resource_attributes=resource_attributes,
        )

        # Call hooks
        if self._config.audit_access_decisions:
            context = AccessContext(
                principal=principal,
                resource=resource,
                action=action,
                resource_attributes=resource_attributes or {},
            )
            for hook in self._config.on_access_decision:
                try:
                    hook(context, decision)
                except Exception:
                    pass

        return decision

    def require(
        self,
        principal: Principal | str,
        resource: str,
        action: str | PermissionAction,
        resource_attributes: dict[str, Any] | None = None,
    ) -> None:
        """Require permission, raising if denied.

        Args:
            principal: Principal or principal ID
            resource: Resource being accessed
            action: Action being performed
            resource_attributes: Resource attributes

        Raises:
            PermissionDeniedError: If access is denied.
        """
        decision = self.check(principal, resource, action, resource_attributes)

        if not decision.allowed:
            if isinstance(principal, str):
                principal_id = principal
            else:
                principal_id = principal.id

            raise PermissionDeniedError(
                decision.reason,
                principal_id=principal_id,
                resource=resource,
                action=action if isinstance(action, str) else action.value,
            )

    def has_permission(
        self,
        principal: Principal | str,
        permission: str | Permission,
    ) -> bool:
        """Check if principal has a specific permission.

        Args:
            principal: Principal or principal ID
            permission: Permission string or object

        Returns:
            True if principal has the permission.
        """
        if isinstance(principal, str):
            principal = self.require_principal(principal)

        if isinstance(permission, str):
            permission = Permission.parse(permission)

        # Check direct permissions
        for perm in principal.direct_permissions:
            if perm.matches(permission):
                return True

        # Check role permissions
        for role_id in principal.roles:
            role_perms = self.get_all_role_permissions(role_id)
            for perm in role_perms:
                if perm.matches(permission):
                    return True

        return False

    def has_role(
        self,
        principal: Principal | str,
        role: str | set[str],
        require_all: bool = False,
    ) -> bool:
        """Check if principal has specific role(s).

        Args:
            principal: Principal or principal ID
            role: Role ID or set of role IDs
            require_all: If True, require all roles

        Returns:
            True if principal has the role(s).
        """
        if isinstance(principal, str):
            principal = self.require_principal(principal)

        required_roles = {role} if isinstance(role, str) else role

        if require_all:
            return required_roles.issubset(principal.roles)
        else:
            return bool(required_roles.intersection(principal.roles))

    # =========================================================================
    # Context Management
    # =========================================================================

    def context(self, principal: Principal | str) -> SecurityContext:
        """Get a context manager for principal operations.

        Args:
            principal: Principal or ID

        Returns:
            Context manager that sets security context.

        Example:
            >>> with manager.context(principal):
            ...     # All operations use this principal
            ...     run_operation()
        """
        if isinstance(principal, str):
            principal = self.require_principal(principal)

        return SecurityContext.set_principal(principal)

    def current_principal(self) -> Principal | None:
        """Get the current principal from context.

        Returns:
            Current Principal or None.
        """
        return SecurityContext.get_current_principal()

    def require_current_principal(self) -> Principal:
        """Get the current principal, raising if not set.

        Returns:
            Current Principal.

        Raises:
            PermissionDeniedError: If no principal in context.
        """
        return SecurityContext.require_principal()

    # =========================================================================
    # Default Roles
    # =========================================================================

    def _init_default_roles(self) -> None:
        """Initialize default system roles."""
        default_roles = [
            Role(
                id="system_admin",
                name="System Admin",
                description="Full system access",
                role_type=RoleType.SYSTEM,
                permissions={Permission.all()},
            ),
            Role(
                id="tenant_admin",
                name="Tenant Admin",
                description="Full access within a tenant",
                role_type=RoleType.SYSTEM,
                permissions={
                    Permission("*", "*", scope="tenant"),
                },
            ),
            Role(
                id="viewer",
                name="Viewer",
                description="Read-only access",
                role_type=RoleType.SYSTEM,
                permissions={
                    Permission("*", "read"),
                    Permission("*", "list"),
                },
            ),
            Role(
                id="editor",
                name="Editor",
                description="Read and write access",
                role_type=RoleType.SYSTEM,
                permissions={
                    Permission("*", "read"),
                    Permission("*", "list"),
                    Permission("*", "create"),
                    Permission("*", "update"),
                },
                parent_roles={"viewer"},
            ),
            Role(
                id="admin",
                name="Admin",
                description="Full access except system settings",
                role_type=RoleType.SYSTEM,
                permissions={
                    Permission("*", "*"),
                },
                parent_roles={"editor"},
            ),
        ]

        for role in default_roles:
            if not self._role_store.exists(role.id):
                self._role_store.save(role)


# =============================================================================
# Global Manager
# =============================================================================


_default_manager: RBACManager | None = None
_manager_lock = threading.Lock()


def get_rbac_manager() -> RBACManager:
    """Get the global RBAC manager.

    Returns:
        RBACManager instance.
    """
    global _default_manager
    with _manager_lock:
        if _default_manager is None:
            _default_manager = RBACManager()
        return _default_manager


def set_rbac_manager(manager: RBACManager) -> None:
    """Set the global RBAC manager.

    Args:
        manager: RBACManager instance to use globally.
    """
    global _default_manager
    with _manager_lock:
        _default_manager = manager


def configure_rbac_manager(
    role_store: RoleStore | None = None,
    principal_store: PrincipalStore | None = None,
    policy_engine: PolicyEngine | None = None,
    config: RBACManagerConfig | None = None,
) -> RBACManager:
    """Configure and set the global RBAC manager.

    Args:
        role_store: Role storage backend
        principal_store: Principal storage backend
        policy_engine: Policy engine
        config: Manager configuration

    Returns:
        Configured RBACManager.
    """
    manager = RBACManager(
        role_store=role_store,
        principal_store=principal_store,
        policy_engine=policy_engine,
        config=config,
    )
    set_rbac_manager(manager)
    return manager


# =============================================================================
# Convenience Functions
# =============================================================================


def create_role(name: str, **kwargs: Any) -> Role:
    """Create a role using the global manager.

    See RBACManager.create_role for full documentation.
    """
    return get_rbac_manager().create_role(name=name, **kwargs)


def get_role(role_id: str) -> Role | None:
    """Get a role using the global manager.

    See RBACManager.get_role for full documentation.
    """
    return get_rbac_manager().get_role(role_id)


def create_principal(name: str, **kwargs: Any) -> Principal:
    """Create a principal using the global manager.

    See RBACManager.create_principal for full documentation.
    """
    return get_rbac_manager().create_principal(name=name, **kwargs)


def get_principal(principal_id: str) -> Principal | None:
    """Get a principal using the global manager.

    See RBACManager.get_principal for full documentation.
    """
    return get_rbac_manager().get_principal(principal_id)


def check_permission(
    principal: Principal | str,
    resource: str,
    action: str,
) -> AccessDecision:
    """Check permission using the global manager.

    See RBACManager.check for full documentation.
    """
    return get_rbac_manager().check(principal, resource, action)


def require_permission(
    principal: Principal | str,
    resource: str,
    action: str,
) -> None:
    """Require permission using the global manager.

    See RBACManager.require for full documentation.
    """
    get_rbac_manager().require(principal, resource, action)


def current_principal() -> Principal | None:
    """Get the current principal from context.

    See RBACManager.current_principal for full documentation.
    """
    return get_rbac_manager().current_principal()
