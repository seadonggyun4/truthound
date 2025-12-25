"""Core types, configuration, and interfaces for RBAC.

This module provides the foundational types and interfaces for Role-Based
Access Control (RBAC) in Truthound, supporting flexible permission models,
hierarchical roles, and attribute-based access control (ABAC) extensions.

Design Principles:
    - Separation of concerns: Permissions, Roles, and Policies are independent
    - Extensibility: Easy to add custom permission types and policy evaluators
    - Performance: Efficient permission checking with caching
    - Security: Deny by default, explicit grants required
    - Multi-tenancy: Full integration with tenant context
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
from enum import Enum, Flag, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    FrozenSet,
    Iterator,
    Mapping,
    Protocol,
    Sequence,
    TypeVar,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Enums and Flags
# =============================================================================


class PermissionAction(Enum):
    """Standard permission actions."""

    # CRUD operations
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"

    # Extended operations
    LIST = "list"
    EXECUTE = "execute"
    APPROVE = "approve"
    EXPORT = "export"
    IMPORT = "import"
    SHARE = "share"

    # Admin operations
    MANAGE = "manage"
    ADMIN = "admin"

    # Wildcard
    ALL = "*"


class ResourceType(Enum):
    """Types of resources that can be protected."""

    # Data resources
    DATASET = "dataset"
    SCHEMA = "schema"
    VALIDATION = "validation"
    CHECKPOINT = "checkpoint"
    REPORT = "report"

    # Configuration resources
    CONFIG = "config"
    SECRET = "secret"
    WEBHOOK = "webhook"

    # User management
    USER = "user"
    ROLE = "role"
    PERMISSION = "permission"
    API_KEY = "api_key"

    # Tenant resources
    TENANT = "tenant"
    QUOTA = "quota"
    BILLING = "billing"

    # System resources
    SYSTEM = "system"
    AUDIT_LOG = "audit_log"

    # Wildcard
    ALL = "*"


class PermissionEffect(Enum):
    """Effect of a permission or policy."""

    ALLOW = "allow"
    DENY = "deny"


class RoleType(Enum):
    """Types of roles."""

    SYSTEM = "system"  # Built-in system roles
    CUSTOM = "custom"  # User-defined roles
    DYNAMIC = "dynamic"  # Dynamically computed roles


class PrincipalType(Enum):
    """Types of principals (entities that can be granted permissions)."""

    USER = "user"
    SERVICE = "service"
    API_KEY = "api_key"
    GROUP = "group"
    ROLE = "role"
    ANONYMOUS = "anonymous"
    SYSTEM = "system"


class ConditionOperator(Enum):
    """Operators for permission conditions."""

    EQUALS = "eq"
    NOT_EQUALS = "neq"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    MATCHES = "matches"  # Regex
    IN = "in"
    NOT_IN = "not_in"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN_OR_EQUAL = "lte"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"


# =============================================================================
# Exceptions
# =============================================================================


class RBACError(Exception):
    """Base exception for RBAC-related errors."""

    def __init__(
        self,
        message: str,
        principal_id: str | None = None,
        resource: str | None = None,
        action: str | None = None,
    ) -> None:
        self.principal_id = principal_id
        self.resource = resource
        self.action = action
        super().__init__(message)


class PermissionDeniedError(RBACError):
    """Raised when permission is denied."""
    pass


class RoleNotFoundError(RBACError):
    """Raised when a role is not found."""
    pass


class PermissionNotFoundError(RBACError):
    """Raised when a permission is not found."""
    pass


class PolicyEvaluationError(RBACError):
    """Raised when policy evaluation fails."""
    pass


class InvalidPermissionError(RBACError):
    """Raised when a permission specification is invalid."""
    pass


class CircularRoleError(RBACError):
    """Raised when circular role inheritance is detected."""
    pass


# =============================================================================
# Core Data Types
# =============================================================================


@dataclass(frozen=True)
class Permission:
    """Represents a single permission.

    Permissions follow the format: resource:action or resource:action:scope
    Examples:
        - dataset:read
        - validation:execute:own
        - checkpoint:*
        - *:read

    Example:
        >>> perm = Permission(
        ...     resource=ResourceType.DATASET,
        ...     action=PermissionAction.READ,
        ... )
        >>> perm.to_string()
        'dataset:read'
    """

    resource: ResourceType | str
    action: PermissionAction | str
    scope: str = ""  # Optional scope: "own", "team", "tenant", "*"
    effect: PermissionEffect = PermissionEffect.ALLOW

    def __post_init__(self) -> None:
        # Normalize resource and action to strings
        object.__setattr__(
            self,
            "resource",
            self.resource.value if isinstance(self.resource, ResourceType) else self.resource,
        )
        object.__setattr__(
            self,
            "action",
            self.action.value if isinstance(self.action, PermissionAction) else self.action,
        )

    def to_string(self) -> str:
        """Convert to string format."""
        if self.scope:
            return f"{self.resource}:{self.action}:{self.scope}"
        return f"{self.resource}:{self.action}"

    def __str__(self) -> str:
        return self.to_string()

    def __hash__(self) -> int:
        return hash((self.resource, self.action, self.scope, self.effect))

    def matches(self, other: "Permission") -> bool:
        """Check if this permission matches another (supports wildcards)."""
        # Check resource match
        if self.resource != "*" and other.resource != "*":
            if self.resource != other.resource:
                return False

        # Check action match
        if self.action != "*" and other.action != "*":
            if self.action != other.action:
                return False

        # Check scope match (empty scope matches any)
        if self.scope and other.scope:
            if self.scope != "*" and other.scope != "*":
                if self.scope != other.scope:
                    return False

        return True

    @classmethod
    def parse(cls, permission_string: str) -> "Permission":
        """Parse a permission string.

        Args:
            permission_string: Format "resource:action" or "resource:action:scope"

        Returns:
            Permission object.

        Example:
            >>> Permission.parse("dataset:read")
            Permission(resource='dataset', action='read', scope='')
        """
        parts = permission_string.split(":")
        if len(parts) < 2:
            raise InvalidPermissionError(f"Invalid permission format: {permission_string}")

        resource = parts[0]
        action = parts[1]
        scope = parts[2] if len(parts) > 2 else ""

        return cls(resource=resource, action=action, scope=scope)

    @classmethod
    def all(cls) -> "Permission":
        """Create a wildcard permission (all resources, all actions)."""
        return cls(resource="*", action="*")


@dataclass
class Condition:
    """A condition for conditional permissions (ABAC).

    Conditions allow fine-grained access control based on attributes
    of the principal, resource, or environment.

    Example:
        >>> condition = Condition(
        ...     field="resource.owner_id",
        ...     operator=ConditionOperator.EQUALS,
        ...     value="${principal.id}",  # Dynamic reference
        ... )
    """

    field: str  # e.g., "resource.owner_id", "principal.department"
    operator: ConditionOperator
    value: Any
    description: str = ""

    def evaluate(self, context: "AccessContext") -> bool:
        """Evaluate the condition against a context."""
        # Get field value from context
        actual_value = self._resolve_field(self.field, context)

        # Resolve value (might be a reference like ${principal.id})
        expected_value = self._resolve_value(self.value, context)

        # Apply operator
        return self._apply_operator(actual_value, expected_value)

    def _resolve_field(self, field: str, context: "AccessContext") -> Any:
        """Resolve a field path to its value."""
        parts = field.split(".")
        if not parts:
            return None

        # Get root object
        root = parts[0]
        if root == "principal":
            obj = context.principal.__dict__ if context.principal else {}
        elif root == "resource":
            obj = context.resource_attributes
        elif root == "environment":
            obj = context.environment
        else:
            return None

        # Navigate path
        for part in parts[1:]:
            if isinstance(obj, dict):
                obj = obj.get(part)
            elif hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return None
            if obj is None:
                return None

        return obj

    def _resolve_value(self, value: Any, context: "AccessContext") -> Any:
        """Resolve a value, handling dynamic references."""
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            field = value[2:-1]
            return self._resolve_field(field, context)
        return value

    def _apply_operator(self, actual: Any, expected: Any) -> bool:
        """Apply the comparison operator."""
        if self.operator == ConditionOperator.EQUALS:
            return actual == expected
        elif self.operator == ConditionOperator.NOT_EQUALS:
            return actual != expected
        elif self.operator == ConditionOperator.CONTAINS:
            return expected in actual if actual else False
        elif self.operator == ConditionOperator.NOT_CONTAINS:
            return expected not in actual if actual else True
        elif self.operator == ConditionOperator.STARTS_WITH:
            return str(actual).startswith(str(expected)) if actual else False
        elif self.operator == ConditionOperator.ENDS_WITH:
            return str(actual).endswith(str(expected)) if actual else False
        elif self.operator == ConditionOperator.IN:
            return actual in expected if expected else False
        elif self.operator == ConditionOperator.NOT_IN:
            return actual not in expected if expected else True
        elif self.operator == ConditionOperator.GREATER_THAN:
            return actual > expected if actual is not None else False
        elif self.operator == ConditionOperator.LESS_THAN:
            return actual < expected if actual is not None else False
        elif self.operator == ConditionOperator.GREATER_THAN_OR_EQUAL:
            return actual >= expected if actual is not None else False
        elif self.operator == ConditionOperator.LESS_THAN_OR_EQUAL:
            return actual <= expected if actual is not None else False
        elif self.operator == ConditionOperator.EXISTS:
            return actual is not None
        elif self.operator == ConditionOperator.NOT_EXISTS:
            return actual is None
        elif self.operator == ConditionOperator.MATCHES:
            import re
            return bool(re.match(str(expected), str(actual))) if actual else False

        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "field": self.field,
            "operator": self.operator.value,
            "value": self.value,
            "description": self.description,
        }


@dataclass
class Role:
    """Represents a role with associated permissions.

    Roles can inherit from other roles, creating a hierarchy.

    Example:
        >>> role = Role(
        ...     id="data_analyst",
        ...     name="Data Analyst",
        ...     permissions={
        ...         Permission.parse("dataset:read"),
        ...         Permission.parse("validation:execute"),
        ...     },
        ... )
    """

    id: str
    name: str
    description: str = ""
    role_type: RoleType = RoleType.CUSTOM

    # Permissions directly assigned to this role
    permissions: set[Permission] = field(default_factory=set)

    # Roles this role inherits from
    parent_roles: set[str] = field(default_factory=set)

    # Conditions for conditional role assignment
    conditions: list[Condition] = field(default_factory=list)

    # Metadata
    tenant_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # Status
    enabled: bool = True

    def add_permission(self, permission: Permission | str) -> None:
        """Add a permission to this role."""
        if isinstance(permission, str):
            permission = Permission.parse(permission)
        self.permissions.add(permission)

    def remove_permission(self, permission: Permission | str) -> None:
        """Remove a permission from this role."""
        if isinstance(permission, str):
            permission = Permission.parse(permission)
        self.permissions.discard(permission)

    def has_permission(self, permission: Permission | str) -> bool:
        """Check if role has a specific permission."""
        if isinstance(permission, str):
            permission = Permission.parse(permission)

        for p in self.permissions:
            if p.matches(permission):
                return True
        return False

    def add_parent_role(self, role_id: str) -> None:
        """Add a parent role for inheritance."""
        self.parent_roles.add(role_id)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "role_type": self.role_type.value,
            "permissions": [p.to_string() for p in self.permissions],
            "parent_roles": list(self.parent_roles),
            "conditions": [c.to_dict() for c in self.conditions],
            "tenant_id": self.tenant_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "metadata": self.metadata,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Role":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            role_type=RoleType(data.get("role_type", "custom")),
            permissions={Permission.parse(p) for p in data.get("permissions", [])},
            parent_roles=set(data.get("parent_roles", [])),
            tenant_id=data.get("tenant_id"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(timezone.utc),
            created_by=data.get("created_by", ""),
            metadata=data.get("metadata", {}),
            enabled=data.get("enabled", True),
        )


@dataclass
class Principal:
    """Represents an entity that can be granted permissions.

    A principal can be a user, service, API key, or any other entity
    that needs to access protected resources.

    Example:
        >>> principal = Principal(
        ...     id="user:123",
        ...     type=PrincipalType.USER,
        ...     name="john.doe@example.com",
        ...     roles={"data_analyst", "viewer"},
        ... )
    """

    id: str
    type: PrincipalType = PrincipalType.USER
    name: str = ""
    email: str = ""

    # Assigned roles
    roles: set[str] = field(default_factory=set)

    # Direct permissions (in addition to role permissions)
    direct_permissions: set[Permission] = field(default_factory=set)

    # Attributes for ABAC
    attributes: dict[str, Any] = field(default_factory=dict)

    # Tenant context
    tenant_id: str | None = None

    # Status
    enabled: bool = True

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def anonymous(cls) -> "Principal":
        """Create an anonymous principal."""
        return cls(
            id="anonymous",
            type=PrincipalType.ANONYMOUS,
            name="Anonymous",
        )

    @classmethod
    def system(cls) -> "Principal":
        """Create a system principal."""
        return cls(
            id="system",
            type=PrincipalType.SYSTEM,
            name="System",
            roles={"system_admin"},
        )

    def add_role(self, role_id: str) -> None:
        """Add a role to this principal."""
        self.roles.add(role_id)

    def remove_role(self, role_id: str) -> None:
        """Remove a role from this principal."""
        self.roles.discard(role_id)

    def has_role(self, role_id: str) -> bool:
        """Check if principal has a specific role."""
        return role_id in self.roles

    def add_permission(self, permission: Permission | str) -> None:
        """Add a direct permission."""
        if isinstance(permission, str):
            permission = Permission.parse(permission)
        self.direct_permissions.add(permission)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "email": self.email,
            "roles": list(self.roles),
            "direct_permissions": [p.to_string() for p in self.direct_permissions],
            "attributes": self.attributes,
            "tenant_id": self.tenant_id,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
            "last_active_at": self.last_active_at.isoformat() if self.last_active_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Principal":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            type=PrincipalType(data.get("type", "user")),
            name=data.get("name", ""),
            email=data.get("email", ""),
            roles=set(data.get("roles", [])),
            direct_permissions={Permission.parse(p) for p in data.get("direct_permissions", [])},
            attributes=data.get("attributes", {}),
            tenant_id=data.get("tenant_id"),
            enabled=data.get("enabled", True),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(timezone.utc),
            last_active_at=datetime.fromisoformat(data["last_active_at"]) if data.get("last_active_at") else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class AccessContext:
    """Context for access control decisions.

    Contains all information needed to make an authorization decision,
    including the principal, resource, action, and environment.

    Example:
        >>> context = AccessContext(
        ...     principal=principal,
        ...     resource="dataset:sales_data",
        ...     action=PermissionAction.READ,
        ...     resource_attributes={"owner_id": "user:456"},
        ... )
    """

    principal: Principal | None
    resource: str
    action: PermissionAction | str
    resource_type: ResourceType | str | None = None

    # Resource attributes for ABAC
    resource_attributes: dict[str, Any] = field(default_factory=dict)

    # Environment attributes (time, IP, etc.)
    environment: dict[str, Any] = field(default_factory=dict)

    # Request context
    request_id: str = ""
    tenant_id: str | None = None
    trace_id: str = ""

    def __post_init__(self) -> None:
        # Normalize action
        if isinstance(self.action, PermissionAction):
            object.__setattr__(self, "action", self.action.value)

        # Extract resource type from resource string if not provided
        if self.resource_type is None and ":" in self.resource:
            parts = self.resource.split(":")
            object.__setattr__(self, "resource_type", parts[0])

        # Add default environment attributes
        if "timestamp" not in self.environment:
            self.environment["timestamp"] = datetime.now(timezone.utc).isoformat()

    def get_required_permission(self) -> Permission:
        """Get the permission required for this access."""
        resource_type = self.resource_type or "*"
        if isinstance(resource_type, ResourceType):
            resource_type = resource_type.value
        return Permission(resource=resource_type, action=self.action)


@dataclass
class AccessDecision:
    """Result of an access control decision.

    Example:
        >>> decision = AccessDecision(
        ...     allowed=True,
        ...     reason="Permission granted by role 'data_analyst'",
        ...     matching_permissions=[Permission.parse("dataset:read")],
        ... )
    """

    allowed: bool
    reason: str = ""
    effect: PermissionEffect = PermissionEffect.DENY

    # Matching permissions that led to this decision
    matching_permissions: list[Permission] = field(default_factory=list)

    # Matching policies
    matching_policies: list[str] = field(default_factory=list)

    # Evaluation time
    evaluation_time_ms: float = 0.0

    # Audit info
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __bool__(self) -> bool:
        return self.allowed

    @classmethod
    def allow(cls, reason: str = "", permissions: list[Permission] | None = None) -> "AccessDecision":
        """Create an allow decision."""
        return cls(
            allowed=True,
            reason=reason,
            effect=PermissionEffect.ALLOW,
            matching_permissions=permissions or [],
        )

    @classmethod
    def deny(cls, reason: str = "") -> "AccessDecision":
        """Create a deny decision."""
        return cls(
            allowed=False,
            reason=reason,
            effect=PermissionEffect.DENY,
        )


# =============================================================================
# Security Context (Thread-Local & ContextVar)
# =============================================================================


_current_principal: ContextVar[Principal | None] = ContextVar("current_principal", default=None)
_current_context: ContextVar[AccessContext | None] = ContextVar("current_access_context", default=None)


@dataclass
class SecurityContext:
    """Context for the current security principal and access context.

    Provides thread-safe and async-safe context management for
    security operations.

    Example:
        >>> with SecurityContext.set_principal(principal):
        ...     current = SecurityContext.get_current_principal()
        ...     assert current.id == principal.id
    """

    principal: Principal
    roles: set[str] = field(default_factory=set)
    permissions: set[Permission] = field(default_factory=set)
    tenant_id: str | None = None
    session_id: str = ""
    authenticated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def get_current_principal(cls) -> Principal | None:
        """Get the current principal from context."""
        return _current_principal.get()

    @classmethod
    def require_principal(cls) -> Principal:
        """Get the current principal, raising if not set."""
        principal = cls.get_current_principal()
        if principal is None:
            raise PermissionDeniedError("No authenticated principal")
        return principal

    @classmethod
    @contextmanager
    def set_principal(cls, principal: Principal) -> Iterator["SecurityContext"]:
        """Set the current principal for the context.

        Example:
            >>> with SecurityContext.set_principal(principal) as ctx:
            ...     process_request()
        """
        token = _current_principal.set(principal)
        context = cls(principal=principal, roles=principal.roles)
        try:
            yield context
        finally:
            _current_principal.reset(token)

    @classmethod
    def clear(cls) -> None:
        """Clear the current security context."""
        _current_principal.set(None)
        _current_context.set(None)


# =============================================================================
# Interfaces (Abstract Base Classes)
# =============================================================================


class RoleStore(ABC):
    """Abstract interface for role storage."""

    @abstractmethod
    def get(self, role_id: str) -> Role | None:
        """Get a role by ID."""
        ...

    @abstractmethod
    def list(
        self,
        tenant_id: str | None = None,
        role_type: RoleType | None = None,
        enabled: bool | None = None,
    ) -> list[Role]:
        """List roles with optional filters."""
        ...

    @abstractmethod
    def save(self, role: Role) -> None:
        """Save a role (create or update)."""
        ...

    @abstractmethod
    def delete(self, role_id: str) -> bool:
        """Delete a role. Returns True if deleted."""
        ...

    @abstractmethod
    def exists(self, role_id: str) -> bool:
        """Check if a role exists."""
        ...

    def get_all_permissions(self, role_id: str, visited: set[str] | None = None) -> set[Permission]:
        """Get all permissions for a role, including inherited ones."""
        if visited is None:
            visited = set()

        if role_id in visited:
            raise CircularRoleError(f"Circular role inheritance detected: {role_id}")

        visited.add(role_id)
        role = self.get(role_id)
        if not role:
            return set()

        permissions = set(role.permissions)

        # Add inherited permissions
        for parent_id in role.parent_roles:
            permissions.update(self.get_all_permissions(parent_id, visited))

        return permissions


class PrincipalStore(ABC):
    """Abstract interface for principal storage."""

    @abstractmethod
    def get(self, principal_id: str) -> Principal | None:
        """Get a principal by ID."""
        ...

    @abstractmethod
    def get_by_email(self, email: str) -> Principal | None:
        """Get a principal by email."""
        ...

    @abstractmethod
    def list(
        self,
        tenant_id: str | None = None,
        principal_type: PrincipalType | None = None,
        role_id: str | None = None,
    ) -> list[Principal]:
        """List principals with optional filters."""
        ...

    @abstractmethod
    def save(self, principal: Principal) -> None:
        """Save a principal (create or update)."""
        ...

    @abstractmethod
    def delete(self, principal_id: str) -> bool:
        """Delete a principal. Returns True if deleted."""
        ...


class PolicyEvaluator(ABC):
    """Abstract interface for policy evaluation."""

    @abstractmethod
    def evaluate(self, context: AccessContext) -> AccessDecision:
        """Evaluate access for the given context."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this evaluator."""
        ...

    @property
    def priority(self) -> int:
        """Priority of this evaluator (higher = evaluated first)."""
        return 0


class PermissionChecker(Protocol):
    """Protocol for permission checking."""

    def check(
        self,
        principal: Principal,
        resource: str,
        action: str | PermissionAction,
    ) -> AccessDecision:
        """Check if principal has permission."""
        ...

    def require(
        self,
        principal: Principal,
        resource: str,
        action: str | PermissionAction,
    ) -> None:
        """Require permission, raising if denied."""
        ...


# =============================================================================
# Utility Functions
# =============================================================================


def generate_role_id(name: str) -> str:
    """Generate a role ID from name."""
    import re
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug


def generate_principal_id(prefix: str = "principal") -> str:
    """Generate a unique principal ID."""
    unique_part = uuid.uuid4().hex[:12]
    return f"{prefix}_{unique_part}"


def current_principal() -> Principal | None:
    """Get the current principal from context."""
    return SecurityContext.get_current_principal()


def require_principal() -> Principal:
    """Get the current principal, raising if not set."""
    return SecurityContext.require_principal()
