"""Policy engine and evaluators for RBAC.

This module provides the policy evaluation engine and various policy
evaluators for making access control decisions.

Architecture:
    The policy engine follows a pipeline design:

    AccessContext
        │
        ├── PolicyEngine
        │       │
        │       ├── PolicyEvaluator 1 (e.g., RoleBasedEvaluator)
        │       │
        │       ├── PolicyEvaluator 2 (e.g., ABACEvaluator)
        │       │
        │       └── PolicyEvaluator N
        │
        v
    AccessDecision

Policy Combination:
    - DENY_OVERRIDES: Any deny results in deny (default, most secure)
    - ALLOW_OVERRIDES: Any allow results in allow
    - FIRST_APPLICABLE: First matching policy decides
    - UNANIMOUS: All must agree
"""

from __future__ import annotations

import threading
import time
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Sequence

from truthound.rbac.core import (
    AccessContext,
    AccessDecision,
    Condition,
    Permission,
    PermissionEffect,
    PolicyEvaluator,
    Principal,
    Role,
    RoleStore,
    PrincipalStore,
    PermissionDeniedError,
    PolicyEvaluationError,
)


# =============================================================================
# Policy Combination Algorithms
# =============================================================================


class PolicyCombination(Enum):
    """Policy combination algorithms."""

    DENY_OVERRIDES = "deny_overrides"  # Any deny wins
    ALLOW_OVERRIDES = "allow_overrides"  # Any allow wins
    FIRST_APPLICABLE = "first_applicable"  # First match wins
    UNANIMOUS = "unanimous"  # All must allow


# =============================================================================
# Policy Types
# =============================================================================


@dataclass
class Policy:
    """A policy that defines access rules.

    Policies bind permissions to subjects (principals, roles) with
    optional conditions.

    Example:
        >>> policy = Policy(
        ...     id="data_analyst_read",
        ...     name="Data Analysts Read Access",
        ...     effect=PermissionEffect.ALLOW,
        ...     subjects=["role:data_analyst"],
        ...     resources=["dataset:*"],
        ...     actions=["read", "list"],
        ... )
    """

    id: str
    name: str
    description: str = ""

    # Effect when policy matches
    effect: PermissionEffect = PermissionEffect.ALLOW

    # Who this policy applies to
    subjects: list[str] = field(default_factory=list)  # e.g., ["role:admin", "user:123"]

    # What resources this policy covers
    resources: list[str] = field(default_factory=list)  # e.g., ["dataset:*", "validation:*"]

    # What actions are allowed/denied
    actions: list[str] = field(default_factory=list)  # e.g., ["read", "write", "*"]

    # Conditions for conditional policies (ABAC)
    conditions: list[Condition] = field(default_factory=list)

    # Priority (higher = evaluated first)
    priority: int = 0

    # Status
    enabled: bool = True

    # Tenant scope
    tenant_id: str | None = None

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def matches_subject(self, principal: Principal) -> bool:
        """Check if the policy matches the principal."""
        if not self.subjects:
            return True  # Empty subjects means applies to all

        for subject in self.subjects:
            if subject == "*":
                return True

            if ":" in subject:
                subject_type, subject_id = subject.split(":", 1)

                if subject_type == "role":
                    if subject_id == "*" or subject_id in principal.roles:
                        return True
                elif subject_type == "user":
                    if subject_id == "*" or subject_id == principal.id:
                        return True
                elif subject_type == "type":
                    if subject_id == principal.type.value:
                        return True
            else:
                # Plain role name
                if subject in principal.roles:
                    return True

        return False

    def matches_resource(self, resource: str) -> bool:
        """Check if the policy matches the resource."""
        if not self.resources:
            return True  # Empty resources means applies to all

        for pattern in self.resources:
            if pattern == "*":
                return True

            if self._matches_pattern(pattern, resource):
                return True

        return False

    def matches_action(self, action: str) -> bool:
        """Check if the policy matches the action."""
        if not self.actions:
            return True  # Empty actions means applies to all

        for allowed_action in self.actions:
            if allowed_action == "*" or allowed_action == action:
                return True

        return False

    def _matches_pattern(self, pattern: str, value: str) -> bool:
        """Check if a pattern matches a value (supports wildcards)."""
        if pattern == "*":
            return True
        if "*" not in pattern:
            return pattern == value

        # Simple wildcard matching
        if pattern.endswith("*"):
            return value.startswith(pattern[:-1])
        if pattern.startswith("*"):
            return value.endswith(pattern[1:])

        # Contains wildcard
        parts = pattern.split("*")
        if len(parts) == 2:
            return value.startswith(parts[0]) and value.endswith(parts[1])

        return pattern == value

    def evaluate_conditions(self, context: AccessContext) -> bool:
        """Evaluate all conditions for this policy."""
        if not self.conditions:
            return True

        for condition in self.conditions:
            if not condition.evaluate(context):
                return False

        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "effect": self.effect.value,
            "subjects": self.subjects,
            "resources": self.resources,
            "actions": self.actions,
            "conditions": [c.to_dict() for c in self.conditions],
            "priority": self.priority,
            "enabled": self.enabled,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


# =============================================================================
# Policy Evaluators
# =============================================================================


class RoleBasedEvaluator(PolicyEvaluator):
    """Role-based access control evaluator.

    Checks permissions based on roles assigned to the principal.

    Example:
        >>> evaluator = RoleBasedEvaluator(role_store)
        >>> decision = evaluator.evaluate(context)
    """

    def __init__(
        self,
        role_store: RoleStore,
        deny_by_default: bool = True,
    ) -> None:
        self._role_store = role_store
        self._deny_by_default = deny_by_default

    @property
    def name(self) -> str:
        return "role_based"

    @property
    def priority(self) -> int:
        return 100

    def evaluate(self, context: AccessContext) -> AccessDecision:
        """Evaluate access based on roles."""
        start_time = time.time()

        if context.principal is None:
            return AccessDecision.deny("No principal in context")

        principal = context.principal
        required_permission = context.get_required_permission()

        # Check direct permissions first
        for perm in principal.direct_permissions:
            if perm.matches(required_permission):
                if perm.effect == PermissionEffect.DENY:
                    return AccessDecision.deny(
                        f"Permission explicitly denied: {perm}"
                    )
                return AccessDecision.allow(
                    f"Direct permission granted: {perm}",
                    [perm],
                )

        # Check role permissions
        matching_permissions: list[Permission] = []
        for role_id in principal.roles:
            permissions = self._role_store.get_all_permissions(role_id)
            for perm in permissions:
                if perm.matches(required_permission):
                    if perm.effect == PermissionEffect.DENY:
                        return AccessDecision.deny(
                            f"Permission denied by role {role_id}: {perm}"
                        )
                    matching_permissions.append(perm)

        if matching_permissions:
            elapsed = (time.time() - start_time) * 1000
            decision = AccessDecision.allow(
                f"Permission granted by roles",
                matching_permissions,
            )
            decision.evaluation_time_ms = elapsed
            return decision

        if self._deny_by_default:
            return AccessDecision.deny(
                f"No matching permission for {required_permission}"
            )

        # Return neutral (let other evaluators decide)
        return AccessDecision(
            allowed=False,
            reason="No matching role permission",
        )


class PolicyBasedEvaluator(PolicyEvaluator):
    """Policy-based access control evaluator.

    Evaluates access based on explicit policies.

    Example:
        >>> evaluator = PolicyBasedEvaluator()
        >>> evaluator.add_policy(Policy(...))
        >>> decision = evaluator.evaluate(context)
    """

    def __init__(
        self,
        combination: PolicyCombination = PolicyCombination.DENY_OVERRIDES,
    ) -> None:
        self._policies: list[Policy] = []
        self._combination = combination
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return "policy_based"

    @property
    def priority(self) -> int:
        return 90

    def add_policy(self, policy: Policy) -> None:
        """Add a policy."""
        with self._lock:
            self._policies.append(policy)
            # Sort by priority (descending)
            self._policies.sort(key=lambda p: p.priority, reverse=True)

    def remove_policy(self, policy_id: str) -> bool:
        """Remove a policy by ID."""
        with self._lock:
            for i, policy in enumerate(self._policies):
                if policy.id == policy_id:
                    del self._policies[i]
                    return True
            return False

    def get_policy(self, policy_id: str) -> Policy | None:
        """Get a policy by ID."""
        with self._lock:
            for policy in self._policies:
                if policy.id == policy_id:
                    return policy
            return None

    def evaluate(self, context: AccessContext) -> AccessDecision:
        """Evaluate access based on policies."""
        start_time = time.time()

        if context.principal is None:
            return AccessDecision.deny("No principal in context")

        resource = context.resource
        action = context.action if isinstance(context.action, str) else context.action.value

        matching_policies: list[tuple[Policy, PermissionEffect]] = []

        with self._lock:
            for policy in self._policies:
                if not policy.enabled:
                    continue

                # Check tenant scope
                if policy.tenant_id and policy.tenant_id != context.tenant_id:
                    continue

                # Check if policy matches
                if not policy.matches_subject(context.principal):
                    continue
                if not policy.matches_resource(resource):
                    continue
                if not policy.matches_action(action):
                    continue
                if not policy.evaluate_conditions(context):
                    continue

                matching_policies.append((policy, policy.effect))

        if not matching_policies:
            return AccessDecision(
                allowed=False,
                reason="No matching policy",
            )

        # Apply combination algorithm
        decision = self._combine_decisions(matching_policies)
        decision.evaluation_time_ms = (time.time() - start_time) * 1000
        return decision

    def _combine_decisions(
        self,
        matches: list[tuple[Policy, PermissionEffect]],
    ) -> AccessDecision:
        """Combine policy decisions based on combination algorithm."""
        if self._combination == PolicyCombination.DENY_OVERRIDES:
            # Any deny wins
            for policy, effect in matches:
                if effect == PermissionEffect.DENY:
                    return AccessDecision.deny(
                        f"Denied by policy: {policy.name}"
                    )
            # All allows
            policy, _ = matches[0]
            return AccessDecision.allow(
                f"Allowed by policy: {policy.name}",
            )

        elif self._combination == PolicyCombination.ALLOW_OVERRIDES:
            # Any allow wins
            for policy, effect in matches:
                if effect == PermissionEffect.ALLOW:
                    return AccessDecision.allow(
                        f"Allowed by policy: {policy.name}",
                    )
            # All denies
            policy, _ = matches[0]
            return AccessDecision.deny(
                f"Denied by policy: {policy.name}"
            )

        elif self._combination == PolicyCombination.FIRST_APPLICABLE:
            # First match wins
            policy, effect = matches[0]
            if effect == PermissionEffect.ALLOW:
                return AccessDecision.allow(
                    f"Allowed by policy: {policy.name}",
                )
            return AccessDecision.deny(
                f"Denied by policy: {policy.name}"
            )

        elif self._combination == PolicyCombination.UNANIMOUS:
            # All must allow
            for policy, effect in matches:
                if effect == PermissionEffect.DENY:
                    return AccessDecision.deny(
                        f"Denied by policy: {policy.name} (unanimous required)"
                    )
            policy, _ = matches[0]
            return AccessDecision.allow(
                f"Unanimously allowed",
            )

        return AccessDecision.deny("Unknown combination algorithm")


class ABACEvaluator(PolicyEvaluator):
    """Attribute-Based Access Control evaluator.

    Evaluates access based on attributes of the principal, resource,
    and environment.

    Example:
        >>> evaluator = ABACEvaluator()
        >>> evaluator.add_rule(
        ...     condition=Condition("resource.owner_id", ConditionOperator.EQUALS, "${principal.id}"),
        ...     effect=PermissionEffect.ALLOW,
        ... )
    """

    def __init__(self) -> None:
        self._rules: list[tuple[list[Condition], PermissionEffect, str]] = []
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return "abac"

    @property
    def priority(self) -> int:
        return 80

    def add_rule(
        self,
        conditions: list[Condition],
        effect: PermissionEffect = PermissionEffect.ALLOW,
        description: str = "",
    ) -> None:
        """Add an ABAC rule."""
        with self._lock:
            self._rules.append((conditions, effect, description))

    def evaluate(self, context: AccessContext) -> AccessDecision:
        """Evaluate access based on attributes."""
        start_time = time.time()

        if context.principal is None:
            return AccessDecision.deny("No principal in context")

        with self._lock:
            for conditions, effect, description in self._rules:
                all_match = True
                for condition in conditions:
                    if not condition.evaluate(context):
                        all_match = False
                        break

                if all_match:
                    elapsed = (time.time() - start_time) * 1000
                    if effect == PermissionEffect.ALLOW:
                        decision = AccessDecision.allow(
                            f"ABAC rule matched: {description}"
                        )
                    else:
                        decision = AccessDecision.deny(
                            f"ABAC rule denied: {description}"
                        )
                    decision.evaluation_time_ms = elapsed
                    return decision

        return AccessDecision(
            allowed=False,
            reason="No matching ABAC rule",
        )


class OwnershipEvaluator(PolicyEvaluator):
    """Ownership-based access control evaluator.

    Grants access if the principal owns the resource.

    Example:
        >>> evaluator = OwnershipEvaluator(owner_field="owner_id")
        >>> # Principal can access resources they own
    """

    def __init__(
        self,
        owner_field: str = "owner_id",
        actions: list[str] | None = None,  # Actions allowed for owners
    ) -> None:
        self._owner_field = owner_field
        self._actions = actions or ["read", "update", "delete"]

    @property
    def name(self) -> str:
        return "ownership"

    @property
    def priority(self) -> int:
        return 70

    def evaluate(self, context: AccessContext) -> AccessDecision:
        """Evaluate access based on ownership."""
        if context.principal is None:
            return AccessDecision.deny("No principal in context")

        action = context.action if isinstance(context.action, str) else context.action.value

        # Check if action is allowed for owners
        if action not in self._actions and "*" not in self._actions:
            return AccessDecision(
                allowed=False,
                reason="Action not covered by ownership evaluator",
            )

        # Get owner from resource attributes
        owner_id = context.resource_attributes.get(self._owner_field)
        if owner_id is None:
            return AccessDecision(
                allowed=False,
                reason="No owner information in resource",
            )

        # Check ownership
        if owner_id == context.principal.id:
            return AccessDecision.allow(
                f"Principal owns the resource"
            )

        return AccessDecision(
            allowed=False,
            reason="Principal does not own the resource",
        )


class TenantIsolationEvaluator(PolicyEvaluator):
    """Tenant isolation evaluator.

    Ensures principals can only access resources within their tenant.

    Example:
        >>> evaluator = TenantIsolationEvaluator()
        >>> # Denies cross-tenant access
    """

    def __init__(
        self,
        tenant_field: str = "tenant_id",
        enforce_strict: bool = True,
    ) -> None:
        self._tenant_field = tenant_field
        self._enforce_strict = enforce_strict

    @property
    def name(self) -> str:
        return "tenant_isolation"

    @property
    def priority(self) -> int:
        return 200  # High priority - check early

    def evaluate(self, context: AccessContext) -> AccessDecision:
        """Evaluate tenant isolation."""
        if context.principal is None:
            return AccessDecision.deny("No principal in context")

        principal_tenant = context.principal.tenant_id
        resource_tenant = context.resource_attributes.get(self._tenant_field)

        # If no tenant info, depends on strict mode
        if resource_tenant is None:
            if self._enforce_strict:
                return AccessDecision(
                    allowed=False,
                    reason="No tenant information in resource",
                )
            return AccessDecision(
                allowed=True,
                reason="Tenant check skipped - no tenant info",
            )

        # Check tenant match
        if principal_tenant != resource_tenant:
            return AccessDecision.deny(
                f"Cross-tenant access denied: {principal_tenant} != {resource_tenant}"
            )

        return AccessDecision.allow("Same tenant access")


class SuperuserEvaluator(PolicyEvaluator):
    """Superuser evaluator.

    Grants all permissions to superuser/admin principals.

    Example:
        >>> evaluator = SuperuserEvaluator(superuser_roles={"superadmin"})
    """

    def __init__(
        self,
        superuser_roles: set[str] | None = None,
        superuser_ids: set[str] | None = None,
    ) -> None:
        self._superuser_roles = superuser_roles or {"superadmin", "system_admin"}
        self._superuser_ids = superuser_ids or {"system"}

    @property
    def name(self) -> str:
        return "superuser"

    @property
    def priority(self) -> int:
        return 1000  # Highest priority

    def evaluate(self, context: AccessContext) -> AccessDecision:
        """Check if principal is a superuser."""
        if context.principal is None:
            return AccessDecision(allowed=False, reason="No principal")

        # Check superuser IDs
        if context.principal.id in self._superuser_ids:
            return AccessDecision.allow("Superuser access (by ID)")

        # Check superuser roles
        if context.principal.roles & self._superuser_roles:
            return AccessDecision.allow("Superuser access (by role)")

        return AccessDecision(
            allowed=False,
            reason="Not a superuser",
        )


# =============================================================================
# Policy Engine
# =============================================================================


@dataclass
class PolicyEngineConfig:
    """Configuration for the policy engine."""

    combination: PolicyCombination = PolicyCombination.DENY_OVERRIDES
    deny_by_default: bool = True
    log_decisions: bool = True
    cache_decisions: bool = True
    cache_ttl_seconds: int = 60


class PolicyEngine:
    """Central policy engine for access control.

    Coordinates multiple policy evaluators to make access decisions.

    Example:
        >>> engine = PolicyEngine()
        >>> engine.add_evaluator(RoleBasedEvaluator(role_store))
        >>> engine.add_evaluator(OwnershipEvaluator())
        >>>
        >>> decision = engine.evaluate(context)
        >>> if decision.allowed:
        ...     process_request()
    """

    def __init__(
        self,
        config: PolicyEngineConfig | None = None,
    ) -> None:
        self._config = config or PolicyEngineConfig()
        self._evaluators: list[PolicyEvaluator] = []
        self._decision_cache: dict[str, tuple[AccessDecision, float]] = {}
        self._lock = threading.RLock()

    def add_evaluator(self, evaluator: PolicyEvaluator) -> None:
        """Add a policy evaluator."""
        with self._lock:
            self._evaluators.append(evaluator)
            # Sort by priority (descending)
            self._evaluators.sort(key=lambda e: e.priority, reverse=True)

    def remove_evaluator(self, name: str) -> bool:
        """Remove an evaluator by name."""
        with self._lock:
            for i, evaluator in enumerate(self._evaluators):
                if evaluator.name == name:
                    del self._evaluators[i]
                    return True
            return False

    def evaluate(self, context: AccessContext) -> AccessDecision:
        """Evaluate access for the given context.

        Args:
            context: Access context containing principal, resource, action

        Returns:
            AccessDecision indicating whether access is allowed.
        """
        start_time = time.time()

        # Check cache
        cache_key = self._get_cache_key(context)
        if self._config.cache_decisions and cache_key:
            cached = self._get_cached_decision(cache_key)
            if cached:
                return cached

        # Collect decisions from all evaluators
        decisions: list[tuple[PolicyEvaluator, AccessDecision]] = []

        with self._lock:
            for evaluator in self._evaluators:
                try:
                    decision = evaluator.evaluate(context)
                    decisions.append((evaluator, decision))
                except Exception as e:
                    # Log error but continue with other evaluators
                    if self._config.log_decisions:
                        pass  # Would log here

        if not decisions:
            return AccessDecision.deny("No evaluators configured")

        # Combine decisions
        final_decision = self._combine_decisions(decisions)
        final_decision.evaluation_time_ms = (time.time() - start_time) * 1000

        # Cache decision
        if self._config.cache_decisions and cache_key:
            self._cache_decision(cache_key, final_decision)

        return final_decision

    def _combine_decisions(
        self,
        decisions: list[tuple[PolicyEvaluator, AccessDecision]],
    ) -> AccessDecision:
        """Combine decisions from multiple evaluators."""
        if self._config.combination == PolicyCombination.DENY_OVERRIDES:
            # Any deny wins
            for evaluator, decision in decisions:
                if decision.allowed is False and decision.effect == PermissionEffect.DENY:
                    return decision

            # Check for any allow
            for evaluator, decision in decisions:
                if decision.allowed:
                    return decision

        elif self._config.combination == PolicyCombination.ALLOW_OVERRIDES:
            # Any allow wins
            for evaluator, decision in decisions:
                if decision.allowed:
                    return decision

        elif self._config.combination == PolicyCombination.FIRST_APPLICABLE:
            # First definitive decision wins
            for evaluator, decision in decisions:
                if decision.effect in (PermissionEffect.ALLOW, PermissionEffect.DENY):
                    return decision

        elif self._config.combination == PolicyCombination.UNANIMOUS:
            # All must allow
            for evaluator, decision in decisions:
                if not decision.allowed:
                    return decision

            # All allowed
            _, decision = decisions[0]
            return decision

        # Default: deny
        if self._config.deny_by_default:
            return AccessDecision.deny("No matching policy (deny by default)")

        return AccessDecision(
            allowed=False,
            reason="No definitive decision",
        )

    def _get_cache_key(self, context: AccessContext) -> str | None:
        """Generate a cache key for the context."""
        if context.principal is None:
            return None

        parts = [
            context.principal.id,
            context.resource,
            context.action if isinstance(context.action, str) else context.action.value,
            context.tenant_id or "",
        ]
        return ":".join(parts)

    def _get_cached_decision(self, cache_key: str) -> AccessDecision | None:
        """Get a cached decision."""
        with self._lock:
            if cache_key in self._decision_cache:
                decision, cached_at = self._decision_cache[cache_key]
                if time.time() - cached_at < self._config.cache_ttl_seconds:
                    return decision
                else:
                    del self._decision_cache[cache_key]
            return None

    def _cache_decision(self, cache_key: str, decision: AccessDecision) -> None:
        """Cache a decision."""
        with self._lock:
            # Limit cache size
            if len(self._decision_cache) > 10000:
                # Remove oldest entries
                sorted_keys = sorted(
                    self._decision_cache.keys(),
                    key=lambda k: self._decision_cache[k][1],
                )
                for key in sorted_keys[:1000]:
                    del self._decision_cache[key]

            self._decision_cache[cache_key] = (decision, time.time())

    def invalidate_cache(
        self,
        principal_id: str | None = None,
        resource: str | None = None,
    ) -> None:
        """Invalidate cached decisions."""
        with self._lock:
            if principal_id is None and resource is None:
                self._decision_cache.clear()
                return

            keys_to_remove = []
            for key in self._decision_cache:
                parts = key.split(":")
                if principal_id and parts[0] == principal_id:
                    keys_to_remove.append(key)
                elif resource and parts[1] == resource:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._decision_cache[key]

    def check(
        self,
        principal: Principal,
        resource: str,
        action: str,
        resource_attributes: dict[str, Any] | None = None,
    ) -> AccessDecision:
        """Convenience method for checking access.

        Args:
            principal: Principal requesting access
            resource: Resource being accessed
            action: Action being performed
            resource_attributes: Optional resource attributes for ABAC

        Returns:
            AccessDecision.
        """
        context = AccessContext(
            principal=principal,
            resource=resource,
            action=action,
            resource_attributes=resource_attributes or {},
            tenant_id=principal.tenant_id,
        )
        return self.evaluate(context)

    def require(
        self,
        principal: Principal,
        resource: str,
        action: str,
        resource_attributes: dict[str, Any] | None = None,
    ) -> None:
        """Check access and raise if denied.

        Args:
            principal: Principal requesting access
            resource: Resource being accessed
            action: Action being performed
            resource_attributes: Optional resource attributes

        Raises:
            PermissionDeniedError: If access is denied.
        """
        decision = self.check(principal, resource, action, resource_attributes)
        if not decision.allowed:
            raise PermissionDeniedError(
                decision.reason,
                principal_id=principal.id,
                resource=resource,
                action=action,
            )
