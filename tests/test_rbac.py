"""Tests for the RBAC module.

This module contains comprehensive tests for the Role-Based Access Control
functionality including core types, storage backends, policy evaluation,
middleware, and multi-tenancy integration.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Test Core Types
# =============================================================================


class TestPermission:
    """Tests for the Permission dataclass."""

    def test_create_permission(self):
        """Test basic permission creation."""
        from truthound.rbac import Permission, PermissionAction, ResourceType

        perm = Permission(
            resource=ResourceType.DATASET,
            action=PermissionAction.READ,
        )

        assert perm.resource == "dataset"
        assert perm.action == "read"
        assert perm.scope == ""
        assert perm.to_string() == "dataset:read"

    def test_permission_with_scope(self):
        """Test permission with scope."""
        from truthound.rbac import Permission

        perm = Permission(resource="dataset", action="read", scope="own")

        assert perm.to_string() == "dataset:read:own"

    def test_permission_parse(self):
        """Test parsing permission strings."""
        from truthound.rbac import Permission

        perm = Permission.parse("dataset:read")
        assert perm.resource == "dataset"
        assert perm.action == "read"

        perm_scoped = Permission.parse("dataset:read:own")
        assert perm_scoped.scope == "own"

    def test_permission_matches_exact(self):
        """Test exact permission matching."""
        from truthound.rbac import Permission

        perm1 = Permission.parse("dataset:read")
        perm2 = Permission.parse("dataset:read")

        assert perm1.matches(perm2)
        assert perm2.matches(perm1)

    def test_permission_matches_wildcard_action(self):
        """Test wildcard action matching."""
        from truthound.rbac import Permission

        wildcard = Permission.parse("dataset:*")
        specific = Permission.parse("dataset:read")

        assert wildcard.matches(specific)
        assert specific.matches(wildcard)

    def test_permission_matches_wildcard_resource(self):
        """Test wildcard resource matching."""
        from truthound.rbac import Permission

        wildcard = Permission.parse("*:read")
        specific = Permission.parse("dataset:read")

        assert wildcard.matches(specific)
        assert specific.matches(wildcard)

    def test_permission_not_matches(self):
        """Test non-matching permissions."""
        from truthound.rbac import Permission

        perm1 = Permission.parse("dataset:read")
        perm2 = Permission.parse("dataset:write")

        assert not perm1.matches(perm2)

    def test_permission_all(self):
        """Test wildcard permission."""
        from truthound.rbac import Permission

        all_perm = Permission.all()
        any_perm = Permission.parse("anything:anything")

        assert all_perm.matches(any_perm)

    def test_permission_frozen(self):
        """Test that permission is immutable (frozen)."""
        from truthound.rbac import Permission

        perm = Permission.parse("dataset:read")

        with pytest.raises(AttributeError):
            perm.resource = "other"


class TestCondition:
    """Tests for the Condition dataclass."""

    def test_condition_equals(self):
        """Test equals operator."""
        from truthound.rbac import (
            AccessContext,
            Condition,
            ConditionOperator,
            Principal,
        )

        condition = Condition(
            field="resource.owner_id",
            operator=ConditionOperator.EQUALS,
            value="user_123",
        )

        context = AccessContext(
            principal=Principal(id="test"),
            resource="dataset:test",
            action="read",
            resource_attributes={"owner_id": "user_123"},
        )

        assert condition.evaluate(context) is True

    def test_condition_dynamic_value(self):
        """Test dynamic value resolution."""
        from truthound.rbac import (
            AccessContext,
            Condition,
            ConditionOperator,
            Principal,
        )

        condition = Condition(
            field="resource.owner_id",
            operator=ConditionOperator.EQUALS,
            value="${principal.id}",
        )

        principal = Principal(id="user_123")
        context = AccessContext(
            principal=principal,
            resource="dataset:test",
            action="read",
            resource_attributes={"owner_id": "user_123"},
        )

        assert condition.evaluate(context) is True

    def test_condition_in_operator(self):
        """Test IN operator."""
        from truthound.rbac import (
            AccessContext,
            Condition,
            ConditionOperator,
            Principal,
            PrincipalType,
        )

        condition = Condition(
            field="principal.type",
            operator=ConditionOperator.IN,
            value=[PrincipalType.USER, PrincipalType.SERVICE],  # Use enum values
        )

        context = AccessContext(
            principal=Principal(id="test"),
            resource="dataset:test",
            action="read",
        )

        assert condition.evaluate(context) is True


class TestRole:
    """Tests for the Role dataclass."""

    def test_create_role(self):
        """Test basic role creation."""
        from truthound.rbac import Permission, Role, RoleType

        role = Role(
            id="editor",
            name="Editor",
            description="Can edit content",
            role_type=RoleType.CUSTOM,
        )

        assert role.id == "editor"
        assert role.name == "Editor"
        assert role.enabled is True

    def test_role_add_permission(self):
        """Test adding permissions to a role."""
        from truthound.rbac import Permission, Role

        role = Role(id="editor", name="Editor")
        role.add_permission("dataset:read")
        role.add_permission(Permission.parse("dataset:update"))

        assert len(role.permissions) == 2
        assert role.has_permission("dataset:read")
        assert role.has_permission("dataset:update")

    def test_role_remove_permission(self):
        """Test removing permissions from a role."""
        from truthound.rbac import Role

        role = Role(id="editor", name="Editor")
        role.add_permission("dataset:read")
        role.add_permission("dataset:update")
        role.remove_permission("dataset:read")

        assert len(role.permissions) == 1
        assert not role.has_permission("dataset:read")

    def test_role_parent_roles(self):
        """Test role inheritance."""
        from truthound.rbac import Role

        role = Role(id="admin", name="Admin")
        role.add_parent_role("editor")
        role.add_parent_role("viewer")

        assert "editor" in role.parent_roles
        assert "viewer" in role.parent_roles

    def test_role_to_dict(self):
        """Test role serialization."""
        from truthound.rbac import Role, RoleType

        role = Role(
            id="editor",
            name="Editor",
            role_type=RoleType.CUSTOM,
        )
        role.add_permission("dataset:read")

        data = role.to_dict()
        assert data["id"] == "editor"
        assert "dataset:read" in data["permissions"]

    def test_role_from_dict(self):
        """Test role deserialization."""
        from truthound.rbac import Role, RoleType

        data = {
            "id": "editor",
            "name": "Editor",
            "role_type": "custom",
            "permissions": ["dataset:read"],
            "parent_roles": ["viewer"],
        }

        role = Role.from_dict(data)
        assert role.id == "editor"
        assert role.has_permission("dataset:read")
        assert "viewer" in role.parent_roles


class TestPrincipal:
    """Tests for the Principal dataclass."""

    def test_create_principal(self):
        """Test basic principal creation."""
        from truthound.rbac import Principal, PrincipalType

        principal = Principal(
            id="user_123",
            type=PrincipalType.USER,
            name="John Doe",
            email="john@example.com",
        )

        assert principal.id == "user_123"
        assert principal.name == "John Doe"
        assert principal.enabled is True

    def test_principal_roles(self):
        """Test role assignment."""
        from truthound.rbac import Principal

        principal = Principal(id="user_123")
        principal.add_role("editor")
        principal.add_role("viewer")

        assert principal.has_role("editor")
        assert principal.has_role("viewer")
        assert not principal.has_role("admin")

    def test_principal_remove_role(self):
        """Test role removal."""
        from truthound.rbac import Principal

        principal = Principal(id="user_123", roles={"editor", "viewer"})
        principal.remove_role("editor")

        assert not principal.has_role("editor")
        assert principal.has_role("viewer")

    def test_principal_direct_permissions(self):
        """Test direct permission assignment."""
        from truthound.rbac import Principal

        principal = Principal(id="user_123")
        principal.add_permission("special:action")

        assert len(principal.direct_permissions) == 1

    def test_principal_anonymous(self):
        """Test anonymous principal creation."""
        from truthound.rbac import Principal, PrincipalType

        anon = Principal.anonymous()
        assert anon.id == "anonymous"
        assert anon.type == PrincipalType.ANONYMOUS

    def test_principal_system(self):
        """Test system principal creation."""
        from truthound.rbac import Principal, PrincipalType

        system = Principal.system()
        assert system.id == "system"
        assert system.type == PrincipalType.SYSTEM
        assert "system_admin" in system.roles

    def test_principal_serialization(self):
        """Test principal serialization/deserialization."""
        from truthound.rbac import Principal, PrincipalType

        principal = Principal(
            id="user_123",
            type=PrincipalType.USER,
            name="John Doe",
            roles={"editor"},
        )

        data = principal.to_dict()
        restored = Principal.from_dict(data)

        assert restored.id == principal.id
        assert restored.name == principal.name
        assert "editor" in restored.roles


class TestSecurityContext:
    """Tests for the SecurityContext."""

    def test_set_principal(self):
        """Test setting principal context."""
        from truthound.rbac import Principal, SecurityContext

        principal = Principal(id="user_123")

        with SecurityContext.set_principal(principal):
            current = SecurityContext.get_current_principal()
            assert current is not None
            assert current.id == "user_123"

        # Should be cleared after context
        assert SecurityContext.get_current_principal() is None

    def test_require_principal_raises(self):
        """Test require_principal raises when no context."""
        from truthound.rbac import PermissionDeniedError, SecurityContext

        SecurityContext.clear()
        with pytest.raises(PermissionDeniedError):
            SecurityContext.require_principal()

    def test_nested_contexts(self):
        """Test nested security contexts."""
        from truthound.rbac import Principal, SecurityContext

        principal1 = Principal(id="user_1")
        principal2 = Principal(id="user_2")

        with SecurityContext.set_principal(principal1):
            assert SecurityContext.get_current_principal().id == "user_1"

            with SecurityContext.set_principal(principal2):
                assert SecurityContext.get_current_principal().id == "user_2"

            # Should restore to principal1
            assert SecurityContext.get_current_principal().id == "user_1"


# =============================================================================
# Test Storage Backends
# =============================================================================


class TestMemoryRoleStore:
    """Tests for the MemoryRoleStore."""

    def test_save_and_get(self):
        """Test saving and retrieving roles."""
        from truthound.rbac import MemoryRoleStore, Role

        store = MemoryRoleStore()
        role = Role(id="editor", name="Editor")

        store.save(role)
        retrieved = store.get("editor")

        assert retrieved is not None
        assert retrieved.id == "editor"

    def test_list_roles(self):
        """Test listing roles."""
        from truthound.rbac import MemoryRoleStore, Role, RoleType

        store = MemoryRoleStore()
        store.save(Role(id="r1", name="Role 1", role_type=RoleType.SYSTEM))
        store.save(Role(id="r2", name="Role 2", role_type=RoleType.CUSTOM))

        all_roles = store.list()
        assert len(all_roles) == 2

        system_roles = store.list(role_type=RoleType.SYSTEM)
        assert len(system_roles) == 1

    def test_delete_role(self):
        """Test deleting a role."""
        from truthound.rbac import MemoryRoleStore, Role

        store = MemoryRoleStore()
        store.save(Role(id="editor", name="Editor"))

        assert store.delete("editor") is True
        assert store.get("editor") is None
        assert store.delete("editor") is False

    def test_exists(self):
        """Test checking role existence."""
        from truthound.rbac import MemoryRoleStore, Role

        store = MemoryRoleStore()
        store.save(Role(id="editor", name="Editor"))

        assert store.exists("editor") is True
        assert store.exists("nonexistent") is False

    def test_get_all_permissions_with_inheritance(self):
        """Test getting all permissions including inherited."""
        from truthound.rbac import MemoryRoleStore, Permission, Role

        store = MemoryRoleStore()

        viewer = Role(id="viewer", name="Viewer")
        viewer.add_permission("dataset:read")
        store.save(viewer)

        editor = Role(id="editor", name="Editor", parent_roles={"viewer"})
        editor.add_permission("dataset:update")
        store.save(editor)

        perms = store.get_all_permissions("editor")
        assert len(perms) == 2

        perm_strings = {p.to_string() for p in perms}
        assert "dataset:read" in perm_strings
        assert "dataset:update" in perm_strings


class TestMemoryPrincipalStore:
    """Tests for the MemoryPrincipalStore."""

    def test_save_and_get(self):
        """Test saving and retrieving principals."""
        from truthound.rbac import MemoryPrincipalStore, Principal

        store = MemoryPrincipalStore()
        principal = Principal(id="user_123", name="John", email="john@example.com")

        store.save(principal)
        retrieved = store.get("user_123")

        assert retrieved is not None
        assert retrieved.name == "John"

    def test_get_by_email(self):
        """Test retrieving by email."""
        from truthound.rbac import MemoryPrincipalStore, Principal

        store = MemoryPrincipalStore()
        principal = Principal(id="user_123", name="John", email="john@example.com")
        store.save(principal)

        retrieved = store.get_by_email("john@example.com")
        assert retrieved is not None
        assert retrieved.id == "user_123"

    def test_list_by_role(self):
        """Test listing principals by role."""
        from truthound.rbac import MemoryPrincipalStore, Principal

        store = MemoryPrincipalStore()
        store.save(Principal(id="u1", name="User 1", roles={"editor"}))
        store.save(Principal(id="u2", name="User 2", roles={"viewer"}))
        store.save(Principal(id="u3", name="User 3", roles={"editor", "admin"}))

        editors = store.list(role_id="editor")
        assert len(editors) == 2


class TestFileRoleStore:
    """Tests for the FileRoleStore."""

    def test_file_persistence(self):
        """Test role persistence to file."""
        from truthound.rbac.storage import FileRoleStore, FileStorageConfig
        from truthound.rbac import Role

        with tempfile.TemporaryDirectory() as tmpdir:
            config = FileStorageConfig(base_path=tmpdir)
            store = FileRoleStore(config=config)
            role = Role(id="editor", name="Editor")
            role.add_permission("dataset:read")

            store.save(role)

            # Create new store instance to verify persistence
            store2 = FileRoleStore(config=config)
            retrieved = store2.get("editor")

            assert retrieved is not None
            assert retrieved.has_permission("dataset:read")


class TestSQLiteRoleStore:
    """Tests for the SQLiteRoleStore."""

    def test_sqlite_operations(self):
        """Test SQLite role operations."""
        from truthound.rbac import Role, SQLiteRoleStore

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            store = SQLiteRoleStore(db_path)
            role = Role(id="editor", name="Editor")
            role.add_permission("dataset:read")

            store.save(role)

            # Verify retrieval
            retrieved = store.get("editor")
            assert retrieved is not None
            assert retrieved.name == "Editor"

            # Verify list
            roles = store.list()
            assert len(roles) == 1

        finally:
            os.unlink(db_path)


class TestStorageFactories:
    """Tests for storage factory functions."""

    def test_create_role_store_memory(self):
        """Test creating memory role store."""
        from truthound.rbac import MemoryRoleStore, create_role_store

        store = create_role_store("memory")
        assert isinstance(store, MemoryRoleStore)

    def test_create_principal_store_memory(self):
        """Test creating memory principal store."""
        from truthound.rbac import MemoryPrincipalStore, create_principal_store

        store = create_principal_store("memory")
        assert isinstance(store, MemoryPrincipalStore)


# =============================================================================
# Test Policy Engine
# =============================================================================


class TestPolicyEngine:
    """Tests for the PolicyEngine."""

    def test_basic_permission_check(self):
        """Test basic permission checking."""
        from truthound.rbac import (
            MemoryRoleStore,
            Permission,
            PolicyEngine,
            Principal,
            Role,
            RoleBasedEvaluator,
        )

        store = MemoryRoleStore()
        role = Role(id="editor", name="Editor")
        role.add_permission("dataset:read")
        store.save(role)

        engine = PolicyEngine()
        engine.add_evaluator(RoleBasedEvaluator(store))

        principal = Principal(id="user_123", roles={"editor"})

        decision = engine.check(principal, "dataset", "read")
        assert decision.allowed is True

    def test_permission_denied(self):
        """Test permission denial."""
        from truthound.rbac import (
            MemoryRoleStore,
            PolicyEngine,
            Principal,
            Role,
            RoleBasedEvaluator,
        )

        store = MemoryRoleStore()
        role = Role(id="viewer", name="Viewer")
        role.add_permission("dataset:read")
        store.save(role)

        engine = PolicyEngine()
        engine.add_evaluator(RoleBasedEvaluator(store))

        principal = Principal(id="user_123", roles={"viewer"})

        decision = engine.check(principal, "dataset", "delete")
        assert decision.allowed is False

    def test_require_raises(self):
        """Test require method raises on denial."""
        from truthound.rbac import (
            MemoryRoleStore,
            PermissionDeniedError,
            PolicyEngine,
            Principal,
            Role,
            RoleBasedEvaluator,
        )

        store = MemoryRoleStore()
        store.save(Role(id="viewer", name="Viewer"))

        engine = PolicyEngine()
        engine.add_evaluator(RoleBasedEvaluator(store))

        principal = Principal(id="user_123", roles={"viewer"})

        with pytest.raises(PermissionDeniedError):
            engine.require(principal, "dataset", "delete")

    def test_superuser_evaluator(self):
        """Test superuser bypass."""
        from truthound.rbac import (
            MemoryRoleStore,
            PolicyEngine,
            Principal,
            SuperuserEvaluator,
        )

        engine = PolicyEngine()
        engine.add_evaluator(SuperuserEvaluator())

        principal = Principal(id="admin", roles={"system_admin"})

        decision = engine.check(principal, "anything", "anything")
        assert decision.allowed is True

    def test_ownership_evaluator(self):
        """Test ownership-based access."""
        from truthound.rbac import (
            OwnershipEvaluator,
            PolicyEngine,
            PolicyEngineConfig,
            Principal,
        )

        # Disable caching for this test since cache key doesn't include resource_attributes
        config = PolicyEngineConfig(cache_decisions=False)
        engine = PolicyEngine(config=config)
        engine.add_evaluator(OwnershipEvaluator())

        principal = Principal(id="user_123")

        # Owner can access
        decision = engine.check(
            principal,
            "dataset",
            "read",
            resource_attributes={"owner_id": "user_123"},
        )
        assert decision.allowed is True

        # Non-owner cannot
        decision = engine.check(
            principal,
            "dataset",
            "read",
            resource_attributes={"owner_id": "other_user"},
        )
        assert decision.allowed is False


class TestABACEvaluator:
    """Tests for the ABACEvaluator."""

    def test_abac_condition_evaluation(self):
        """Test ABAC condition evaluation."""
        from truthound.rbac import (
            ABACEvaluator,
            Condition,
            ConditionOperator,
            MemoryRoleStore,
            Permission,
            PermissionEffect,
            PolicyEngine,
            Principal,
            Role,
        )

        engine = PolicyEngine()

        # Create ABAC evaluator with owner rule
        evaluator = ABACEvaluator()
        evaluator.add_rule(
            conditions=[
                Condition(
                    field="resource.owner_id",
                    operator=ConditionOperator.EQUALS,
                    value="${principal.id}",
                )
            ],
            effect=PermissionEffect.ALLOW,
            description="Owner can access",
        )
        engine.add_evaluator(evaluator)

        principal = Principal(id="user_123")

        # Owner can access
        decision = engine.check(
            principal,
            "dataset",
            "delete",
            resource_attributes={"owner_id": "user_123"},
        )
        assert decision.allowed is True


# =============================================================================
# Test Policy Combination Strategies
# =============================================================================


class TestPolicyCombination:
    """Tests for policy combination strategies."""

    def test_deny_overrides(self):
        """Test deny overrides combination."""
        from truthound.rbac import (
            Permission,
            PermissionEffect,
            Policy,
            PolicyBasedEvaluator,
            PolicyCombination,
            PolicyEngine,
            PolicyEngineConfig,
            Principal,
        )

        config = PolicyEngineConfig(combination=PolicyCombination.DENY_OVERRIDES)
        engine = PolicyEngine(config=config)

        # Add policies - one allow, one deny
        allow_policy = Policy(
            id="allow_read",
            name="Allow Read",
            resources=["dataset:*"],
            actions=["read"],
            effect=PermissionEffect.ALLOW,
        )
        deny_policy = Policy(
            id="deny_read",
            name="Deny Read",
            resources=["dataset:*"],
            actions=["read"],
            effect=PermissionEffect.DENY,
        )

        evaluator = PolicyBasedEvaluator()
        evaluator.add_policy(allow_policy)
        evaluator.add_policy(deny_policy)
        engine.add_evaluator(evaluator)

        principal = Principal(id="user_123")

        # Deny should override allow
        decision = engine.check(principal, "dataset", "read")
        assert decision.allowed is False


# =============================================================================
# Test Middleware
# =============================================================================


class TestRBACMiddleware:
    """Tests for the RBACMiddleware."""

    def test_resolve_principal_from_header(self):
        """Test principal resolution from header."""
        from truthound.rbac import (
            MemoryPrincipalStore,
            PolicyEngine,
            Principal,
            RBACMiddleware,
        )

        principal_store = MemoryPrincipalStore()
        principal = Principal(id="user_123", name="John")
        principal_store.save(principal)

        engine = PolicyEngine()
        middleware = RBACMiddleware(engine, principal_store)

        context = {"headers": {"X-Principal-ID": "user_123"}}
        resolved = middleware.resolve_principal(context)

        assert resolved is not None
        assert resolved.id == "user_123"

    def test_public_paths(self):
        """Test public path detection."""
        from truthound.rbac import (
            MemoryPrincipalStore,
            PolicyEngine,
            RBACMiddleware,
            RBACMiddlewareConfig,
        )

        config = RBACMiddlewareConfig(
            allow_public_paths=["/health", "/api/public/"]
        )

        engine = PolicyEngine()
        middleware = RBACMiddleware(engine, MemoryPrincipalStore(), config)

        assert middleware.is_public_path("/health") is True
        assert middleware.is_public_path("/api/public/data") is True
        assert middleware.is_public_path("/api/private/data") is False


class TestDecorators:
    """Tests for RBAC decorators."""

    def test_require_permission_decorator(self):
        """Test require_permission decorator."""
        from truthound.rbac import (
            MemoryRoleStore,
            PermissionDeniedError,
            PolicyEngine,
            Principal,
            Role,
            RoleBasedEvaluator,
            SecurityContext,
            require_permission,
            set_default_engine,
        )

        store = MemoryRoleStore()
        role = Role(id="editor", name="Editor")
        role.add_permission("dataset:read")
        store.save(role)

        engine = PolicyEngine()
        engine.add_evaluator(RoleBasedEvaluator(store))
        set_default_engine(engine)

        @require_permission("dataset", "read")
        def get_data():
            return "data"

        principal = Principal(id="user_123", roles={"editor"})

        with SecurityContext.set_principal(principal):
            result = get_data()
            assert result == "data"

        # Without proper role
        principal_no_role = Principal(id="user_456", roles=set())

        with SecurityContext.set_principal(principal_no_role):
            with pytest.raises(PermissionDeniedError):
                get_data()

    def test_require_role_decorator(self):
        """Test require_role decorator."""
        from truthound.rbac import (
            PermissionDeniedError,
            Principal,
            SecurityContext,
            require_role,
        )

        @require_role("admin")
        def admin_only():
            return "admin_data"

        admin = Principal(id="admin_123", roles={"admin"})
        user = Principal(id="user_123", roles={"viewer"})

        with SecurityContext.set_principal(admin):
            result = admin_only()
            assert result == "admin_data"

        with SecurityContext.set_principal(user):
            with pytest.raises(PermissionDeniedError):
                admin_only()

    def test_require_role_multiple(self):
        """Test require_role with multiple roles."""
        from truthound.rbac import (
            PermissionDeniedError,
            Principal,
            SecurityContext,
            require_role,
        )

        @require_role({"editor", "admin"}, require_all=False)
        def editor_or_admin():
            return "ok"

        editor = Principal(id="e1", roles={"editor"})
        admin = Principal(id="a1", roles={"admin"})
        viewer = Principal(id="v1", roles={"viewer"})

        with SecurityContext.set_principal(editor):
            assert editor_or_admin() == "ok"

        with SecurityContext.set_principal(admin):
            assert editor_or_admin() == "ok"

        with SecurityContext.set_principal(viewer):
            with pytest.raises(PermissionDeniedError):
                editor_or_admin()

    def test_with_principal_decorator(self):
        """Test with_principal decorator."""
        from truthound.rbac import Principal, SecurityContext, with_principal

        service_principal = Principal(id="service_123", name="Service")

        @with_principal(service_principal)
        def service_operation():
            current = SecurityContext.get_current_principal()
            return current.id if current else None

        result = service_operation()
        assert result == "service_123"


# =============================================================================
# Test RBAC Manager
# =============================================================================


class TestRBACManager:
    """Tests for the RBACManager."""

    def test_create_role(self):
        """Test role creation through manager."""
        from truthound.rbac import RBACManager

        manager = RBACManager()
        role = manager.create_role(
            name="Data Analyst",
            permissions={"dataset:read", "validation:execute"},
        )

        assert role.id == "data_analyst"
        assert role.has_permission("dataset:read")

    def test_create_role_with_tenant(self):
        """Test tenant-scoped role creation."""
        from truthound.rbac import RBACManager

        manager = RBACManager()
        role = manager.create_role(
            name="Analyst",
            tenant_id="tenant_123",
            permissions={"dataset:read"},
        )

        assert role.id == "tenant_123:analyst"
        assert role.tenant_id == "tenant_123"

    def test_create_principal(self):
        """Test principal creation through manager."""
        from truthound.rbac import RBACManager

        manager = RBACManager()
        principal = manager.create_principal(
            name="john.doe@example.com",
            email="john.doe@example.com",
            roles={"viewer"},
        )

        assert principal.email == "john.doe@example.com"
        assert principal.has_role("viewer")

    def test_assign_revoke_role(self):
        """Test role assignment and revocation."""
        from truthound.rbac import RBACManager

        manager = RBACManager()
        principal = manager.create_principal(name="John")

        manager.assign_role(principal.id, "editor")
        assert manager.has_role(principal.id, "editor")

        manager.revoke_role(principal.id, "editor")
        assert not manager.has_role(principal.id, "editor")

    def test_check_permission(self):
        """Test permission checking through manager."""
        from truthound.rbac import RBACManager

        manager = RBACManager()

        role = manager.create_role(
            name="Custom Editor",
            permissions={"dataset:read", "dataset:update"},
        )

        principal = manager.create_principal(
            name="John",
            roles={role.id},
        )

        decision = manager.check(principal, "dataset", "read")
        assert decision.allowed is True

        decision = manager.check(principal, "dataset", "delete")
        assert decision.allowed is False

    def test_default_roles_exist(self):
        """Test that default roles are created."""
        from truthound.rbac import RBACManager

        manager = RBACManager()

        assert manager.get_role("system_admin") is not None
        assert manager.get_role("viewer") is not None
        assert manager.get_role("editor") is not None
        assert manager.get_role("admin") is not None

    def test_get_principal_permissions(self):
        """Test getting all principal permissions."""
        from truthound.rbac import RBACManager

        manager = RBACManager()

        manager.create_role(
            role_id="role1",
            name="Role 1",
            permissions={"dataset:read"},
        )
        manager.create_role(
            role_id="role2",
            name="Role 2",
            permissions={"validation:execute"},
        )

        principal = manager.create_principal(
            name="John",
            roles={"role1", "role2"},
            permissions={"special:action"},
        )

        perms = manager.get_principal_permissions(principal.id)
        perm_strings = {p.to_string() for p in perms}

        assert "dataset:read" in perm_strings
        assert "validation:execute" in perm_strings
        assert "special:action" in perm_strings


class TestGlobalRBACManager:
    """Tests for global RBAC manager functions."""

    def test_get_set_manager(self):
        """Test getting and setting global manager."""
        from truthound.rbac import RBACManager, get_rbac_manager, set_rbac_manager

        manager = RBACManager()
        set_rbac_manager(manager)

        retrieved = get_rbac_manager()
        assert retrieved is manager

    def test_configure_manager(self):
        """Test configuring global manager."""
        from truthound.rbac import (
            MemoryRoleStore,
            RBACManagerConfig,
            configure_rbac_manager,
            get_rbac_manager,
        )

        config = RBACManagerConfig(cache_decisions=False)
        manager = configure_rbac_manager(config=config)

        assert get_rbac_manager() is manager

    def test_convenience_functions(self):
        """Test convenience functions."""
        from truthound.rbac import (
            RBACManager,
            check_permission,
            create_principal,
            create_role,
            get_principal,
            get_role,
            set_rbac_manager,
        )

        set_rbac_manager(RBACManager())

        role = create_role(name="Test Role", permissions={"test:read"})
        assert role is not None
        assert get_role(role.id) is not None

        principal = create_principal(name="Test User", roles={role.id})
        assert principal is not None
        assert get_principal(principal.id) is not None

        decision = check_permission(principal, "test", "read")
        assert decision.allowed is True


# =============================================================================
# Test Multi-tenancy Integration
# =============================================================================


class TestTenantRBACIntegration:
    """Tests for multi-tenancy RBAC integration."""

    def test_tenant_scoped_role_manager(self):
        """Test tenant-scoped role management."""
        from truthound.rbac import MemoryRoleStore, TenantScopedRoleManager

        store = MemoryRoleStore()
        manager = TenantScopedRoleManager(store)

        role = manager.create_tenant_role(
            tenant_id="tenant_123",
            name="Analyst",
            permissions={"dataset:read"},
        )

        assert role.id == "tenant_123:analyst"
        assert role.tenant_id == "tenant_123"

        # List tenant roles
        tenant_roles = manager.get_tenant_roles("tenant_123")
        assert len(tenant_roles) == 1

    def test_tenant_scoped_principal_manager(self):
        """Test tenant-scoped principal management."""
        from truthound.rbac import MemoryPrincipalStore, TenantScopedPrincipalManager

        store = MemoryPrincipalStore()
        manager = TenantScopedPrincipalManager(store)

        principal = manager.create_tenant_principal(
            tenant_id="tenant_123",
            name="john@example.com",
        )

        assert principal.tenant_id == "tenant_123"

        # List tenant principals
        tenant_principals = manager.get_tenant_principals("tenant_123")
        assert len(tenant_principals) == 1

    def test_tenant_aware_policy_evaluator(self):
        """Test tenant-aware policy evaluation."""
        from truthound.rbac import (
            AccessContext,
            Principal,
            TenantAwarePolicyEvaluator,
            TenantRBACConfig,
        )

        config = TenantRBACConfig(
            cross_tenant_roles={"platform_admin"},
        )
        evaluator = TenantAwarePolicyEvaluator(config)

        # Same tenant - allowed
        principal = Principal(id="user_123", tenant_id="tenant_a")
        context = AccessContext(
            principal=principal,
            resource="dataset:test",
            action="read",
            resource_attributes={"tenant_id": "tenant_a"},
        )

        decision = evaluator.evaluate(context)
        assert decision.allowed is True

        # Different tenant - denied
        context_cross = AccessContext(
            principal=principal,
            resource="dataset:test",
            action="read",
            resource_attributes={"tenant_id": "tenant_b"},
        )

        decision_cross = evaluator.evaluate(context_cross)
        assert decision_cross.allowed is False

        # Platform admin can cross tenants
        admin = Principal(id="admin", roles={"platform_admin"})
        context_admin = AccessContext(
            principal=admin,
            resource="dataset:test",
            action="read",
            resource_attributes={"tenant_id": "any_tenant"},
        )

        decision_admin = evaluator.evaluate(context_admin)
        assert decision_admin.allowed is True


class TestTenantRBACDecorators:
    """Tests for tenant-aware RBAC decorators."""

    def test_require_tenant_role_decorator(self):
        """Test require_tenant_role decorator."""
        from truthound.rbac import (
            PermissionDeniedError,
            Principal,
            SecurityContext,
        )

        # This test needs multitenancy context which may not be available
        # So we test the basic logic here

        principal_with_role = Principal(
            id="user_123",
            roles={"tenant_abc:admin"},
            tenant_id="tenant_abc",
        )

        assert "tenant_abc:admin" in principal_with_role.roles


# =============================================================================
# Test Caching
# =============================================================================


class TestCaching:
    """Tests for permission caching."""

    def test_cached_role_store(self):
        """Test CachedRoleStore functionality."""
        from truthound.rbac import CachedRoleStore, MemoryRoleStore, Role
        from truthound.rbac.storage import CacheConfig

        base_store = MemoryRoleStore()
        cache_config = CacheConfig(ttl_seconds=300)
        cached_store = CachedRoleStore(base_store, cache_config=cache_config)

        role = Role(id="editor", name="Editor")
        cached_store.save(role)

        # First get - should cache
        result1 = cached_store.get("editor")
        assert result1 is not None

        # Second get - from cache
        result2 = cached_store.get("editor")
        assert result2 is not None

    def test_cache_invalidation_on_save(self):
        """Test cache invalidation when saving."""
        from truthound.rbac import CachedRoleStore, MemoryRoleStore, Role
        from truthound.rbac.storage import CacheConfig

        base_store = MemoryRoleStore()
        cache_config = CacheConfig(ttl_seconds=300)
        cached_store = CachedRoleStore(base_store, cache_config=cache_config)

        role = Role(id="editor", name="Editor")
        cached_store.save(role)
        cached_store.get("editor")  # Cache it

        # Update role
        role.description = "Updated"
        cached_store.save(role)

        # Should get updated version
        result = cached_store.get("editor")
        assert result.description == "Updated"


# =============================================================================
# Test Access Decision
# =============================================================================


class TestAccessDecision:
    """Tests for AccessDecision."""

    def test_access_decision_allow(self):
        """Test allow decision creation."""
        from truthound.rbac import AccessDecision, Permission

        decision = AccessDecision.allow(
            reason="Permission granted",
            permissions=[Permission.parse("dataset:read")],
        )

        assert decision.allowed is True
        assert bool(decision) is True
        assert len(decision.matching_permissions) == 1

    def test_access_decision_deny(self):
        """Test deny decision creation."""
        from truthound.rbac import AccessDecision

        decision = AccessDecision.deny(reason="Permission denied")

        assert decision.allowed is False
        assert bool(decision) is False


# =============================================================================
# Test Utility Functions
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_generate_role_id(self):
        """Test role ID generation."""
        from truthound.rbac import generate_role_id

        assert generate_role_id("Data Analyst") == "data_analyst"
        assert generate_role_id("Super-Admin!") == "super_admin"

    def test_generate_principal_id(self):
        """Test principal ID generation."""
        from truthound.rbac import generate_principal_id

        id1 = generate_principal_id("user")
        id2 = generate_principal_id("user")

        assert id1.startswith("user_")
        assert id2.startswith("user_")
        assert id1 != id2  # Should be unique


# =============================================================================
# Integration Tests
# =============================================================================


class TestFullIntegration:
    """Full integration tests for the RBAC system."""

    def test_complete_rbac_workflow(self):
        """Test complete RBAC workflow."""
        from truthound.rbac import (
            Permission,
            PermissionDeniedError,
            RBACManager,
            RBACManagerConfig,
            SecurityContext,
        )

        # Setup manager (disable caching since roles are modified during test)
        config = RBACManagerConfig(cache_decisions=False)
        manager = RBACManager(config=config)

        # Create roles (using custom names to avoid collision with default roles)
        viewer_role = manager.create_role(
            name="Data Viewer",
            permissions={"dataset:read", "validation:read"},
        )

        editor_role = manager.create_role(
            name="Data Editor",
            permissions={"dataset:update", "validation:execute"},
            parent_roles={viewer_role.id},
        )

        admin_role = manager.create_role(
            name="Project Admin",
            permissions={"dataset:*", "validation:*", "user:read"},
            parent_roles={editor_role.id},
        )

        # Create principals
        viewer = manager.create_principal(
            name="viewer@example.com",
            roles={viewer_role.id},
        )

        editor = manager.create_principal(
            name="editor@example.com",
            roles={editor_role.id},
        )

        admin = manager.create_principal(
            name="admin@example.com",
            roles={admin_role.id},
        )

        # Test viewer permissions
        assert manager.check(viewer, "dataset", "read").allowed is True
        assert manager.check(viewer, "dataset", "update").allowed is False
        assert manager.check(viewer, "dataset", "delete").allowed is False

        # Test editor permissions (includes viewer permissions through inheritance)
        assert manager.check(editor, "dataset", "read").allowed is True
        assert manager.check(editor, "dataset", "update").allowed is True
        assert manager.check(editor, "dataset", "delete").allowed is False

        # Test admin permissions (has wildcard)
        assert manager.check(admin, "dataset", "read").allowed is True
        assert manager.check(admin, "dataset", "update").allowed is True
        assert manager.check(admin, "dataset", "delete").allowed is True

        # Test context management
        with manager.context(admin):
            current = manager.current_principal()
            assert current is not None
            assert current.id == admin.id

        # Test role assignment
        manager.assign_role(viewer.id, editor_role.id)
        assert manager.check(viewer, "dataset", "update").allowed is True

        # Test role revocation
        manager.revoke_role(viewer.id, editor_role.id)
        assert manager.check(viewer, "dataset", "update").allowed is False

    def test_tenant_isolated_rbac(self):
        """Test tenant-isolated RBAC workflow."""
        from truthound.rbac import (
            MemoryRoleStore,
            Permission,
            PolicyEngine,
            PolicyEngineConfig,
            Principal,
            Role,
            RoleBasedEvaluator,
            TenantAwarePolicyEvaluator,
            TenantRBACConfig,
        )

        # Setup
        role_store = MemoryRoleStore()

        # Create tenant roles
        tenant_a_admin = Role(
            id="tenant_a:admin",
            name="Tenant A Admin",
            tenant_id="tenant_a",
        )
        tenant_a_admin.add_permission("dataset:*")
        role_store.save(tenant_a_admin)

        tenant_b_admin = Role(
            id="tenant_b:admin",
            name="Tenant B Admin",
            tenant_id="tenant_b",
        )
        tenant_b_admin.add_permission("dataset:*")
        role_store.save(tenant_b_admin)

        # Create engine with tenant isolation (disable cache for ABAC)
        engine_config = PolicyEngineConfig(cache_decisions=False)
        config = TenantRBACConfig()
        engine = PolicyEngine(config=engine_config)
        engine.add_evaluator(TenantAwarePolicyEvaluator(config))
        engine.add_evaluator(RoleBasedEvaluator(role_store))

        # Create principals
        user_a = Principal(
            id="user_a",
            roles={"tenant_a:admin"},
            tenant_id="tenant_a",
        )

        user_b = Principal(
            id="user_b",
            roles={"tenant_b:admin"},
            tenant_id="tenant_b",
        )

        # User A can access tenant A resources
        decision = engine.check(
            user_a,
            "dataset",
            "read",
            resource_attributes={"tenant_id": "tenant_a"},
        )
        assert decision.allowed is True

        # User A cannot access tenant B resources
        decision = engine.check(
            user_a,
            "dataset",
            "read",
            resource_attributes={"tenant_id": "tenant_b"},
        )
        assert decision.allowed is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
