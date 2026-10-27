"""Native permission changes must invalidate cached grants and denials immediately."""

import pytest


@pytest.mark.parametrize(
    "mutation", ["revoke", "remove_permission", "delete_role", "disable_role", "disable_principal"]
)
def test_native_rbac_revocation_does_not_reuse_an_earlier_allow(mutation):
    from truthound.rbac import RBACManager

    manager = RBACManager()
    role = manager.create_role("Reader", role_id="local:reader", permissions={"dataset:read"})
    principal = manager.create_principal("Reviewer", principal_id="local:reviewer", roles={role.id})
    assert manager.check(principal.id, "dataset", "read").allowed
    operations = {
        "revoke": lambda: manager.revoke_role(principal.id, role.id),
        "remove_permission": lambda: manager.remove_permission_from_role(role.id, "dataset:read"),
        "delete_role": lambda: manager.delete_role(role.id),
        "disable_role": lambda: manager.update_role(role.id, enabled=False),
        "disable_principal": lambda: manager.update_principal(principal.id, enabled=False),
    }
    operations[mutation]()
    assert not manager.check(principal.id, "dataset", "read").allowed


def test_native_rbac_new_grant_does_not_reuse_an_earlier_deny():
    from truthound.rbac import RBACManager

    manager = RBACManager()
    role = manager.create_role("Reader", role_id="local:reader", permissions={"dataset:read"})
    principal = manager.create_principal("Reviewer", principal_id="local:reviewer", roles=set())
    assert not manager.check(principal.id, "dataset", "read").allowed
    manager.assign_role(principal.id, role.id)
    assert manager.check(principal.id, "dataset", "read").allowed


def test_plain_resource_name_is_not_a_wildcard_permission_request():
    from truthound.rbac import RBACManager

    manager = RBACManager()
    role = manager.create_role("Reader", role_id="local:reader", permissions={"dataset:read"})
    principal = manager.create_principal("Reviewer", principal_id="local:reviewer", roles={role.id})
    assert manager.check(principal.id, "dataset", "read").allowed
    assert not manager.check(principal.id, "secret", "read").allowed


def test_native_diamond_inheritance_does_not_report_a_false_cycle():
    from truthound.rbac import RBACManager
    from truthound.rbac.core import CircularRoleError

    manager = RBACManager()
    root = manager.create_role("root", role_id="root", permissions={"dataset:read"})
    left = manager.create_role("left", role_id="left", parent_roles={root.id})
    right = manager.create_role("right", role_id="right", parent_roles={root.id})
    leaf = manager.create_role("leaf", role_id="leaf", parent_roles={left.id, right.id})
    assert manager.get_all_role_permissions(leaf.id) == root.permissions
    # A real back edge must still fail; do not relax cycle detection.
    root.parent_roles.add(leaf.id)
    manager.role_store.save(root)
    with pytest.raises(CircularRoleError):
        manager.get_all_role_permissions(leaf.id)


def test_native_deny_overrides_direct_allow_in_both_permission_helpers_and_engine():
    from truthound.rbac import RBACManager
    from truthound.rbac.core import Permission, PermissionEffect, Role

    allow = Permission.parse("dataset:read")
    deny = Permission("dataset", "read", effect=PermissionEffect.DENY)
    manager = RBACManager()
    role = manager.create_role("deny", role_id="deny", permissions={deny})
    principal = manager.create_principal("reviewer", roles={role.id}, permissions={allow})
    assert not manager.check(principal.id, "dataset", "read").allowed
    assert not manager.has_permission(principal.id, allow)
    assert not Role(id="mixed", name="mixed", permissions={allow, deny}).has_permission(allow)


def test_permission_helper_refuses_disabled_principal_and_role():
    from truthound.rbac import RBACManager

    manager = RBACManager()
    role = manager.create_role("reader", permissions={"dataset:read"})
    principal = manager.create_principal("reader", roles={role.id})
    manager.update_principal(principal.id, enabled=False)
    assert not manager.has_permission(principal.id, "dataset:read")
    manager.update_principal(principal.id, enabled=True)
    manager.update_role(role.id, enabled=False)
    assert not manager.has_permission(principal.id, "dataset:read")
    assert not role.has_permission("dataset:read")


def test_attribute_policy_rechecks_changed_resource_attributes_with_default_cache():
    from truthound.rbac.core import Condition, ConditionOperator, Principal
    from truthound.rbac.policy import ABACEvaluator, PolicyEngine

    evaluator = ABACEvaluator()
    evaluator.add_rule(
        [Condition("resource.owner_id", ConditionOperator.EQUALS, "${principal.id}")]
    )
    engine = PolicyEngine()
    engine.add_evaluator(evaluator)
    principal = Principal(id="reviewer")
    assert engine.check(principal, "dataset:orders", "read", {"owner_id": principal.id}).allowed
    assert not engine.check(principal, "dataset:orders", "read", {"owner_id": "other"}).allowed


def test_added_deny_evaluator_invalidates_a_cached_native_role_allow():
    from truthound.rbac import RBACManager
    from truthound.rbac.core import PermissionEffect
    from truthound.rbac.policy import ABACEvaluator

    manager = RBACManager()
    principal = manager.create_principal("reviewer", permissions={"dataset:read"})
    assert manager.check(principal.id, "dataset", "read").allowed
    evaluator = ABACEvaluator()
    evaluator.add_rule([], effect=PermissionEffect.DENY)
    manager.engine.add_evaluator(evaluator)
    assert not manager.check(principal.id, "dataset", "read").allowed


def test_cache_separates_resource_type_and_colon_containing_identity_components():
    from truthound.rbac import RBACManager
    from truthound.rbac.core import AccessContext

    manager = RBACManager()
    first = manager.create_principal("first", principal_id="user:a", permissions={"dataset:read"})
    second = manager.create_principal("second", principal_id="user")
    engine = manager.engine
    assert engine.evaluate(AccessContext(first, "b", "read", resource_type="dataset")).allowed
    assert not engine.evaluate(AccessContext(first, "b", "read", resource_type="secret")).allowed
    assert not engine.evaluate(AccessContext(second, "a:b", "read")).allowed


@pytest.mark.parametrize("backend", ["file", "sqlite"])
@pytest.mark.parametrize("source", ["role", "direct"])
def test_native_persisted_denies_survive_fresh_store_and_manager(tmp_path, backend, source):
    from truthound.rbac import RBACManager
    from truthound.rbac.core import Permission, PermissionEffect
    from truthound.rbac.storage import (
        FilePrincipalStore,
        FileRoleStore,
        FileStorageConfig,
        SQLitePrincipalStore,
        SQLiteRoleStore,
    )

    def open_stores():
        if backend == "file":
            config = FileStorageConfig(base_path=tmp_path)
            return FileRoleStore(config), FilePrincipalStore(config)
        return SQLiteRoleStore(tmp_path / "roles.db"), SQLitePrincipalStore(
            tmp_path / "principals.db"
        )

    stores = open_stores()
    try:
        manager = RBACManager(*stores)
        deny = Permission("dataset", "read", effect=PermissionEffect.DENY)
        role = manager.create_role(
            "deny", role_id="deny", permissions={deny} if source == "role" else set()
        )
        manager.create_principal(
            "reviewer",
            principal_id="reviewer",
            roles={role.id},
            permissions={"dataset:read", deny} if source == "direct" else set(),
        )
    finally:
        for store in stores:
            if hasattr(store, "close"):
                store.close()
    stores = open_stores()
    try:
        restored = RBACManager(*stores)
        assert not restored.check("reviewer", "dataset", "read").allowed
        assert not restored.has_permission("reviewer", "dataset:read")
    finally:
        for store in stores:
            if hasattr(store, "close"):
                store.close()


@pytest.mark.parametrize("backend", ["file", "sqlite"])
def test_conditional_role_is_preserved_and_applied_after_reopen(tmp_path, backend):
    from truthound.rbac import RBACManager
    from truthound.rbac.core import Condition, ConditionOperator
    from truthound.rbac.storage import FileRoleStore, FileStorageConfig, SQLiteRoleStore

    condition = Condition("resource.owner_id", ConditionOperator.EQUALS, "${principal.id}")
    open_roles = (
        (lambda: FileRoleStore(FileStorageConfig(base_path=tmp_path)))
        if backend == "file"
        else (lambda: SQLiteRoleStore(tmp_path / "roles.db"))
    )
    store = open_roles()
    try:
        manager = RBACManager(role_store=store)
        role = manager.create_role("owner", role_id="owner", permissions={"dataset:read"})
        role.conditions = [condition]
        store.save(role)
    finally:
        if hasattr(store, "close"):
            store.close()
    store = open_roles()
    try:
        manager = RBACManager(role_store=store)
        principal = manager.create_principal("reviewer", principal_id="reviewer", roles={role.id})
        assert manager.check(
            principal.id, "dataset:orders", "read", {"owner_id": principal.id}
        ).allowed
        assert not manager.check(
            principal.id, "dataset:orders", "read", {"owner_id": "other"}
        ).allowed
        assert manager.require_role(role.id).conditions
    finally:
        if hasattr(store, "close"):
            store.close()


@pytest.mark.parametrize("kind", ["role", "principal"])
def test_corrupt_file_store_refuses_mutation_without_destroying_evidence(tmp_path, kind):
    import json

    from truthound.rbac.core import Principal, RBACError, Role
    from truthound.rbac.storage import FilePrincipalStore, FileRoleStore, FileStorageConfig

    target = tmp_path / ("roles.json" if kind == "role" else "principals.json")
    original = b'{"broken":'
    target.write_bytes(original)
    cls, item = (
        (FileRoleStore, Role(id="new", name="new"))
        if kind == "role"
        else (FilePrincipalStore, Principal(id="new"))
    )
    store = cls(FileStorageConfig(base_path=tmp_path))
    with pytest.raises((json.JSONDecodeError, RBACError)):
        store.save(item)
    assert target.read_bytes() == original


@pytest.mark.parametrize("kind", ["role", "principal"])
def test_file_store_serialization_failure_preserves_disk_and_invalidates_mutated_cache(
    tmp_path, kind
):
    from truthound.rbac.core import Principal, RBACError, Role
    from truthound.rbac.storage import FilePrincipalStore, FileRoleStore, FileStorageConfig

    class InvalidValue:
        def __str__(self):
            raise RuntimeError("synthetic serialization failure")

    cls, constructor, name = (
        (FileRoleStore, lambda key: Role(id=key, name=key), "roles.json")
        if kind == "role"
        else (FilePrincipalStore, lambda key: Principal(id=key), "principals.json")
    )
    store = cls(FileStorageConfig(base_path=tmp_path))
    store.save(constructor("existing"))
    original = (tmp_path / name).read_bytes()
    item = constructor("failed")
    item.metadata = {"invalid": InvalidValue()}
    with pytest.raises((RuntimeError, RBACError)):
        store.save(item)
    assert (tmp_path / name).read_bytes() == original
    assert store.get("failed") is None


@pytest.mark.parametrize("combination", ["deny_overrides", "allow_overrides"])
def test_evaluator_failure_cannot_be_silently_replaced_by_another_allow(combination):
    from truthound.rbac import RBACManager
    from truthound.rbac.policy import (
        ABACEvaluator,
        PolicyCombination,
        PolicyEngine,
        PolicyEngineConfig,
        RoleBasedEvaluator,
    )

    manager = RBACManager()
    role = manager.create_role("broken", role_id="broken", permissions={"dataset:read"})
    role.parent_roles.add(role.id)
    manager.role_store.save(role)
    principal = manager.create_principal("reviewer", roles={role.id})
    engine = PolicyEngine(PolicyEngineConfig(combination=PolicyCombination(combination)))
    engine.add_evaluator(RoleBasedEvaluator(manager.role_store))
    other = ABACEvaluator()
    other.add_rule([])
    engine.add_evaluator(other)
    assert not engine.check(principal, "dataset", "read").allowed
