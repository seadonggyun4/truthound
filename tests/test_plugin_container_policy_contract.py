"""Container grants reject ambiguous paths before runtime side effects."""

from dataclasses import replace

import pytest

from truthound.plugins.security.protocols import SecurityPolicy
from truthound.plugins.security.sandbox.engines.container import ContainerSandboxEngine


def test_host_grants_reject_symlink_alias_and_readonly_write_conflict(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    policy = SecurityPolicy.development()
    engine = ContainerSandboxEngine()
    for selected in (
        replace(
            policy, resource_limits=replace(policy.resource_limits, allowed_paths=(str(alias),))
        ),
        replace(
            policy,
            allow_file_write=False,
            resource_limits=replace(
                policy.resource_limits, writable_paths=(str(target.resolve()),)
            ),
        ),
        replace(policy, resource_limits=replace(policy.resource_limits, allowed_paths=("/",))),
        replace(policy, allow_network="false"),
        replace(
            policy,
            resource_limits=replace(policy.resource_limits, denied_syscalls=("unknown-syscall",)),
        ),
    ):
        with pytest.raises(ValueError):
            engine.create_sandbox("invalid-grant", selected)
    assert not engine._contexts and not engine._container_ids
