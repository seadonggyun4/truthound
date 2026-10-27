"""Resident artifact re-entry must preserve a dependency DAG and clean release."""

import pytest

from truthound.mcp.policy import MCPError, MCPPolicy
from truthound.mcp.resources import SessionResources

pytestmark = pytest.mark.fault


@pytest.mark.parametrize("indirect", [False, True])
def test_resident_binding_rejects_descendant_input_before_mutating_owner(tmp_path, indirect):
    resources = SessionResources(lambda: MCPPolicy((tmp_path,)))
    try:
        def resource(kind, parents=()):
            identity, path = resources.reserve(kind)
            path.write_bytes(b"synthetic")
            return resources.commit(identity, kind=kind, media_type="text/plain", parents=parents)

        owner = resource("monitor")
        child = resource("artifact", (owner.id,))
        if indirect:
            child = resource("artifact", (child.id,))
        with pytest.raises(MCPError, match="invalid_arguments"):
            resources.bind_resident_inputs(owner.id, (child.id,), kind="monitor")
        assert resources.get(owner.id).parents == ()
        resources.release(owner.id)
        assert not resources._entries
    finally:
        resources.close()
