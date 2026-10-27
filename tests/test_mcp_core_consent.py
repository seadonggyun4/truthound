"""Revocable operator grants for the separately installed Core plugin."""

import json
from pathlib import Path

import pytest

from truthound.mcp.consent import ConsentPolicy, policy_from, write_config
from truthound.mcp.policy import MCPError

pytestmark = pytest.mark.contract


def configuration(data, **changes):
    return {
        "schema_version": 2,
        "allowed_directories": [str(data)],
        "workspace_directories": [],
        "control_directory": None,
        "allow_sensitive_results": False,
        **changes,
    }


def test_revocation_denies_existing_handle_authority_and_preserves_input(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    source = data / "input.csv"
    source.write_text("x\n1\n")
    empty = tmp_path / "empty"
    empty.mkdir()
    config = tmp_path / "config" / "permissions.json"
    unconfigured = ConsentPolicy(config, empty)
    with pytest.raises(MCPError):
        unconfigured.resolve(str(source))
    write_config(config, configuration(data))
    policy = ConsentPolicy(config, empty)
    assert policy.resolve(str(source)) == source.resolve()
    write_config(config, configuration(data, allowed_directories=[]))
    assert not policy.unchanged()
    for operation in (lambda: policy.profile, lambda: policy.authorize_origin(source)):
        with pytest.raises(MCPError, match="access_denied"):
            operation()
    assert source.read_text() == "x\n1\n"
    import asyncio

    from truthound.mcp.service import MCPService

    service = MCPService(policy)
    try:
        result = asyncio.run(service.call_operation_async("capabilities_list", {}))
        assert result["execution_status"] == "failed"
        assert result["issues"] == [{"code": "access_denied"}]
    finally:
        service.close()


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": True},
        {"allowed_directories": ["relative"]},
        {"allowed_directories": [False]},
        {"allow_sensitive_results": "true"},
        {"workspace_directories": ["relative"]},
        {"control_directory": False},
        {"allowed_directories": [str(Path.cwd().anchor)]},
        {"extra": True},
    ],
)
def test_invalid_grants_cannot_replace_saved_consent(tmp_path, change):
    config = tmp_path / "state" / "permissions.json"
    write_config(config, configuration(tmp_path))
    before = config.read_bytes()
    with pytest.raises((ValueError, OSError)):
        write_config(config, configuration(tmp_path, **change))
    assert config.read_bytes() == before


def test_control_and_workspace_grants_remain_independent(tmp_path):
    data, workspace, control, empty = (
        tmp_path / name for name in ("data", "workspace", "control", "empty")
    )
    for path in (data, workspace, control, empty):
        path.mkdir(mode=0o700)
    value = configuration(
        data, workspace_directories=[str(workspace)], control_directory=str(control)
    )
    policy = policy_from(json.dumps(value).encode(), empty)
    assert policy.resolve_workspace(str(workspace)) == workspace.resolve()
    with pytest.raises(MCPError):
        policy.resolve_workspace(str(data))
    assert not policy.allow_sensitive_results
    with pytest.raises(ValueError):
        policy_from(json.dumps({**value, "control_directory": str(data)}).encode(), empty)
