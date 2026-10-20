"""Additional regression/failure checks for the optional adapter."""

import asyncio
import sys

import pytest

pytest.importorskip("mcp")
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from truthound.mcp.policy import MCPPolicy
from truthound.mcp.service import MCPService

pytestmark = pytest.mark.fault


def test_platform_alias_above_root_and_symlink_inside_root(tmp_path):
    real = tmp_path / "real"
    root = real / "data"
    root.mkdir(parents=True)
    file = root / "input.csv"
    file.write_text("x\n1\n")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    policy = MCPPolicy((root,))
    assert policy.resolve(str(alias / "data/input.csv")) == file
    (root / "self").symlink_to(root, target_is_directory=True)
    from truthound.mcp.policy import MCPError

    with pytest.raises(MCPError):
        policy.resolve(str(root / "self/input.csv"))


def test_busy_and_response_bound(tmp_path):
    source = tmp_path / "wide.csv"
    source.write_text("a" * 2000 + "\n1\n")
    service = MCPService(MCPPolicy((tmp_path,), max_response_bytes=1024))
    result = service.call("profile", str(source))
    assert result["error"]["code"] == "response_too_large"
    service._busy.acquire()
    try:
        assert service.call("profile", str(source))["error"]["code"] == "busy"
    finally:
        service._busy.release()


@pytest.mark.parametrize("suffix", [".parquet", ".json", ".ndjson", ".jsonl"])
def test_file_format_core_parity(tmp_path, suffix):
    import polars as pl

    frame = pl.DataFrame({"x": [1, 2, 3]})
    path = tmp_path / ("data" + suffix)
    if suffix == ".parquet":
        frame.write_parquet(path)
    elif suffix == ".json":
        frame.write_json(path)
    else:
        frame.write_ndjson(path)
    result = MCPService(MCPPolicy((tmp_path,))).call("profile", str(path))
    assert result["ok"], result
    assert result["data"]["row_count"] == 3


def test_invalid_tool_input_does_not_echo_payload(tmp_path):
    async def scenario():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "truthound.mcp", "serve", "--allow-dir", str(tmp_path)],
        )
        async with (
            stdio_client(params) as (read, write),
            ClientSession(read, write) as client,
        ):
            await client.initialize()
            for arguments in (
                {"path": "unused.csv", "secret": "private-marker"},
                {"path": 42},
                {},
            ):
                result = await client.call_tool("profile_file", arguments)
                assert result.isError
                assert "private-marker" not in result.model_dump_json()

    asyncio.run(scenario())


def test_bundle_rejects_different_wheel_and_existing_output(tmp_path):
    import zipfile

    from truthound.mcp.bundle import build_bundle

    wheel = tmp_path / "truthound-0.0.1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        z.writestr("truthound-0.0.1.dist-info/METADATA", "Name: truthound\nVersion: 0.0.1\n")
    with pytest.raises(ValueError):
        build_bundle(tmp_path / "wrong.mcpb", wheel=wheel)
    output = tmp_path / "existing.mcpb"
    output.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        build_bundle(output)
    assert output.read_bytes() == b"preserve"
