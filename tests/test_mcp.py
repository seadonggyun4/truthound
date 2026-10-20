"""Real Core and MCP boundaries; synthetic input only."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from truthound.mcp.policy import MCPPolicy, MCPError
from truthound.mcp.service import MCPService

pytestmark = pytest.mark.contract


@pytest.fixture
def dataset(tmp_path):
    p = tmp_path / "synthetic.csv"
    p.write_text("email,age\na@example.com,1\na@example.com,1\n,3\n")
    return p


def test_core_parity_and_no_workspace_writes(dataset):
    import truthound as th

    service = MCPService(MCPPolicy((dataset.parent,)))
    checked = service.call("check", str(dataset))
    profiled = service.call("profile", str(dataset))
    scanned = service.call("scan", str(dataset))
    assert checked["ok"] and not checked["data"]["success"]
    assert profiled["data"]["row_count"] == th.profile(str(dataset)).row_count
    assert scanned["data"]["finding_count"] == len(th.scan(str(dataset)).findings)
    assert checked["data"]["row_count"] == 3
    assert not (dataset.parent / ".truthound").exists()
    assert "a@example.com" not in json.dumps([checked, profiled, scanned])
    assert str(dataset.parent) not in json.dumps([checked, profiled, scanned])


@pytest.mark.parametrize(
    "value", ["../escape.csv", "https://example.com/data.csv", "file:///etc/passwd", "*.csv"]
)
def test_reject_untrusted_paths(tmp_path, value):
    with pytest.raises(MCPError):
        MCPPolicy((tmp_path,)).resolve(value)


def test_symlink_escape_and_unsupported_file(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text("x\n1\n")
    (allowed / "link.csv").symlink_to(outside)
    policy = MCPPolicy((allowed,))
    with pytest.raises(MCPError):
        policy.resolve(str(allowed / "link.csv"))
    (allowed / "payload.py").write_text("raise RuntimeError")
    with pytest.raises(MCPError):
        policy.resolve("payload.py")
    with pytest.raises(ValueError):
        MCPPolicy(())


def test_size_limit_and_sanitized_failure(dataset):
    service = MCPService(MCPPolicy((dataset.parent,), max_file_bytes=1))
    result = service.call("profile", str(dataset))
    assert not result["ok"] and result["error"]["code"] == "file_too_large"
    assert str(dataset) not in json.dumps(result)
    broken = dataset.parent / "broken.parquet"
    broken.write_bytes(b"not-parquet-private-marker")
    result = MCPService(MCPPolicy((dataset.parent,))).call("profile", str(broken))
    assert not result["ok"]
    assert "private-marker" not in json.dumps(result)


def test_result_limit_preserves_total(dataset):
    result = MCPService(MCPPolicy((dataset.parent,), max_items=1)).call("profile", str(dataset))
    assert result["ok"]
    assert result["data"]["column_count"] == 2
    assert len(result["data"]["columns"]) == 1
    assert result["truncated"]


def test_timeout_does_not_claim_validation_success(dataset, monkeypatch):
    import anyio

    async def timeout(*args, **kwargs):
        raise TimeoutError

    monkeypatch.setattr(anyio, "run_process", timeout)
    result = MCPService(MCPPolicy((dataset.parent,))).call("check", str(dataset))
    assert not result["ok"] and result["error"]["code"] == "timeout"


def test_stdio_initialize_discovery_call_and_errors(dataset):
    async def scenario():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "truthound.mcp", "serve", "--allow-dir", str(dataset.parent)],
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                initialized = await client.initialize()
                assert initialized.instructions is not None
                assert "No raw values are returned" in initialized.instructions
                assert initialized.protocolVersion == "2025-11-25"
                tools = await client.list_tools()
                assert {t.name for t in tools.tools} == {"check_file", "profile_file", "scan_pii"}
                result = await client.call_tool("profile_file", {"path": str(dataset)})
                assert not result.isError
                assert result.structuredContent["data"]["row_count"] == 3
                result = await client.call_tool("check_file", {"path": str(dataset)})
                assert not result.isError and not result.structuredContent["data"]["success"]
                result = await client.call_tool("scan_pii", {"path": str(dataset)})
                assert not result.isError
                result = await client.call_tool(
                    "profile_file", {"path": "https://invalid/data.csv"}
                )
                assert result.isError
                assert "a@example.com" not in result.model_dump_json()

    asyncio.run(scenario())
