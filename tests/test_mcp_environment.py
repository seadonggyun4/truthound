"""Keep OS identity needed by native runtimes without inheriting credentials."""

import json
from types import SimpleNamespace

import pytest

from truthound.mcp.policy import MCPPolicy
from truthound.mcp.service import MCPService

pytestmark = pytest.mark.fault


def test_worker_keeps_windows_architecture_but_not_secrets_or_cpu_bypass(tmp_path, monkeypatch):
    data = tmp_path / "input.csv"
    data.write_text("x\n1\n", encoding="utf-8")
    monkeypatch.setenv("PROCESSOR_ARCHITECTURE", "AMD64")
    monkeypatch.setenv("PROCESSOR_ARCHITEW6432", "ARM64")
    monkeypatch.setenv("POLARS_SKIP_CPU_CHECK", "1")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "synthetic-test-marker")
    monkeypatch.setenv("PYTHONPATH", "synthetic-test-marker")

    observed = []

    async def run(argv, **kwargs):
        observed.append(True)
        env = kwargs["env"]
        assert env["PROCESSOR_ARCHITECTURE"] == "AMD64"
        assert env["PROCESSOR_ARCHITEW6432"] == "ARM64"
        assert not {"POLARS_SKIP_CPU_CHECK", "AWS_SECRET_ACCESS_KEY", "PYTHONPATH"} & env.keys()
        assert "-I" in argv
        assert env["POLARS_MAX_THREADS"] == "1"
        from truthound.mcp.projection import response

        return SimpleNamespace(returncode=0, stdout=json.dumps(response("check", data={})).encode())

    import anyio

    monkeypatch.setattr(anyio, "run_process", run)
    assert MCPService(MCPPolicy((tmp_path,))).call("check", str(data))["ok"]
    assert observed == [True]
