from __future__ import annotations

import builtins
import importlib
import sys

import pytest
from typer.testing import CliRunner

from truthound.cli import app


@pytest.mark.contract
def test_truthound_ai_namespace_requires_optional_dependency(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pydantic" or name.startswith("pydantic."):
            raise ImportError("No module named 'pydantic'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    for module_name in list(sys.modules):
        if (
            module_name == "truthound.ai"
            or module_name.startswith("truthound.ai.")
            or module_name == "pydantic"
            or module_name.startswith("pydantic.")
        ):
            sys.modules.pop(module_name, None)

    assert importlib.import_module("truthound") is not None
    with pytest.raises(ImportError, match=r"truthound\[ai\]"):
        importlib.import_module("truthound.ai")


@pytest.mark.contract
def test_truthound_root_ai_support_probe_is_safe_without_optional_dependency(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            name == "pydantic"
            or name.startswith("pydantic.")
            or name == "openai"
            or name.startswith("openai.")
        ):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    for module_name in list(sys.modules):
        if (
            module_name == "truthound.ai"
            or module_name.startswith("truthound.ai.")
            or module_name == "pydantic"
            or module_name.startswith("pydantic.")
            or module_name == "openai"
            or module_name.startswith("openai.")
        ):
            sys.modules.pop(module_name, None)

    truthound = importlib.import_module("truthound")
    status = truthound.get_ai_support_status()

    assert truthound.has_ai_support() is False
    assert status.ready is False
    assert "pydantic" in status.missing_dependencies
    assert "openai" in status.missing_dependencies
    assert "truthound[ai]" in status.install_hint


@pytest.mark.contract
def test_truthound_cli_help_shows_ai_group():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "ai" in result.output


@pytest.mark.contract
def test_truthound_ai_cli_command_surfaces_install_hint_when_optional_dependency_missing(
    monkeypatch,
    tmp_path,
):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pydantic" or name.startswith("pydantic."):
            raise ImportError("No module named 'pydantic'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    for module_name in list(sys.modules):
        if (
            module_name == "truthound.ai"
            or module_name.startswith("truthound.ai.")
            or module_name == "pydantic"
            or module_name.startswith("pydantic.")
        ):
            sys.modules.pop(module_name, None)
    data_path = tmp_path / "data.csv"
    data_path.write_text("id,value\n1,10\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["ai", "suggest-suite", str(data_path), "--prompt", "validate id uniqueness", "--model", "gpt-test"],
    )

    assert result.exit_code == 40
    assert "truthound[ai]" in result.output
