"""Generated native reporters must honor filtering and canonical Core results."""

import os
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

import truthound as th
from truthound.cli_modules.scaffolding import PluginScaffold, ReporterScaffold, ScaffoldConfig
from truthound.reporters.presentation import build_run_presentation


def test_shared_legacy_projection_includes_only_actual_successful_checks_when_requested():
    run = th.check(pl.DataFrame({"x": [None, 2]}), validators=["null", "unique"])
    presentation = build_run_presentation(run)
    hidden = presentation.to_legacy_view()
    shown = presentation.to_legacy_view(include_passed=True)
    expected = [check.name for check in run.checks if check.success]
    assert expected and hidden.results
    assert [row.validator_name for row in shown.results if row.success] == expected
    assert tuple(row for row in shown.results if not row.success) == hidden.results
    assert shown.statistics == hidden.statistics


@pytest.mark.parametrize(
    "factory,variant",
    [
        (ReporterScaffold, "basic"),
        (ReporterScaffold, "full"),
        (PluginScaffold, "reporter"),
        (PluginScaffold, "full"),
    ],
)
def test_generated_reporter_executes_pass_and_zero_limit_controls(tmp_path, factory, variant):
    result = factory().generate(
        ScaffoldConfig(
            name="synthetic_component",
            template_variant=variant,
            include_tests=False,
        )
    )
    assert result.success, result.errors
    result.write_files(tmp_path)
    module = "reporter" if factory is ReporterScaffold else "plugin"
    script = f"""
import json
import polars as pl
import truthound as th
from synthetic_component.{module} import SyntheticComponentReporter as Reporter, SyntheticComponentReporterConfig as Config
passed = th.check(pl.DataFrame({{"x": [1, 2]}}), validators=["null"])
hidden = Reporter(Config(include_passed=False)).render(passed)
shown = Reporter(Config(include_passed=True)).render(passed)
assert shown != hidden
assert passed.checks[0].name in shown
"""
    if factory is ReporterScaffold:
        script += """
failed = th.check(pl.DataFrame({"x": [None, 2]}), validators=["null"])
limited = Reporter(Config(max_issues=0)).render(failed)
"""
        script += (
            'assert not json.loads(limited)["issues"]\n'
            if variant == "full"
            else 'assert "\\nIssues\\n" not in limited\n'
        )
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            [
                str(tmp_path),
                str(Path(__file__).resolve().parents[1] / "src"),
            ]
        ),
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    "variant,options",
    [
        ("basic", {"severity": "NOT_A_LEVEL"}),
        ("column", {"columns": "abc"}),
        ("range", {"min_value": float("nan")}),
        ("range", {"min_value": True}),
        ("range", {"min_value": 10, "max_value": 1}),
        ("pattern", {"pattern": "["}),
        ("pattern", {"pattern": "(?=x)"}),
        ("basic", {"min_value": 1}),
        ("range", {"pattern": "x"}),
        ("basic", {"columns": ["x"]}),
    ],
)
def test_invalid_validator_options_do_not_emit_broken_files(variant, options):
    from truthound.cli_modules.scaffolding import ValidatorScaffold

    result = ValidatorScaffold().generate(
        ScaffoldConfig(name="synthetic_component", template_variant=variant, extra=options)
    )
    assert not result.success and result.errors
    assert not result.files


@pytest.mark.parametrize("variant", ["validator", "reporter", "hook", "custom"])
def test_plugins_cli_created_packages_load_against_current_core(tmp_path, variant):
    import tomllib

    from typer.testing import CliRunner

    from truthound.cli import app

    author = 'Synthetic "Author"'
    result = CliRunner().invoke(
        app,
        [
            "plugins",
            "create",
            "synthetic-plugin",
            "--type",
            variant,
            "--author",
            author,
            "--output",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    root = tmp_path / "truthound-plugin-synthetic-plugin"
    metadata = tomllib.loads((root / "pyproject.toml").read_text())
    assert metadata["project"]["authors"][0]["name"] == author
    script = """
from synthetic_plugin.plugin import SyntheticPluginPlugin
plugin = SyntheticPluginPlugin()
assert plugin.name == "synthetic-plugin"
plugin.setup()
try:
"""
    script += (
        """
    import polars as pl
    import truthound as th
    from dataclasses import replace
    reporter_type = plugin.get_reporters()["synthetic-plugin"]
    reporter = reporter_type()
    reporter._config = replace(reporter.config, include_passed=True)
    run = th.check(pl.DataFrame({"x": [1, 2]}), validators=["null"])
    assert run.checks[0].name in reporter.render(run)
"""
        if variant == "reporter"
        else '    assert plugin.info.name == "synthetic-plugin"\n'
    )
    script += "finally:\n    plugin.teardown()\n"
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            [
                str(root),
                str(Path(__file__).resolve().parents[1] / "src"),
            ]
        ),
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_plugins_cli_refuses_to_overwrite_existing_user_code(tmp_path):
    from typer.testing import CliRunner

    from truthound.cli import app

    args = ["plugins", "create", "synthetic-plugin", "--output", str(tmp_path)]
    runner = CliRunner()
    assert runner.invoke(app, args).exit_code == 0
    root = tmp_path / "truthound-plugin-synthetic-plugin"
    code = root / "synthetic_plugin/plugin.py"
    code.write_text("# existing user code\n")
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = runner.invoke(app, args)
    assert result.exit_code != 0
    assert {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    } == before
