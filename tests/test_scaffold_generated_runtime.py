"""Execute the generated component test suites in owned directories."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from truthound.cli_modules.scaffolding import (
    PluginScaffold,
    ReporterScaffold,
    ScaffoldConfig,
    ScaffoldType,
    ValidatorScaffold,
)

CASES = [
    (factory, variant)
    for factory in (ValidatorScaffold, ReporterScaffold, PluginScaffold)
    for variant in factory.TEMPLATE_VARIANTS
]


@pytest.mark.parametrize(
    "factory,variant", CASES, ids=[f"{factory.name}-{variant}" for factory, variant in CASES]
)
def test_generated_component_tests_run_against_current_core(tmp_path, factory, variant):
    config = ScaffoldConfig(
        name="synthetic_component",
        description='Synthetic "quoted" description with """ docstrings and backslash \\\nnext line',
        author='Synthetic "Author"',
        output_dir=tmp_path,
        scaffold_type=ScaffoldType[factory.name.upper()],
        template_variant=variant,
        include_tests=True,
        include_docs=True,
        include_examples=True,
        extra=(
            {"content_type": 'text/plain; charset="utf-8"', "file_extension": ".custom"}
            if factory is ReporterScaffold
            else {"category": 'custom "quoted" category'}
            if factory is ValidatorScaffold
            else {}
        ),
    )
    result = factory().generate(config)
    assert result.success, result.errors
    paths = result.write_files(tmp_path)
    assert paths
    for path in paths:
        if path.suffix == ".py":
            compile(path.read_text(), str(path), "exec")
    test_files = [
        str(path) for path in paths if path.name.startswith("test_") and path.suffix == ".py"
    ]
    assert test_files
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(tmp_path / "src"), str(Path(__file__).resolve().parents[1] / "src")]
    )
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *test_files],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "skipped" not in completed.stdout
    examples = [path for path in paths if "examples" in path.parts and path.suffix == ".py"]
    assert examples, "include_examples must produce runnable examples"
    for example in examples:
        executed = subprocess.run(
            [sys.executable, str(example)],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert executed.returncode == 0, executed.stdout + executed.stderr
