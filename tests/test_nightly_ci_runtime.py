"""Exercise the actual existing manifest runner, not just fabricated reports."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / "verification/ci"
sys.path.insert(0, str(CI))
SPEC = importlib.util.spec_from_file_location("nightly_plan", CI / "nightly_plan.py")
assert SPEC and SPEC.loader
nightly = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nightly)


@pytest.mark.parametrize("scenario", ["normal", "failure", "timeout", "wrong-sha"])
def test_real_manifest_executor_preserves_failure_timeout_and_identity(tmp_path, scenario):
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    name = "test_required_failure" if scenario == "failure" else "test_boundary"
    nodeid = f"tests/fixtures/nightly_case.py::{name}"
    plan = nightly.build_plan([dict(nodeid=nodeid, markers=["contract"], skip_reason=None)], sha)
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "manifest.txt").write_text(nodeid + "\n")
    output = tmp_path / "execution.json"
    junit = tmp_path / "junit.xml"
    env = {
        **os.environ,
        "GITHUB_SHA": "b" * 40 if scenario == "wrong-sha" else sha,
        "GITHUB_RUN_ID": "synthetic42",
        "NIGHTLY_CASE_SECONDS": "5" if scenario == "timeout" else "0.3",
    }
    result = subprocess.run(
        [
            sys.executable,
            str(CI / "run_pytest_manifest.py"),
            "--manifest",
            str(tmp_path / "manifest.txt"),
            "--nightly-plan",
            str(tmp_path / "plan.json"),
            "--lane",
            "contract-0",
            "--execution-report",
            str(output),
            "--junitxml",
            str(junit),
            "--pytest-arg=--timeout=1",
            "--pytest-arg=--timeout-method=thread",
            "--pytest-arg=-o",
            "--pytest-arg=faulthandler_timeout=0.1",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if scenario == "normal":
        assert result.returncode == 0, result.stderr
        assert (
            nightly.verify_lane(
                plan, "contract-0", json.loads(output.read_text()), junit, run_id="synthetic42"
            )["passed"]
            == 1
        )
        # Stack emission is diagnostic, not evidence of an enforced deadline.
        assert "Timeout" in result.stderr
    else:
        assert result.returncode != 0
        if scenario == "failure":
            with pytest.raises(ValueError):
                nightly.verify_lane(
                    plan, "contract-0", json.loads(output.read_text()), junit, run_id="synthetic42"
                )
        elif scenario == "timeout":
            assert not output.exists()
            assert '"event": "start"' in output.with_suffix(".ndjson").read_text()
        else:
            assert not output.exists()
