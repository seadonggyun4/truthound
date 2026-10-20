"""Exercise archive parsing and fail-closed release evidence, without network."""

import hashlib
import io
import json
import zipfile

import pytest

from tests.test_mcp_release_ci_regressions import load
from tests.test_nightly_ci import SHA, execution, nightly


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "weekly",
        "flags",
        "digest",
        "duplicate-basename",
        "missing-junit",
        "failed",
        "drift",
        "missing-sdk",
        "summary-incomplete",
        "summary-count",
    ],
)
def test_actual_archive_gate_requires_full_execution_and_digest(tmp_path, monkeypatch, mutation):
    gate = load("archive_publish_gate", "verification/mcp_publish_gate.py")
    records = [
        dict(
            nodeid="tests/test_synthetic.py::test_boundary", markers=["contract"], skip_reason=None
        )
    ]
    plan = nightly.build_plan(records, SHA)
    receipt = execution(plan, "contract-0")
    environment = dict(
        pytest="9.0.2",
        polars="1.37.1",
        truthound="3.2.0",
        mcp="1.30.0",
        **{"pytest-timeout": "2.4.0"},
    )
    receipt["environment"] = environment.copy()
    junit = b'<testsuites><testsuite tests="1"><testcase name="boundary"/></testsuite></testsuites>'
    report = tmp_path / "one.xml"
    report.write_bytes(junit)
    summary = dict(
        sha=SHA,
        mode="manual-full",
        complete=True,
        overlap=0,
        selected=1,
        plan_digest=plan["digest"],
        lanes={"contract-0": nightly.verify_lane(plan, "contract-0", receipt, report, run_id="42")},
    )
    if mutation == "weekly":
        plan["mode"] = "weekly-integration-core"
    elif mutation == "flags":
        plan["flags"] = ["--run-integration"]
    elif mutation == "failed":
        receipt["exit_code"] = 1
    elif mutation == "drift":
        receipt["environment"]["mcp"] = "different"
    elif mutation == "missing-sdk":
        environment["mcp"] = None
    elif mutation == "summary-incomplete":
        summary["complete"] = False
    elif mutation == "summary-count":
        summary["selected"] = 0
    files = {
        "plan.json": json.dumps(plan).encode(),
        "collection-environment.json": json.dumps(dict(environment=environment)).encode(),
        "execution-contract-0.json": json.dumps(receipt).encode(),
        "nightly-summary.json": json.dumps(summary).encode(),
        "junit-contract-0.xml": junit,
    }
    if mutation == "missing-junit":
        del files["junit-contract-0.xml"]
    if mutation == "duplicate-basename":
        files["collision/plan.json"] = files["plan.json"]
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, data in files.items():
            bundle.writestr(name, data)
    raw = archive.getvalue()
    metadata = dict(
        total_count=1,
        artifacts=[
            dict(
                id=123,
                name="truthound-nightly-test-artifacts",
                expired=False,
                size_in_bytes=len(raw),
                digest="sha256:" + hashlib.sha256(raw).hexdigest(),
            )
        ],
    )
    if mutation == "digest":
        metadata["artifacts"][0]["digest"] = "sha256:" + "0" * 64
    monkeypatch.setattr(
        gate.subprocess,
        "check_output",
        lambda command, **kwargs: raw
        if command[-1].endswith("/zip")
        else json.dumps(metadata).encode(),
    )
    if mutation is None:
        assert gate.verify_full_artifact(42, SHA)["contract-0"]["passed"] == 1
    else:
        with pytest.raises((AssertionError, ValueError, KeyError)):
            gate.verify_full_artifact(42, SHA)
