"""Fail-closed partition and actual-execution regressions, not timing mocks."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

CI = Path(__file__).resolve().parents[1] / "verification" / "ci"
sys.path.insert(0, str(CI))
SPEC = importlib.util.spec_from_file_location("nightly_plan", CI / "nightly_plan.py")
assert SPEC and SPEC.loader
nightly = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nightly)
SHA = "a" * 40


def collection():
    return [
        {
            "nodeid": f"tests/test_contract_{i}.py::test_boundary",
            "markers": ["contract"],
            "skip_reason": None,
        }
        for i in range(4)
    ] + [
        {
            "nodeid": "tests/test_fault.py::test_recovery",
            "markers": ["contract", "fault", "e2e"],
            "skip_reason": None,
        },
        {
            "nodeid": "tests/integration/test_provider.py::test_roundtrip",
            "markers": ["integration"],
            "skip_reason": None,
        },
        {
            "nodeid": "tests/stress/test_load.py::test_load",
            "markers": ["stress", "soak", "integration"],
            "skip_reason": None,
        },
        {
            "nodeid": "tests/test_scale.py::test_volume",
            "markers": ["scale_100m", "slow", "soak"],
            "skip_reason": None,
        },
    ]


def execution(plan, lane):
    ids = plan["lanes"][lane]["nodes"]
    return {
        "sha": SHA,
        "run_id": "42",
        "plan_digest": plan["digest"],
        "lane": lane,
        "collected": ids,
        "exit_code": 0,
        "reports": [
            {
                "nodeid": node,
                "when": phase,
                "outcome": "passed",
                "duration": 0.01,
                "skip_reason": None,
                "xfail": False,
            }
            for node in ids
            for phase in ("setup", "call", "teardown")
        ],
    }


def junit(tmp_path, count=1, extra=""):
    path = tmp_path / "junit.xml"
    path.write_text(
        "<testsuites><testsuite>"
        + '<testcase name="boundary"/>' * count
        + extra
        + "</testsuite></testsuites>"
    )
    return path


def test_overlapping_markers_have_one_cost_owner_and_four_contract_shards():
    plan = nightly.build_plan(collection(), SHA)
    nightly.validate_plan(plan)
    lanes = plan["lanes"]
    assert len(lanes) == 8
    assert lanes["scale-100m"]["nodes"] == ["tests/test_scale.py::test_volume"]
    assert lanes["soak-stress"]["nodes"] == ["tests/stress/test_load.py::test_load"]
    assert lanes["fault-e2e"]["nodes"] == ["tests/test_fault.py::test_recovery"]
    nodes = [n for lane in lanes.values() for n in lane["nodes"]]
    assert len(nodes) == len(set(nodes)) == len(collection())
    assert set(nodes) == {r["nodeid"] for r in collection()}
    assert plan == nightly.build_plan(list(reversed(collection())), SHA)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra", "wrong-sha", "empty"])
def test_corrupt_or_unbound_plan_never_passes(mutation):
    plan = nightly.build_plan(collection(), SHA)
    if mutation == "missing":
        plan["lanes"]["scale-100m"]["nodes"] = []
    if mutation == "duplicate":
        plan["collection"].append(copy.deepcopy(plan["collection"][0]))
    if mutation == "extra":
        plan["lanes"]["integration"]["nodes"].append("tests/test_other.py::test_extra")
    if mutation == "wrong-sha":
        plan["sha"] = "branch-name"
    if mutation == "empty":
        plan["collection"] = []
    with pytest.raises(ValueError):
        nightly.validate_plan(plan)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-call",
        "duplicate-call",
        "skip",
        "failed",
        "cancelled",
        "wrong-sha",
        "wrong-plan",
        "xfail",
        "deselected",
    ],
)
def test_missing_cancelled_skipped_or_mismatched_execution_is_not_success(tmp_path, mutation):
    plan = nightly.build_plan(collection(), SHA)
    result = execution(plan, "scale-100m")
    if mutation == "missing-call":
        result["reports"] = result["reports"][:1]
    if mutation == "duplicate-call":
        result["reports"].append(copy.deepcopy(result["reports"][1]))
    if mutation == "skip":
        result["reports"][1]["outcome"] = "skipped"
    if mutation == "failed":
        result["reports"][1]["outcome"] = "failed"
    if mutation == "cancelled":
        result["exit_code"] = 2
    if mutation == "wrong-sha":
        result["sha"] = "b" * 40
    if mutation == "wrong-plan":
        result["plan_digest"] = "0" * 64
    if mutation == "xfail":
        result["reports"][1]["xfail"] = True
    if mutation == "deselected":
        result["collected"] = []
    with pytest.raises(ValueError):
        nightly.verify_lane(plan, "scale-100m", result, junit(tmp_path), run_id="42")


def test_full_execution_requires_both_exact_reports_and_junit(tmp_path):
    plan = nightly.build_plan(collection(), SHA)
    result = execution(plan, "scale-100m")
    assert (
        nightly.verify_lane(plan, "scale-100m", result, junit(tmp_path), run_id="42")["passed"] == 1
    )
    with pytest.raises(ValueError):
        nightly.verify_lane(plan, "scale-100m", result, junit(tmp_path, count=0), run_id="42")
    with pytest.raises(ValueError):
        nightly.verify_lane(
            plan, "scale-100m", result, junit(tmp_path, extra="<failure/>"), run_id="42"
        )
    with pytest.raises(ValueError):
        nightly.verify_lane(plan, "scale-100m", result, junit(tmp_path), run_id="43")


def test_explicit_optional_static_skip_is_counted_as_skip_never_pass(tmp_path):
    items = collection()
    items[-3]["skip_reason"] = "optional backend is not installed"
    plan = nightly.build_plan(items, SHA)
    result = execution(plan, "integration")
    result["reports"][0].update(outcome="skipped", skip_reason="optional backend is not installed")
    result["reports"] = [result["reports"][0], result["reports"][2]]
    path = tmp_path / "skip.xml"
    path.write_text(
        "<testsuites><testsuite><testcase><skipped/></testcase></testsuite></testsuites>"
    )
    verified = nightly.verify_lane(plan, "integration", result, path, run_id="42")
    assert verified["passed"] == 0 and verified["expected_skipped"] == 1
    # Scale and MCP requirements cannot gain a skip waiver from source marks.
    items[-1]["skip_reason"] = "optional backend is not installed"
    with pytest.raises(ValueError):
        nightly.build_plan(items, SHA)
