"""Deterministic full-suite partition and fail-closed execution evidence.

Uses the PR file-balancing algorithm and manifest executor. No test computation
or service is shared across jobs. A plan is evidence, not an execution result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from build_quality_shards import _build_contract_shards

FULL = "contract or fault or integration or soak or stress or scale_100m or e2e"
FLAGS = ["--run-integration", "--run-expensive", "--run-soak"]
BUDGETS = {
    "contract": (35, 120),
    "fault-e2e": (25, 180),
    "integration": (35, 180),
    "soak-stress": (35, 300),
    "scale-100m": (120, 600),
}


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def owner(markers: list[str]) -> str:
    marks = set(markers)
    if "scale_100m" in marks:
        return "scale-100m"
    if marks & {"soak", "stress"}:
        return "soak-stress"
    if "integration" in marks:
        return "integration"
    if marks & {"fault", "e2e"}:
        return "fault-e2e"
    if "contract" in marks:
        return "contract"
    raise ValueError("unassigned_full_test")


def build_plan(
    records: list[dict[str, Any]], sha: str, mode: str = "manual-full"
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", sha) or mode not in {
        "manual-full",
        "weekly-integration-core",
    }:
        raise ValueError("invalid_plan_identity")
    records = sorted(records, key=lambda r: r["nodeid"])
    ids = [r["nodeid"] for r in records]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("empty_or_duplicate_collection")
    groups = {key: [] for key in BUDGETS}
    for record in records:
        if not record["nodeid"].startswith("tests/") or "::" not in record["nodeid"]:
            raise ValueError("invalid_node_identity")
        if "manual_verification" in record["markers"]:
            raise ValueError("manual_test_in_full")
        lane = owner(record["markers"])
        if mode != "manual-full" and "integration" not in record["markers"]:
            raise ValueError("non_integration_in_weekly")
        if record["skip_reason"] and (
            lane == "scale-100m"
            or Path(record["nodeid"].split("::")[0]).name.startswith("test_mcp")
        ):
            raise ValueError("required_boundary_skip")
        groups[lane].append(record["nodeid"])
    lanes = {}
    for key, nodes in groups.items():
        if not nodes:
            continue
        if key == "contract":
            count = min(4, len({n.split("::")[0] for n in nodes}))
            partitions = {
                f"contract-{s.shard_id}": s.nodeids
                for s in _build_contract_shards(nodes, shard_count=count)
            }
        else:
            partitions = {key: nodes}
        for lane, selected in partitions.items():
            minutes, test_seconds = BUDGETS[key]
            lanes[lane] = dict(
                nodes=selected,
                job_minutes=minutes,
                test_seconds=test_seconds,
                stack_seconds=min(120, test_seconds // 2),
                runner="ubuntu-latest",
            )
    selected = [n for lane in lanes.values() for n in lane["nodes"]]
    if len(selected) != len(set(selected)) or set(selected) != set(ids):
        raise ValueError("incomplete_or_overlapping_partition")
    plan = dict(
        schema_version=1,
        sha=sha,
        mode=mode,
        marker_expression=FULL if mode == "manual-full" else "integration",
        flags=FLAGS if mode == "manual-full" else ["--run-integration"],
        collection=records,
        lanes=lanes,
    )
    return {**plan, "digest": digest(plan)}


def validate_plan(plan: dict[str, Any]) -> None:
    expected = build_plan(plan["collection"], plan["sha"], plan["mode"])
    if expected != plan:
        raise ValueError("altered_or_incomplete_plan")


def verify_lane(
    plan: dict[str, Any], lane: str, result: dict[str, Any], junit: Path, *, run_id: str
) -> dict[str, int]:
    validate_plan(plan)
    wanted = plan["lanes"][lane]["nodes"]
    if (
        result.get("sha") != plan["sha"]
        or result.get("plan_digest") != plan["digest"]
        or result.get("run_id") != run_id
        or result.get("lane") != lane
        or result.get("exit_code") != 0
    ):
        raise ValueError("failed_or_mismatched_execution")
    if sorted(result["collected"]) != sorted(wanted):
        raise ValueError("execution_collection_drift")
    expected_sources = {
        r["nodeid"]: r["source"]
        for r in plan["collection"]
        if r["nodeid"] in wanted and "source" in r
    }
    if expected_sources and result.get("sources") != expected_sources:
        raise ValueError("execution_source_identity_drift")
    if result.get("collection_errors"):
        raise ValueError("failed_collection")
    reports = {}
    for report in result["reports"]:
        key = (report["nodeid"], report["when"])
        if (
            key in reports
            or report["nodeid"] not in wanted
            or report["when"] not in {"setup", "call", "teardown"}
        ):
            raise ValueError("duplicate_or_extra_phase")
        if report["outcome"] not in {"passed", "skipped"} or report.get("xfail"):
            raise ValueError("failed_or_xfailed_phase")
        reports[key] = report
    expected_skips = {r["nodeid"]: r["skip_reason"] for r in plan["collection"] if r["skip_reason"]}
    skipped = 0
    for node in wanted:
        phases = {
            phase: reports[(node, phase)]
            for phase in ("setup", "call", "teardown")
            if (node, phase) in reports
        }
        if node in expected_skips:
            if (
                set(phases) != {"setup", "teardown"}
                or phases["setup"]["outcome"] != "skipped"
                or phases["setup"]["skip_reason"] != expected_skips[node]
            ):
                raise ValueError("expected_skip_changed")
            if phases["teardown"]["outcome"] != "passed":
                raise ValueError("failed_skip_teardown")
            skipped += 1
        elif set(phases) != {"setup", "call", "teardown"} or any(
            r["outcome"] != "passed" for r in phases.values()
        ):
            raise ValueError("missing_or_skipped_required_phase")
    root = ET.parse(junit).getroot()
    if list(root.iter("failure")) or list(root.iter("error")):
        raise ValueError("failed_junit")
    if (
        len(list(root.iter("testcase"))) != len(wanted)
        or len(list(root.iter("skipped"))) != skipped
    ):
        raise ValueError("junit_execution_disagrees")
    return dict(selected=len(wanted), passed=len(wanted) - skipped, expected_skipped=skipped)


def environment() -> dict[str, str | None]:
    names = ["pytest", "polars", "truthound", "mcp", "pytest-timeout"]
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def require_environment(observed):
    if any(not value for value in observed.values()):
        raise ValueError("missing_required_nightly_dependency")


def source_identity(item):
    root = Path(__file__).resolve().parents[2]
    collected = Path(item.path).resolve()
    module = Path(item.module.__file__).resolve()
    if module != collected:
        raise ValueError("collected_module_source_mismatch:" + item.nodeid)
    return dict(
        path=collected.relative_to(root).as_posix(),
        sha256=hashlib.sha256(collected.read_bytes()).hexdigest(),
    )


class Collection:
    def __init__(self):
        self.records = []
        self.module_skips = []

    def pytest_collection_finish(self, session):
        # Use pytest's own skip evaluation, never a second eval implementation.
        from _pytest.skipping import evaluate_skip_marks

        for item in session.items:
            skip = evaluate_skip_marks(item)
            self.records.append(
                dict(
                    nodeid=item.nodeid,
                    markers=sorted({m.name for m in item.iter_markers()}),
                    skip_reason=skip.reason if skip else None,
                    source=source_identity(item),
                )
            )

    def pytest_collectreport(self, report):
        if report.skipped:
            self.module_skips.append(report.nodeid)


class Execution:
    def __init__(self, plan, lane, output):
        import pytest

        self.plan, self.lane, self.output = plan, lane, output
        self.reports = []
        self.collected = []
        self.sources = {}
        self.collection_errors = []
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.output.with_suffix(".ndjson").open("w", encoding="utf-8", buffering=1)
        self.pytest = pytest

    def pytest_collection_finish(self, session):
        self.collected = [item.nodeid for item in session.items]
        # Preserve the original collection error instead of masking it in finally.
        if session.testsfailed or self.collection_errors:
            return
        self.sources = {item.nodeid: source_identity(item) for item in session.items}
        expected = {r["nodeid"]: r.get("source") for r in self.plan["collection"]}
        if any(
            expected[node] is not None and expected[node] != source
            for node, source in self.sources.items()
        ):
            raise self.pytest.UsageError("nightly execution source identity differs from plan")
        if sorted(self.collected) != sorted(self.plan["lanes"][self.lane]["nodes"]):
            raise self.pytest.UsageError(
                "nightly execution collection differs from approved manifest"
            )

    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_errors.append(report.nodeid)

    def pytest_runtest_logstart(self, nodeid, location):
        self.stream.write(json.dumps(dict(nodeid=nodeid, event="start")) + "\n")
        print(f"NIGHTLY_START {nodeid}", flush=True)

    def pytest_runtest_logreport(self, report):
        reason = None
        if report.skipped and isinstance(report.longrepr, tuple):
            reason = str(report.longrepr[2]).removeprefix("Skipped: ")
        record = dict(
            nodeid=report.nodeid,
            when=report.when,
            outcome=report.outcome,
            duration=report.duration,
            skip_reason=reason,
            xfail=bool(getattr(report, "wasxfail", False)),
        )
        self.reports.append(record)
        self.stream.write(json.dumps(record) + "\n")

    def pytest_sessionfinish(self, session, exitstatus):
        self.stream.close()
        self.output.write_text(
            json.dumps(
                dict(
                    schema_version=1,
                    sha=self.plan["sha"],
                    run_id=os.environ.get("GITHUB_RUN_ID", "local"),
                    plan_digest=self.plan["digest"],
                    lane=self.lane,
                    collected=self.collected,
                    sources=self.sources,
                    collection_errors=self.collection_errors,
                    reports=self.reports,
                    exit_code=int(exitstatus),
                    environment=environment(),
                ),
                indent=2,
            ),
            encoding="utf-8",
        )


def resource_probe(lane):
    import psutil

    memory = psutil.virtual_memory()
    report = dict(
        cpu_count=os.cpu_count(),
        memory_total=memory.total,
        memory_available=memory.available,
        disk_free=shutil.disk_usage(".").free,
    )
    if lane == "scale-100m" and (
        report["cpu_count"] < 4
        or memory.total < 16_000_000_000
        or memory.available < 12_000_000_000
        or report["disk_free"] < 10_000_000_000
    ):
        raise ValueError("scale_runner_resources_below_documented_requirement")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["collect", "verify", "resources"])
    parser.add_argument("--root", type=Path, default=Path("test-artifacts/nightly"))
    parser.add_argument(
        "--mode", choices=["manual-full", "weekly-integration-core"], default="manual-full"
    )
    parser.add_argument("--lane", default="integration")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    sha = (
        os.environ.get("GITHUB_SHA")
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    )
    if args.operation == "collect":
        import pytest

        plugin = Collection()
        flags = FLAGS if args.mode == "manual-full" else ["--run-integration"]
        expression = FULL if args.mode == "manual-full" else "integration"
        code = pytest.main(
            ["tests", "--collect-only", "-q", "-p", "no:cacheprovider", "-m", expression, *flags],
            plugins=[plugin],
        )
        if code != 0:
            return int(code)
        plan = build_plan(plugin.records, sha, args.mode)
        require_environment(environment())
        (args.root / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        (args.root / "collection-environment.json").write_text(
            json.dumps(dict(environment=environment(), module_skips=plugin.module_skips)),
            encoding="utf-8",
        )
        matrix = []
        for lane, policy in plan["lanes"].items():
            (args.root / f"{lane}.txt").write_text(
                "\n".join(policy["nodes"]) + "\n", encoding="utf-8"
            )
            matrix.append(dict(lane=lane, **{k: v for k, v in policy.items() if k != "nodes"}))
        output = dict(
            matrix=dict(include=matrix), selected=len(plan["collection"]), digest=plan["digest"]
        )
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
                handle.write("matrix=" + json.dumps(output["matrix"], separators=(",", ":")) + "\n")
        print(json.dumps(output))
    elif args.operation == "resources":
        (args.root / f"resources-{args.lane}.json").write_text(
            json.dumps(resource_probe(args.lane)), encoding="utf-8"
        )
    else:
        plan = json.loads((args.root / "plan.json").read_text())
        validate_plan(plan)
        if plan["sha"] != sha:
            raise ValueError("gate_wrong_sha")
        if plan["mode"] != args.mode:
            raise ValueError("gate_wrong_mode")
        needs = json.loads(os.environ["NIGHTLY_NEEDS"])
        if set(needs) != {"nightly-collect", "nightly-lane"} or any(
            value["result"] != "success" for value in needs.values()
        ):
            raise ValueError("failed_cancelled_or_missing_job")
        expected_environment = json.loads((args.root / "collection-environment.json").read_text())[
            "environment"
        ]
        require_environment(expected_environment)
        summaries = {}
        for lane in plan["lanes"]:
            result = json.loads((args.root / f"execution-{lane}.json").read_text())
            if result["environment"] != expected_environment:
                raise ValueError("lane_dependency_drift")
            if not (args.root / f"resources-{lane}.json").is_file():
                raise ValueError("missing_resource_evidence")
            summaries[lane] = verify_lane(
                plan,
                lane,
                result,
                args.root / f"junit-{lane}.xml",
                run_id=os.environ["GITHUB_RUN_ID"],
            )
        expected_files = {f"execution-{lane}.json" for lane in plan["lanes"]}
        if {p.name for p in args.root.glob("execution-*.json")} != expected_files:
            raise ValueError("extra_execution_artifact")
        report = dict(
            sha=sha,
            mode=plan["mode"],
            plan_digest=plan["digest"],
            selected=len(plan["collection"]),
            lanes=summaries,
            complete=True,
            overlap=0,
        )
        (args.root / "nightly-summary.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
