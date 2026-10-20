"""Read-only release identity checks plus actual MCP contracts, with bounded output.

Use after publication. This script does not upload, dispatch, authenticate or
claim approval: it checks public GitHub/PyPI metadata for one immutable commit.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


def read_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "truthound-release-contract"})
    with urllib.request.urlopen(request, timeout=5) as stream:
        raw = stream.read(2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError("Metadata exceeds contract limit")
    return json.loads(raw)


REQUIRED_WORKFLOWS = {
    ".github/workflows/" + name + ".yml"
    for name in (
        "tests-pr",
        "docs",
        "benchmarks-pr",
        "mcp",
        "tests-nightly",
        "benchmarks-nightly",
        "benchmarks-release",
        "release-pypi",
    )
}


def validate_workflows(runs, sha, *, require_release=True):
    required = REQUIRED_WORKFLOWS - (
        set() if require_release else {".github/workflows/release-pypi.yml"}
    )
    applicable = [r for r in runs if r.get("path") in required]
    if required != {r["path"] for r in applicable}:
        raise ValueError("missing_required_workflow")
    if any(
        r.get("head_sha") != sha
        or r.get("status") != "completed"
        or r.get("conclusion") != "success"
        for r in applicable
    ):
        raise ValueError("failed_cancelled_pending_or_wrong_sha_workflow")
    full = [
        r
        for r in applicable
        if r["path"] == ".github/workflows/tests-nightly.yml"
        and r.get("event") == "workflow_dispatch"
        and "manual-full" in r.get("display_title", "")
    ]
    if not full:
        raise ValueError("missing_actual_manual_full")
    return full


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", help="Exact commit; defaults to this source checkout HEAD")
    parser.add_argument("--version", default="3.2.0")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.commit is None:
        args.commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit) or not re.fullmatch(
        r"\d+\.\d+\.\d+", args.version
    ):
        parser.error("Expected an immutable commit and stable numeric version")
    results = dict.fromkeys(("api", "safety", "package", "docs"), "FAIL")
    try:
        with tempfile.TemporaryDirectory(prefix="truthound-release-contract-") as td:
            junit = Path(td) / "result.xml"
            run = subprocess.run(
                [
                    str(root / ".venv/bin/python"),
                    "verification/mcp_ci_gate.py",
                    "--junitxml",
                    str(junit),
                ],
                cwd=root,
                capture_output=True,
                timeout=45,
            )
            suites = ET.parse(junit).getroot().findall("testsuite")
            assert run.returncode == 0 and suites
            assert sum(int(s.get("tests", 0)) for s in suites) >= 45
            assert all(int(s.get("skipped", 0)) == 0 for s in suites)
            results["api"] = "PASS"
            ci_junit = Path(td) / "ci.xml"
            ci_run = subprocess.run(
                [
                    str(root / ".venv/bin/python"),
                    "-m",
                    "pytest",
                    "-q",
                    "tests/test_nightly_ci.py",
                    "tests/test_nightly_ci_runtime.py",
                    "-p",
                    "no:cacheprovider",
                    f"--junitxml={ci_junit}",
                ],
                cwd=root,
                capture_output=True,
                timeout=15,
            )
            ci_cases = list(ET.parse(ci_junit).getroot().iter("testcase"))
            assert ci_run.returncode == 0 and len(ci_cases) >= 21
            assert not any(
                case.find(tag) is not None
                for case in ci_cases
                for tag in ("skipped", "failure", "error")
            )
        api = "https://api.github.com/repos/seadonggyun4/truthound"
        workflows = read_json(f"{api}/actions/runs?head_sha={args.commit}&per_page=100")[
            "workflow_runs"
        ]
        full_runs = validate_workflows(workflows, args.commit)
        required_jobs = {
            "nightly-collect",
            "nightly-gate",
            "nightly / fault-e2e",
            "nightly / integration",
            "nightly / soak-stress",
            "nightly / scale-100m",
        } | {f"nightly / contract-{i}" for i in range(4)}
        for full_run in full_runs:
            jobs = read_json(f"{api}/actions/runs/{full_run['id']}/jobs?per_page=100")["jobs"]
            assert required_jobs == {job["name"] for job in jobs}
            assert all(
                job["status"] == "completed" and job["conclusion"] == "success" for job in jobs
            )
        results["safety"] = "PASS"
        package = read_json(f"https://pypi.org/pypi/truthound/{args.version}/json")
        assert package["info"]["version"] == args.version
        files = {item["filename"]: item for item in package["urls"]}
        wheel = f"truthound-{args.version}-py3-none-any.whl"
        sdist = f"truthound-{args.version}.tar.gz"
        assert {wheel, sdist} <= files.keys()
        assert not any(item["yanked"] for item in files.values())
        results["package"] = "PASS"
        release = read_json(f"{api}/releases/tags/v{args.version}")
        assert not release["draft"] and not release["prerelease"]
        tag = read_json(f"{api}/git/ref/tags/v{args.version}")["object"]
        if tag["type"] == "tag":
            tag = read_json(f"{api}/git/tags/{tag['sha']}")["object"]
        assert tag["type"] == "commit" and tag["sha"] == args.commit
        assets = {item["name"]: item for item in release["assets"]}
        assert {
            wheel,
            sdist,
            f"truthound-{args.version}.mcpb",
            f"truthound-{args.version}-local.mcpb",
            "SHA256SUMS",
        } <= assets.keys()
        for name in (wheel, sdist):
            assert assets[name]["digest"] == "sha256:" + files[name]["digests"]["sha256"]
        results["docs"] = "PASS"
    except Exception:
        # Never echo dependency errors, file data, remote bodies or raw test logs.
        pass
    print(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [{"id": key, "status": value} for key, value in results.items()],
            }
        )
    )
    return 0 if all(value == "PASS" for value in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
