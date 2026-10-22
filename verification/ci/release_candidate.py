"""Read-only pre-publication gate for an explicitly identified verification run."""

import argparse
import json
import os
import re
import subprocess


def verify(run, sha):
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("invalid_sha")
    if (
        run.get("head_sha") != sha
        or run.get("path") != ".github/workflows/release-verify.yml"
        or run.get("event") != "workflow_dispatch"
        or run.get("status") != "completed"
        or run.get("conclusion") != "success"
    ):
        raise ValueError("release_candidate_not_verified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id.isascii() or not args.run_id.isdecimal():
        parser.error("Expected numeric verification run ID")
    repo = os.environ["GITHUB_REPOSITORY"]
    run = json.loads(
        subprocess.check_output(
            ["gh", "api", f"repos/{repo}/actions/runs/{args.run_id}"], text=True
        )
    )
    verify(run, os.environ["GITHUB_SHA"])
    print("Same-revision release verification passed")


if __name__ == "__main__":
    main()
