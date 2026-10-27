"""Remove only the temporary benchmark runtime claimed by this workflow run."""

import argparse
import json
import os
import re
import shutil
from pathlib import Path


def runtime_identity(environment):
    run, attempt, sha = (environment[key] for key in (
        "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_SHA"
    ))
    if (
        not run.isascii() or not run.isdecimal()
        or not attempt.isascii() or not attempt.isdecimal()
        or re.fullmatch(r"[0-9a-f]{40}", sha) is None
    ):
        raise ValueError("invalid_benchmark_runtime_identity")
    temporary = Path(environment["RUNNER_TEMP"]).resolve(strict=True)
    root = temporary / f"truthound-release-{run}-{attempt}"
    selected = Path(environment["RELEASE_VENV"])
    if (
        selected != root / "venv" or root.is_symlink() or selected.is_symlink()
        or root.resolve() != root or selected.resolve() != selected
    ):
        raise ValueError("unowned_benchmark_runtime")
    return root, {"run": run, "attempt": attempt, "sha": sha}


def claim(environment):
    root, identity = runtime_identity(environment)
    if not (root / "venv").is_dir():
        raise ValueError("missing_benchmark_runtime")
    with (root / ".truthound-owned-runtime").open("x", encoding="utf-8") as output:
        json.dump(identity, output)


def cleanup(environment):
    root, identity = runtime_identity(environment)
    marker = root / ".truthound-owned-runtime"
    if marker.is_symlink() or json.loads(marker.read_text(encoding="utf-8")) != identity:
        raise ValueError("benchmark_runtime_owner_mismatch")
    shutil.rmtree(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("claim", "cleanup"))
    operation = parser.parse_args().operation
    (claim if operation == "claim" else cleanup)(os.environ)
    print("Owned benchmark runtime " + ("claimed" if operation == "claim" else "removed"))
