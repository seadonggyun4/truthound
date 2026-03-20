from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest


def _load_manifest(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pytest directly from a manifest of collected node ids.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--junitxml", type=Path, required=True)
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="Additional pytest argument to pass before the node ids.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    targets = _load_manifest(args.manifest)
    if not targets:
        print(
            f"error: manifest {args.manifest} does not contain any pytest targets",
            file=sys.stderr,
        )
        return 2

    args.junitxml.parent.mkdir(parents=True, exist_ok=True)
    files = {target.split('::', 1)[0] for target in targets}
    print(
        f"Running pytest manifest {args.manifest} with {len(targets)} targets "
        f"from {len(files)} files",
    )

    pytest_args = [
        "-q",
        "-p",
        "no:cacheprovider",
        f"--junitxml={args.junitxml}",
        *args.pytest_arg,
        *targets,
    ]
    return int(pytest.main(pytest_args))


if __name__ == "__main__":
    raise SystemExit(main())
