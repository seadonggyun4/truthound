from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = Path(__file__).with_name("ruff_ratchet_targets.toml")


@dataclass(frozen=True)
class RuffRatchetTarget:
    name: str
    paths: tuple[str, ...]


def load_targets(manifest_path: Path = DEFAULT_MANIFEST_PATH) -> list[RuffRatchetTarget]:
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    raw_targets = manifest.get("target", [])
    targets: list[RuffRatchetTarget] = []
    seen_names: set[str] = set()

    for raw_target in raw_targets:
        name = raw_target["name"]
        paths = tuple(raw_target["paths"])
        if name in seen_names:
            raise ValueError(f"Duplicate ruff ratchet target '{name}' in {manifest_path}")
        if not paths:
            raise ValueError(f"Ruff ratchet target '{name}' does not define any paths")
        seen_names.add(name)
        targets.append(RuffRatchetTarget(name=name, paths=paths))

    if not targets:
        raise ValueError(f"Ruff ratchet manifest {manifest_path} does not define any targets")

    return targets


def resolve_target(target_name: str, targets: list[RuffRatchetTarget]) -> RuffRatchetTarget:
    for target in targets:
        if target.name == target_name:
            return target
    available = ", ".join(target.name for target in targets)
    raise ValueError(
        f"Unknown ruff ratchet target '{target_name}'. Available targets: {available}"
    )


def verify_target_paths(target: RuffRatchetTarget, repo_root: Path = REPO_ROOT) -> None:
    missing = [path for path in target.paths if not (repo_root / path).exists()]
    if missing:
        missing_paths = ", ".join(missing)
        raise FileNotFoundError(
            f"Ruff ratchet target '{target.name}' references missing paths: {missing_paths}"
        )


def run_target(target: RuffRatchetTarget, repo_root: Path = REPO_ROOT) -> int:
    verify_target_paths(target, repo_root=repo_root)
    print(f"Running ruff ratchet target '{target.name}'")
    for path in target.paths:
        print(f"  - {path}")
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", *target.paths],
        cwd=repo_root,
        check=False,
        text=True,
    )
    return result.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run folder-level ruff ratchets for clean, contract-boundary surfaces."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target", help="Run a single named ruff ratchet target")
    group.add_argument("--all", action="store_true", help="Run every configured ratchet target")
    group.add_argument("--list", action="store_true", help="List configured ratchet targets")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        targets = load_targets()
        if args.list:
            for target in targets:
                print(target.name)
                for path in target.paths:
                    print(f"  - {path}")
            return 0

        selected = targets if args.all else [resolve_target(args.target, targets)]
        exit_code = 0
        for target in selected:
            exit_code = max(exit_code, run_target(target))
        return exit_code
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
