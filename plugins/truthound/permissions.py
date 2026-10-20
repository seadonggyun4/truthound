"""Local consent storage only; all file validation delegates to Core's MCP policy."""

from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
from pathlib import Path

MAX_CONFIG_BYTES = 65536


def config_path() -> Path:
    # Not inside the installed plugin, repository, package or input workspace.
    path = Path(
        os.environ.get(
            "TRUTHOUND_OPENAI_CONFIG", Path.home() / ".truthound-openai" / "permissions.json"
        )
    )
    if not path.is_absolute():
        raise ValueError("absolute_config_required")
    return path


def read_directories(path: Path) -> tuple[Path, ...]:
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("unsafe_permissions")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_CONFIG_BYTES:
            raise ValueError("unsafe_permissions")
        if os.name != "nt" and (info.st_uid != os.getuid() or info.st_mode & 0o022):
            raise ValueError("unsafe_permissions")
        payload = stream.read(MAX_CONFIG_BYTES + 1)
        if len(payload) > MAX_CONFIG_BYTES:
            raise ValueError("unsafe_permissions")
        try:
            value = json.loads(payload)
        except (ValueError, RecursionError):
            raise ValueError("invalid_permissions") from None
    if not isinstance(value, dict) or set(value) != {"schema_version", "allowed_directories"}:
        raise ValueError("invalid_permissions")
    roots = value["allowed_directories"]
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("invalid_permissions")
    if (
        not isinstance(roots, list)
        or len(roots) > 16
        or any(not isinstance(p, str) or not Path(p).is_absolute() for p in roots)
    ):
        raise ValueError("invalid_permissions")
    paths = tuple(Path(p) for p in roots)
    if any(p == Path(p.anchor) or ".." in p.parts for p in paths):
        raise ValueError("unsafe_directory")
    return paths


def write_directories(path: Path, roots: list[str]) -> None:
    if not path.is_absolute() or path.parent.is_symlink() or path.is_symlink():
        raise ValueError("unsafe_permissions")
    if len(roots) > 16:
        raise ValueError("too_many_directories")
    resolved = []
    for root in roots:
        p = Path(root)
        if not p.is_absolute() or ".." in p.parts:
            raise ValueError("absolute_directory_required")
        p = p.resolve(strict=True)
        if not p.is_dir() or p == Path(p.anchor):
            raise ValueError("unsafe_directory")
        resolved.append(str(p))
    payload = (
        json.dumps({"schema_version": 1, "allowed_directories": list(dict.fromkeys(resolved))})
        + "\n"
    ).encode()
    if len(payload) > MAX_CONFIG_BYTES:
        raise ValueError("permissions_too_large")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="permissions-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace or revoke Truthound local directory consent; never inspect data."
    )
    parser.add_argument("--config", type=Path, default=config_path())
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--allow-dir", action="append")
    group.add_argument("--revoke", action="store_true")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Operator explicitly approves the exact directory list",
    )
    args = parser.parse_args()
    if args.allow_dir and not args.confirm:
        parser.error("Directory consent requires --confirm")
    try:
        write_directories(args.config, args.allow_dir or [])
    except (OSError, ValueError):
        parser.error("Invalid directory consent; no data was inspected")
    print("Directory consent saved; the next tool call uses the new policy.")


if __name__ == "__main__":
    main()
