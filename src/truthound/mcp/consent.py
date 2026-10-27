"""Private operator configuration for the installable Core plugin.

Configuration changes invalidate the entire session, including retained handles.
The adapter grants capabilities only; computation remains in Core.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import stat
import tempfile
from contextlib import suppress
from pathlib import Path

from truthound.mcp.policy import MCPError, MCPPolicy

MAX_BYTES = 65536


def config_path():
    path = Path(
        os.environ.get(
            "TRUTHOUND_CORE_CONFIG", Path.home() / ".truthound-core" / "permissions.json"
        )
    )
    if not path.is_absolute():
        raise ValueError("absolute_config_required")
    return path


def read_config(path):
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("unsafe_permissions")
    try:
        fd = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
    except FileNotFoundError:
        return None
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BYTES:
            raise ValueError("unsafe_permissions")
        if os.name != "nt" and (info.st_uid != os.getuid() or info.st_mode & 0o077):
            raise ValueError("unsafe_permissions")
        payload = stream.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise ValueError("unsafe_permissions")
    return payload


def decode_config(payload):
    value = json.loads(payload)
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "allowed_directories",
        "workspace_directories",
        "control_directory",
        "allow_sensitive_results",
    }:
        raise ValueError("invalid_permissions")
    if type(value["schema_version"]) is not int or value["schema_version"] != 2:
        raise ValueError("invalid_permissions")
    if type(value["allow_sensitive_results"]) is not bool:
        raise ValueError("invalid_permissions")
    for key in ("allowed_directories", "workspace_directories"):
        roots = value[key]
        if not isinstance(roots, list) or len(roots) > 16:
            raise ValueError("invalid_permissions")
        for root in roots:
            validate_directory(root)
    control = value["control_directory"]
    if control is not None:
        validate_directory(control)
    return value


def validate_directory(value):
    if not isinstance(value, str) or not value:
        raise ValueError("invalid_directory")
    path = Path(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or path == Path(path.anchor)
        or path.is_symlink()
        or any(parent.is_symlink() for parent in path.parents)
        or not path.is_dir()
    ):
        raise ValueError("invalid_directory")
    return path


def policy_from(payload, empty):
    if payload is None:
        return MCPPolicy((empty,), profile="core")
    value = decode_config(payload)
    return MCPPolicy(
        tuple(Path(root) for root in value["allowed_directories"]) or (empty,),
        profile="core",
        allow_sensitive_results=value["allow_sensitive_results"],
        workspace_directories=tuple(Path(root) for root in value["workspace_directories"]),
        control_directory=Path(value["control_directory"]) if value["control_directory"] else None,
    )


def write_config(path, value):
    if not path.is_absolute() or path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError("unsafe_permissions")
    payload = (json.dumps(value) + "\n").encode()
    if len(payload) > MAX_BYTES:
        raise ValueError("unsafe_permissions")
    # Validate all grants before replacing the current configuration.
    with tempfile.TemporaryDirectory(prefix="truthound-consent-check-") as empty:
        policy_from(payload, Path(empty))
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="permissions-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class ConsentPolicy:
    def __init__(self, path, empty):
        self.path = path
        self.snapshot = read_config(path)
        self.delegate = policy_from(self.snapshot, empty)

    def unchanged(self):
        try:
            return read_config(self.path) == self.snapshot
        except (OSError, ValueError):
            return False

    def __getattr__(self, name):
        if not self.unchanged():
            raise MCPError("access_denied")
        return getattr(self.delegate, name)


async def run(path):
    from truthound.mcp.server import serve

    with tempfile.TemporaryDirectory(prefix="truthound-unconfigured-") as directory:
        policy = ConsentPolicy(path, Path(directory))
        task = asyncio.create_task(serve(policy))
        try:
            while not task.done():
                await asyncio.sleep(0.1)
                if not policy.unchanged():
                    task.cancel()
                    break
        finally:
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError):
                await task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("serve", "configure", "revoke"))
    parser.add_argument("--config", type=Path, default=config_path())
    parser.add_argument("--allow-dir", action="append", default=[])
    parser.add_argument("--workspace-dir", action="append", default=[])
    parser.add_argument("--control-dir")
    parser.add_argument("--allow-sensitive-results", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if args.command != "configure" and (
        args.allow_dir or args.workspace_dir or args.control_dir or args.allow_sensitive_results
    ):
        parser.error("Grants require the configure command")
    try:
        if args.command == "serve":
            asyncio.run(run(args.config))
        else:
            if args.command == "configure" and not args.confirm:
                parser.error("Explicit operator confirmation is required")
            write_config(
                args.config,
                {
                    "schema_version": 2,
                    "allowed_directories": args.allow_dir,
                    "workspace_directories": args.workspace_dir,
                    "control_directory": args.control_dir,
                    "allow_sensitive_results": args.allow_sensitive_results,
                },
            )
            print("Core consent saved; reconnect the MCP client.")
    except (OSError, ValueError, RecursionError, MCPError):
        parser.error("Invalid Core consent; no data was inspected")


if __name__ == "__main__":
    main()
