"""Dedicated entry point avoids unrelated CLI plugin discovery on stdio."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="truthound-mcp")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Run the local stdio MCP server")
    serve.add_argument("--allow-dir", action="append", required=True, type=Path)
    serve.add_argument("--max-file-bytes", type=int, default=100 * 1024 * 1024)
    serve.add_argument("--max-items", type=int, default=100)
    serve.add_argument("--timeout-seconds", type=int, default=30)
    bundle = commands.add_parser("bundle", help="Build a Claude Desktop MCPB")
    bundle.add_argument("--output", required=True, type=Path)
    bundle.add_argument(
        "--wheel", type=Path, help="Bundle an exact local wheel before PyPI publication"
    )
    args = parser.parse_args()
    if args.command == "bundle":
        from truthound.mcp.bundle import build_bundle

        print(build_bundle(args.output, wheel=args.wheel))
        return
    try:
        from truthound.mcp.policy import MCPPolicy
        from truthound.mcp.server import serve as run

        policy = MCPPolicy(
            tuple(args.allow_dir),
            max_file_bytes=args.max_file_bytes,
            max_items=args.max_items,
            timeout_seconds=args.timeout_seconds,
        )
        asyncio.run(run(policy))
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == "mcp" or exc.name.startswith("mcp.")):
            print('Install the MCP extra: pip install "truthound[mcp]"', file=sys.stderr)
            raise SystemExit(2) from None
        raise
    except (ValueError, OSError):
        parser.error("Invalid MCP policy; provide existing allowed directories and valid limits")


if __name__ == "__main__":
    main()
