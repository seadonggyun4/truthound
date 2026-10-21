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
    serve.add_argument("--workspace-dir", action="append", default=[], type=Path)
    serve.add_argument("--control-dir", type=Path)
    serve.add_argument("--max-handles", type=int, default=128)
    serve.add_argument("--max-session-bytes", type=int, default=1024**3)
    serve.add_argument("--handle-ttl-seconds", type=int, default=900)
    serve.add_argument("--max-file-bytes", type=int, default=100 * 1024 * 1024)
    serve.add_argument("--max-items", type=int, default=100)
    serve.add_argument("--timeout-seconds", type=int, default=30)
    serve.add_argument("--profile", choices=("summary", "core"), default="summary")
    serve.add_argument(
        "--allow-sensitive-results",
        action="store_true",
        help="Operator consent to disclose raw values and full Core evidence within allowed directories",
    )
    bundle = commands.add_parser("bundle", help="Build a Claude Desktop MCPB")
    bundle.add_argument("--output", required=True, type=Path)
    bundle.add_argument(
        "--wheel", type=Path, help="Bundle an exact local wheel before PyPI publication"
    )
    for name in ("review-plan", "approve-plan", "revoke-plan"):
        operator = commands.add_parser(
            name, help="Trusted local operator control; never an MCP tool"
        )
        operator.add_argument("--control-dir", required=True, type=Path)
        operator.add_argument("--plan-id", required=True)
    fingerprint = commands.add_parser(
        "plugin-fingerprint",
        help="Inspect installed plugin metadata/files without importing its code",
    )
    fingerprint.add_argument("--distribution", required=True)
    args = parser.parse_args()
    if args.command == "plugin-fingerprint":
        import json

        from truthound.mcp.plugin_bindings import fingerprint
        from truthound.mcp.policy import MCPError

        try:
            package, digest, _ = fingerprint(args.distribution)
            print(
                json.dumps(
                    {
                        "distribution": package.metadata["Name"],
                        "version": package.version,
                        "sha256": digest,
                        "entry_points": [
                            entry.name
                            for entry in package.entry_points
                            if entry.group == "truthound.plugins"
                        ],
                    }
                )
            )
        except (MCPError, ValueError, OSError):
            parser.error(
                "Plugin metadata inspection failed; use an installed non-editable distribution"
            )
        return
    if args.command in {"review-plan", "approve-plan", "revoke-plan"}:
        import json

        from truthound.mcp.approval import ApprovalStore
        from truthound.mcp.policy import MCPError

        try:
            store = ApprovalStore(args.control_dir)
            try:
                if args.command == "review-plan":
                    print(json.dumps(store.review(args.plan_id), ensure_ascii=True))
                else:
                    (store.approve if args.command == "approve-plan" else store.revoke)(
                        args.plan_id
                    )
                    print(json.dumps({"plan_id": args.plan_id, "action": args.command}))
            finally:
                store.close()
        except (MCPError, ValueError, OSError):
            parser.error(
                "Operator plan action failed; verify the private control directory and plan"
            )
        return
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
            profile=args.profile,
            allow_sensitive_results=args.allow_sensitive_results,
            workspace_directories=tuple(args.workspace_dir),
            control_directory=args.control_dir,
            max_handles=args.max_handles,
            max_session_bytes=args.max_session_bytes,
            handle_ttl_seconds=args.handle_ttl_seconds,
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
