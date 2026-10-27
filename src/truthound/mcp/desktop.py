"""Translate desktop configuration into explicit Core CLI grants."""

from __future__ import annotations

import argparse
import sys


def arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--allowed-directory", required=True)
    parser.add_argument("--workspace-directory", default="")
    parser.add_argument("--control-directory", default="")
    parser.add_argument(
        "--sensitive-results", choices=("true", "false"), default="false", type=str.lower
    )
    options = parser.parse_args(argv)
    if not options.allowed_directory.strip():
        parser.error("An explicit data directory is required")
    result = ["serve", "--profile", "core", "--allow-dir", options.allowed_directory]
    for value, flag in (
        (options.workspace_directory, "--workspace-dir"),
        (options.control_directory, "--control-dir"),
    ):
        if value.strip():
            result.extend([flag, value])
    if options.sensitive_results == "true":
        result.append("--allow-sensitive-results")
    return result


def main():
    from truthound.mcp.__main__ import main as serve

    sys.argv = [sys.argv[0], *arguments()]
    serve()
