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
    if not options.allowed_directory.strip() or options.allowed_directory == "${user_config.allowed_directory}":
        parser.error("An explicit data directory is required")
    result = ["serve", "--profile", "core", "--allow-dir", options.allowed_directory]
    for value, flag, placeholder in (
        (options.workspace_directory, "--workspace-dir", "${user_config.workspace_directory}"),
        (options.control_directory, "--control-dir", "${user_config.control_directory}"),
    ):
        # Desktop may leave an empty optional setting unexpanded. It grants nothing.
        if value.strip() and value != placeholder:
            result.extend([flag, value])
    if options.sensitive_results == "true":
        result.append("--allow-sensitive-results")
    return result


def main():
    from truthound.mcp.__main__ import main as serve

    sys.argv = [sys.argv[0], *arguments()]
    serve()
