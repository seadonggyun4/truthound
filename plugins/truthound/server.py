"""Launch the pinned public Core server with revocable local directory consent."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from permissions import config_path, read_directories

from truthound.mcp.policy import MCPError, MCPPolicy
from truthound.mcp.server import serve


class ConsentPolicy:
    # Match the pinned Core 3.2.0 defaults; no quality computation is duplicated.
    max_file_bytes = 100 * 1024 * 1024
    max_items = 100
    max_response_bytes = 65536
    timeout_seconds = 30

    def __init__(self, path: Path):
        self.path = path

    def resolve(self, value: str) -> Path:
        try:
            roots = read_directories(self.path)
            if not roots:
                raise ValueError("unconfigured")
            policy = MCPPolicy(
                roots,
                max_file_bytes=self.max_file_bytes,
                max_items=self.max_items,
                max_response_bytes=self.max_response_bytes,
                timeout_seconds=self.timeout_seconds,
            )
        except (OSError, ValueError):
            raise MCPError("access_denied") from None
        return policy.resolve(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=config_path())
    args = parser.parse_args()
    # No config is still a usable MCP connection: all three tools deny file access.
    asyncio.run(serve(ConsentPolicy(args.config)))


if __name__ == "__main__":
    main()
