"""Real production stdio server with a fixed synthetic child-worker fixture."""

import asyncio
import sys
from pathlib import Path

from truthound.mcp import server
from truthound.mcp.policy import MCPPolicy
from truthound.mcp.service import MCPService

worker = Path(__file__).with_name("mcp_lifecycle_worker.py")
command = [sys.executable, "-I", str(worker), sys.argv[2]]


class FixtureService(MCPService):
    def _worker_argv(self):
        return command


server.MCPService = FixtureService
asyncio.run(server.serve(MCPPolicy((Path(sys.argv[1]),), timeout_seconds=int(sys.argv[3]))))
