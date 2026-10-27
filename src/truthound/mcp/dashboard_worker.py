"""Native dashboard owner; no model-supplied host, port, path or Python code."""

import asyncio

from truthound.mcp.policy import MCPError


class DashboardOwner:
    def __init__(self, runtime, request):
        from truthound.profiler.dashboard.server import DashboardServer

        if not isinstance(runtime, DashboardServer) or runtime.status()["running"]:
            raise MCPError("invalid_binding")
        if runtime.config.host not in {"127.0.0.1", "localhost", "::1"}:
            raise MCPError("access_denied")
        self.runtime = runtime

    async def execute(self, args, budget):
        if args["command"] == "start":
            return await asyncio.to_thread(
                self.runtime.start, timeout_seconds=args.get("timeout_seconds", 10)
            )
        if args["command"] == "stop":
            return await asyncio.to_thread(self.runtime.stop)
        if args["command"] == "status":
            return await asyncio.to_thread(self.runtime.status)
        raise MCPError("invalid_arguments")

    async def close(self):
        await asyncio.to_thread(self.runtime.close)


if __name__ == "__main__":
    from truthound.mcp.broker_worker import main

    main(DashboardOwner)
