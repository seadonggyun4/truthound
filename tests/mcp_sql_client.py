"""Real stdio client for a fixed synthetic tenant source; grants stay operator-side."""

import asyncio
import base64
import json


def read_tenant_source_over_stdio(runtime, *, on_live=None):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from truthound.mcp.approval import ApprovalStore
    from truthound.mcp.tables import decode

    async def scenario():
        parameters = StdioServerParameters(command=runtime[0]._worker_argv()[0], args=[
            "-I", "-m", "truthound.mcp", "serve", "--profile", "core",
            "--allow-dir", str(runtime[0].policy.allowed_directories[0]),
            "--control-dir", str(runtime[1]), "--allow-sensitive-results"])
        async with stdio_client(parameters) as (read, write), ClientSession(read, write) as client:
            await client.initialize()

            async def call(name, arguments):
                reply = await client.call_tool(name, arguments)
                assert reply.structuredContent is not None
                return reply.structuredContent

            async def approved_call(name, arguments):
                pending = await call(name, arguments)
                assert pending["issues"] == [{"code": "approval_required"}], pending
                store = ApprovalStore(runtime[1])
                try:
                    store.approve(pending["data"]["plan_id"])
                finally:
                    store.close()
                result = await call(name, arguments)
                assert result["execution_status"] == "succeeded", result
                return result

            opened = await approved_call("tenant_services_open", {
                "binding_id": "local", "idempotency_key": "tenant-sdk-open"})
            result = await approved_call("tenant_services_execute", {
                "binding_id": "local", "service_id": opened["data"]["resource_id"],
                "command": "source_read", "tenant_id": "structured", "source_id": "records",
                "limit": 1, "idempotency_key": "tenant-sdk-read"})
            chunks, offset = [], 0
            while offset is not None:
                part = await call("artifacts_read", {"resource_id": result["data"]["resource_id"], "offset": offset})
                assert part["execution_status"] == "succeeded", part
                chunks.append(base64.b64decode(part["data"]["base64"]))
                offset = part["data"]["next_offset"]
            assert decode(json.loads(b"".join(chunks)))["rows"] == [{"x": 1}]
            if on_live:
                on_live()
            # Deliberately leave the source open: frontend shutdown owns teardown.

    asyncio.run(scenario())
