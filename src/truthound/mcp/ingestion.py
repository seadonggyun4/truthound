"""Bounded admission shared by disposable requests and retained assets."""

from __future__ import annotations

import hashlib
import os
import stat

from truthound.mcp.policy import MCPError


async def capture(policy, source, target):
    import anyio

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    digest = hashlib.sha256()
    fd = os.open(source, flags)
    with os.fdopen(fd, "rb") as stream, target.open("xb") as output:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise MCPError("access_denied")
        count = 0
        while chunk := stream.read(1024 * 1024):
            count += len(chunk)
            if count > policy.max_file_bytes:
                raise MCPError("file_too_large")
            output.write(chunk)
            digest.update(chunk)
            await anyio.lowlevel.checkpoint()
        after = os.fstat(stream.fileno())
        if count != before.st_size or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise MCPError("source_changed")
    return digest
