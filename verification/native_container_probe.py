"""Real Docker/Podman worker verification; absence is failure, never a skipped pass."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

from truthound.plugins.security.exceptions import SandboxError, SandboxTimeoutError
from truthound.plugins.security.protocols import ResourceLimits, SecurityPolicy
from truthound.plugins.security.sandbox.engines.container import ContainerSandboxEngine


async def verify(runtime, image, report):
    identity = subprocess.run(
        [runtime, "image", "inspect", image, "--format", "{{.Id}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report["image_id"] = identity
    engine = ContainerSandboxEngine(runtime=runtime, image=identity)
    policy = replace(
        SecurityPolicy.development(),
        allow_network=False,
        resource_limits=ResourceLimits(
            max_memory_mb=128, max_cpu_percent=100, max_execution_time_sec=15
        ),
    )
    try:
        value = await engine.execute(
            engine.create_sandbox("native-values", policy), eval, "(b'abc', {'x': 3})"
        )
        report["native_values"] = value == (b"abc", {"x": 3})
        assert report["native_values"]
        if hasattr(os, "getuid"):
            identity = await engine.execute(
                engine.create_sandbox("native-ipc-owner", policy),
                eval,
                "(__import__('os').getuid(), __import__('os').stat('/workspace').st_mode & 0o777)",
            )
            assert identity == (os.getuid(), 0o700), identity
            report["private_ipc_owner_preserved"] = True
        rejected = False
        try:
            value = await engine.execute(
                engine.create_sandbox("native-reducer", policy),
                eval,
                "type('Payload', (), {'__reduce__': lambda self: (__import__('os').getpid, ())})()",
            )
            report["parent_reducer_executed"] = value == os.getpid()
        except SandboxError:
            rejected = True
        report["executable_result_rejected"] = rejected
        assert rejected, "Child return values must never execute pickle reducers in the parent"
        context = engine.create_sandbox("native-cancel", policy)
        task = asyncio.create_task(engine.execute(context, eval, "__import__('time').sleep(60)"))
        for _ in range(200):
            if context.container_id is not None or task.done():
                break
            await asyncio.sleep(0.05)
        assert context.container_id is not None, "Container was not created"
        identity = context.container_id
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("Cancelled execution did not propagate cancellation")
        inspection = subprocess.run(
            [runtime, "inspect", identity],
            capture_output=True,
            check=False,
        )
        report["cancelled_container_removed"] = inspection.returncode != 0
        assert report["cancelled_container_removed"]
        assert not engine._container_ids
        short_policy = replace(
            policy, resource_limits=replace(policy.resource_limits, max_execution_time_sec=0.2)
        )
        context = engine.create_sandbox("native-timeout", short_policy)
        try:
            await engine.execute(context, eval, "__import__('time').sleep(60)")
        except SandboxTimeoutError:
            report["wall_timeout"] = True
        else:
            raise AssertionError("Native deadline was not enforced")
        assert not engine._container_ids
        assert not engine._temp_dirs
        inspection = subprocess.run(
            [runtime, "inspect", context.container_id], capture_output=True, check=False
        )
        assert inspection.returncode != 0
        report["timed_out_container_removed"] = True
        limited = replace(policy, allow_file_write=False)
        context = engine.create_sandbox("native-limits", limited)
        observed = await engine.execute(
            context,
            eval,
            "{'memory': open('/sys/fs/cgroup/memory.max').read().strip(), "
            "'cpu': open('/sys/fs/cgroup/cpu.max').read().strip(), "
            "'interfaces': {name: int(open('/sys/class/net/' + name + '/flags').read(), 16) "
            "for _, name in __import__('socket').if_nameindex()}, "
            "'routes': open('/proc/net/route').read().splitlines()[1:], "
            "'descriptors': __import__('resource').getrlimit(__import__('resource').RLIMIT_NOFILE)}",
        )
        report["observed_limits"] = observed
        assert int(observed["memory"]) == 128 * 1024 * 1024
        quota, period = map(int, observed["cpu"].split())
        assert quota == period
        assert "lo" in observed["interfaces"]
        assert all(not flags & 1 for name, flags in observed["interfaces"].items() if name != "lo")
        assert not observed["routes"], "Network-disabled worker has an IPv4 route"
        assert observed["descriptors"] == (limited.resource_limits.max_file_descriptors,) * 2
        report["native_cgroups_network_descriptors"] = True
        try:
            await engine.execute(
                engine.create_sandbox("native-readonly", limited),
                eval,
                "open('/truthound-owned-write-probe', 'w')",
            )
        except SandboxError:
            report["readonly_root"] = True
        else:
            raise AssertionError("Read-only root accepted a write")
        try:
            await engine.execute(
                engine.create_sandbox("native-subprocess-denied", policy),
                eval,
                "__import__('subprocess').run(['/usr/local/bin/python', '-c', 'pass'], check=True).returncode",
            )
        except SandboxError:
            report["subprocess_denied"] = True
        else:
            raise AssertionError("Configured denied syscalls allowed a subprocess")
        permissive = replace(
            policy,
            allow_subprocess=True,
            resource_limits=replace(policy.resource_limits, denied_syscalls=()),
        )
        value = await engine.execute(
            engine.create_sandbox("native-subprocess-allowed", permissive),
            eval,
            "__import__('subprocess').run(['/usr/local/bin/python', '-c', 'pass'], check=True).returncode",
        )
        assert value == 0
        report["explicit_subprocess_allow"] = True
        selected = replace(
            permissive,
            resource_limits=replace(permissive.resource_limits, denied_syscalls=("getpid",)),
        )
        value = await engine.execute(
            engine.create_sandbox("native-selected-syscall", selected),
            eval,
            "__import__('os').getpid()",
        )
        assert value == -1
        report["selected_syscall_denied"] = True

        class LostCreateAcknowledgement(ContainerSandboxEngine):
            async def _create_container(self, command):
                await super()._create_container(command)
                raise OSError("Synthetic lost create acknowledgement")

        uncertain = LostCreateAcknowledgement(runtime=runtime, image=report["image_id"])
        try:
            try:
                await uncertain.execute(uncertain.create_sandbox("lost-create", policy), abs, -1)
            except OSError:
                pass
            else:
                raise AssertionError("Lost acknowledgement did not surface")
            assert not uncertain._container_ids, "Failed creation left an owned container"
            assert not uncertain._temp_dirs
            report["lost_create_acknowledgement_reconciled"] = True
        finally:
            await uncertain.cleanup()
        with tempfile.TemporaryDirectory(prefix="truthound-container-grant-") as directory:
            root = Path(directory).resolve()
            read_only = root / "read-only"
            writable = root / "writable"
            read_only.mkdir()
            writable.mkdir()
            (read_only / "input.txt").write_text("synthetic-input")
            grants = replace(
                policy,
                resource_limits=replace(
                    policy.resource_limits,
                    allowed_paths=(str(read_only),),
                    writable_paths=(str(writable),),
                ),
            )
            value = await engine.execute(
                engine.create_sandbox("native-host-read", grants),
                eval,
                f"__import__('pathlib').Path({str(read_only / 'input.txt')!r}).read_text()",
            )
            assert value == "synthetic-input"
            try:
                await engine.execute(
                    engine.create_sandbox("native-host-readonly", grants),
                    eval,
                    f"__import__('pathlib').Path({str(read_only / 'input.txt')!r}).write_text('changed')",
                )
            except SandboxError:
                pass
            else:
                raise AssertionError("Read-only host grant accepted a write")
            value = await engine.execute(
                engine.create_sandbox("native-host-write", grants),
                eval,
                f"__import__('pathlib').Path({str(writable / 'output.txt')!r}).write_text('synthetic-output')",
            )
            assert value == len("synthetic-output")
            assert (writable / "output.txt").read_text() == "synthetic-output"
            assert (read_only / "input.txt").read_text() == "synthetic-input"
            report["explicit_host_grants"] = True

    finally:
        await engine.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", default="docker")
    parser.add_argument("--image", default="python:3.11-slim")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"runtime": args.runtime, "success": False}
    try:
        asyncio.run(verify(args.runtime, args.image, report))
        report["success"] = True
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
