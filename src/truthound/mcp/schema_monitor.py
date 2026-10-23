"""Explicit approval of pinned, session-owned native schema polling and sending."""

import threading
from contextlib import suppress
from typing import Annotated, Literal

from pydantic import Field

from truthound.mcp.bindings import load
from truthound.mcp.contracts import Name, WireModel
from truthound.mcp.model_process import OwnedModelProcess
from truthound.mcp.operations import envelope
from truthound.mcp.plugin_bindings import fingerprint
from truthound.mcp.policy import MCPError
from truthound.mcp.session_contracts import Handle


class MonitorOpen(WireModel):
    binding_id: Name
    mode: Literal["sync", "async"] = "sync"
    poll_interval: Annotated[float, Field(ge=0.1, le=3600, allow_inf_nan=False)] = 60
    max_events: Annotated[int, Field(ge=1, le=1000)] = 100
    idempotency_key: Annotated[str, Field(min_length=8, max_length=128)]


class MonitorInput(WireModel):
    monitor_id: Handle


def resolve(policy, binding_id):
    document, digest = load(policy)
    binding = document.schema_monitors.get(binding_id)
    if binding is None:
        raise MCPError("invalid_binding")
    needed = {binding.source, binding.notifier} - {None}
    plugins, found = {}, set()
    for name, spec in document.plugins.items():
        exposed = set(spec.capabilities) & needed
        if not exposed:
            continue
        if exposed & found:
            raise MCPError("invalid_binding")
        package, actual, _ = fingerprint(spec.distribution)
        if package.version != spec.version or actual != spec.sha256:
            raise MCPError("plugin_identity_mismatch")
        found |= exposed
        plugins[name] = {
            **spec.model_dump(),
            "capabilities": {key: spec.capabilities[key] for key in exposed},
        }
    if found != needed:
        raise MCPError("invalid_binding")
    return binding, digest, plugins


def authorize(session, metadata):
    if (
        str(session.policy.control_directory) != metadata["binding_control_root"]
        or session._policy_stamp() != metadata["monitor_policy"]
        or resolve(session.policy, metadata["monitor_binding"])[1] != metadata["binding_digest"]
    ):
        raise MCPError("access_denied")


class MonitorProcess(OwnedModelProcess):
    worker_module = "truthound.mcp.schema_monitor_worker"

    def __init__(self, ttl):
        self._watch_stop = threading.Event()
        super().__init__(ttl)

    def watch_policy(self, authorized):
        def supervise():
            while not self._watch_stop.wait(0.1):
                if self.closed:
                    return
                if not authorized():
                    self.close()
                    return

        threading.Thread(target=supervise, daemon=True).start()

    def close(self):
        self._watch_stop.set()
        super().close()

    def validate_reply(self, result):
        if result != {"ok": True}:
            raise MCPError("worker_error")
        return result


class MonitorSession:
    def __init__(self, session):
        self.session = session
        self.workers = {}

    def release(self, identity):
        worker = self.workers.pop(identity, None)
        if worker is not None:
            worker.close()

    def close(self):
        for identity in list(self.workers):
            self.release(identity)

    def reap(self):
        for identity, worker in list(self.workers.items()):
            try:
                self.session.resources.get(identity, "monitor")
                if worker.closed:
                    raise MCPError("invalid_handle")
            except (MCPError, OSError, ValueError):
                self.release(identity)
                with suppress(MCPError):
                    self.session.resources.release_owned(identity)

    async def call(self, operation, arguments):
        session = self.session
        resources, policy = session.resources, session.policy
        if operation == "schema_monitor_open":
            if arguments.max_events > policy.max_items:
                raise MCPError("resource_exhausted")
            binding, digest, plugins = resolve(policy, arguments.binding_id)
            stamp, store = session._policy_stamp(), session.approvals()
            plan = store.prepare(
                operation,
                key=arguments.idempotency_key,
                request={
                    **arguments.model_dump(exclude={"idempotency_key"}),
                    "bindings_digest": digest,
                },
                policy=stamp,
                session=session.identity,
                effects=["installed_extension_execute", "continuous_schema_read"]
                + (["external_schema_alert_send"] if binding.notifier else []),
            )
            if "committed_result" in plan:
                result = plan["committed_result"]
                resources.get(result["data"]["resource_id"], "monitor")
                return result
            if plan["state"] != "approved":
                return envelope(operation, data=plan, error="approval_required")
            identity, path = resources.reserve("monitor")
            worker, started, entry = None, False, None
            try:
                metadata = {
                    "monitor_binding": arguments.binding_id,
                    "monitor_policy": stamp,
                    "binding_digest": digest,
                    "binding_control_root": str(policy.control_directory),
                }
                path.write_text(arguments.model_dump_json(), encoding="utf-8")
                authorize(session, metadata)
                store.begin(plan["plan_id"], session=session.identity, policy=stamp)
                started = True
                worker = MonitorProcess(policy.handle_ttl_seconds)
                worker.watch_policy(lambda: session.policy == policy)
                await worker.call(
                    {
                        "open": arguments.model_dump(),
                        "binding": binding.model_dump(),
                        "plugins": plugins,
                        "control": str(policy.control_directory),
                        "digest": digest,
                        "budget": min(
                            policy.max_file_bytes, policy.max_session_bytes, 16 * 1024**2
                        ),
                    },
                    policy.timeout_seconds,
                )
                authorize(session, metadata)
                entry = resources.commit(
                    identity,
                    kind="monitor",
                    media_type="application/vnd.truthound.monitor+json",
                    metadata=metadata,
                )
                worker.expire_after(max(0, entry.expires - resources.clock()))
                self.workers[identity] = worker
                result = envelope(operation, data=session._info(entry))
                store.finish(plan["plan_id"], result)
                return result
            except BaseException:
                if worker:
                    worker.close()
                self.workers.pop(identity, None)
                if entry:
                    with suppress(MCPError):
                        resources.release_owned(identity)
                if started:
                    store.uncertain(plan["plan_id"])
                    raise MCPError("outcome_unknown") from None
                raise
            finally:
                resources.discard(identity)
        entry = resources.get(arguments.monitor_id, "monitor")
        worker = self.workers.get(entry.id)
        if worker is None or worker.closed:
            raise MCPError("invalid_handle")
        identity, path = resources.reserve("artifact")
        try:
            await worker.call(
                {
                    "operation": operation,
                    "output_path": str(path),
                    "budget": resources.available_bytes,
                },
                policy.timeout_seconds,
            )
            resources.get(entry.id, "monitor")
            artifact = resources.commit(
                identity,
                kind="artifact",
                parents=(entry.id,),
                media_type="application/vnd.truthound.monitor-result+json",
            )
            return envelope(operation, data=session._info(artifact))
        except BaseException:
            self.release(entry.id)
            with suppress(MCPError):
                resources.release_owned(entry.id)
            raise
        finally:
            resources.discard(identity)
