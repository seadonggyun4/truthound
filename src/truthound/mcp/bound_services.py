"""Approved resident native services with pinned factory and session ownership."""

import json
import time
from contextlib import suppress

from truthound.mcp.approved_sources import resolve
from truthound.mcp.operations import envelope
from truthound.mcp.policy import MCPError
from truthound.mcp.schema_monitor import MonitorSession


def authorize(session, metadata):
    if (
        str(session.policy.control_directory) != metadata["binding_control_root"]
        or session._policy_stamp() != metadata["native_policy"]
        or resolve(session.policy, metadata["native_binding"], group=metadata["native_group"])[0]
        != metadata["binding_digest"]
    ):
        raise MCPError("access_denied")


class BoundServiceSession(MonitorSession):
    """Subclasses declare a closed group, worker and effects; no reflective methods."""

    group = None
    process_type = None
    effects = ()
    limit_commands = set()

    def input_resources(self, arguments):
        """Trusted subclasses select immutable session inputs, never wire paths."""
        return {}

    def artifact_media_type(self, arguments):
        return "application/vnd.truthound." + self.group + "+json"

    def response_media_type(self, arguments, reply):
        """Native failure receipts may use JSON instead of a requested binary format."""
        return self.artifact_media_type(arguments)

    def reply_error(self, arguments, reply):
        """Closed subclasses may retain a verified native failure artifact."""
        return None

    async def call(self, operation, arguments):
        session, policy, resources = self.session, self.session.policy, self.session.resources
        opening = operation == self.group + "_open"
        digest, capability, plugins = resolve(policy, arguments.binding_id, group=self.group)
        if (
            getattr(arguments, "command", None) in self.limit_commands
            and arguments.limit > policy.max_items
        ):
            raise MCPError("resource_exhausted")
        entry, worker = None, None
        if not opening:
            entry = resources.get(arguments.service_id, "monitor")
            if (
                entry.metadata.get("native_binding") != arguments.binding_id
                or entry.metadata.get("native_group") != self.group
            ):
                raise MCPError("invalid_binding")
            worker = self.workers.get(entry.id)
            if worker is None or worker.closed:
                raise MCPError("invalid_handle")
            if getattr(worker, "native_shutdown", False) and arguments.command != "shutdown":
                raise MCPError("invalid_handle")
        inputs = {} if opening else self.input_resources(arguments)
        if inputs:
            resources.validate_resident_inputs(
                entry.id, tuple(item.id for item in inputs.values()), kind="monitor"
            )
        store, stamp = session.approvals(), session._policy_stamp()
        plan = store.prepare(
            operation,
            key=arguments.idempotency_key,
            request={
                **arguments.model_dump(
                    mode="json", exclude={"idempotency_key"}, exclude_unset=True
                ),
                "bindings_digest": digest,
                **(
                    {"input_digests": {key: item.sha256 for key, item in inputs.items()}}
                    if inputs
                    else {}
                ),
            },
            policy=stamp,
            session=session.identity,
            effects=list(self.effects),
        )
        if "committed_result" in plan:
            response = plan["committed_result"]
            resources.get(response["data"]["resource_id"], "monitor" if opening else "artifact")
            return response
        if worker is not None and getattr(worker, "native_shutdown", False):
            raise MCPError("invalid_handle")
        if plan["state"] != "approved":
            return envelope(operation, data=plan, error="approval_required")
        metadata = (
            entry.metadata
            if entry
            else {
                "native_binding": arguments.binding_id,
                "native_group": self.group,
                "native_policy": stamp,
                "binding_digest": digest,
                "binding_control_root": str(policy.control_directory),
            }
        )
        identity, path = resources.reserve("monitor" if opening else "artifact")
        started, committed = False, None
        try:
            authorize(session, metadata)
            for item in inputs.values():
                resources.get(item.id)
            deadline = time.monotonic() + policy.timeout_seconds
            if opening:
                worker = self.process_type(policy.handle_ttl_seconds)

                def authorized():
                    try:
                        authorize(session, metadata)
                        return True
                    except (MCPError, OSError, ValueError):
                        return False

                worker.watch_policy(authorized)
                # Imports/bootstrap cannot run an operator factory. Retain the
                # approval until this effect-free exchange has succeeded.
                await worker.call({"prepare": True}, max(0, deadline - time.monotonic()))
                authorize(session, metadata)
                if time.monotonic() >= deadline:
                    raise TimeoutError
            if opening:
                path.write_text(json.dumps({"binding_id": arguments.binding_id}), encoding="utf-8")
                request = {
                    "open": True,
                    "plugins": plugins,
                    "capability": capability,
                    "control": str(policy.control_directory),
                    "digest": digest,
                    "native_group": self.group,
                    "workspace_roots": list(map(str, policy.workspace_directories)),
                }
            else:
                request = {"arguments": arguments.model_dump(mode="json", exclude_unset=True)}
                if inputs:
                    bound = resources.bind_resident_inputs(
                        entry.id, tuple(item.id for item in inputs.values()), kind="monitor"
                    )
                    worker.expire_after(max(0, bound.expires - resources.clock()))
                    request["inputs"] = {key: str(item.path) for key, item in inputs.items()}
            request.update(output_path=str(path), budget=resources.available_bytes)
            store.begin(plan["plan_id"], session=session.identity, policy=stamp)
            started = True
            reply = await worker.call(request, max(0, deadline - time.monotonic()))
            authorize(session, metadata)
            for item in inputs.values():
                resources.get(item.id)
            committed = resources.commit(
                identity,
                kind="monitor" if opening else "artifact",
                media_type=("application/vnd.truthound." + self.group + "+json")
                if opening
                else self.response_media_type(arguments, reply),
                metadata=metadata if opening else {},
                parents=() if opening else (entry.id,),
            )
            if opening:
                worker.expire_after(max(0, committed.expires - resources.clock()))
                self.workers[identity] = worker
            response = envelope(
                operation,
                data=session._info(committed),
                error=None if opening else self.reply_error(arguments, reply),
            )
            store.finish(plan["plan_id"], response)
            return response
        except BaseException:
            if worker is not None:
                worker.close()
            if entry is not None:
                self.release(entry.id)
                with suppress(MCPError):
                    resources.release_owned(entry.id)
            if committed is not None:
                with suppress(MCPError):
                    resources.release_owned(identity)
            if started:
                store.uncertain(plan["plan_id"])
                raise MCPError("outcome_unknown") from None
            raise
        finally:
            resources.discard(identity)
