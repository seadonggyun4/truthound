"""Approved, session-owned native runner and escalation lifecycle controls."""

import json
from contextlib import suppress
from typing import Annotated, Literal

from pydantic import Field, model_validator

from truthound.mcp.approved_sources import SourceBindingOpen, resolve
from truthound.mcp.checkpoints import authorize
from truthound.mcp.contracts import Name
from truthound.mcp.operations import envelope
from truthound.mcp.policy import MCPError
from truthound.mcp.schema_monitor import MonitorProcess, MonitorSession
from truthound.mcp.session_contracts import Handle


class CheckpointServiceOpen(SourceBindingOpen):
    kind: Literal["runner", "escalation"]


class CheckpointServiceCommand(SourceBindingOpen):
    # The binding is repeated intentionally: each approval pins both identity
    # and the live handle, rather than authorizing a portable command.
    service_id: Handle
    command: Literal[
        "start",
        "stop",
        "status",
        "results",
        "trigger",
        "record",
        "acknowledge",
        "resolve",
        "cancel",
        "escalate",
    ]
    record_id: Name | None = None
    incident_id: Name | None = None
    policy_name: Name | None = None
    max_results: Annotated[int, Field(ge=1, le=1000)] = 100

    @model_validator(mode="after")
    def shape(self):
        record = self.command in {"record", "acknowledge", "resolve", "cancel", "escalate"}
        if record != (self.record_id is not None):
            raise ValueError("record_id is required only for record commands")
        if (self.command == "trigger") != (self.incident_id is not None):
            raise ValueError("incident_id is required only for trigger")
        if self.policy_name is not None and self.command != "trigger":
            raise ValueError("policy_name is only supported by trigger")
        if "max_results" in self.model_fields_set and self.command != "results":
            raise ValueError("max_results is only supported by results")
        return self


class ServiceProcess(MonitorProcess):
    worker_module = "truthound.mcp.checkpoint_service_worker"


class CheckpointServices(MonitorSession):
    async def call(self, operation, arguments):
        session = self.session
        resources, policy = session.resources, session.policy
        opening = operation == "checkpoint_services_open"
        digest, capability, plugins = resolve(policy, arguments.binding_id, checkpoint=True)
        stamp = session._policy_stamp()
        entry = None
        if not opening:
            entry = resources.get(arguments.service_id, "monitor")
            metadata = entry.metadata
            if metadata.get("checkpoint_binding") != arguments.binding_id:
                raise MCPError("invalid_binding")
            allowed = {"start", "stop", "status"} | (
                {"results"}
                if metadata["service_kind"] == "runner"
                else {"trigger", "record", "acknowledge", "resolve", "cancel", "escalate"}
            )
            if arguments.command not in allowed:
                raise MCPError("invalid_arguments")
            if arguments.command == "results" and arguments.max_results > policy.max_items:
                raise MCPError("resource_exhausted")
            worker = self.workers.get(entry.id)
            if worker is None or worker.closed:
                raise MCPError("invalid_handle")
        store = session.approvals()
        plan = store.prepare(
            operation,
            key=arguments.idempotency_key,
            request={
                **arguments.model_dump(exclude={"idempotency_key"}),
                "bindings_digest": digest,
            },
            policy=stamp,
            session=session.identity,
            effects=[
                "installed_extension_execute",
                "native_checkpoint_service_control",
                "operator_configured_actions_and_external_notifications",
                "operator_configured_native_action_retries",
            ],
        )
        if "committed_result" in plan:
            response = plan["committed_result"]
            resources.get(response["data"]["resource_id"], "monitor" if opening else "artifact")
            return response
        if plan["state"] != "approved":
            return envelope(operation, data=plan, error="approval_required")
        identity, path = resources.reserve("monitor" if opening else "artifact")
        started, committed = False, None
        if opening:
            worker = None
            metadata = {
                "checkpoint_binding": arguments.binding_id,
                "checkpoint_policy": stamp,
                "binding_digest": digest,
                "binding_control_root": str(policy.control_directory),
                "service_kind": arguments.kind,
            }
        try:
            authorize(session, metadata)
            store.begin(plan["plan_id"], session=session.identity, policy=stamp)
            started = True
            if opening:
                path.write_text(json.dumps({"kind": arguments.kind}), encoding="utf-8")
                worker = ServiceProcess(policy.handle_ttl_seconds)

                def authorized():
                    try:
                        authorize(session, metadata)
                        return True
                    except (MCPError, OSError, ValueError):
                        return False

                worker.watch_policy(authorized)
                request = {
                    "open": arguments.kind,
                    "plugins": plugins,
                    "capability": capability,
                    "control": str(policy.control_directory),
                    "digest": digest,
                }
            else:
                request = {
                    "arguments": arguments.model_dump(),
                    "output_path": str(path),
                    "budget": resources.available_bytes,
                }
            await worker.call(request, policy.timeout_seconds)
            authorize(session, metadata)
            committed = resources.commit(
                identity,
                kind="monitor" if opening else "artifact",
                media_type="application/vnd.truthound.checkpoint-service+json",
                metadata=metadata if opening else {},
                parents=() if opening else (entry.id,),
            )
            if opening:
                worker.expire_after(max(0, committed.expires - resources.clock()))
                self.workers[identity] = worker
            response = envelope(operation, data=session._info(committed))
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
