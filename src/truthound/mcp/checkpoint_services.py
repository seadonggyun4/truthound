"""Approved, session-owned native runner and escalation lifecycle controls."""

import json
from contextlib import suppress
from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from truthound.mcp.approved_sources import SourceBindingOpen, resolve
from truthound.mcp.checkpoints import authorize
from truthound.mcp.contracts import Name, WireModel
from truthound.mcp.operations import envelope
from truthound.mcp.policy import MCPError
from truthound.mcp.schema_monitor import MonitorProcess, MonitorSession
from truthound.mcp.session_contracts import Handle, RegistryCheck


class CheckpointServiceOpen(SourceBindingOpen):
    kind: Literal["runner", "escalation"]
    snapshot_id: Annotated[str, Field(pattern=r"^servicesnapshot_[0-9a-f]{64}$")] | None = None


Trigger = Literal[
    "unacknowledged",
    "unresolved",
    "severity_upgrade",
    "repeated_failure",
    "threshold_breach",
    "manual",
    "scheduled",
    "custom",
]


class EscalationTargetSpec(WireModel):
    type: Literal["user", "team", "channel", "schedule", "webhook", "email", "phone", "custom"]
    identifier: Name
    name: Annotated[str, Field(max_length=512)] = ""
    priority: int = 0
    metadata: Annotated[dict[Name, JsonValue], Field(max_length=100)] = Field(default_factory=dict)


class EscalationLevelSpec(WireModel):
    level: Annotated[int, Field(ge=1)]
    delay_minutes: Annotated[int, Field(ge=0)] = 0
    targets: Annotated[list[EscalationTargetSpec], Field(max_length=100)] = Field(
        default_factory=list
    )
    repeat_count: Annotated[int, Field(ge=0)] = 0
    repeat_interval_minutes: Annotated[int, Field(ge=0)] = 5
    require_ack: bool = True
    auto_resolve_minutes: Annotated[int, Field(ge=0)] = 0
    conditions: Annotated[dict[Name, JsonValue], Field(max_length=100)] = Field(
        default_factory=dict
    )
    metadata: Annotated[dict[Name, JsonValue], Field(max_length=100)] = Field(default_factory=dict)


class EscalationPolicyUpdate(WireModel):
    """Declarative native policy edits; callbacks remain operator-owned."""

    enabled: bool | None = None
    severity_filter: list[Literal["critical", "high", "medium", "low", "info"]] | None = None
    max_escalations: Annotated[int, Field(ge=0)] | None = None
    cooldown_minutes: Annotated[int, Field(ge=0)] | None = None
    business_hours_only: bool | None = None
    business_hours_start: Annotated[int, Field(ge=0, le=23)] | None = None
    business_hours_end: Annotated[int, Field(ge=0, le=23)] | None = None
    business_days: list[Annotated[int, Field(ge=0, le=6)]] | None = None
    timezone: Name | None = None
    description: Annotated[str, Field(max_length=4096)] | None = None
    triggers: Annotated[list[Trigger], Field(max_length=8)] | None = None
    levels: Annotated[list[EscalationLevelSpec], Field(max_length=100)] | None = None
    metadata: Annotated[dict[Name, JsonValue], Field(max_length=100)] | None = None

    @model_validator(mode="after")
    def supplied(self):
        if not self.model_fields_set or any(
            getattr(self, k) is None for k in self.model_fields_set
        ):
            raise ValueError("supply at least one non-null policy setting")
        if self.levels is not None and len({level.level for level in self.levels}) != len(
            self.levels
        ):
            raise ValueError("Level numbers must be unique")
        return self


class ScheduleUpdate(WireModel):
    expression: Annotated[str, Field(min_length=1, max_length=256)] | None = None
    enabled: bool | None = None
    max_runs: Annotated[int, Field(ge=0)] | None = None
    run_immediately: bool | None = None
    catch_up: bool | None = None
    max_concurrent: Annotated[int, Field(ge=1)] | None = None
    metadata: Annotated[dict[Name, JsonValue], Field(max_length=100)] | None = None
    interval_seconds: Annotated[int, Field(ge=0)] | None = None
    interval_minutes: Annotated[int, Field(ge=0)] | None = None
    interval_hours: Annotated[int, Field(ge=0)] | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    run_on_weekdays: (
        Annotated[list[Annotated[int, Field(ge=0, le=6)]], Field(min_length=1)] | None
    ) = None
    timezone: Name | None = None

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def datetime_from_wire(cls, value):
        return datetime.fromisoformat(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def supplied(self):
        nullable = {"start_time", "end_time", "run_on_weekdays", "timezone"}
        if not self.model_fields_set or any(
            getattr(self, key) is None for key in self.model_fields_set - nullable
        ):
            raise ValueError("Supply schedule settings; only optional boundaries may be cleared")
        for value in (self.start_time, self.end_time):
            if value is not None and value.utcoffset() is not None:
                raise ValueError("Use naive local datetimes, as required by Core")
        return self


class CheckpointDefinitionUpdate(WireModel):
    checks: Annotated[list[RegistryCheck], Field(min_length=1, max_length=1000)] | None = None
    min_severity: Literal["low", "medium", "high", "critical"] | None = None
    auto_schema: bool | None = None
    run_name_template: Annotated[str, Field(min_length=1, max_length=256)] | None = None
    tags: (
        Annotated[dict[Name, Annotated[str, Field(max_length=512)]], Field(max_length=100)] | None
    ) = None
    metadata: Annotated[dict[Name, JsonValue], Field(max_length=100)] | None = None
    fail_on_critical: bool | None = None
    fail_on_high: bool | None = None
    timeout_seconds: Annotated[int, Field(ge=1, le=86400)] | None = None
    sample_size: Annotated[int, Field(ge=1)] | None = None

    @model_validator(mode="after")
    def supplied(self):
        if not self.model_fields_set or any(
            getattr(self, key) is None
            for key in self.model_fields_set - {"min_severity", "sample_size"}
        ):
            raise ValueError("Supply checkpoint settings; only optional limits may be cleared")
        if any(check.binding_id is not None for check in self.checks or ()):
            raise ValueError("Native callback validators remain owned by the factory")
        return self


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
        "run_once",
        "run_all",
        "remove_checkpoint",
        "schedule_update",
        "checkpoint_update",
        "policy",
        "policy_update",
        "policy_create",
        "persist_configuration",
        "policy_remove",
        "active_records",
        "incident_records",
        "recovery_status",
        "recovery_resolve",
        "recovery_forget",
    ]
    record_id: Name | None = None
    incident_id: Name | None = None
    trigger_type: Trigger = "unacknowledged"
    context: Annotated[dict[Name, JsonValue], Field(max_length=100)] = Field(default_factory=dict)
    policy_name: Name | None = None
    checkpoint_name: Name | None = None
    trigger_name: Name | None = None
    schedule_update: ScheduleUpdate | None = None
    checkpoint_update: CheckpointDefinitionUpdate | None = None
    policy_update: EscalationPolicyUpdate | None = None
    policy_names: Annotated[list[Name], Field(min_length=1, max_length=100)] | None = None
    checkpoint_names: Annotated[list[Name], Field(min_length=1, max_length=100)] | None = None
    attempt_id: Annotated[str, Field(min_length=1, max_length=256)] | None = None
    resolution: Literal["completed", "cancelled"] | None = None
    attempt_ids: (
        Annotated[
            list[Annotated[str, Field(min_length=1, max_length=256)]],
            Field(min_length=1, max_length=100),
        ]
        | None
    ) = None
    recovery_limit: Annotated[int, Field(ge=1, le=1000)] = 100
    recovery_offset: Annotated[int, Field(ge=0)] = 0
    max_results: Annotated[int, Field(ge=1, le=1000)] = 100

    @model_validator(mode="after")
    def shape(self):
        record = self.command in {"record", "acknowledge", "resolve", "cancel", "escalate"}
        if record != (self.record_id is not None):
            raise ValueError("record_id is required only for record commands")
        if (self.command in {"trigger", "incident_records"}) != (self.incident_id is not None):
            raise ValueError("incident_id is required only for incident commands")
        if self.command != "trigger" and self.model_fields_set & {"trigger_type", "context"}:
            raise ValueError("trigger_type and context are only supported by trigger")
        policy_commands = {"policy", "policy_update", "policy_create", "policy_remove"}
        if self.command in policy_commands and self.policy_name is None:
            raise ValueError("policy_name is required")
        if self.policy_name is not None and self.command not in policy_commands | {
            "trigger",
            "active_records",
        }:
            raise ValueError("policy_name is not supported by this command")
        if (self.command in {"policy_update", "policy_create"}) != (self.policy_update is not None):
            raise ValueError("policy_update is required only for policy_update or policy_create")
        selections = int(self.policy_names is not None) + int(self.checkpoint_names is not None)
        if selections != int(self.command == "persist_configuration"):
            raise ValueError("persist_configuration requires exactly one kind of named definitions")
        if self.checkpoint_names is not None and len(set(self.checkpoint_names)) != len(
            self.checkpoint_names
        ):
            raise ValueError("checkpoint_names must be unique")
        if self.policy_names is not None and len(set(self.policy_names)) != len(self.policy_names):
            raise ValueError("policy_names must be unique")
        if (self.command == "schedule_update") != (self.trigger_name is not None):
            raise ValueError("trigger_name is required only for schedule_update")
        if (self.command == "schedule_update") != (self.schedule_update is not None):
            raise ValueError("schedule_update is required only for schedule_update")
        if (self.command == "checkpoint_update") != (self.checkpoint_update is not None):
            raise ValueError("checkpoint_update is required only for checkpoint_update")
        if (
            self.command
            in {"run_once", "remove_checkpoint", "schedule_update", "checkpoint_update"}
        ) != (self.checkpoint_name is not None):
            raise ValueError("checkpoint_name is required only for named runner commands")
        if (self.command == "recovery_resolve") != (
            self.attempt_id is not None and self.resolution is not None
        ):
            raise ValueError("attempt_id and resolution are required for recovery_resolve")
        if self.command != "recovery_resolve" and (
            self.attempt_id is not None or self.resolution is not None
        ):
            raise ValueError("attempt_id and resolution are only supported by recovery_resolve")
        if (self.command == "recovery_forget") != (self.attempt_ids is not None):
            raise ValueError("attempt_ids is required only for recovery_forget")
        if self.attempt_ids is not None and len(set(self.attempt_ids)) != len(self.attempt_ids):
            raise ValueError("attempt_ids must be unique")
        if self.command != "recovery_status" and self.model_fields_set & {
            "recovery_limit",
            "recovery_offset",
        }:
            raise ValueError("recovery pagination is only supported by recovery_status")
        if "max_results" in self.model_fields_set and self.command != "results":
            raise ValueError("max_results is only supported by results")
        return self


class ServiceProcess(MonitorProcess):
    worker_module = "truthound.mcp.checkpoint_service_worker"

    def validate_reply(self, result):
        if result == {"ok": True, "shutdown": True}:
            self.native_shutdown = True
            return result
        return super().validate_reply(result)

    def _request_shutdown(self):
        if self.process.poll() is None:
            with suppress(OSError, ValueError):
                self.process.stdin.write(b'{"close":true}\n')
                self.process.stdin.flush()
            with suppress(Exception):
                self.process.wait(timeout=1)

    async def call(self, request, timeout):
        try:
            return await super().call(request, timeout)
        except BaseException as error:
            # Private owner diagnostics; the public operation still becomes
            # outcome_unknown and must not be retried after effects start.
            self.failure_category = (
                error.code
                if isinstance(error, MCPError)
                else "timeout"
                if isinstance(error, TimeoutError)
                else "exchange_failed"
            )
            # This is observed after superclass cleanup and is not necessarily
            # the worker's original exit status (cleanup may have killed it).
            self.failure_exit_status_after_cleanup = self.process.poll()
            raise


class CheckpointServices(MonitorSession):
    async def call(self, operation, arguments):
        session = self.session
        resources, policy = session.resources, session.policy
        opening = operation == "checkpoint_services_open"
        digest, capability, plugins = resolve(policy, arguments.binding_id, checkpoint=True)
        stamp = session._policy_stamp()
        snapshot = None
        if opening and arguments.snapshot_id is not None:
            from truthound.mcp.durable_checkpoint_services import read

            snapshot = read(
                session, arguments.snapshot_id, arguments.binding_id, kind=arguments.kind
            )
        entry = None
        if not opening:
            entry = resources.get(arguments.service_id, "monitor")
            metadata = entry.metadata
            if metadata.get("checkpoint_binding") != arguments.binding_id:
                raise MCPError("invalid_binding")
            allowed = {
                "start",
                "stop",
                "status",
                "recovery_status",
                "recovery_resolve",
                "recovery_forget",
            } | (
                {
                    "results",
                    "run_once",
                    "run_all",
                    "remove_checkpoint",
                    "schedule_update",
                    "checkpoint_update",
                    "persist_configuration",
                }
                if metadata["service_kind"] == "runner"
                else {
                    "trigger",
                    "record",
                    "acknowledge",
                    "resolve",
                    "cancel",
                    "escalate",
                    "policy",
                    "policy_update",
                    "policy_create",
                    "persist_configuration",
                    "policy_remove",
                    "active_records",
                    "incident_records",
                    "recovery_status",
                }
            )
            if arguments.command not in allowed:
                raise MCPError("invalid_arguments")
            if arguments.command == "persist_configuration" and (
                (metadata["service_kind"] == "runner") != (arguments.checkpoint_names is not None)
            ):
                raise MCPError("invalid_arguments")
            if (
                arguments.command == "results"
                and arguments.max_results > policy.max_items
                or arguments.command == "recovery_status"
                and arguments.recovery_limit > policy.max_items
            ):
                raise MCPError("resource_exhausted")
            worker = self.workers.get(entry.id)
            if worker is None or worker.closed:
                raise MCPError("invalid_handle")
        store = session.approvals()
        plan = store.prepare(
            operation,
            key=arguments.idempotency_key,
            request={
                **arguments.model_dump(
                    mode="json", exclude={"idempotency_key"}, exclude_unset=True
                ),
                "bindings_digest": digest,
            },
            policy=stamp,
            session=session.identity,
            effects=[
                "installed_extension_execute",
                "native_checkpoint_service_control",
                "operator_configured_actions_and_external_notifications",
                "operator_configured_native_action_retries",
                "approved_notification_target_configuration",
                "persistent_configuration_storage_or_restore",
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
                if snapshot is None:
                    path.write_text(json.dumps({"kind": arguments.kind}), encoding="utf-8")
                else:
                    if len(snapshot) > resources.available_bytes:
                        raise MCPError("resource_exhausted")
                    path.write_bytes(snapshot)
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
                    "recovery_scope": store.digest(
                        {
                            "policy": stamp,
                            "binding": arguments.binding_id,
                            "bindings_digest": digest,
                            "kind": arguments.kind,
                        }
                    ),
                    "configuration_path": str(path) if snapshot is not None else None,
                }
            else:
                request = {
                    "arguments": arguments.model_dump(mode="json", exclude_unset=True),
                    "output_path": str(path),
                    "budget": resources.available_bytes,
                }
            await worker.call(request, policy.timeout_seconds)
            authorize(session, metadata)
            if not opening and arguments.command == "persist_configuration":
                from truthound.mcp.durable_checkpoint_services import persist
                from truthound.mcp.encoding import encode

                payload = path.read_bytes()
                receipt = persist(session, plan, metadata, payload)
                path.write_text(json.dumps(encode(receipt)), encoding="utf-8")
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
