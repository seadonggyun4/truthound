"""Bounded durable one-shot validation jobs, using native validation and cron.

There is no background daemon or automatic retry. A client explicitly pumps a
due job. A lost execution lease becomes indeterminate, never successful.
"""

import hashlib
import hmac
import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from truthound.mcp.approval import canonical
from truthound.mcp.contracts import WireModel
from truthound.mcp.durable_models import ancestry
from truthound.mcp.operations import envelope
from truthound.mcp.policy import MCPError
from truthound.mcp.scheduling import SchedulePreview
from truthound.mcp.session_contracts import AssetValidation, ResourceInfo


class JobSubmit(WireModel):
    validation: AssetValidation
    schedule: SchedulePreview | None = None
    idempotency_key: Annotated[str, Field(min_length=8, max_length=128)]

    @model_validator(mode="after")
    def closed_job(self):
        from truthound.mcp.validator_wire import frame_handles

        value = self.validation
        if value.context_id or value.schema_id or value.idempotency_key:
            raise ValueError("Durable jobs cannot reference session context/schema handles")
        for check in value.checks or ():
            if check.binding_id or frame_handles(check.validator, check.settings):
                raise ValueError("Durable jobs cannot persist provider bindings or frame handles")
        if self.schedule and (
            self.schedule.count != 1
            or datetime.fromisoformat(self.schedule.after).utcoffset() is None
        ):
            raise ValueError("One-shot schedules require count=1 and an explicit UTC offset")
        return self


class JobInput(WireModel):
    job_id: Annotated[str, Field(pattern=r"^job_[0-9a-f]{64}$")] | None = None
    idempotency_key: Annotated[str, Field(min_length=8, max_length=128)] | None = None

    @model_validator(mode="after")
    def one_identity(self):
        if (self.job_id is None) == (self.idempotency_key is None):
            raise ValueError("Select exactly one job identity or original intent key")
        return self


class JobState(WireModel):
    job_id: str
    state: Literal["queued", "running", "succeeded", "failed", "cancelled", "outcome_unknown"]
    due_at: float
    lease_until: float | None
    result_sha256: str | None
    artifact: ResourceInfo | None = None


def storage(session):
    store = session.approvals()
    store.connection.execute("""CREATE TABLE IF NOT EXISTS validation_jobs (
        identity TEXT PRIMARY KEY, record TEXT NOT NULL, input BLOB NOT NULL,
        output BLOB, signature TEXT NOT NULL
    )""")
    return store


def write(store, identity, record, payload, output):
    store.connection.execute(
        "INSERT OR REPLACE INTO validation_jobs VALUES (?,?,?,?,?)",
        (
            identity,
            canonical(record),
            payload,
            output,
            store.digest({"identity": identity, "record": record}),
        ),
    )


def read(store, identity):
    row = store.connection.execute(
        "SELECT * FROM validation_jobs WHERE identity=?", (identity,)
    ).fetchone()
    if row is None:
        raise MCPError("invalid_handle")
    record = json.loads(row["record"])
    if (
        not hmac.compare_digest(
            row["signature"], store.digest({"identity": identity, "record": record})
        )
        or hashlib.sha256(row["input"]).hexdigest() != record["input_sha256"]
        or (
            row["output"] is not None
            and hashlib.sha256(row["output"]).hexdigest() != record["result_sha256"]
        )
    ):
        raise MCPError("invalid_control_state")
    return record, row["input"], row["output"]


def authorize(session, record):
    if session._policy_stamp() != record["policy"]:
        raise MCPError("access_denied")
    for origin in record["origins"]:
        session.policy.authorize_origin(Path(origin))


def info(identity, record):
    return {
        "job_id": identity,
        **{key: record[key] for key in ("state", "due_at", "lease_until", "result_sha256")},
        "artifact": None,
    }


async def call(session, name, arguments):
    store = storage(session)
    resources = session.resources
    if name == "jobs_submit":
        entry = resources.get(arguments.validation.asset_id, "asset")
        inputs = ancestry(resources, entry)
        if any(
            "binding_digest" in item.metadata
            or "workspace_root" in item.metadata
            or item.metadata.get("durable_workspaces")
            for item in inputs
        ):
            raise MCPError("unsupported_operation")
        due = store.clock()
        if arguments.schedule:
            result = await session.call("schedules_preview", arguments.schedule)
            if result["execution_status"] != "succeeded":
                return envelope(name, error="core_error")
            times = result["data"]["next_times"]
            if not result["data"]["valid"] or len(times) != 1:
                raise MCPError("invalid_arguments")
            due = datetime.fromisoformat(times[0]).timestamp()
        policy = session._policy_stamp()
        intent = {
            "arguments": arguments.model_dump(exclude={"idempotency_key"}),
            "input_sha256": entry.sha256,
            "policy": policy,
        }
        plan = store.prepare(
            name,
            key=arguments.idempotency_key,
            request=intent,
            policy=policy,
            session=session.identity,
            effects=["persistent_input_storage", "deferred_validation"],
        )
        if "committed_result" in plan:
            return plan["committed_result"]
        if plan["state"] != "approved":
            return envelope(name, data=plan, error="approval_required")
        if entry.size > min(session.policy.max_file_bytes, 8 * 1024**2):
            raise MCPError("resource_exhausted")
        payload = entry.path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry.sha256:
            raise MCPError("resource_changed")
        identity = "job_" + store.digest({"plan": plan["plan_id"]})
        record = {
            "validation": arguments.validation.model_dump(),
            "policy": policy,
            "origins": sorted({str(origin) for item in inputs for origin in item.origins}),
            "input_sha256": entry.sha256,
            "state": "queued",
            "due_at": due,
            "lease_until": None,
            "result_sha256": None,
        }
        result = envelope(name, data=info(identity, record))
        store.begin(plan["plan_id"], session=session.identity, policy=session._policy_stamp())
        try:
            for item in inputs:
                resources.get(item.id)
            with store.transaction():
                size, count = store.connection.execute(
                    "SELECT COALESCE(SUM(length(input)+COALESCE(length(output),0)),0),COUNT(*) FROM validation_jobs"
                ).fetchone()
                if count >= 128 or size + len(payload) > 16 * 1024**2:
                    raise MCPError("resource_exhausted")
                write(store, identity, record, payload, None)
                store.finish_in_transaction(plan["plan_id"], result)
            return result
        except Exception:
            store.uncertain(plan["plan_id"])
            raise MCPError("outcome_unknown") from None
        except BaseException:
            store.uncertain(plan["plan_id"])
            raise
    identity = arguments.job_id or store.committed_identity(
        arguments.idempotency_key, "jobs_submit", "job_id"
    )
    record, payload, output = read(store, identity)
    authorize(session, record)
    if name in {"jobs_cancel", "jobs_recover"}:
        with store.transaction():
            record, payload, output = read(store, identity)
            if name == "jobs_cancel":
                if record["state"] == "running":
                    raise MCPError("busy")
                if record["state"] == "queued":
                    record["state"] = "cancelled"
            elif record["state"] == "running" and store.clock() >= record["lease_until"]:
                record["state"] = "outcome_unknown"
            write(store, identity, record, payload, output)
    elif name == "jobs_execute" and record["state"] == "queued":
        with store.transaction():
            record, payload, output = read(store, identity)
            if record["state"] != "queued":
                raise MCPError("busy")
            if store.clock() < record["due_at"]:
                raise MCPError("job_not_due")
            record.update(
                state="running", lease_until=store.clock() + session.policy.timeout_seconds + 30
            )
            write(store, identity, record, payload, None)
        asset_id = None
        try:
            asset_id, path = resources.reserve("asset")
            if len(payload) > resources.available_bytes:
                raise MCPError("resource_exhausted")
            path.write_bytes(payload)
            resources.commit(
                asset_id,
                kind="asset",
                media_type="application/vnd.apache.arrow.file",
                origins=tuple(Path(origin) for origin in record["origins"]),
            )
            validation = AssetValidation.model_validate(
                {**record["validation"], "asset_id": asset_id}
            )
            response = await session.call("assets_validate", validation)
            artifact = (response.get("data") or {}).get("artifact")
            output = (
                resources.get(artifact["resource_id"], "artifact").path.read_bytes()
                if artifact
                else None
            )
            if output is not None and len(output) > 8 * 1024**2:
                raise MCPError("resource_exhausted")
            authorize(session, record)
            # Native computation and owned-resource cleanup must both finish
            # before the durable receipt can attest success.
            resources.release(asset_id)
            asset_id = None
            with store.transaction():
                current, _, _ = read(store, identity)
                if current["state"] != "running":
                    raise MCPError("outcome_unknown")
                size = store.connection.execute(
                    "SELECT COALESCE(SUM(length(input)+COALESCE(length(output),0)),0) FROM validation_jobs"
                ).fetchone()[0]
                if size + len(output or b"") > 16 * 1024**2:
                    raise MCPError("resource_exhausted")
                record.update(
                    state="succeeded"
                    if response["execution_status"] == "succeeded" and output is not None
                    else "failed",
                    result_sha256=hashlib.sha256(output).hexdigest()
                    if output is not None
                    else None,
                )
                write(store, identity, record, payload, output)
        except BaseException:
            with store.transaction():
                current, _, _ = read(store, identity)
                if current["state"] == "running":
                    current["state"] = "outcome_unknown"
                    write(store, identity, current, payload, None)
            raise
        finally:
            if asset_id:
                resources.discard(asset_id)
                if asset_id in resources._entries:
                    resources.release(asset_id)
    result = info(identity, record)
    if name == "jobs_result" and output is not None:
        artifact_id, path = resources.reserve("artifact")
        try:
            if len(output) > resources.available_bytes:
                raise MCPError("resource_exhausted")
            path.write_bytes(output)
            retained = resources.commit(
                artifact_id,
                kind="artifact",
                media_type="application/vnd.truthound.validation+json",
                origins=tuple(Path(origin) for origin in record["origins"]),
                metadata={
                    "durable_policy": record["policy"],
                    "durable_control": str(session.policy.control_directory),
                    "durable_workspaces": [],
                },
            )
            result["artifact"] = session._info(retained)
        finally:
            resources.discard(artifact_id)
    return envelope(name, data=result)
