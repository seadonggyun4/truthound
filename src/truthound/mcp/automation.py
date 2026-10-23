"""Operator-approved recurring definitions; native computation remains in Core."""

import asyncio
import hmac
import json
from contextlib import suppress
from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, model_validator

from truthound.mcp.approval import canonical
from truthound.mcp.contracts import WireModel
from truthound.mcp.operations import envelope
from truthound.mcp.policy import MCPError
from truthound.mcp.retention import RetentionPreview


class ControlInput(WireModel):
    control_id: Annotated[str, Field(pattern=r"^automation_[0-9a-f]{64}$")] | None = None
    idempotency_key: Annotated[str, Field(min_length=8, max_length=128)] | None = None

    @model_validator(mode="after")
    def one_identity(self):
        if (self.control_id is None) == (self.idempotency_key is None):
            raise ValueError("Select exactly one control identity or original intent key")
        return self


class PolicyRegister(RetentionPreview):
    idempotency_key: Annotated[str, Field(min_length=8, max_length=128)]


class ControlState(WireModel):
    control_id: str
    kind: Literal["schedule", "retention"]
    state: Literal["active", "disabled", "paused"]
    job_id: str | None = None
    generation: int = 0
    deleted: int = 0


def storage(session):
    store = session.approvals()
    store.connection.execute("""CREATE TABLE IF NOT EXISTS mcp_automations (
        identity TEXT PRIMARY KEY, record TEXT NOT NULL, signature TEXT NOT NULL
    )""")
    return store


def write(store, identity, record):
    store.connection.execute(
        "INSERT OR REPLACE INTO mcp_automations VALUES (?,?,?)",
        (identity, canonical(record), store.digest({"identity": identity, "record": record})),
    )


def read(session, identity):
    from truthound.mcp.durable_jobs import authorize

    store = storage(session)
    row = store.connection.execute(
        "SELECT * FROM mcp_automations WHERE identity=?", (identity,)
    ).fetchone()
    if row is None:
        raise MCPError("invalid_handle")
    record = json.loads(row["record"])
    if not hmac.compare_digest(
        row["signature"], store.digest({"identity": identity, "record": record})
    ):
        raise MCPError("invalid_control_state")
    authorize(session, record)
    return record


def info(identity, record):
    return dict(
        control_id=identity,
        kind=record["kind"],
        state=record["state"],
        job_id=record.get("job_id"),
        generation=record.get("generation", 0),
        deleted=record.get("deleted", 0),
    )


def capacity(store):
    if store.connection.execute("SELECT COUNT(*) FROM mcp_automations").fetchone()[0] >= 64:
        raise MCPError("resource_exhausted")


def pinned(session, job_id):
    store = storage(session)
    for row in store.connection.execute("SELECT identity FROM mcp_automations"):
        record = read(session, row[0])
        if (
            record["kind"] == "schedule"
            and record["state"] != "disabled"
            and record["job_id"] == job_id
        ):
            return True
    return False


async def call(session, name, arguments):
    store = storage(session)
    if name == "storage_policy_register":
        policy = arguments.model_dump(exclude={"idempotency_key"})
        plan = store.prepare(
            name,
            key=arguments.idempotency_key,
            request={"retention": policy, "policy": session._policy_stamp()},
            policy=session._policy_stamp(),
            session=session.identity,
            effects=["persistent_automatic_retention", "future_payload_deletion"],
        )
        if "committed_result" in plan:
            return plan["committed_result"]
        if plan["state"] != "approved":
            return envelope(name, data=plan, error="approval_required")
        identity = "automation_" + store.digest({"plan": plan["plan_id"]})
        record = dict(
            kind="retention",
            state="active",
            retention=policy,
            policy=session._policy_stamp(),
            origins=[],
            deleted=0,
        )
        result = envelope(name, data=info(identity, record))
        store.begin(plan["plan_id"], session=session.identity, policy=session._policy_stamp())
        try:
            with store.transaction():
                capacity(store)
                write(store, identity, record)
                store.finish_in_transaction(plan["plan_id"], result)
        except Exception:
            store.uncertain(plan["plan_id"])
            raise MCPError("outcome_unknown") from None
        except BaseException:
            store.uncertain(plan["plan_id"])
            raise
        return result
    registration = (
        "schedules_register" if name.startswith("schedules_") else "storage_policy_register"
    )
    identity = arguments.control_id or store.committed_identity(
        arguments.idempotency_key, registration, "control_id"
    )
    with store.transaction():
        record = read(session, identity)
        expected_kind = "schedule" if name.startswith("schedules_") else "retention"
        if record["kind"] != expected_kind:
            raise MCPError("invalid_handle")
        if name.endswith("_disable") and record["state"] != "disabled":
            if record["kind"] == "schedule":
                from truthound.mcp.durable_jobs import read as job_read
                from truthound.mcp.durable_jobs import write as job_write

                job, payload, output = job_read(store, record["job_id"])
                if job["state"] == "queued":
                    job["state"] = "cancelled"
                    job_write(store, record["job_id"], job, payload, output)
            record["state"] = "disabled"
            write(store, identity, record)
    return envelope(name, data=info(identity, record))


async def tick(session, stop=None):
    """One bounded sweep, safe to run from multiple independently owned workers."""
    from truthound.mcp import durable_jobs as jobs
    from truthound.mcp.retention import call as retention_call
    from truthound.mcp.retention import eligible
    from truthound.mcp.retention import read as retained_read
    from truthound.mcp.scheduling import SchedulePreview

    store = storage(session)
    jobs.storage(session)
    processed = 0
    identities = [
        row[0]
        for row in store.connection.execute(
            "SELECT identity FROM mcp_automations ORDER BY identity"
        )
    ]
    # Approved cleanup first permits indefinite schedules within fixed storage quotas.
    identities.sort(key=lambda identity: read(session, identity)["kind"] != "retention")
    for identity in identities:
        if stop is not None and stop.is_set():
            break
        record = read(session, identity)
        if record["state"] != "active":
            continue
        if record["kind"] == "retention":
            with store.transaction():
                record = read(session, identity)
                if record["state"] != "active":
                    continue
                preview = await retention_call(
                    session,
                    "storage_retention_preview",
                    RetentionPreview.model_validate(record["retention"]),
                )
                for candidate in preview["data"]["candidates"]:
                    value, _, table = retained_read(session, candidate)
                    if not eligible(value) or pinned(session, candidate):
                        raise MCPError("busy")
                    store.connection.execute(f"DELETE FROM {table} WHERE identity=?", (candidate,))
                    record["deleted"] += 1
                write(store, identity, record)
            processed += 1
            continue
        job_id = record["job_id"]
        current, payload, _ = jobs.read(store, job_id)
        jobs.authorize(session, current)
        if current["state"] == "running":
            await jobs.call(session, "jobs_recover", jobs.JobInput(job_id=job_id))
            current, payload, _ = jobs.read(store, job_id)
        if current["state"] in {"failed", "cancelled", "outcome_unknown"}:
            with store.transaction():
                latest = read(session, identity)
                if latest["job_id"] == job_id and latest["state"] == "active":
                    latest["state"] = "paused"
                    write(store, identity, latest)
            continue
        if current["state"] == "succeeded":
            zone = datetime.fromisoformat(record["schedule"]["after"]).tzinfo
            preview = await session.call(
                "schedules_preview",
                SchedulePreview(
                    expression=record["schedule"]["expression"],
                    after=datetime.fromtimestamp(current["due_at"], zone).isoformat(),
                    count=1,
                ),
            )
            if (
                preview["execution_status"] != "succeeded"
                or len(preview["data"]["next_times"]) != 1
            ):
                raise MCPError("schedule_exhausted")
            with store.transaction():
                latest = read(session, identity)
                if latest["state"] != "active" or latest["job_id"] != job_id:
                    continue
                size, count = store.connection.execute(
                    "SELECT COALESCE(SUM(length(input)+COALESCE(length(output),0)),0),COUNT(*) FROM validation_jobs"
                ).fetchone()
                if count >= 128 or size + len(payload) > 16 * 1024**2:
                    raise MCPError("resource_exhausted")
                latest["generation"] += 1
                job_id = "job_" + store.digest(
                    {"automation": identity, "generation": latest["generation"]}
                )
                current = {
                    **current,
                    "created_at": store.clock(),
                    "state": "queued",
                    "due_at": datetime.fromisoformat(preview["data"]["next_times"][0]).timestamp(),
                    "lease_until": None,
                    "result_sha256": None,
                    "occurrence_ids": [job_id],
                }
                jobs.write(store, job_id, current, payload, None)
                latest["job_id"] = job_id
                write(store, identity, latest)
        if current["state"] == "queued" and current["due_at"] <= store.clock():
            try:
                await jobs.call(session, "jobs_execute", jobs.JobInput(job_id=job_id))
            except MCPError as exc:
                if exc.code != "busy":
                    raise
            processed += 1
    return processed


async def run(policy, *, once=False, interval=5):
    """Foreground operator process; never installs a system service or self-starts."""
    import signal

    from truthound.mcp.service import MCPService
    from truthound.mcp.session import MCPSession

    if (
        policy.profile != "core"
        or not policy.allow_sensitive_results
        or policy.control_directory is None
    ):
        raise ValueError("Worker requires Core, sensitive consent and private control directory")
    if not isinstance(interval, int) or not 1 <= interval <= 3600:
        raise ValueError("Polling interval must be an integer from 1 to 3600 seconds")
    service = MCPService(policy)
    session = MCPSession(service)
    service._session = session
    stop = asyncio.Event()
    previous = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, lambda *_: stop.set())
        while not stop.is_set():
            await tick(session, stop)
            from truthound.mcp.durable_schema import tick as schema_tick

            await schema_tick(session, stop)
            if once:
                return
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=interval)
    finally:
        service.close()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
