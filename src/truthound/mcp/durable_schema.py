"""Signed durable monitor checkpoints and an at-most-once-attempt notification outbox."""

import hmac
import json
import os
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from truthound.mcp.approval import canonical
from truthound.mcp.contracts import WireModel
from truthound.mcp.encoding import encode
from truthound.mcp.operations import envelope
from truthound.mcp.policy import MCPError
from truthound.mcp.schema_monitor import MonitorOpen, MonitorProcess, resolve
from truthound.mcp.session_contracts import ResourceInfo


class WatchRegister(MonitorOpen):
    lifetime_seconds: Annotated[int, Field(ge=1, le=2592000)] = 86400


class WatchInput(WireModel):
    watch_id: Annotated[str, Field(pattern=r"^watch_[0-9a-f]{64}$")]
    include_snapshot: bool = False


class WatchState(WireModel):
    watch_id: str
    state: Literal["active", "paused", "disabled", "failed", "expired"]
    revision: str
    retention: dict | None = None
    pruned_notifications: int = 0
    expires_at: float
    polls: int
    notifications: dict[str, int]
    artifact: ResourceInfo | None = None


class PollProcess(MonitorProcess):
    worker_module = "truthound.mcp.durable_schema_worker"


def storage(session):
    store = session.approvals()
    store.connection.execute("""CREATE TABLE IF NOT EXISTS mcp_schema_watches (
        identity TEXT PRIMARY KEY, record TEXT NOT NULL, signature TEXT NOT NULL
    )""")
    return store


def write(store, identity, record):
    raw = canonical(record)
    if len(raw.encode()) > record["budget"]:
        raise MCPError("resource_exhausted")
    store.connection.execute(
        "INSERT OR REPLACE INTO mcp_schema_watches VALUES (?,?,?)",
        (identity, raw, store.digest({"identity": identity, "record": record})),
    )


def read(store, identity):
    row = store.connection.execute(
        "SELECT * FROM mcp_schema_watches WHERE identity=?", (identity,)
    ).fetchone()
    if row is None:
        raise MCPError("invalid_handle")
    record = json.loads(row["record"])
    if not hmac.compare_digest(
        row["signature"], store.digest({"identity": identity, "record": record})
    ):
        raise MCPError("invalid_control_state")
    return record


def authorize(session, record):
    if record["policy"] != session._policy_stamp():
        raise MCPError("access_denied")
    binding, digest, plugins = resolve(session.policy, record["options"]["binding_id"])
    if digest != record["digest"]:
        raise MCPError("access_denied")
    return binding, plugins


def info(identity, record, store):
    return dict(
        watch_id=identity,
        state=record["state"],
        revision=store.digest(record),
        retention=record.get("retention"),
        pruned_notifications=record.get("pruned_notifications", 0),
        expires_at=record["expires_at"],
        polls=(record["checkpoint"] or {}).get("polls", 0),
        notifications=dict(Counter(item["state"] for item in record["outbox"])),
        artifact=None,
    )


async def call(session, name, arguments):
    if name == "schema_watch_update":
        from truthound.mcp.schema_watch_control import update

        return await update(session, arguments)
    store = storage(session)
    if name == "schema_watch_register":
        if arguments.max_events > session.policy.max_items:
            raise MCPError("resource_exhausted")
        binding, digest, _ = resolve(session.policy, arguments.binding_id)
        options = arguments.model_dump(exclude={"idempotency_key"})
        stamp = session._policy_stamp()
        plan = store.prepare(
            name,
            key=arguments.idempotency_key,
            request={"options": options, "digest": digest, "policy": stamp},
            policy=stamp,
            session=session.identity,
            effects=[
                "persistent_schema_monitor",
                "automatic_restart_reads",
                "installed_extension_execute",
            ]
            + (["persistent_external_notification_queue"] if binding.notifier else []),
        )
        if "committed_result" in plan:
            return plan["committed_result"]
        if plan["state"] != "approved":
            return envelope(name, data=plan, error="approval_required")
        identity = "watch_" + store.digest({"plan": plan["plan_id"]})
        record = dict(
            options=options,
            policy=stamp,
            digest=digest,
            state="active",
            expires_at=store.clock() + arguments.lifetime_seconds,
            due_at=store.clock(),
            checkpoint=None,
            outbox=[],
            busy=None,
            budget=min(1024**2, session.policy.max_file_bytes, session.policy.max_session_bytes),
        )
        result = envelope(name, data=info(identity, record, store))
        store.begin(plan["plan_id"], session=session.identity, policy=stamp)
        try:
            with store.transaction():
                if (
                    store.connection.execute("SELECT COUNT(*) FROM mcp_schema_watches").fetchone()[
                        0
                    ]
                    >= 32
                ):
                    raise MCPError("resource_exhausted")
                write(store, identity, record)
                store.finish_in_transaction(plan["plan_id"], result)
        except BaseException:
            store.uncertain(plan["plan_id"])
            raise
        return result
    with store.transaction():
        record = read(store, arguments.watch_id)
        authorize(session, record)
        if name == "schema_watch_disable":
            record["state"] = "disabled"
            write(store, arguments.watch_id, record)
    result = info(arguments.watch_id, record, store)
    if arguments.include_snapshot:
        resources = session.resources
        identity, path = resources.reserve("artifact")
        try:
            path.write_text(
                json.dumps(
                    encode({"checkpoint": record["checkpoint"], "outbox": record["outbox"]})
                ),
                encoding="utf-8",
            )
            entry = resources.commit(
                identity,
                kind="artifact",
                media_type="application/json",
                metadata={
                    "monitor_policy": session._policy_stamp(),
                    "binding_control_root": str(session.policy.control_directory),
                    "monitor_binding": record["options"]["binding_id"],
                    "binding_digest": record["digest"],
                },
            )
            result["artifact"] = session._info(entry)
        finally:
            resources.discard(identity)
    return envelope(name, data=result)


def alive(owner):
    import psutil

    try:
        process = psutil.Process(owner["pid"])
        return (
            process.create_time() == owner["created"] and process.status() != psutil.STATUS_ZOMBIE
        )
    except psutil.NoSuchProcess:
        return False


async def tick(session, stop=None):
    """Bounded sweep. Lease owner identity prevents simultaneous poll/send attempts."""
    import psutil

    from truthound.mcp.schema_watch_control import prune

    store = storage(session)
    for row in store.connection.execute(
        "SELECT identity FROM mcp_schema_watches ORDER BY identity"
    ).fetchall():
        if stop is not None and stop.is_set():
            return
        identity = row[0]
        with store.transaction():
            record = read(store, identity)
            busy = record["busy"]
            if busy:
                if alive(busy):
                    continue
                if busy["task"] == "deliver":
                    record["outbox"][busy["index"]]["state"] = "unknown"
                record["busy"] = None
            if record.get("retention"):
                authorize(session, record)
                prune(record)
            if store.clock() >= record["expires_at"] and record["state"] in {"active", "paused"}:
                record["state"] = "expired"
            if record["state"] != "active":
                write(store, identity, record)
                continue
            binding, plugins = authorize(session, record)
            if record["checkpoint"] is not None and record["checkpoint"].get("schema_format") != 2:
                record["state"] = "failed"
                write(store, identity, record)
                continue
            pending = next(
                (i for i, item in enumerate(record["outbox"]) if item["state"] == "pending"), None
            )
            task = "deliver" if pending is not None else "poll"
            if task == "poll" and record["due_at"] > store.clock():
                write(store, identity, record)
                continue
            busy = dict(
                pid=os.getpid(), created=psutil.Process().create_time(), task=task, index=pending
            )
            record["busy"] = busy
            if pending is not None:
                record["outbox"][pending]["state"] = "sending"
                record["outbox"][pending]["attempts"] = (
                    record["outbox"][pending].get("attempts", 0) + 1
                )
            write(store, identity, record)  # Commit sending BEFORE attempting external delivery.
        policy = session.policy
        worker = None
        try:
            worker = PollProcess(policy.timeout_seconds)
            worker.watch_policy(lambda policy=policy: session.policy == policy)
            path = Path(worker.directory.name) / "result.json"
            request = dict(
                task=task,
                control=str(policy.control_directory),
                digest=record["digest"],
                plugins=plugins,
                binding=binding.model_dump(),
                options=record["options"],
                checkpoint=record["checkpoint"],
                output_path=str(path),
                budget=record["budget"],
            )
            if pending is not None:
                request["alert"] = record["outbox"][pending]["alert"]
            await worker.call(request, policy.timeout_seconds)
            result = json.loads(path.read_text(encoding="utf-8"))
            authorize(session, record)
            with store.transaction():
                latest = read(store, identity)
                if latest["busy"] != busy:
                    raise MCPError("invalid_control_state")
                if task == "deliver":
                    if result != {"delivered": True}:
                        raise MCPError("notification_delivery_unknown")
                    latest["outbox"][pending]["state"] = "sent"
                    latest["outbox"][pending]["resolved_at"] = store.clock()
                else:
                    previous_ids = {
                        alert["alert_id"]
                        for alert in (latest["checkpoint"] or {}).get("alerts", [])
                    }
                    if binding.notifier:
                        latest["outbox"].extend(
                            {"state": "pending", "alert": alert}
                            for alert in reversed(result["alerts"])
                            if alert["alert_id"] not in previous_ids
                        )
                    latest["checkpoint"] = result
                    latest["due_at"] = store.clock() + record["options"]["poll_interval"]
                latest["busy"] = None
                write(store, identity, latest)  # Checkpoint and new outbox entries are atomic.
        except BaseException:
            with store.transaction():
                latest = read(store, identity)
                if latest["busy"] == busy:
                    if task == "deliver":
                        latest["outbox"][pending]["state"] = "unknown"
                    else:
                        latest["state"] = "failed"
                    latest["busy"] = None
                    write(store, identity, latest)
            raise
        finally:
            if worker is not None:
                worker.close()
