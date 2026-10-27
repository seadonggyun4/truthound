"""Approved durable watch transitions and native terminal-queue retention."""

import hashlib
from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, model_validator

from truthound.mcp.approval import private_read
from truthound.mcp.contracts import WireModel
from truthound.mcp.operations import envelope
from truthound.mcp.policy import MCPError


class QueueRetention(WireModel):
    max_age_minutes: Annotated[int, Field(ge=1, le=5256000)] | None = None
    max_count: Annotated[int, Field(ge=1, le=1000)] | None = None
    mode: Literal["all", "any"] = "all"

    @model_validator(mode="after")
    def required(self):
        if self.max_age_minutes is None and self.max_count is None:
            raise ValueError("At least one retention limit is required")
        return self


class WatchUpdate(WireModel):
    watch_id: Annotated[str, Field(pattern=r"^watch_[0-9a-f]{64}$")]
    expected_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    idempotency_key: Annotated[str, Field(min_length=8, max_length=128)]
    action: Literal[
        "pause",
        "resume",
        "confirm_delivery",
        "discard",
        "retry",
        "retention",
        "renew",
        "restart",
        "delete",
        "history_retention",
    ]
    alert_id: Annotated[str, Field(min_length=1, max_length=256)] | None = None
    evidence_reference: Annotated[str, Field(min_length=1, max_length=256)] | None = None
    accept_duplicate_risk: bool = False
    retention: QueueRetention | None = None
    accept_history_loss: bool = False
    discard_unresolved: bool = False
    reset_checkpoint: bool = False
    lifetime_seconds: Annotated[int, Field(ge=1, le=2592000)] | None = None

    @model_validator(mode="after")
    def consistent(self):
        if self.reset_checkpoint and self.action != "restart":
            raise ValueError("Checkpoint reset is only available during restart")
        if self.accept_history_loss != (self.action == "delete" or self.reset_checkpoint):
            raise ValueError(
                "Deletion and checkpoint reset require explicit history-loss acceptance"
            )
        if self.discard_unresolved and self.action != "delete":
            raise ValueError("Unresolved outbox deletion requires definition deletion")
        if (self.action == "renew") != (self.lifetime_seconds is not None):
            raise ValueError("Only renewal requires an explicit lifetime")
        reconciliation = self.action in {"confirm_delivery", "discard", "retry"}
        if reconciliation != (self.alert_id is not None and self.evidence_reference is not None):
            raise ValueError("Reconciliation requires an alert and operator evidence reference")
        if not reconciliation and (
            self.alert_id is not None or self.evidence_reference is not None
        ):
            raise ValueError("Unexpected reconciliation fields")
        if self.accept_duplicate_risk != (self.action == "retry"):
            raise ValueError("Retry requires explicit duplicate-risk acceptance")
        if self.action not in {"retention", "history_retention"} and self.retention is not None:
            raise ValueError("Unexpected retention policy")
        return self


def prune(record):
    """Only terminal, timestamped queue records; never source history or unknown sends."""
    from truthound.stores.retention.base import ItemMetadata, PolicyMode
    from truthound.stores.retention.policies import (
        CompositePolicy,
        CountBasedPolicy,
        TimeBasedPolicy,
    )

    policy = record.get("retention")
    if policy is None or record["busy"]:
        return
    items = [
        ItemMetadata(
            item["alert"]["alert_id"], "schema-outbox", datetime.fromtimestamp(item["resolved_at"])
        )
        for item in record["outbox"]
        if item["state"] in {"sent", "confirmed", "discarded"} and "resolved_at" in item
    ]
    policies = []
    if policy["max_age_minutes"] is not None:
        policies.append(TimeBasedPolicy(max_age_minutes=policy["max_age_minutes"]))
    if policy["max_count"] is not None:
        policies.append(CountBasedPolicy(policy["max_count"]))
    combined = CompositePolicy(policies, mode=PolicyMode(policy["mode"]))
    combined.prepare_batch(items)
    removed = {item.item_id for item in items if not combined.should_retain(item)}
    record["outbox"] = [
        item for item in record["outbox"] if item["alert"]["alert_id"] not in removed
    ]
    record["pruned_notifications"] = record.get("pruned_notifications", 0) + len(removed)


def alert_counter(record):
    previous = record.get("checkpoint") or {}
    alerts = list(previous.get("alerts", [])) + [item["alert"] for item in record["outbox"]]
    return max(
        record.get("alert_counter", 0),
        previous.get("alert_counter", 0),
        *(
            int(item["alert_id"][6:])
            for item in alerts
            if item["alert_id"].startswith("ALERT-") and item["alert_id"][6:].isdigit()
        ),
    )


def prune_history(record):
    """Use Core retention decisions without resetting the polling baseline or counters."""
    from truthound.stores.retention.base import ItemMetadata, PolicyMode
    from truthound.stores.retention.policies import (
        CompositePolicy,
        CountBasedPolicy,
        TimeBasedPolicy,
    )

    policy = record.get("history_retention")
    checkpoint = record.get("checkpoint")
    if policy is None or checkpoint is None or record["busy"]:
        return
    record["alert_counter"] = alert_counter(record)
    checkpoint["alert_counter"] = record["alert_counter"]
    protected_alerts = {item["alert"]["alert_id"] for item in record["outbox"]}
    for field, identity, timestamp in (
        ("versions", "version_id", "created_at"),
        ("alerts", "alert_id", "detected_at"),
    ):
        values = checkpoint[field]
        policies = []
        if policy["max_count"] is not None:
            policies.append(CountBasedPolicy(policy["max_count"]))
        if policy["max_age_minutes"] is not None:
            policies.append(TimeBasedPolicy(max_age_minutes=policy["max_age_minutes"]))
        combined = CompositePolicy(policies, mode=PolicyMode(policy["mode"]))
        items = [
            ItemMetadata(
                value[identity], "schema-" + field, datetime.fromisoformat(value[timestamp])
            )
            for value in values
        ]
        combined.prepare_batch(items)
        retained = {item.item_id for item in items if combined.should_retain(item)}
        if field == "versions" and values:
            retained.add(values[0][identity])  # Native history lists newest first.
        if field == "alerts":
            retained.update(protected_alerts)
        checkpoint[field] = [value for value in values if value[identity] in retained]
        record["pruned_history"] = (
            record.get("pruned_history", 0) + len(values) - len(checkpoint[field])
        )
        if field == "versions":
            checkpoint["version_schemas"] = {
                key: value
                for key, value in checkpoint["version_schemas"].items()
                if key in retained
            }


async def update(session, arguments):
    from truthound.mcp.durable_schema import authorize, info, read, storage, write

    store = storage(session)
    stamp = session._policy_stamp()
    request = {
        "update": arguments.model_dump(exclude={"idempotency_key"}),
        "policy": stamp,
        "bindings": hashlib.sha256(
            private_read(session.policy.control_directory / "bindings.json", 1024**2)
        ).hexdigest(),
    }
    if arguments.action == "delete":
        replay = store.replay_committed(
            "schema_watch_update", key=arguments.idempotency_key, request=request
        )
        if replay is not None:
            return replay
    record = read(store, arguments.watch_id)
    authorize(session, record)
    plan = store.prepare(
        "schema_watch_update",
        key=arguments.idempotency_key,
        request=request,
        policy=stamp,
        session=session.identity,
        effects=["durable_watch_" + arguments.action]
        + (["checkpoint_history_deletion"] if arguments.accept_history_loss else [])
        + (["unresolved_notification_deletion"] if arguments.discard_unresolved else [])
        + (
            ["external_notification_retry", "possible_duplicate_delivery"]
            if arguments.action == "retry"
            else ["future_" + arguments.action + "_payload_deletion"]
            if arguments.action in {"retention", "history_retention"} and arguments.retention
            else []
        ),
    )
    if "committed_result" in plan:
        return plan["committed_result"]
    if store.digest(record) != arguments.expected_revision:
        raise MCPError("plan_stale")
    if plan["state"] != "approved":
        return envelope("schema_watch_update", data=plan, error="approval_required")
    # No external effect: transaction includes approval claim, state and result.
    with store.transaction():
        record = read(store, arguments.watch_id)
        authorize(session, record)
        if store.digest(record) != arguments.expected_revision:
            raise MCPError("plan_stale")
        if record["busy"] and arguments.action != "pause":
            raise MCPError("busy")
        action = arguments.action
        if action in {"pause", "resume"}:
            required = "active" if action == "pause" else "paused"
            if record["state"] != required or store.clock() >= record["expires_at"]:
                raise MCPError("invalid_control_state")
            record["state"] = "paused" if action == "pause" else "active"
        elif action == "renew":
            if record["state"] not in {"active", "paused", "expired", "failed"}:
                raise MCPError("invalid_control_state")
            expired = store.clock() >= record["expires_at"]
            record["expires_at"] = store.clock() + arguments.lifetime_seconds
            if record["state"] == "expired" or (expired and record["state"] == "active"):
                record["state"] = "paused"
        elif action == "restart":
            if record["state"] != "failed" or store.clock() >= record["expires_at"]:
                raise MCPError("invalid_control_state")
            if arguments.reset_checkpoint:
                record["alert_counter"] = alert_counter(record)
                record["checkpoint"] = None
            if record["checkpoint"] is not None and record["checkpoint"].get("schema_format") != 2:
                raise MCPError("invalid_control_state")
            record["state"] = "paused"
            record["due_at"] = store.clock()
        elif action == "delete":
            if record["state"] == "active":
                raise MCPError("invalid_control_state")
            if not arguments.discard_unresolved and any(
                item["state"] not in {"sent", "confirmed", "discarded"} for item in record["outbox"]
            ):
                raise MCPError("invalid_control_state")
            record["state"] = "deleted"
        elif action == "history_retention":
            record["history_retention"] = (
                arguments.retention.model_dump() if arguments.retention else None
            )
            prune_history(record)
        elif action == "retention":
            record["retention"] = arguments.retention.model_dump() if arguments.retention else None
            prune(record)
        else:
            matches = [
                item for item in record["outbox"] if item["alert"]["alert_id"] == arguments.alert_id
            ]
            if len(matches) != 1 or matches[0]["state"] != "unknown":
                raise MCPError("invalid_control_state")
            if action == "retry" and (
                record["state"] not in {"active", "paused"} or store.clock() >= record["expires_at"]
            ):
                raise MCPError("invalid_control_state")
            item = matches[0]
            item.setdefault("decisions", []).append(
                {
                    "action": action,
                    "evidence_reference": arguments.evidence_reference,
                    "plan_id": plan["plan_id"],
                    "at": store.clock(),
                }
            )
            item["state"] = {
                "retry": "pending",
                "confirm_delivery": "confirmed",
                "discard": "discarded",
            }[action]
            if action != "retry":
                item["resolved_at"] = store.clock()
        result = envelope("schema_watch_update", data=info(arguments.watch_id, record, store))
        replay = store.begin(
            plan["plan_id"], session=session.identity, policy=stamp, in_transaction=True
        )
        if replay is not None:
            return replay
        if action == "delete":
            store.connection.execute(
                "DELETE FROM mcp_schema_watches WHERE identity=?", (arguments.watch_id,)
            )
        else:
            write(store, arguments.watch_id, record)
        store.finish_in_transaction(plan["plan_id"], result)
    return result
