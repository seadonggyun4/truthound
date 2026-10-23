"""Approved durable watch transitions and native terminal-queue retention."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, model_validator

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
    action: Literal["pause", "resume", "confirm_delivery", "discard", "retry", "retention"]
    alert_id: Annotated[str, Field(min_length=1, max_length=256)] | None = None
    evidence_reference: Annotated[str, Field(min_length=1, max_length=256)] | None = None
    accept_duplicate_risk: bool = False
    retention: QueueRetention | None = None

    @model_validator(mode="after")
    def consistent(self):
        reconciliation = self.action in {"confirm_delivery", "discard", "retry"}
        if reconciliation != (self.alert_id is not None and self.evidence_reference is not None):
            raise ValueError("Reconciliation requires an alert and operator evidence reference")
        if not reconciliation and (
            self.alert_id is not None or self.evidence_reference is not None
        ):
            raise ValueError("Unexpected reconciliation fields")
        if self.accept_duplicate_risk != (self.action == "retry"):
            raise ValueError("Retry requires explicit duplicate-risk acceptance")
        if self.action != "retention" and self.retention is not None:
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


async def update(session, arguments):
    from truthound.mcp.durable_schema import authorize, info, read, storage, write

    store = storage(session)
    record = read(store, arguments.watch_id)
    authorize(session, record)
    stamp = session._policy_stamp()
    plan = store.prepare(
        "schema_watch_update",
        key=arguments.idempotency_key,
        request={"update": arguments.model_dump(exclude={"idempotency_key"}), "policy": stamp},
        policy=stamp,
        session=session.identity,
        effects=["durable_watch_" + arguments.action]
        + (
            ["external_notification_retry", "possible_duplicate_delivery"]
            if arguments.action == "retry"
            else ["future_terminal_queue_payload_deletion"]
            if arguments.action == "retention" and arguments.retention
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
        write(store, arguments.watch_id, record)
        store.finish_in_transaction(plan["plan_id"], result)
    return result
