"""Explicit native retention selection and approved transactional payload deletion."""

import hashlib
import hmac
import json
from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, model_validator

from truthound.mcp.contracts import WireModel
from truthound.mcp.operations import envelope
from truthound.mcp.policy import MCPError

Identity = Annotated[str, Field(pattern=r"^(snapshot|job)_[0-9a-f]{64}$")]


class RetentionPreview(WireModel):
    kind: Literal["models", "jobs"]
    max_age_minutes: Annotated[int, Field(ge=1, le=5256000)] | None = None
    max_count: Annotated[int, Field(ge=1, le=128)] | None = None

    @model_validator(mode="after")
    def policy(self):
        if self.max_age_minutes is None and self.max_count is None:
            raise ValueError("At least one native retention policy is required")
        return self


class StorageDelete(WireModel):
    identities: Annotated[list[Identity], Field(min_length=1, max_length=128)]
    idempotency_key: Annotated[str, Field(min_length=8, max_length=128)]

    @model_validator(mode="after")
    def unique(self):
        if len(set(self.identities)) != len(self.identities):
            raise ValueError("Duplicate storage identity")
        return self


class RetentionResult(WireModel):
    candidates: list[str]
    preserved: list[str]
    excluded: list[str]


class DeleteResult(WireModel):
    deleted: list[str]


def read(session, identity):
    from truthound.mcp.durable_jobs import authorize
    from truthound.mcp.durable_jobs import read as read_job
    from truthound.mcp.durable_models import storage

    store = storage(session)
    if identity.startswith("job_"):
        record, payload, output = read_job(store, identity)
        authorize(session, record)
        return record, len(payload) + len(output or b""), "validation_jobs"
    row = store.connection.execute(
        "SELECT * FROM model_snapshots WHERE identity=?", (identity,)
    ).fetchone()
    if row is None:
        raise MCPError("invalid_handle")
    record = json.loads(row["metadata"])
    if (
        not hmac.compare_digest(
            row["signature"], store.digest({"identity": identity, "metadata": record})
        )
        or hashlib.sha256(row["payload"]).hexdigest() != record["sha256"]
    ):
        raise MCPError("invalid_control_state")
    authorize(session, record)
    for root in record["workspaces"]:
        session.policy.resolve_workspace(root)
    return record, len(row["payload"]), "model_snapshots"


def eligible(record):
    return record.get("state") not in {"queued", "running", "outcome_unknown"}


async def call(session, name, arguments):
    from truthound.mcp.durable_jobs import storage
    from truthound.mcp.durable_models import storage as model_storage
    from truthound.stores.retention.base import ItemMetadata
    from truthound.stores.retention.policies import CountBasedPolicy, TimeBasedPolicy

    store = storage(session)
    model_storage(session)
    if name == "storage_retention_preview":
        table = "model_snapshots" if arguments.kind == "models" else "validation_jobs"
        excluded, items = [], []
        for row in store.connection.execute(f"SELECT identity FROM {table} ORDER BY identity"):
            identity = row[0]
            record, size, _ = read(session, identity)
            if not eligible(record) or "created_at" not in record:
                excluded.append(identity)
                continue
            items.append(
                ItemMetadata(
                    identity, arguments.kind, datetime.fromtimestamp(record["created_at"]), size
                )
            )
        policies = []
        if arguments.max_age_minutes is not None:
            policies.append(TimeBasedPolicy(max_age_minutes=arguments.max_age_minutes))
        if arguments.max_count is not None:
            count = CountBasedPolicy(arguments.max_count)
            count.prepare_batch(items)
            policies.append(count)
        candidates, preserved = [], []
        for item in items:
            (
                preserved if all(policy.should_retain(item) for policy in policies) else candidates
            ).append(item.item_id)
        return envelope(
            name, data=dict(candidates=candidates, preserved=preserved, excluded=excluded)
        )
    identities = sorted(arguments.identities)
    # Intent is stable after deletion so a lost successful reply is recoverable.
    plan = store.prepare(
        name,
        key=arguments.idempotency_key,
        request={"identities": identities, "policy": session._policy_stamp()},
        policy=session._policy_stamp(),
        session=session.identity,
        effects=["permanent_payload_deletion"],
    )
    if "committed_result" in plan:
        return plan["committed_result"]
    for identity in identities:
        record, _, _ = read(session, identity)
        if not eligible(record):
            raise MCPError("busy")
    if plan["state"] != "approved":
        return envelope(name, data=plan, error="approval_required")
    store.begin(plan["plan_id"], session=session.identity, policy=session._policy_stamp())
    result = envelope(name, data={"deleted": identities})
    try:
        with store.transaction():
            for identity in identities:
                record, _, table = read(session, identity)
                if not eligible(record):
                    raise MCPError("busy")
                store.connection.execute(f"DELETE FROM {table} WHERE identity=?", (identity,))
            store.finish_in_transaction(plan["plan_id"], result)
    except Exception:
        store.uncertain(plan["plan_id"])
        raise MCPError("outcome_unknown") from None
    except BaseException:
        store.uncertain(plan["plan_id"])
        raise
    return result
