"""Approved, signed native file-history checkpoints, never live shared writers."""

import hashlib
import hmac
import json
from pathlib import Path
from typing import Annotated

from pydantic import Field, model_validator

from truthound.mcp.approval import canonical
from truthound.mcp.contracts import WireModel
from truthound.mcp.durable_models import ancestry
from truthound.mcp.operations import envelope
from truthound.mcp.policy import MCPError
from truthound.mcp.schema_history import HistoryAction, HistoryCreate
from truthound.mcp.session_contracts import Handle


class HistoryPersist(WireModel):
    history_id: Handle
    expected_revision: Annotated[int, Field(ge=0)]
    idempotency_key: Annotated[str, Field(min_length=8, max_length=128)]


class HistoryResume(WireModel):
    snapshot_id: Annotated[str, Field(pattern=r"^historysnapshot_[0-9a-f]{64}$")] | None = None
    idempotency_key: Annotated[str, Field(min_length=8, max_length=128)] | None = None

    @model_validator(mode="after")
    def identity(self):
        if (self.snapshot_id is None) == (self.idempotency_key is None):
            raise ValueError("Select one snapshot identity or original intent key")
        return self


def storage(session):
    store = session.approvals()
    store.connection.execute("""CREATE TABLE IF NOT EXISTS history_snapshots (
        identity TEXT PRIMARY KEY, metadata TEXT NOT NULL, payload BLOB NOT NULL,
        signature TEXT NOT NULL
    )""")
    return store


def read(session, identity):
    from truthound.mcp.durable_jobs import authorize

    store = storage(session)
    row = store.connection.execute(
        "SELECT * FROM history_snapshots WHERE identity=?", (identity,)
    ).fetchone()
    if row is None:
        raise MCPError("invalid_handle")
    metadata = json.loads(row["metadata"])
    if (
        not hmac.compare_digest(
            row["signature"], store.digest({"identity": identity, "metadata": metadata})
        )
        or hashlib.sha256(row["payload"]).hexdigest() != metadata["sha256"]
    ):
        raise MCPError("invalid_control_state")
    authorize(session, metadata)
    for root in metadata["workspaces"]:
        session.policy.resolve_workspace(root)
    return metadata, row["payload"]


async def call(session, name, arguments, histories):
    resources, policy = session.resources, session._policy_stamp()
    store = storage(session)
    if name == "schema_history_persist":
        entry = resources.get(arguments.history_id, "history")
        configuration = HistoryCreate.model_validate(entry.metadata["history_configuration"])
        if configuration.backend != "file":
            raise MCPError("unsupported_operation")
        if histories.revisions.get(entry.id) != arguments.expected_revision:
            raise MCPError("revision_conflict")
        inputs = ancestry(resources, entry)
        if any(
            "binding_digest" in item.metadata or "source_binding" in item.metadata
            for item in inputs
        ):
            raise MCPError("unsupported_operation")
        metadata = {
            "configuration": configuration.model_dump(),
            "policy": policy,
            "origins": sorted({str(origin) for item in inputs for origin in item.origins}),
            "workspaces": sorted(
                {
                    root
                    for item in inputs
                    for root in (
                        (
                            [item.metadata["workspace_root"]]
                            if "workspace_root" in item.metadata
                            else []
                        )
                        + item.metadata.get("durable_workspaces", [])
                    )
                }
            ),
        }
        plan = store.prepare(
            name,
            key=arguments.idempotency_key,
            request={
                "history": entry.id,
                "revision": arguments.expected_revision,
                "snapshot": metadata,
            },
            policy=policy,
            session=session.identity,
            effects=["persistent_schema_history_storage", "session_lifetime_transfer"],
        )
        if "committed_result" in plan:
            return plan["committed_result"]
        if plan["state"] != "approved":
            return envelope(name, data=plan, error="approval_required")
        exported = await histories.call(
            "schema_history_execute",
            HistoryAction(
                history_id=entry.id,
                expected_revision=arguments.expected_revision,
                command={"action": "snapshot"},
            ),
        )
        artifact = resources.get(exported["data"]["artifact"]["resource_id"], "artifact")
        try:
            if artifact.size > min(session.policy.max_file_bytes, 16 * 1024**2):
                raise MCPError("resource_exhausted")
            payload = artifact.path.read_bytes()
            metadata.update(sha256=hashlib.sha256(payload).hexdigest(), created_at=store.clock())
            identity = "historysnapshot_" + store.digest({"plan": plan["plan_id"]})
            result = envelope(
                name,
                data={"snapshot_id": identity, "sha256": metadata["sha256"], "bytes": len(payload)},
            )
            store.begin(plan["plan_id"], session=session.identity, policy=session._policy_stamp())
            try:
                for item in inputs:
                    resources.get(item.id)
                with store.transaction():
                    size, count = store.connection.execute(
                        "SELECT COALESCE(SUM(length(payload)),0),COUNT(*) FROM history_snapshots"
                    ).fetchone()
                    if count >= 128 or size + len(payload) > 32 * 1024**2:
                        raise MCPError("resource_exhausted")
                    store.connection.execute(
                        "INSERT INTO history_snapshots VALUES (?,?,?,?)",
                        (
                            identity,
                            canonical(metadata),
                            payload,
                            store.digest({"identity": identity, "metadata": metadata}),
                        ),
                    )
                    store.finish_in_transaction(plan["plan_id"], result)
                return result
            except BaseException as exc:
                store.uncertain(plan["plan_id"])
                if isinstance(exc, Exception):
                    raise MCPError("outcome_unknown") from None
                raise
        finally:
            resources.release_owned(artifact.id)
    identity = arguments.snapshot_id or store.committed_identity(
        arguments.idempotency_key, "schema_history_persist", "snapshot_id"
    )
    metadata, payload = read(session, identity)
    configuration = HistoryCreate.model_validate(metadata["configuration"])
    resource_id, path = resources.reserve("artifact")
    try:
        if len(payload) > resources.available_bytes:
            raise MCPError("resource_exhausted")
        path.write_bytes(payload)
        entry = resources.commit(
            resource_id,
            kind="artifact",
            media_type="application/vnd.truthound.schema-history-snapshot+json",
            origins=tuple(Path(origin) for origin in metadata["origins"]),
            metadata={
                "durable_policy": policy,
                "durable_control": str(session.policy.control_directory),
                "durable_workspaces": metadata["workspaces"],
                "durable_identity": identity,
            },
        )
        result = await histories.call("schema_history_create", configuration, restore=entry)
        result["operation_id"] = name
        return result
    except BaseException:
        if resource_id in resources._entries:
            resources.release_owned(resource_id)
        raise
    finally:
        resources.discard(resource_id)
