"""Approved data-only snapshots; new sessions never resurrect old handles."""

import hashlib
import hmac
import json
from pathlib import Path
from typing import Annotated

from pydantic import Field, model_validator

from truthound.mcp.advanced_contracts import ModelRestore, ModelSpec
from truthound.mcp.approval import canonical
from truthound.mcp.contracts import WireModel
from truthound.mcp.operations import envelope
from truthound.mcp.policy import MCPError
from truthound.mcp.session_contracts import Handle


class ModelPersist(WireModel):
    resource_id: Handle
    idempotency_key: Annotated[str, Field(min_length=8, max_length=128)]


class ModelResume(WireModel):
    snapshot_id: Annotated[str, Field(pattern=r"^snapshot_[0-9a-f]{64}$")] | None = None
    idempotency_key: Annotated[str, Field(min_length=8, max_length=128)] | None = None

    @model_validator(mode="after")
    def one_identity(self):
        if (self.snapshot_id is None) == (self.idempotency_key is None):
            raise ValueError("Select exactly one snapshot identity or original intent key")
        return self


class ModelSnapshot(WireModel):
    snapshot_id: str
    sha256: str
    bytes: int


def storage(session):
    store = session.approvals()
    store.connection.execute("""CREATE TABLE IF NOT EXISTS model_snapshots (
        identity TEXT PRIMARY KEY, metadata TEXT NOT NULL, payload BLOB NOT NULL,
        signature TEXT NOT NULL
    )""")
    return store


def ancestry(resources, entry):
    entries = {}

    def visit(item):
        if item.id in entries:
            return
        entries[item.id] = item
        for parent in item.parents:
            visit(resources.get(parent))

    visit(entry)
    return list(entries.values())


async def call(session, name, arguments, models):
    store = storage(session)
    policy = session._policy_stamp()
    resources = session.resources
    if name == "models_persist":
        entry = resources.get(arguments.resource_id, "artifact")
        if entry.media_type != "application/vnd.truthound.model+json":
            raise MCPError("invalid_handle")
        inputs = ancestry(resources, entry)
        if any("binding_digest" in item.metadata for item in inputs):
            raise MCPError("unsupported_operation")
        metadata = {
            "specification": entry.metadata["model_specification"],
            "trained": entry.metadata.get("trained", True),
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
            "sha256": entry.sha256,
        }
        ModelSpec.model_validate(metadata["specification"])
        plan = store.prepare(
            name,
            key=arguments.idempotency_key,
            request={"source": entry.id, "snapshot": metadata},
            policy=policy,
            session=session.identity,
            effects=["persistent_model_storage", "session_lifetime_transfer"],
        )
        if "committed_result" in plan:
            return plan["committed_result"]
        if plan["state"] != "approved":
            return envelope(name, data=plan, error="approval_required")
        if entry.size > min(session.policy.max_file_bytes, 16 * 1024**2):
            raise MCPError("resource_exhausted")
        payload = entry.path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry.sha256:
            raise MCPError("resource_changed")
        identity = "snapshot_" + store.digest({"plan": plan["plan_id"]})
        result = envelope(
            name, data={"snapshot_id": identity, "sha256": entry.sha256, "bytes": len(payload)}
        )
        store.begin(plan["plan_id"], session=session.identity, policy=session._policy_stamp())
        try:
            for item in inputs:
                resources.get(item.id)
            with store.transaction():
                size, count = store.connection.execute(
                    "SELECT COALESCE(SUM(length(payload)),0),COUNT(*) FROM model_snapshots"
                ).fetchone()
                if count >= 128 or size + len(payload) > 32 * 1024**2:
                    raise MCPError("resource_exhausted")
                store.connection.execute(
                    "INSERT INTO model_snapshots VALUES (?,?,?,?)",
                    (
                        identity,
                        canonical(metadata),
                        payload,
                        store.digest({"identity": identity, "metadata": metadata}),
                    ),
                )
                store.finish_in_transaction(plan["plan_id"], result)
            return result
        except Exception:
            store.uncertain(plan["plan_id"])
            raise MCPError("outcome_unknown") from None
        except BaseException:
            store.uncertain(plan["plan_id"])
            raise
    snapshot_id = arguments.snapshot_id or store.committed_identity(
        arguments.idempotency_key, "models_persist", "snapshot_id"
    )
    row = store.connection.execute(
        "SELECT * FROM model_snapshots WHERE identity=?", (snapshot_id,)
    ).fetchone()
    if row is None:
        raise MCPError("invalid_handle")
    metadata = json.loads(row["metadata"])
    if not hmac.compare_digest(
        row["signature"], store.digest({"identity": row["identity"], "metadata": metadata})
    ):
        raise MCPError("invalid_control_state")
    if metadata["policy"] != policy:
        raise MCPError("access_denied")
    for origin in metadata["origins"]:
        session.policy.authorize_origin(Path(origin))
    for root in metadata["workspaces"]:
        session.policy.resolve_workspace(root)
    if hashlib.sha256(row["payload"]).hexdigest() != metadata["sha256"]:
        raise MCPError("invalid_control_state")
    ModelSpec.model_validate(metadata["specification"])
    identity, path = resources.reserve("artifact")
    try:
        if len(row["payload"]) > resources.available_bytes:
            raise MCPError("resource_exhausted")
        path.write_bytes(row["payload"])
        resources.commit(
            identity,
            kind="artifact",
            media_type="application/vnd.truthound.model+json",
            origins=tuple(Path(origin) for origin in metadata["origins"]),
            metadata={
                "model_specification": metadata["specification"],
                "trained": metadata["trained"],
                "durable_policy": policy,
                "durable_control": str(session.policy.control_directory),
                "durable_workspaces": metadata["workspaces"],
            },
        )
        result = await models.call("models_restore", ModelRestore(resource_id=identity))
        result["operation_id"] = name
        return result
    except BaseException:
        if identity in resources._entries:
            resources.release(identity)
        raise
    finally:
        resources.discard(identity)
