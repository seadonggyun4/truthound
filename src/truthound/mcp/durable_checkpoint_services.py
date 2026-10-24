"""Signed, immutable policy snapshots; never replay notifications or scheduler jobs."""

import hashlib
import hmac
import json

from truthound.mcp.approval import canonical
from truthound.mcp.checkpoints import authorize
from truthound.mcp.policy import MCPError


def json_configuration(value, depth=0):
    """Reject rich native metadata rather than changing its type in storage."""
    if depth > 32:
        raise MCPError("unsupported_value")
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise MCPError("unsupported_value")
        for item in value.values():
            json_configuration(item, depth + 1)
    elif type(value) is list:
        for item in value:
            json_configuration(item, depth + 1)
    elif value is not None and type(value) not in (str, int, float, bool):
        raise MCPError("unsupported_value")
    return value


def storage(session):
    store = session.approvals()
    store.connection.execute("""CREATE TABLE IF NOT EXISTS checkpoint_service_snapshots (
        identity TEXT PRIMARY KEY, metadata TEXT NOT NULL, payload BLOB NOT NULL,
        signature TEXT NOT NULL
    )""")
    return store


def persist(session, plan, metadata, payload):
    store = storage(session)
    if len(payload) > min(1024**2, session.policy.max_file_bytes):
        raise MCPError("resource_exhausted")
    identity = "servicesnapshot_" + store.digest({"plan": plan["plan_id"]})
    metadata = {**metadata, "sha256": hashlib.sha256(payload).hexdigest()}
    with store.transaction():
        size, count = store.connection.execute(
            "SELECT COALESCE(SUM(length(payload)),0),COUNT(*) FROM checkpoint_service_snapshots"
        ).fetchone()
        if count >= 128 or size + len(payload) > 16 * 1024**2:
            raise MCPError("resource_exhausted")
        store.connection.execute(
            "INSERT INTO checkpoint_service_snapshots VALUES (?,?,?,?)",
            (
                identity,
                canonical(metadata),
                payload,
                store.digest({"identity": identity, "metadata": metadata}),
            ),
        )
    return {"snapshot_id": identity, "sha256": metadata["sha256"], "bytes": len(payload)}


def read(session, identity, binding_id):
    store = storage(session)
    row = store.connection.execute(
        "SELECT * FROM checkpoint_service_snapshots WHERE identity=?", (identity,)
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
    if metadata["checkpoint_binding"] != binding_id or metadata["service_kind"] != "escalation":
        raise MCPError("access_denied")
    if len(row["payload"]) > min(1024**2, session.policy.max_file_bytes):
        raise MCPError("resource_exhausted")
    return row["payload"]
