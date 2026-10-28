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
    store.connection.execute("""CREATE TABLE IF NOT EXISTS checkpoint_snapshot_retention (
        identity TEXT PRIMARY KEY, metadata TEXT NOT NULL, policy TEXT NOT NULL,
        signature TEXT NOT NULL
    )""")
    return store


def persist(session, plan, metadata, payload):
    store = storage(session)
    if len(payload) > min(1024**2, session.policy.max_file_bytes):
        raise MCPError("resource_exhausted")
    identity = "servicesnapshot_" + store.digest({"plan": plan["plan_id"]})
    metadata = {**metadata, "sha256": hashlib.sha256(payload).hexdigest(), "created_at": store.clock()}
    with store.transaction():
        sweep(session, metadata)
        store.connection.execute(
            "INSERT INTO checkpoint_service_snapshots VALUES (?,?,?,?)",
            (
                identity,
                canonical(metadata),
                payload,
                store.digest({"identity": identity, "metadata": metadata}),
            ),
        )
        sweep(session, metadata)
        size, count = store.connection.execute(
            "SELECT COALESCE(SUM(length(payload)),0),COUNT(*) FROM checkpoint_service_snapshots"
        ).fetchone()
        if count > 128 or size > 16 * 1024**2:
            raise MCPError("resource_exhausted")
        retained = store.connection.execute(
            "SELECT 1 FROM checkpoint_service_snapshots WHERE identity=?",(identity,)
        ).fetchone() is not None
    return {"snapshot_id": identity, "sha256": metadata["sha256"], "bytes": len(payload), "retained":retained}


def read(session, identity, binding_id, *, kind="escalation"):
    store = storage(session)
    row = store.connection.execute(
        "SELECT * FROM checkpoint_service_snapshots WHERE identity=?", (identity,)
    ).fetchone()
    if row is None:
        raise MCPError("invalid_handle")
    return verified(session, row, binding_id, kind=kind)[1]


def verified(session, row, binding_id, *, kind):
    """Authenticate every selected row before disclosure, restore or deletion."""
    store = storage(session)
    try:
        metadata = json.loads(row["metadata"])
        valid = (
            isinstance(metadata, dict)
            and hmac.compare_digest(
                row["signature"], store.digest({"identity": row["identity"], "metadata": metadata})
            )
            and hashlib.sha256(row["payload"]).hexdigest() == metadata["sha256"]
        )
    except (KeyError, TypeError, ValueError):
        raise MCPError("invalid_control_state") from None
    if not valid:
        raise MCPError("invalid_control_state")
    if metadata.get("checkpoint_binding") != binding_id or metadata.get("service_kind") != kind:
        raise MCPError("access_denied")
    authorize(session, metadata)
    if len(row["payload"]) > min(1024**2, session.policy.max_file_bytes):
        raise MCPError("resource_exhausted")
    return metadata, row["payload"]


def manage(session, metadata, arguments, path, budget):
    """Approved metadata-only discovery and atomic, explicitly selected cleanup."""
    from truthound.mcp.resources import BoundedWriter
    from truthound.mcp.encoding import encode

    store = storage(session)
    binding, kind = metadata["checkpoint_binding"], metadata["service_kind"]

    def write(value):
        with BoundedWriter(path, budget) as sink:
            sink.write(json.dumps(encode(value), allow_nan=False).encode())

    with store.transaction():
        if arguments.command.startswith("configuration_retention_"):
            return retention_command(session, metadata, arguments, write)
        if arguments.command == "configuration_list":
            sweep(session, metadata)
            snapshots = []
            for row in store.connection.execute(
                "SELECT * FROM checkpoint_service_snapshots ORDER BY identity"
            ).fetchall():
                # This store is shared by bindings and service kinds. Do not
                # expose foreign entries or their payload/configuration fields.
                try:
                    candidate = json.loads(row["metadata"])
                except (TypeError, ValueError):
                    raise MCPError("invalid_control_state") from None
                if not isinstance(candidate, dict):
                    raise MCPError("invalid_control_state")
                if candidate.get("checkpoint_binding") != binding or candidate.get("service_kind") != kind:
                    continue
                signed, payload = verified(session, row, binding, kind=kind)
                snapshots.append({"snapshot_id": row["identity"], "sha256": signed["sha256"],
                                  "bytes": len(payload), "service_kind": kind})
            start, limit = arguments.configuration_offset, arguments.configuration_limit
            page = snapshots[start:start + limit]
            end = start + len(page)
            write({"snapshots": page, "total": len(snapshots),
                   "next_offset": end if end < len(snapshots) else None})
        else:
            # Verify the whole selection first. Missing, foreign or corrupted
            # rows roll back the batch; no valid prefix may be removed.
            for identity in arguments.snapshot_ids:
                row = store.connection.execute(
                    "SELECT * FROM checkpoint_service_snapshots WHERE identity=?", (identity,)
                ).fetchone()
                if row is None:
                    raise MCPError("invalid_handle")
                verified(session, row, binding, kind=kind)
            # Check the receipt budget before the first irreversible effect.
            write({"deleted": len(arguments.snapshot_ids), "snapshot_ids": arguments.snapshot_ids})
            store.connection.executemany(
                "DELETE FROM checkpoint_service_snapshots WHERE identity=?",
                [(identity,) for identity in arguments.snapshot_ids],
            )


def policy_identity(store, metadata):
    return store.digest({"binding": metadata["checkpoint_binding"], "kind": metadata["service_kind"]})


def retention_policy(session, metadata):
    store = storage(session)
    identity = policy_identity(store, metadata)
    row = store.connection.execute("SELECT * FROM checkpoint_snapshot_retention WHERE identity=?",(identity,)).fetchone()
    if row is None:
        return None
    try:
        signed, policy = json.loads(row["metadata"]), json.loads(row["policy"])
        if not hmac.compare_digest(row["signature"],store.digest({"identity":identity,"metadata":signed,"policy":policy})):
            raise MCPError("invalid_control_state")
        authorize(session,signed)
        from truthound.mcp.checkpoint_services import ConfigurationRetention
        return ConfigurationRetention.model_validate(policy).model_dump()
    except (ValueError,TypeError,KeyError):
        raise MCPError("invalid_control_state") from None


def sweep(session, metadata):
    """Run native retention policies under the caller's SQLite transaction."""
    from datetime import datetime
    import math
    from truthound.stores.retention.base import ItemMetadata, PolicyMode
    from truthound.stores.retention.policies import (TimeBasedPolicy, CountBasedPolicy,
        SizeBasedPolicy, CompositePolicy)
    store=storage(session)
    policy=retention_policy(session,metadata)
    if not policy or not policy["enabled"] or policy["expires_at"] <= store.clock():
        return {"deleted":[],"excluded":[],"active":False}
    items, excluded = [], []
    for row in store.connection.execute("SELECT * FROM checkpoint_service_snapshots ORDER BY identity").fetchall():
        try:
            candidate=json.loads(row["metadata"])
        except (ValueError,TypeError):
            raise MCPError("invalid_control_state") from None
        if not isinstance(candidate,dict):
            raise MCPError("invalid_control_state")
        if (candidate.get("checkpoint_binding"),candidate.get("service_kind")) != (metadata["checkpoint_binding"],metadata["service_kind"]):
            continue
        signed,payload=verified(session,row,metadata["checkpoint_binding"],kind=metadata["service_kind"])
        created=signed.get("created_at")
        if type(created) not in (float,int) or not math.isfinite(created):
            excluded.append(row["identity"])
            continue
        items.append(ItemMetadata(row["identity"],metadata["checkpoint_binding"],datetime.fromtimestamp(created),len(payload)))
    policies=[]
    if policy["max_age_minutes"] is not None:policies.append(TimeBasedPolicy(max_age_minutes=policy["max_age_minutes"]))
    if policy["max_count"] is not None:policies.append(CountBasedPolicy(policy["max_count"]))
    if policy["max_size_bytes"] is not None:policies.append(SizeBasedPolicy(max_size_bytes=policy["max_size_bytes"]))
    native=CompositePolicy(policies,mode=PolicyMode(policy["mode"]))
    native.prepare_batch(items)
    deleted=[item.item_id for item in items if not native.should_retain(item)]
    store.connection.executemany("DELETE FROM checkpoint_service_snapshots WHERE identity=?",[(identity,) for identity in deleted])
    return {"deleted":deleted,"excluded":excluded,"active":True}


def retention_command(session, metadata, arguments, write):
    store=storage(session)
    if arguments.command == "configuration_retention_set":
        policy=arguments.retention.model_dump()
        if policy["expires_at"] <= store.clock():
            raise MCPError("invalid_arguments")
        identity=policy_identity(store,metadata)
        signature=store.digest({"identity":identity,"metadata":metadata,"policy":policy})
        write({"configured":True,"policy":policy})
        store.connection.execute("INSERT OR REPLACE INTO checkpoint_snapshot_retention VALUES (?,?,?,?)",
            (identity,canonical(metadata),canonical(policy),signature))
    elif arguments.command == "configuration_retention_status":
        policy=retention_policy(session,metadata)
        write({"policy":policy,"active":bool(policy and policy["enabled"] and policy["expires_at"]>store.clock())})
    else:
        # Budget and transaction failures must not publish a partial cleanup.
        result=sweep(session,metadata)
        write(result)
