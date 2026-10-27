"""Durable multipart intent/receipt ownership and explicit cleanup.

Unknown creation receipts are never inferred from a bucket-wide upload list.
An operator may provide a receipt for the exact journaled key after independent
inspection; this is an ownership assertion, not automatic discovery.
"""

from copy import deepcopy
from datetime import datetime

FIELD = "_truthound_multipart_v1"


def capture(session, document, config):
    entries = document.get(FIELD, {})
    prefix = f"{config.prefix.rstrip('/')}/{session.run_id}/"
    if not isinstance(entries, dict):
        raise ValueError("Invalid native multipart journal")
    for identity, entry in entries.items():
        if (
            not isinstance(identity, str)
            or not identity
            or not isinstance(entry, dict)
            or set(entry) != {"key", "upload_id"}
        ):
            raise ValueError("Invalid native multipart intent")
        key, receipt = entry["key"], entry["upload_id"]
        if (
            not isinstance(key, str)
            or not key.startswith(prefix)
            or not key[len(prefix) :]
            or "/" in key[len(prefix) :]
        ):
            raise ValueError("Native multipart intent is outside its run")
        if receipt is not None and (not isinstance(receipt, str) or not receipt):
            raise ValueError("Invalid native multipart receipt")
    if len({entry["key"] for entry in entries.values()}) != len(entries):
        raise ValueError("Ambiguous native multipart ownership")
    session._multipart_intents = deepcopy(entries)


def document(session, *, recovery_document=None):
    result = session.to_dict()
    if recovery_document is not None and "_truthound_recovery_v1" in recovery_document:
        result["_truthound_recovery_v1"] = deepcopy(recovery_document["_truthound_recovery_v1"])
    entries = getattr(session, "_multipart_intents", {})
    if entries:
        result[FIELD] = deepcopy(entries)
    return result


def pending(session):
    return getattr(session, "_multipart_intents", {})


def require_clear(session):
    if pending(session):
        raise RuntimeError("Native S3 multipart cleanup is pending; explicit recovery is required")


def abort(client, config, key, upload_id, guard):
    guard()
    try:
        response = client.abort_multipart_upload(Bucket=config.bucket, Key=key, UploadId=upload_id)
    except Exception as error:
        missing = getattr(client.exceptions, "NoSuchUpload", None)
        if missing is None or not isinstance(error, missing):
            raise
        acknowledged = False
    else:
        if response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 204:
            raise RuntimeError("Native multipart abort has no confirmed receipt")
        acknowledged = True
    guard()
    return acknowledged


def recover(store, run_id, *, expected_fingerprint, receipts=None):
    from truthound.stores.streaming.s3 import _publish_manifest
    from truthound.stores.streaming.s3_recovery import _capture, status

    session, captured, _, fingerprint = _capture(store, run_id)
    if fingerprint != expected_fingerprint:
        raise RuntimeError("Native S3 recovery evidence changed")
    entries = pending(session)
    supplied = receipts or {}
    if not isinstance(supplied, dict) or set(supplied) - {
        entry["key"] for entry in entries.values()
    }:
        raise ValueError("Multipart receipts must match journaled run keys")
    selected = {}
    # Validate every operator receipt before any native cleanup effect.
    for identity, entry in entries.items():
        receipt = entry["upload_id"]
        asserted = supplied.get(entry["key"])
        if receipt is not None and asserted is not None and asserted != receipt:
            raise ValueError("Operator receipt conflicts with the owned receipt")
        receipt = receipt if receipt is not None else asserted
        if not isinstance(receipt, str) or not receipt:
            raise RuntimeError(
                "An unresolved multipart creation requires an operator-selected receipt"
            )
        selected[identity] = {"key": entry["key"], "upload_id": receipt}
    client, config = store._get_s3_client(), store.config
    if selected:
        # Persist newly asserted receipts before abort. A process loss after
        # abort can then safely repeat only this exact owned upload cleanup.
        session.updated_at = datetime.now()
        session._multipart_intents = selected
        _publish_manifest(client, config, session, store._guard_io, recovery_document=captured)
    acknowledged = 0
    for identity, entry in selected.items():
        acknowledged += abort(client, config, entry["key"], entry["upload_id"], store._guard_io)
        remaining = dict(pending(session))
        del remaining[identity]
        session._multipart_intents = remaining
        _publish_manifest(client, config, session, store._guard_io, recovery_document=captured)
    result = status(store, run_id)
    result["aborted_uploads"] = acknowledged
    result["resolved_uploads"] = len(selected)
    return result
