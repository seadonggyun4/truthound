"""Native schema-audit documents and reports with owned, typed artifacts."""

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Annotated

from pydantic import Field, model_validator

from truthound.mcp.contracts import Name, WireModel
from truthound.mcp.operations import envelope
from truthound.mcp.policy import MCPError
from truthound.mcp.resources import BoundedWriter
from truthound.mcp.session_contracts import Handle

DOCUMENT_MEDIA = "application/vnd.truthound.schema-audit-document+json"


class AuditCreate(WireModel):
    document_json: Annotated[str, Field(min_length=2, max_length=65536)]


class AuditDocument(WireModel):
    resource_id: Handle


class TermBinding(WireModel):
    object_id: Name
    term_id: Name
    standard_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    design_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    approval_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class AuditStandards(WireModel):
    standard_id: Handle
    design_id: Handle
    binding_id: Name | None = None
    idempotency_key: Name | None = None

    @model_validator(mode="after")
    def approval_key(self):
        if self.binding_id is not None and self.idempotency_key is None:
            raise ValueError("An operator mapping requires an idempotency key")
        return self


class AuditImplementation(WireModel):
    design_id: Handle
    catalog_id: Handle


async def call(session, name, arguments):
    resources = session.resources
    entries, bindings, binding_digest = [], [], None
    request = {
        "operation": name,
        "arguments": arguments.model_dump(),
        "contract_version": 1,
        "max_response_bytes": session.policy.max_response_bytes,
        "max_items": session.policy.max_items,
    }
    for field in ("resource_id", "standard_id", "design_id", "catalog_id"):
        identity = getattr(arguments, field, None)
        if identity is not None:
            entry = resources.get(identity, "artifact")
            if entry.media_type != DOCUMENT_MEDIA:
                raise MCPError("invalid_handle")
            entries.append(entry)
            request[field.removesuffix("_id") + "_path"] = str(entry.path)
    if getattr(arguments, "binding_id", None) is not None:
        from truthound.mcp.bindings import load

        document, binding_digest = load(session.policy)
        if arguments.binding_id not in document.schema_audit:
            raise MCPError("invalid_binding")
        bindings = [row.model_dump() for row in document.schema_audit[arguments.binding_id]]
    request["bindings"] = bindings
    store = plan = None
    if binding_digest is not None:
        store = session.approvals()
        policy = session._policy_stamp(bindings=binding_digest)
        plan = store.prepare(
            name,
            key=arguments.idempotency_key,
            request={"arguments": arguments.model_dump(), "inputs": [e.sha256 for e in entries]},
            policy=policy,
            session=session.identity,
            effects=["apply_approved_mapping"],
        )
        if "committed_result" in plan:
            result = plan["committed_result"]
            resources.get(result["data"]["resource_id"], "artifact")
            return result
        if plan["state"] != "approved":
            return envelope(name, data=plan, error="approval_required")
        store.begin(plan["plan_id"], session=session.identity, policy=policy)
    identity = None
    try:
        identity, path = resources.reserve("artifact")
        if resources.available_bytes <= 0:
            raise MCPError("resource_exhausted")
        request.update(output_path=str(path), output_budget=resources.available_bytes)
        with tempfile.TemporaryDirectory(prefix="truthound-schema-audit-") as directory:
            result = await session.service._execute_request(request, directory)
        if set(result) != {"media_type", "quality_status", "coverage"}:
            if result.get("execution_status") == "failed":
                from truthound.mcp.catalog import CATALOG

                CATALOG[name].output_type().model_validate(result)
                if store is not None:
                    store.uncertain(plan["plan_id"])
                return result
            raise MCPError("worker_error")
        metadata = {}
        if binding_digest is not None:
            from truthound.mcp.bindings import load

            if load(session.policy)[1] != binding_digest:
                raise MCPError("access_denied")
            metadata.update(
                binding_digest=binding_digest,
                binding_control_root=str(session.policy.control_directory),
            )
        entry = resources.commit(
            identity,
            kind="artifact",
            media_type=result["media_type"],
            parents=tuple(dict.fromkeys(e.id for e in entries)),
            metadata=metadata,
        )
        digest = (
            hashlib.sha256("".join(e.sha256 for e in entries).encode()).hexdigest()
            if entries
            else hashlib.sha256(arguments.document_json.encode("utf-8")).hexdigest()
            if name == "schema_audit_create"
            else None
        )
        response = envelope(
            name,
            data=session._info(entry),
            quality=result["quality_status"],
            coverage=result["coverage"],
            provenance=session._provenance(digest),
        )
        if store is not None:
            store.finish(plan["plan_id"], response)
        return response
    except BaseException:
        if store is not None:
            store.uncertain(plan["plan_id"])
        raise
    finally:
        if identity is not None:
            resources.discard(identity)


def execute(request, arguments):
    from truthound.mcp.encoding import encode
    from truthound.schema_audit import (
        TermBinding as NativeBinding,
    )
    from truthound.schema_audit import (
        audit_implementation,
        audit_standards,
        dump_document,
        json_schema,
        load_document,
        raw_digest,
        semantic_digest,
        to_dict,
    )

    name = request["operation"]
    quality, coverage = "not_applicable", {}
    media = "application/vnd.truthound.schema-audit-result+json"

    def document(key):
        return load_document(Path(request[key + "_path"]).read_bytes())

    if name == "schema_audit_contract":
        value = json_schema()
    elif name == "schema_audit_create":
        value = to_dict(load_document(arguments.document_json.encode("utf-8")))
        media = DOCUMENT_MEDIA
    elif name == "schema_audit_inspect":
        native = document("resource")
        raw = Path(request["resource_path"]).read_bytes()
        value = {
            "document": to_dict(native),
            "raw_digest": raw_digest(raw),
            "semantic_digest": semantic_digest(native),
            "canonical_json": dump_document(native).decode("utf-8"),
        }
    else:
        if name == "schema_audit_standards":
            result = audit_standards(
                document("standard"),
                document("design"),
                bindings=tuple(NativeBinding(**row) for row in request["bindings"]),
            )
        elif name == "schema_audit_implementation":
            result = audit_implementation(document("design"), document("catalog"))
        else:
            raise MCPError("unknown_operation")
        value = result.as_dict()
        quality = {"CONFORMANT": "pass", "NONCONFORMANT": "fail", "INDETERMINATE": "unknown"}[
            result.verdict
        ]
        coverage = {
            "target_count": result.target_count,
            "counts": value["counts"],
            "rules": value["coverage"],
        }
    raw = json.dumps(
        value if media == DOCUMENT_MEDIA else encode(value), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    with BoundedWriter(Path(request["output_path"]), request["output_budget"]) as output:
        output.write(raw)
    return {"media_type": media, "quality_status": quality, "coverage": coverage}
