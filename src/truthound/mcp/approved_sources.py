"""Operator-bound native source factories with explicit, session-owned effects."""

import json
from contextlib import suppress
from typing import Annotated

from pydantic import Field

from truthound.mcp.bindings import load
from truthound.mcp.contracts import Name, WireModel
from truthound.mcp.operations import envelope
from truthound.mcp.plugin_bindings import fingerprint
from truthound.mcp.policy import MCPError


class SourceBindingOpen(WireModel):
    binding_id: Name
    idempotency_key: Annotated[str, Field(min_length=8, max_length=128)]


def resolve(policy, binding_id, *, checkpoint=False, group=None):
    document, digest = load(policy)
    groups = {
        "sources": document.sources,
        "checkpoints": document.checkpoints,
        "brokers": document.brokers,
        "dashboards": document.dashboards,
        "security_services": document.security_services,
        "rate_limit_services": document.rate_limit_services,
        "secret_services": document.secret_services,
        "execution_services": document.execution_services,
        "store_services": document.store_services,
        "profiler_services": document.profiler_services,
        "ml_services": document.ml_services,
        "store_utilities": document.store_utilities,
        "streaming_store_services": document.streaming_store_services,
        "tenant_services": document.tenant_services,
        "plugin_services": document.plugin_services,
        "ai_services": document.ai_services,
        "documentation_services": document.documentation_services,
        "benchmark_services": document.benchmark_services,
        "openlineage_services": document.openlineage_services,
        "lineage_services": document.lineage_services,
        "audit_services": document.audit_services,
        "telemetry_services": document.telemetry_services,
    }
    selected = group or ("checkpoints" if checkpoint else "sources")
    if selected not in groups:
        raise MCPError("invalid_binding")
    capability = groups[selected].get(binding_id)
    matches = [
        (name, spec)
        for name, spec in document.plugins.items()
        if capability is not None and capability in spec.capabilities
    ]
    if len(matches) != 1:
        raise MCPError("invalid_binding")
    name, spec = matches[0]
    try:
        package, actual, _ = fingerprint(spec.distribution)
    except (OSError, ValueError):
        raise MCPError("plugin_identity_mismatch") from None
    if package.version != spec.version or actual != spec.sha256:
        raise MCPError("plugin_identity_mismatch")
    return (
        digest,
        capability,
        {name: {**spec.model_dump(), "capabilities": {capability: spec.capabilities[capability]}}},
    )


def authorize(session, metadata):
    if (
        str(session.policy.control_directory) != metadata["binding_control_root"]
        or session._policy_stamp() != metadata["source_policy"]
    ):
        raise MCPError("access_denied")
    if resolve(session.policy, metadata["source_binding"])[0] != metadata["binding_digest"]:
        raise MCPError("access_denied")


async def open_source(controller, arguments):
    from truthound.mcp.sources import SourceProcess

    session = controller.session
    resources, policy = session.resources, session.policy
    digest, capability, plugins = resolve(policy, arguments.binding_id)
    store = session.approvals()
    stamp = session._policy_stamp()
    plan = store.prepare(
        "sources_open_binding",
        key=arguments.idempotency_key,
        request={"binding_id": arguments.binding_id, "bindings_digest": digest},
        policy=stamp,
        session=session.identity,
        effects=[
            "installed_extension_execute",
            "native_datasource_connect",
            "native_datasource_read",
            "native_datasource_close",
        ],
    )
    if "committed_result" in plan:
        response = plan["committed_result"]
        resources.get(response["data"]["resource_id"], "source")
        return response
    if plan["state"] != "approved":
        return envelope("sources_open_binding", data=plan, error="approval_required")
    identity, path = resources.reserve("source")
    worker = entry = None
    started = False
    try:
        path.write_text(json.dumps({"binding_id": arguments.binding_id}), encoding="utf-8")
        metadata = {
            "source_binding": arguments.binding_id,
            "source_policy": stamp,
            "binding_digest": digest,
            "binding_control_root": str(policy.control_directory),
            "asset_id": identity,
            "input_kind": "source",
        }
        authorize(session, metadata)
        store.begin(plan["plan_id"], session=session.identity, policy=stamp)
        started = True
        worker = SourceProcess(policy.handle_ttl_seconds)
        await worker.call(
            {
                "operation": "sources_open_binding",
                "plugins": plugins,
                "capability": capability,
                "public_name": arguments.binding_id,
            },
            policy.timeout_seconds,
        )
        authorize(session, metadata)
        entry = resources.commit(
            identity,
            kind="source",
            media_type="application/vnd.truthound.source+json",
            metadata=metadata,
        )
        worker.expire_after(max(0, entry.expires - resources.clock()))
        controller.workers[identity] = worker
        response = envelope("sources_open_binding", data=session._info(entry))
        store.finish(plan["plan_id"], response)
        return response
    except BaseException:
        if worker is not None:
            worker.close()
        controller.workers.pop(identity, None)
        if entry is not None:
            with suppress(MCPError):
                resources.release_owned(identity)
        if started:
            store.uncertain(plan["plan_id"])
            raise MCPError("outcome_unknown") from None
        raise
    finally:
        resources.discard(identity)
