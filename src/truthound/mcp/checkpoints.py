"""Approved one-shot native Checkpoints, including operator-selected actions."""

from contextlib import suppress

from truthound.mcp.approved_sources import SourceBindingOpen, resolve
from truthound.mcp.operations import envelope
from truthound.mcp.policy import MCPError
from truthound.mcp.schema_monitor import MonitorProcess


class CheckpointRun(SourceBindingOpen):
    use_router: bool = True


class CheckpointProcess(MonitorProcess):
    worker_module = "truthound.mcp.checkpoint_worker"


def authorize(session, metadata):
    if (
        str(session.policy.control_directory) != metadata["binding_control_root"]
        or session._policy_stamp() != metadata["checkpoint_policy"]
        or resolve(session.policy, metadata["checkpoint_binding"], checkpoint=True)[0]
        != metadata["binding_digest"]
    ):
        raise MCPError("access_denied")


async def run(session, arguments):
    resources, policy = session.resources, session.policy
    digest, capability, plugins = resolve(policy, arguments.binding_id, checkpoint=True)
    stamp, store = session._policy_stamp(), session.approvals()
    plan = store.prepare(
        "checkpoints_run",
        key=arguments.idempotency_key,
        request={
            **arguments.model_dump(exclude={"idempotency_key"}),
            "bindings_digest": digest,
        },
        policy=stamp,
        session=session.identity,
        effects=[
            "installed_extension_execute",
            "native_checkpoint_validation",
            "operator_configured_actions_and_external_notifications",
            "operator_configured_native_action_retries",
            "native_checkpoint_resource_cleanup",
        ],
    )
    if "committed_result" in plan:
        response = plan["committed_result"]
        resources.get(response["data"]["resource_id"], "artifact")
        return response
    if plan["state"] != "approved":
        return envelope("checkpoints_run", data=plan, error="approval_required")
    identity, path = resources.reserve("artifact")
    worker = entry = None
    started = False
    metadata = {
        "checkpoint_binding": arguments.binding_id,
        "checkpoint_policy": stamp,
        "binding_digest": digest,
        "binding_control_root": str(policy.control_directory),
    }
    try:
        authorize(session, metadata)
        store.begin(plan["plan_id"], session=session.identity, policy=stamp)
        started = True
        worker = CheckpointProcess(policy.handle_ttl_seconds)
        session._checkpoint_workers.add(worker)

        def authorized():
            try:
                authorize(session, metadata)
                return True
            except (MCPError, OSError, ValueError):
                return False

        worker.watch_policy(authorized)
        await worker.call(
            {
                "plugins": plugins,
                "capability": capability,
                "control": str(policy.control_directory),
                "digest": digest,
                "run_id": plan["plan_id"],
                "use_router": arguments.use_router,
                "output_path": str(path),
                "budget": resources.available_bytes,
            },
            policy.timeout_seconds,
        )
        authorize(session, metadata)
        entry = resources.commit(
            identity,
            kind="artifact",
            media_type="application/vnd.truthound.checkpoint-result+json",
            metadata=metadata,
        )
        response = envelope("checkpoints_run", data=session._info(entry))
        store.finish(plan["plan_id"], response)
        return response
    except BaseException:
        if entry is not None:
            with suppress(MCPError):
                resources.release_owned(identity)
        if started:
            store.uncertain(plan["plan_id"])
            raise MCPError("outcome_unknown") from None
        raise
    finally:
        if worker is not None:
            worker.close()
            session._checkpoint_workers.discard(worker)
        resources.discard(identity)
