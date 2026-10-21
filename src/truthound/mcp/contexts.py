"""Explicit Core context lifecycle, with approved effects and owned result artifacts."""

import json
import re
import tempfile
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator

from truthound.mcp.contracts import Name, WireModel
from truthound.mcp.operations import envelope
from truthound.mcp.policy import MCPError
from truthound.mcp.session_contracts import ContextInitialize, Handle, ResourceInfo


class AssetAction(WireModel):
    action: Literal["resolve_source_key", "resolve_fingerprint", "track_asset"]
    asset_id: Handle


class BaselineAction(WireModel):
    action: Literal["get_or_create_schema_baseline"]
    asset_id: Handle
    infer_constraints: bool = True


class AppliedRead(WireModel):
    action: Literal["read_applied_suite"]
    source_key: Name | None = None
    asset_id: Handle | None = None

    @model_validator(mode="after")
    def source(self):
        if (self.source_key is None) == (self.asset_id is None):
            raise ValueError("Select exactly one source identity")
        return self


class AppliedActor(WireModel):
    actor_id: Name
    actor_name: Name


class AppliedCheck(WireModel):
    check_key: Name
    validator_name: Name
    category: Name
    columns: Annotated[list[Name], Field(max_length=100)]
    params: dict[str, JsonValue]
    rationale: str

    @model_validator(mode="after")
    def closed_params(self):
        from truthound.mcp.validator_wire import frame_handles, settings

        values = {**self.params, "columns": self.columns} if self.columns else self.params
        settings(self.validator_name, values)
        if frame_handles(self.validator_name, values):
            raise ValueError("Session frame references cannot become persistent suite references")
        return self


class AppliedRecord(WireModel):
    source_key: Name
    proposal_id: Name
    diff_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    applied_by: AppliedActor
    applied_at: Annotated[str, Field(max_length=100)]
    checks: Annotated[list[AppliedCheck], Field(min_length=1, max_length=1000)]
    effective_suite_snapshot: dict[str, JsonValue]


class AppliedWrite(WireModel):
    action: Literal["write_applied_suite"]
    record: AppliedRecord


class PersistRun(WireModel):
    action: Literal["persist_run", "persist_docs"]
    resource_id: Handle


class MetricGet(WireModel):
    action: Literal["metric_get"]
    key: Name


class MetricWrite(WireModel):
    action: Literal["metric_put", "metric_append_history"]
    key: Name
    value: JsonValue


class WorkspaceList(WireModel):
    action: Literal["list_runs", "list_docs", "list_baselines", "list_applied_suites"]
    offset: Annotated[int, Field(ge=0, le=10000)] = 0
    limit: Annotated[int, Field(ge=1, le=100)] = 50


class ContextAction(ContextInitialize):
    operation: Annotated[
        AssetAction
        | BaselineAction
        | AppliedRead
        | AppliedWrite
        | PersistRun
        | MetricGet
        | MetricWrite
        | WorkspaceList,
        Field(discriminator="action"),
    ]


class ContextActionResult(WireModel):
    artifact: ResourceInfo | None
    artifact_available: bool
    replayed: bool
    plan_id: str


def validate_workspace_references(root):
    """Core indexes contain relative filenames, never ambient filesystem grants."""
    for relative, field in (
        ("baselines/index.json", "schema_file"),
        ("suites/index.json", "suite_file"),
    ):
        path = root / ".truthound" / relative
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            # The separately approved Core operation may quarantine corruption.
            continue
        if isinstance(payload, dict):
            for value in payload.values():
                name = value.get(field) if isinstance(value, dict) else None
                if name is not None and (
                    not isinstance(name, str)
                    or not re.fullmatch(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*", name)
                ):
                    raise MCPError("invalid_workspace_reference")


async def call(session, arguments):
    operation = "contexts_execute"
    resources = session.resources
    context = resources.get(arguments.context_id, "context")
    root = session.policy.resolve_workspace(context.metadata["workspace_root"])
    session._workspace_stamp(root)
    validate_workspace_references(root)
    parents = [context]
    request = {
        "operation": operation,
        "arguments": arguments.model_dump(),
        "context": context.metadata,
        "contract_version": 1,
        "max_items": session.policy.max_items,
        "max_response_bytes": session.policy.max_response_bytes,
    }
    action = arguments.operation
    if hasattr(action, "asset_id") and action.asset_id:
        asset = resources.get(action.asset_id, "asset")
        parents.append(asset)
        request["asset_path"] = str(asset.path)
    if isinstance(action, PersistRun):
        run = resources.get(action.resource_id, "artifact")
        if run.media_type != "application/vnd.truthound.validation+json":
            raise MCPError("invalid_handle")
        parents.append(run)
        request["run_path"] = str(run.path)
    if isinstance(action, WorkspaceList) and action.limit > session.policy.max_items:
        raise MCPError("resource_exhausted")
    store = session.approvals()
    plan = store.prepare(
        operation,
        key=arguments.idempotency_key,
        request={
            "context_id": context.id,
            "action": action.action,
            "arguments_digest": store.digest(arguments.model_dump(exclude={"idempotency_key"})),
            "inputs": {entry.id: entry.sha256 for entry in parents},
        },
        session=session.identity,
        policy=session._policy_stamp(root),
        effects=["workspace_write", "native_context_lifecycle"],
    )
    if "committed_result" in plan:
        response = plan["committed_result"]
        artifact = response["data"]["artifact"]
        response["data"]["replayed"] = True
        if artifact:
            try:
                entry = resources.get(artifact["resource_id"], "artifact")
            except MCPError:
                response["data"].update(artifact=None, artifact_available=False)
            else:
                response["data"]["artifact"] = session._info(entry)
        return response
    if plan["state"] != "approved":
        return envelope(operation, data=plan, error="approval_required")
    identity = None
    store.begin(plan["plan_id"], session=session.identity, policy=session._policy_stamp(root))
    try:
        identity, path = resources.reserve("artifact")
        request.update(output_path=str(path), output_budget=resources.available_bytes)
        with tempfile.TemporaryDirectory(prefix="truthound-mcp-context-effect-") as directory:
            result = await session.service._execute_request(request, directory)
        if result != {"context_action_completed": True}:
            raise MCPError("worker_error")
        session._workspace_stamp(root)
        entry = resources.commit(
            identity,
            kind="artifact",
            media_type="application/vnd.truthound.context-result+json",
            parents=tuple(item.id for item in parents),
        )
        response = envelope(
            operation,
            data={
                "artifact": session._info(entry),
                "artifact_available": True,
                "replayed": False,
                "plan_id": plan["plan_id"],
            },
            provenance={
                "context_mode": "persistent",
                "authorization_plan_id": plan["plan_id"],
                "configuration_identity": store.digest(
                    arguments.model_dump(exclude={"idempotency_key"})
                ),
            },
        )
        if len(json.dumps(response).encode()) > session.policy.max_response_bytes:
            raise MCPError("response_too_large")
        store.finish(plan["plan_id"], response)
        return response
    except Exception:
        store.uncertain(plan["plan_id"])
        return envelope(
            operation,
            data=store.status(plan["plan_id"], session=session.identity),
            error="outcome_unknown",
        )
    except BaseException:
        store.uncertain(plan["plan_id"])
        raise
    finally:
        if identity:
            resources.discard(identity)


def execute(request, arguments):
    from truthound._applied_suite import AppliedSuiteRecord
    from truthound.context import TruthoundContext, TruthoundContextConfig
    from truthound.core.results import ValidationRunResult
    from truthound.mcp.encoding import encode
    from truthound.mcp.resources import BoundedWriter
    from truthound.mcp.tables import decode

    action = arguments.operation
    root = Path(request["context"]["workspace_root"])
    validate_workspace_references(root)
    context = TruthoundContext(root, config=TruthoundContextConfig(**request["context"]["config"]))
    frame = None
    if "asset_path" in request:
        import polars as pl

        frame = pl.read_ipc(request["asset_path"], memory_map=False)
    if action.action == "resolve_source_key":
        result = context.resolve_source_key(data=frame)
    elif action.action == "resolve_fingerprint":
        result = context.resolve_fingerprint(data=frame)
    elif action.action == "track_asset":
        result = context.track_asset(data=frame)
    elif action.action == "get_or_create_schema_baseline":
        schema, created = context.get_or_create_schema_baseline(
            data=frame, infer_constraints=action.infer_constraints
        )
        result = {"schema": schema.to_dict(), "created": created}
    elif action.action == "read_applied_suite":
        value = context.read_applied_suite(source_key=action.source_key, data=frame)
        result = value.to_dict() if value is not None else None
    elif action.action == "write_applied_suite":
        path = context.write_applied_suite(AppliedSuiteRecord.from_dict(action.record.model_dump()))
        result = {"relative_path": path.relative_to(context.workspace_dir).as_posix()}
    elif isinstance(action, PersistRun):
        run = ValidationRunResult.from_dict(
            decode(json.loads(Path(request["run_path"]).read_bytes()))
        )
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", run.run_id):
            raise MCPError("invalid_run_identity")
        path = (
            context.persist_run(run)
            if action.action == "persist_run"
            else context.persist_docs(run)
        )
        result = {"relative_path": path.relative_to(context.workspace_dir).as_posix()}
    elif action.action == "metric_get":
        result = context.metric_repository.get(action.key)
    elif action.action == "metric_put":
        context.metric_repository.put(action.key, action.value)
        result = None
    elif action.action == "metric_append_history":
        context.metric_repository.append_history(action.key, action.value)
        result = None
    elif isinstance(action, WorkspaceList):
        directory = {
            "list_runs": context.runs_dir,
            "list_docs": context.docs_dir,
            "list_baselines": context.baselines_dir,
            "list_applied_suites": context.suites_dir,
        }[action.action]
        paths = (
            sorted(path.name for path in directory.iterdir() if path.is_file())
            if directory.is_dir()
            else []
        )
        page = paths[action.offset : action.offset + action.limit]
        end = action.offset + len(page)
        result = {
            "names": page,
            "total": len(paths),
            "next_offset": end if end < len(paths) else None,
        }
    else:
        raise MCPError("unsupported_operation")
    with BoundedWriter(Path(request["output_path"]), request["output_budget"]) as output:
        output.write(
            json.dumps(
                encode({"action": action.action, "result": result}), allow_nan=False
            ).encode()
        )
    return {"context_action_completed": True}
