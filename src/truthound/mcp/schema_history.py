"""Session-owned native schema history; no caller paths or external stores."""

from contextlib import suppress
from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, JsonValue, field_validator

from truthound.mcp.contracts import WireModel
from truthound.mcp.model_process import OwnedModelProcess
from truthound.mcp.operations import envelope
from truthound.mcp.policy import MCPError
from truthound.mcp.session_contracts import Handle, ResourceInfo


class HistoryCreate(WireModel):
    max_versions: Annotated[int, Field(ge=1, le=1000)] = 100
    strategy: Literal["semantic", "incremental", "timestamp", "git_like"] = "semantic"


class Save(WireModel):
    action: Literal["save"]
    asset_id: Handle
    version: Annotated[str, Field(max_length=256)] | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict, max_length=100)


class VersionAction(WireModel):
    action: Literal["get", "delete", "breaking_since"]
    version_id: Annotated[str, Field(min_length=1, max_length=128)]


class Rollback(WireModel):
    action: Literal["rollback"]
    version_id: Annotated[str, Field(min_length=1, max_length=128)]
    reason: Annotated[str, Field(max_length=1024)] = "manual_rollback"


class Latest(WireModel):
    action: Literal["latest", "baseline", "count"]


class ByVersion(WireModel):
    action: Literal["by_version"]
    version: Annotated[str, Field(max_length=256)]


class ListVersions(WireModel):
    action: Literal["list"]
    limit: Annotated[int, Field(ge=0, le=1000)] | None = None
    since: Annotated[str, Field(max_length=64)] | None = None
    until: Annotated[str, Field(max_length=64)] | None = None

    @field_validator("since", "until")
    @classmethod
    def datetime_input(cls, value):
        if value is not None:
            datetime.fromisoformat(value)
        return value


class Diff(WireModel):
    action: Literal["diff"]
    source_id: Annotated[str, Field(min_length=1, max_length=128)]
    target_id: Annotated[str, Field(min_length=1, max_length=128)] | None = None


class HistoryAction(WireModel):
    history_id: Handle
    expected_revision: Annotated[int, Field(ge=0)]
    command: Annotated[
        Save | VersionAction | Rollback | Latest | ByVersion | ListVersions | Diff,
        Field(discriminator="action"),
    ]


class HistoryState(WireModel):
    history: ResourceInfo
    revision: int
    artifact: ResourceInfo | None = None


class HistoryProcess(OwnedModelProcess):
    worker_module = "truthound.mcp.schema_history_worker"

    def validate_reply(self, result):
        if set(result) != {"ok", "revision"} or type(result["revision"]) is not int:
            raise MCPError("worker_error")
        return result


class HistorySession:
    def __init__(self, session):
        self.session = session
        self.workers = {}
        self.revisions = {}

    def release(self, identity):
        worker = self.workers.pop(identity, None)
        self.revisions.pop(identity, None)
        if worker is not None:
            worker.close()

    def close(self):
        for identity in list(self.workers):
            self.release(identity)

    def reap(self):
        for identity, worker in list(self.workers.items()):
            try:
                self.session.resources.get(identity, "history")
                if worker.closed:
                    raise MCPError("invalid_handle")
            except MCPError:
                self.release(identity)
                with suppress(MCPError):
                    self.session.resources.release_owned(identity)

    async def call(self, operation, arguments):
        session = self.session
        resources, policy = session.resources, session.policy
        if operation == "schema_history_create":
            if arguments.max_versions > policy.max_items:
                raise MCPError("resource_exhausted")
            identity, path = resources.reserve("history")
            worker = None
            try:
                path.write_text(arguments.model_dump_json(), encoding="utf-8")
                worker = HistoryProcess(policy.handle_ttl_seconds)
                await worker.call({"create": arguments.model_dump()}, policy.timeout_seconds)
                entry = resources.commit(
                    identity,
                    kind="history",
                    media_type="application/vnd.truthound.schema-history+json",
                )
                worker.expire_after(max(0, entry.expires - resources.clock()))
                self.workers[identity], self.revisions[identity] = worker, 0
                return envelope(operation, data={"history": session._info(entry), "revision": 0})
            except BaseException:
                if worker is not None:
                    worker.close()
                raise
            finally:
                resources.discard(identity)
        entry = resources.get(arguments.history_id, "history")
        worker = self.workers.get(entry.id)
        if worker is None or worker.closed:
            raise MCPError("invalid_handle")
        if arguments.expected_revision != self.revisions[entry.id]:
            raise MCPError("revision_conflict")
        asset = None
        if arguments.command.action == "save":
            asset = resources.get(arguments.command.asset_id, "asset")
            entry = resources.bind_resident_inputs(entry.id, (asset.id,), kind="history")
            worker.expire_after(max(0, entry.expires - resources.clock()))
        identity, path = resources.reserve("artifact")
        try:
            result = await worker.call(
                {
                    "arguments": arguments.model_dump(mode="json"),
                    "asset_path": str(asset.path) if asset else None,
                    "output_path": str(path),
                    "output_budget": resources.available_bytes,
                },
                policy.timeout_seconds,
            )
            resources.get(entry.id, "history")
            artifact = resources.commit(
                identity,
                kind="artifact",
                parents=(entry.id,),
                media_type="application/vnd.truthound.schema-history-result+json",
            )
            self.revisions[entry.id] = result["revision"]
            return envelope(
                operation,
                data={
                    "history": session._info(resources.get(entry.id)),
                    "revision": result["revision"],
                    "artifact": session._info(artifact),
                },
            )
        except BaseException:
            # A lost mutation reply must never leave a silently advanced handle.
            self.release(entry.id)
            with suppress(MCPError):
                resources.release_owned(entry.id)
            raise
        finally:
            resources.discard(identity)
