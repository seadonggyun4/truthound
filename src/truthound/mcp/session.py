"""Session ownership and supervision around explicit retained Core operations."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import secrets
import tempfile
from pathlib import Path

from truthound.mcp.ingestion import capture
from truthound.mcp.operations import envelope
from truthound.mcp.policy import MCPError
from truthound.mcp.resources import SessionResources


class MCPSession:
    def __init__(self, service):
        self.service = service
        self.resources = SessionResources(
            lambda: service.policy,
            before_release=self._before_release,
            authorize_metadata=self._authorize_metadata,
        )
        self.identity = secrets.token_urlsafe(24)
        self._approvals = None
        self._models = None
        self._sources = None
        self._streams = None

    @property
    def policy(self):
        return self.service.policy

    def _authorize_metadata(self, metadata):
        if "source_binding" in metadata:
            from truthound.mcp.approved_sources import authorize

            authorize(self, metadata)
            return
        if (
            str(self.policy.control_directory) != metadata["durable_control"]
            or self._policy_stamp() != metadata["durable_policy"]
        ):
            raise MCPError("access_denied")
        for root in metadata["durable_workspaces"]:
            self.policy.resolve_workspace(root)
        if "durable_identity" in metadata:
            from truthound.mcp.retention import read

            read(self, metadata["durable_identity"])

    def _before_release(self, identity):
        # Windows cannot unlink SQLite or mapped stream inputs while their
        # native reader is alive. Cleanup uses ownership, not a fresh read grant.
        for controller in (self._streams, self._sources, self._models):
            if controller is not None:
                controller.release(identity)

    def close(self):
        if self._streams is not None:
            self._streams.close()
        if self._sources is not None:
            self._sources.close()
        if self._models is not None:
            self._models.close()
        self.resources.close()
        if self._approvals is not None:
            self._approvals.close()

    def approvals(self):
        from truthound.mcp.approval import ApprovalStore

        if self.policy.control_directory is None:
            raise MCPError("operator_configuration_required")
        if (
            self._approvals is not None
            and self._approvals.directory != self.policy.control_directory
        ):
            self._approvals.close()
            self._approvals = None
        if self._approvals is None:
            self._approvals = ApprovalStore(self.policy.control_directory)
        return self._approvals

    def _workspace_stamp(self, root):
        """Reject static workspace link escapes and bind the actual pre-effect state."""
        paths = [root / "truthound.yaml", root / ".truthound"]
        workspace = root / ".truthound"
        if workspace.is_symlink():
            raise MCPError("access_denied")
        if workspace.is_dir():
            for path in workspace.rglob("*"):
                paths.append(path)
                if len(paths) > 10000:
                    raise MCPError("resource_exhausted")
        digest = hashlib.sha256()
        size = 0
        for path in sorted(paths):
            if path.is_symlink():
                raise MCPError("access_denied")
            if not path.exists():
                continue
            digest.update(str(path.relative_to(root)).encode())
            if path.is_file():
                size += path.stat().st_size
                if size > 200 * 1024**2:
                    raise MCPError("resource_exhausted")
                with path.open("rb") as stream:
                    digest.update(hashlib.file_digest(stream, "sha256").digest())
        from truthound.mcp.contexts import validate_workspace_references

        validate_workspace_references(root)
        return digest.hexdigest()

    def _context_info(self, entry):
        self._workspace_stamp(Path(entry.metadata["workspace_root"]))
        return {
            "context_id": entry.id,
            "config": entry.metadata["config"],
            "workspace_exists": (Path(entry.metadata["workspace_root"]) / ".truthound").is_dir(),
        }

    def _policy_stamp(self, root=None, *, bindings=None):
        return self.approvals().digest(
            {
                "data_roots": list(map(str, self.policy.allowed_directories)),
                "workspace_roots": list(map(str, self.policy.workspace_directories)),
                "sensitive": self.policy.allow_sensitive_results,
                "timeout": self.policy.timeout_seconds,
                "profile": self.policy.profile,
                "limits": {
                    key: getattr(self.policy, key)
                    for key in (
                        "max_handles",
                        "max_file_bytes",
                        "max_session_bytes",
                        "max_items",
                        "max_response_bytes",
                        "handle_ttl_seconds",
                    )
                },
                "workspace_state": self._workspace_stamp(root) if root else None,
                "bindings": bindings,
                "control_root": str(self.policy.control_directory)
                if self.policy.control_directory
                else None,
            }
        )

    async def _initialize_context(self, arguments):
        entry = self.resources.get(arguments.context_id, "context")
        root = self.policy.resolve_workspace(entry.metadata["workspace_root"])
        store = self.approvals()
        stamp = self._policy_stamp(root)
        plan = store.prepare(
            "contexts_initialize",
            key=arguments.idempotency_key,
            request={"root": str(root), "config": entry.metadata["config"]},
            policy=stamp,
            session=self.identity,
            effects=["workspace_write"],
        )
        if "committed_result" in plan:
            return envelope(
                "contexts_initialize",
                data={
                    **self._context_info(entry),
                    "plan_id": plan["committed_result"]["plan_id"],
                    "replayed": True,
                },
            )
        if plan["state"] != "approved":
            return envelope("contexts_initialize", data=plan, error="approval_required")
        replay = store.begin(
            plan["plan_id"], session=self.identity, policy=self._policy_stamp(root)
        )
        if replay is not None:
            return envelope(
                "contexts_initialize",
                data={**self._context_info(entry), "plan_id": plan["plan_id"], "replayed": True},
            )
        try:
            with tempfile.TemporaryDirectory(prefix="truthound-mcp-context-") as directory:
                result = await self.service._execute_request(
                    {
                        "operation": "contexts_initialize",
                        "contract_version": 1,
                        "arguments": arguments.model_dump(),
                        "context": entry.metadata,
                        "max_response_bytes": self.policy.max_response_bytes,
                        "max_items": self.policy.max_items,
                    },
                    directory,
                )
            if result != {"workspace_initialized": True}:
                raise MCPError("worker_error")
            self.policy.resolve_workspace(str(root))
            self._workspace_stamp(root)
            store.finish(
                plan["plan_id"], {"plan_id": plan["plan_id"], "workspace_initialized": True}
            )
            return envelope(
                "contexts_initialize",
                data={**self._context_info(entry), "plan_id": plan["plan_id"], "replayed": False},
            )
        except Exception:
            store.uncertain(plan["plan_id"])
            return envelope(
                "contexts_initialize",
                data=store.status(plan["plan_id"], session=self.identity),
                error="outcome_unknown",
            )
        except BaseException:
            store.uncertain(plan["plan_id"])
            raise

    def _info(self, entry):
        return {
            "resource_id": entry.id,
            "kind": entry.kind,
            "media_type": entry.media_type,
            "bytes": entry.size,
            "ttl_seconds": max(0, math.ceil(entry.expires - self.resources.clock())),
            **({"sha256": entry.sha256} if self.policy.allow_sensitive_results else {}),
            **{
                key: entry.metadata[key]
                for key in ("row_count", "column_count")
                if key in entry.metadata
            },
        }

    def _provenance(self, digest=None):
        from truthound import __version__

        return {
            "core_version": __version__,
            "context_mode": "retained",
            **(
                {
                    "input_identity": hmac.new(
                        self.service._identity_key, bytes.fromhex(digest), hashlib.sha256
                    ).hexdigest()
                }
                if digest
                else {}
            ),
            **({"input_sha256": digest} if digest and self.policy.allow_sensitive_results else {}),
        }

    async def _validate(
        self,
        arguments,
        *,
        operation="assets_validate",
        suite_entry=None,
        suite=None,
        public_arguments=None,
        source_entry=None,
        source_worker=None,
        source_input_kind="asset",
    ):
        from truthound.mcp.bindings import load as load_bindings
        from truthound.mcp.bindings import resolve as resolve_bindings
        from truthound.mcp.validator_wire import frame_handles

        if arguments.checks is not None and len(arguments.checks) > self.policy.max_items:
            raise MCPError("resource_exhausted")
        asset = self.resources.get(arguments.asset_id, source_input_kind)
        entries = {asset.id: asset}
        if source_entry is not None:
            entries[source_entry.id] = source_entry
        if suite_entry is not None:
            entries[suite_entry.id] = suite_entry
        frames = {}
        resolved, trusted, binding_digest, plugins, parameter_objects = resolve_bindings(
            self.policy, arguments.checks
        )
        callbacks = {key: True for plugin in plugins.values() for key in plugin["capabilities"]}
        for index, check in enumerate(arguments.checks or ()):
            for handle in frame_handles(
                check.validator,
                resolved[index],
                trusted=index in trusted,
                callbacks=callbacks,
                partial=bool(parameter_objects[index]),
            ):
                reference = self.resources.get(handle, "asset")
                entries[handle] = reference
                frames[handle] = str(reference.path)
        schema = self.resources.get(arguments.schema_id, "schema") if arguments.schema_id else None
        if schema:
            if arguments.schema_spec is not None:
                raise MCPError("invalid_arguments")
            entries[schema.id] = schema
        context = (
            self.resources.get(arguments.context_id, "context") if arguments.context_id else None
        )
        store = plan = None
        root = None
        if context:
            entries[context.id] = context
            root = self.policy.resolve_workspace(context.metadata["workspace_root"])
        if context or trusted:
            store = self.approvals()
            plan = store.prepare(
                operation,
                key=arguments.idempotency_key,
                request={
                    "arguments": arguments.model_dump(
                        exclude={"idempotency_key", "checks", "schema_spec"}
                    ),
                    "arguments_digest": store.digest(
                        arguments.model_dump(exclude={"idempotency_key"})
                    ),
                    "checks": [
                        {
                            "validator": check.validator,
                            "binding_id": check.binding_id,
                            "settings_digest": store.digest(check.settings),
                        }
                        for check in arguments.checks or ()
                    ],
                    "context": {
                        "root": str(root),
                        "config_digest": store.digest(context.metadata["config"]),
                    }
                    if context
                    else None,
                    "inputs": {key: entry.sha256 for key, entry in entries.items()},
                    "constructor_digest": store.digest(resolved),
                    "object_parameter_digest": store.digest(parameter_objects),
                    "binding_digest": binding_digest,
                },
                policy=self._policy_stamp(root, bindings=binding_digest),
                session=self.identity,
                effects=(
                    [
                        "workspace_write",
                        "catalog_write",
                        "baseline_write",
                        "run_and_docs_write",
                    ]
                    if context
                    else []
                )
                + (["execute_trusted_binding"] if trusted else [])
                + (["installed_extension_execute"] if plugins else []),
            )
            if "committed_result" in plan:
                response = plan["committed_result"]
                receipt_identity = {
                    key: response.get("provenance", {})[key]
                    for key in ("authorization_plan_id", "configuration_identity")
                    if key in response.get("provenance", {})
                }
                response["provenance"] = {
                    **receipt_identity,
                    **self._provenance(asset.sha256),
                    "context_mode": "persistent" if context else "retained",
                }
                response["data"]["replayed"] = True
                artifact = response["data"]["artifact"]
                if artifact:
                    try:
                        retained = self.resources.get(artifact["resource_id"], "artifact")
                    except MCPError:
                        # The durable receipt is not permission to repeat a
                        # completed write when its old session artifact expired.
                        response["data"].update(artifact=None, artifact_available=False)
                    else:
                        response["data"]["artifact"] = self._info(retained)
                return response
            if plan["state"] != "approved":
                return envelope(operation, data=plan, error="approval_required")
            if binding_digest and load_bindings(self.policy)[1] != binding_digest:
                raise MCPError("stale_plan")
            store.begin(
                plan["plan_id"],
                session=self.identity,
                policy=self._policy_stamp(root, bindings=binding_digest),
            )
        identity = None
        try:
            identity, output = self.resources.reserve("artifact")
            budget = self.resources.available_bytes
            if budget <= 0:
                raise MCPError("resource_exhausted")
            with tempfile.TemporaryDirectory(prefix="truthound-mcp-validation-") as directory:
                request = {
                    "operation": operation,
                    "contract_version": 1,
                    "arguments": public_arguments.model_dump()
                    if public_arguments
                    else arguments.model_dump(),
                    "asset_path": str(asset.path),
                    "frames": frames,
                    "constructor_settings": resolved,
                    "trusted_indices": sorted(trusted),
                    "plugin_bindings": plugins,
                    "parameter_objects": parameter_objects,
                    "scratch_path": str(Path(directory) / "input.ipc"),
                    "output_path": str(output),
                    "output_budget": budget,
                    "max_items": self.policy.max_items,
                    "max_response_bytes": self.policy.max_response_bytes,
                    "provenance": self._provenance(asset.sha256),
                }
                if schema:
                    request["schema_path"] = str(schema.path)
                if suite is not None:
                    request["kernel_suite"] = suite.model_dump()
                if context:
                    request["context"] = context.metadata
                if source_worker is not None:
                    request["native_asset_id"] = asset.id
                    result = await source_worker.call(request, self.policy.timeout_seconds)
                else:
                    result = await self.service._execute_request(request, directory)
            if (
                set(result)
                != {"validation_created", "summary", "coverage", "core_execution_failed"}
                or result["validation_created"] is not True
            ):
                if not store and result.get("execution_status") == "failed":
                    return result
                raise MCPError("worker_error")
            # A revoked input invalidates publication even if the child finished.
            for entry in entries.values():
                self.resources.get(entry.id)
            artifact = self.resources.commit(
                identity,
                kind="artifact",
                media_type="application/vnd.truthound.validation+json",
                parents=tuple(entries),
                origins=tuple({origin for entry in entries.values() for origin in entry.origins}),
                metadata={
                    "binding_digest": binding_digest,
                    "binding_control_root": str(self.policy.control_directory),
                }
                if binding_digest
                else None,
            )
            response = envelope(
                operation,
                data={
                    "summary": result["summary"],
                    "artifact": self._info(artifact),
                    "artifact_available": True,
                    "replayed": False,
                },
                quality="unknown",
                error="core_execution_failed" if result["core_execution_failed"] else None,
                coverage=result["coverage"],
                provenance=self._provenance(asset.sha256),
            )
            if context:
                response["provenance"]["context_mode"] = "persistent"
            if plan:
                response["provenance"].update(
                    authorization_plan_id=plan["plan_id"],
                    configuration_identity=store.digest(
                        {
                            "settings": resolved,
                            "objects": parameter_objects,
                            "bindings": binding_digest,
                        }
                    ),
                )
            # The full canonical run stays in its owned artifact. Bound only
            # the inline summary, before writing a durable effect receipt.
            while (
                len(json.dumps(response, allow_nan=False).encode()) > self.policy.max_response_bytes
            ):
                checks = response["data"]["summary"]["checks"]
                if not checks:
                    raise MCPError("response_too_large")
                checks.pop()
                response["coverage"].update(returned_checks=len(checks), truncated=True)
            if store:
                if root:
                    self._workspace_stamp(root)
                if binding_digest and load_bindings(self.policy)[1] != binding_digest:
                    raise MCPError("binding_changed")
                store.finish(plan["plan_id"], response)
            return response
        except Exception:
            if not store:
                raise
            store.uncertain(plan["plan_id"])
            return envelope(
                operation,
                data=store.status(plan["plan_id"], session=self.identity),
                error="outcome_unknown",
            )
        except BaseException:
            if store:
                store.uncertain(plan["plan_id"])
            raise
        finally:
            if identity:
                self.resources.discard(identity)

    async def call(self, name, arguments):
        from truthound.mcp.catalog import CATALOG

        if self._models is not None:
            self._models.reap()
        if self._sources is not None:
            self._sources.reap()
        if self._streams is not None:
            self._streams.reap()
        if name == "schedules_preview":
            from truthound.mcp.scheduling import call as schedule_call

            return await schedule_call(self, name, arguments)
        if name.startswith("jobs_") or name == "schedules_register":
            from truthound.mcp.durable_jobs import call as job_call

            return await job_call(self, name, arguments)
        if name.startswith("storage_policy_") or name in {"schedules_status", "schedules_disable"}:
            from truthound.mcp.automation import call as automation_call

            return await automation_call(self, name, arguments)
        if name.startswith("storage_"):
            from truthound.mcp.retention import call as retention_call

            return await retention_call(self, name, arguments)
        if name.startswith("streams_"):
            from truthound.mcp.streams import StreamSession

            if self._streams is None:
                self._streams = StreamSession(self)
            return await self._streams.call(name, arguments)
        if name.startswith("sources_"):
            from truthound.mcp.sources import SourceSession

            if self._sources is None:
                self._sources = SourceSession(self)
            return await self._sources.call(name, arguments)
        if name.startswith("schema_audit_"):
            from truthound.mcp.schema_audit import call as audit_call

            return await audit_call(self, name, arguments)
        if name == "lineage_catalog":
            from truthound.mcp.encoding import encode
            from truthound.mcp.validator_wire import contracts, record_definitions

            types = contracts()["lineage"]
            return envelope(
                name,
                data={
                    "encoding": "tagged-v1",
                    "result": encode(
                        {"constructors": types, "record_types": record_definitions(types)}
                    ),
                },
            )
        if name.startswith("models_"):
            from truthound.mcp.models import ModelSession

            if self._models is None:
                self._models = ModelSession(self)
            return await self._models.call(name, arguments)
        if name == "profilers_catalog":
            from truthound.mcp.encoding import encode
            from truthound.mcp.validator_wire import contracts, record_definitions

            row = contracts()["profiling"]
            return envelope(
                name,
                data={
                    "encoding": "tagged-v1",
                    "result": encode(
                        {
                            **row,
                            "record_types": record_definitions(row),
                        }
                    ),
                },
            )
        if name == "drift_catalog":
            from truthound.mcp.encoding import encode
            from truthound.mcp.validator_wire import contracts

            rows = contracts()["drift_detectors"]
            keys = list(rows)[arguments.offset : arguments.offset + arguments.limit]
            end = arguments.offset + len(keys)
            return envelope(
                name,
                data={
                    "encoding": "tagged-v1",
                    "result": encode(
                        {
                            "detectors": [{"name": key, **rows[key]} for key in keys],
                            "total": len(rows),
                            "next_offset": end if end < len(rows) else None,
                        }
                    ),
                },
            )
        if name in {"validators_list", "validators_describe"}:
            from truthound.mcp.encoding import encode
            from truthound.mcp.validator_wire import contracts, record_definitions, spec

            if name == "validators_describe":
                result = {
                    "validator": arguments.validator,
                    **spec(arguments.validator),
                    "record_types": record_definitions(spec(arguments.validator)["fields"]),
                    "blocked_inputs": [
                        "untrusted SQL",
                        "Python callables",
                        "caller filesystem paths",
                        "unrestricted Python regex",
                    ],
                    "equivalence_verified": False,
                }
            else:
                rows = contracts()["validators"]
                names = list(rows)[arguments.offset : arguments.offset + arguments.limit]
                end = arguments.offset + len(names)
                result = {
                    "validators": [
                        {"name": key, "category": rows[key]["category"]} for key in names
                    ],
                    "total": len(rows),
                    "next_offset": end if end < len(rows) else None,
                    "abstract_aliases": contracts()["abstract_aliases"],
                }
            return envelope(name, data={"encoding": "tagged-v1", "result": encode(result)})
        if name == "assets_validate":
            return await self._validate(arguments)
        if name == "suites_validate":
            from truthound.mcp.suites import SuiteSpec, effective_arguments

            entry = self.resources.get(arguments.suite_id, "suite")
            suite = SuiteSpec.model_validate_json(entry.path.read_bytes())
            return await self._validate(
                effective_arguments(arguments, suite),
                operation=name,
                suite_entry=entry,
                suite=suite,
                public_arguments=arguments,
            )
        if name == "suites_create":
            from truthound.mcp.validator_wire import frame_handles

            if len(arguments.spec.checks) > self.policy.max_items:
                raise MCPError("resource_exhausted")
            parents = []
            for check in arguments.spec.checks:
                parents.extend(
                    frame_handles(check.validator, check.settings, partial=bool(check.binding_id))
                )
            parents = tuple(dict.fromkeys(parents))
            for identity in parents:
                self.resources.get(identity, "asset")
            if arguments.spec.schema_id:
                self.resources.get(arguments.spec.schema_id, "schema")
                parents = (*parents, arguments.spec.schema_id)
            identity, path = self.resources.reserve("suite")
            try:
                path.write_text(arguments.spec.model_dump_json(), encoding="utf-8")
                entry = self.resources.commit(
                    identity,
                    kind="suite",
                    media_type="application/vnd.truthound.suite-spec+json",
                    parents=parents,
                )
            finally:
                self.resources.discard(identity)
            return envelope(name, data=self._info(entry))
        if name == "suites_inspect":
            from truthound.mcp.encoding import encode

            entry = self.resources.get(arguments.resource_id, "suite")
            if entry.size > self.policy.max_response_bytes:
                raise MCPError("response_too_large")
            return envelope(
                name,
                data={
                    "encoding": "tagged-v1",
                    "result": encode(json.loads(entry.path.read_bytes())),
                },
                provenance=self._provenance(entry.sha256),
            )
        if name == "contexts_open":
            from truthound.context import TruthoundContextConfig
            from truthound.mcp.session_contracts import ContextConfig

            root = self.policy.resolve_workspace(arguments.path)
            self._workspace_stamp(root)
            config = arguments.config or ContextConfig.model_validate(
                TruthoundContextConfig.from_project_file(root).to_dict()
            )
            identity, path = self.resources.reserve("context")
            try:
                path.write_text(
                    json.dumps({"root": str(root), "config": config.model_dump()}), encoding="utf-8"
                )
                entry = self.resources.commit(
                    identity,
                    kind="context",
                    media_type="application/vnd.truthound.context+json",
                    metadata={"workspace_root": str(root), "config": config.model_dump()},
                )
            finally:
                self.resources.discard(identity)
            return envelope(name, data=self._context_info(entry))
        if name == "contexts_inspect":
            return envelope(
                name, data=self._context_info(self.resources.get(arguments.context_id, "context"))
            )
        if name == "contexts_initialize":
            return await self._initialize_context(arguments)
        if name == "contexts_execute":
            from truthound.mcp.contexts import call as execute_context

            return await execute_context(self, arguments)
        if name == "plans_status":
            return envelope(
                name, data=self.approvals().status(arguments.plan_id, session=self.identity)
            )

        if name == "resources_release":
            # Destruction requires session ownership, not renewed permission
            # to read denied or damaged input bytes.
            self.resources.release_owned(arguments.resource_id)
            if self._streams is not None:
                self._streams.reap()
            if self._sources is not None:
                self._sources.reap()
            if self._models is not None:
                self._models.reap()
            return envelope(name, data={"resource_id": arguments.resource_id, "released": True})
        if name == "resources_describe":
            return envelope(name, data=self._info(self.resources.get(arguments.resource_id)))
        if name == "artifacts_read":
            entry = self.resources.get(arguments.resource_id)
            if entry.kind not in {"artifact", "schema", "suite"}:
                raise MCPError("invalid_handle")
            with entry.path.open("rb") as stream:
                stream.seek(arguments.offset)
                data = stream.read(arguments.limit)
            end = arguments.offset + len(data)
            return envelope(
                name,
                data={
                    "resource_id": entry.id,
                    "media_type": entry.media_type,
                    "offset": arguments.offset,
                    "total_bytes": entry.size,
                    "base64": base64.b64encode(data).decode("ascii"),
                    "sha256": entry.sha256,
                    "next_offset": end if end < entry.size else None,
                },
                provenance=self._provenance(entry.sha256),
            )
        if name == "schemas_inspect":
            entry = self.resources.get(arguments.resource_id, "schema")
            if entry.size > self.policy.max_response_bytes:
                raise MCPError("response_too_large")
            return envelope(
                name,
                data={
                    "encoding": "tagged-v1",
                    "result": json.loads(entry.path.read_text(encoding="utf-8")),
                },
                provenance=self._provenance(entry.sha256),
            )

        producers = {
            "lineage_load": "graph",
            "lineage_export": "artifact",
            "lineage_create": "graph",
            "lineage_edit": "graph",
            "lineage_track": "graph",
            "lineage_query": "artifact",
            "lineage_render": "artifact",
            "lineage_impact": "artifact",
            "lineage_critical_paths": "artifact",
            "lineage_what_if_delete": "artifact",
            "lineage_schema_impact": "artifact",
            "lineage_dependency_chain": "artifact",
            "drift_compare": "artifact",
            "drift_detect": "artifact",
            "assets_profile": "artifact",
            "profilers_profile": "artifact",
            "profilers_generate_rules": "artifact",
            "profilers_schema_evolution": "artifact",
            "profilers_column_renames": "artifact",
            "profilers_schema_compatibility": "artifact",
            "assets_scan": "artifact",
            "artifacts_render": "artifact",
            "artifacts_datadocs": "artifact",
            "assets_sqlite": "asset",
            "assets_capture": "asset",
            "assets_inline": "asset",
            "assets_mask": "asset",
            "assets_sample": "asset",
            "schemas_learn": "schema",
            "schemas_create": "schema",
            "schemas_export": "artifact",
            "artifacts_export": "artifact",
        }
        identity = None
        parents = ()
        origins = ()
        digest = None
        request = {
            "operation": name,
            "arguments": arguments.model_dump(),
            "contract_version": 1,
            "max_items": self.policy.max_items,
            "max_response_bytes": self.policy.max_response_bytes,
        }
        try:
            with tempfile.TemporaryDirectory(prefix="truthound-mcp-operation-") as directory:
                if name.startswith("lineage_"):
                    if name == "lineage_load":
                        source = self.policy.resolve(arguments.path, suffixes={".json"})
                        target = Path(directory) / "lineage.json"
                        digest = (await capture(self.policy, source, target)).hexdigest()
                        request["path"] = str(target)
                        origins = (source,)
                    elif name != "lineage_create":
                        handle = (
                            arguments.query.graph_id
                            if name == "lineage_query"
                            else arguments.graph_id
                        )
                        entry = self.resources.get(handle, "graph")
                        request["graph_path"] = str(entry.path)
                        parents, origins, digest = (entry.id,), entry.origins, entry.sha256
                elif name in {
                    "drift_compare",
                    "drift_detect",
                    "profilers_schema_evolution",
                    "profilers_column_renames",
                    "profilers_schema_compatibility",
                }:
                    baseline = self.resources.get(arguments.baseline_id, "asset")
                    current = self.resources.get(arguments.current_id, "asset")
                    request.update(baseline_path=str(baseline.path), current_path=str(current.path))
                    parents = tuple(dict.fromkeys((baseline.id, current.id)))
                    origins = tuple(set(baseline.origins + current.origins))
                    # Bind both actual snapshots, not just one filename or mtime.
                    digest = hashlib.sha256(
                        bytes.fromhex(baseline.sha256) + bytes.fromhex(current.sha256)
                    ).hexdigest()
                elif name == "assets_sqlite":
                    source = self.policy.resolve(
                        arguments.path, suffixes={".db", ".sqlite", ".sqlite3"}
                    )
                    request.update(
                        source_path=str(source),
                        snapshot_path=str(Path(directory) / "snapshot.sqlite"),
                        snapshot_budget=self.policy.max_file_bytes,
                    )
                    origins = (source,)
                elif name == "assets_capture":
                    source = self.policy.resolve(arguments.path)
                    target = Path(directory) / ("input" + source.suffix.lower())
                    digest = (await capture(self.policy, source, target)).hexdigest()
                    request["path"] = str(target)
                    origins = (source,)
                elif name in {"artifacts_render", "artifacts_datadocs"}:
                    entry = self.resources.get(arguments.resource_id, "artifact")
                    if entry.media_type != "application/vnd.truthound.validation+json":
                        raise MCPError("invalid_handle")
                    request["run_path"] = str(entry.path)
                    parents, origins, digest = (entry.id,), entry.origins, entry.sha256
                elif name == "schemas_export":
                    entry = self.resources.get(arguments.resource_id, "schema")
                    request["schema_path"] = str(entry.path)
                    parents, origins, digest = (entry.id,), entry.origins, entry.sha256
                elif name not in {"assets_inline", "schemas_create"}:
                    handle = (
                        arguments.resource_id if name == "artifacts_export" else arguments.asset_id
                    )
                    entry = self.resources.get(handle, "asset")
                    request["asset_path"] = str(entry.path)
                    parents, origins, digest = (entry.id,), entry.origins, entry.sha256
                if getattr(arguments, "limit", 1) > self.policy.max_items:
                    raise MCPError("resource_exhausted")
                if name in producers:
                    identity, output = self.resources.reserve(producers[name])
                    budget = self.resources.available_bytes
                    if budget <= 0:
                        raise MCPError("resource_exhausted")
                    request.update(output_path=str(output), output_budget=budget)
                request["provenance"] = self._provenance(digest)
                result = await self.service._execute_request(request, directory)
                if identity is not None:
                    expected = {"resource_created", "media_type", "row_count", "column_count"}
                    if set(result) != expected or result["resource_created"] is not True:
                        if result.get("execution_status") == "failed":
                            return result
                        raise MCPError("worker_error")
                    entry = self.resources.commit(
                        identity,
                        kind=producers[name],
                        media_type=result["media_type"],
                        origins=origins,
                        parents=parents,
                        metadata={key: result[key] for key in ("row_count", "column_count")},
                    )
                    result = envelope(
                        name, data=self._info(entry), provenance=request["provenance"]
                    )
                CATALOG[name].output_type().model_validate(result)
                if (
                    len(json.dumps(result, allow_nan=False).encode())
                    > self.policy.max_response_bytes
                ):
                    raise MCPError("response_too_large")
                return result
        finally:
            if identity is not None:
                self.resources.discard(identity)
