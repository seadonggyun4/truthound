"""Session-local Core model lifecycle with explicit revisions and owned artifacts."""

import json
from contextlib import suppress

from truthound.mcp.model_process import OwnedModelProcess
from truthound.mcp.operations import envelope
from truthound.mcp.policy import MCPError


class ModelSession:
    def __init__(self, session):
        self.session = session
        self.workers = {}
        self.states = {}

    def release(self, identity):
        worker = self.workers.get(identity)
        if worker is not None:
            worker.close()
            self.workers.pop(identity, None)
        self.states.pop(identity, None)

    def close(self):
        for worker in self.workers.values():
            worker.close()
        self.workers.clear()
        self.states.clear()

    def reap(self):
        for identity, worker in list(self.workers.items()):
            try:
                self.session.resources.get(identity, "model")
                if worker.closed:
                    raise MCPError("invalid_handle")
            except MCPError:
                self.release(identity)
                with suppress(MCPError):
                    self.session.resources.release(identity)

    async def call(self, name, arguments):
        resources, policy = self.session.resources, self.session.policy
        if name == "models_catalog":
            from truthound.mcp.encoding import encode
            from truthound.mcp.validator_wire import contracts, record_definitions, record_fields

            rows = contracts()["models"]
            keys = list(rows)[arguments.offset : arguments.offset + arguments.limit]
            end = arguments.offset + len(keys)
            return envelope(
                name,
                data={
                    "encoding": "tagged-v1",
                    "result": encode(
                        {
                            "models": [
                                {
                                    "algorithm": key,
                                    "config_fields": record_fields(rows[key]["config"]),
                                    "supports_online_learning": rows[key][
                                        "supports_online_learning"
                                    ],
                                    "native_methods": rows[key]["native_methods"],
                                }
                                for key in keys
                            ],
                            "total": len(rows),
                            "record_types": record_definitions(
                                [rows[key]["config"] for key in keys]
                            ),
                            "next_offset": end if end < len(rows) else None,
                        }
                    ),
                },
            )
        if name in {"models_create", "models_restore"}:
            if len(self.workers) >= min(policy.max_handles, 4):
                raise MCPError("resource_exhausted")
            identity, path = resources.reserve("model")
            worker = None
            try:
                source = None
                if name == "models_restore":
                    source = resources.get(arguments.resource_id, "artifact")
                    if source.media_type != "application/vnd.truthound.model+json":
                        raise MCPError("invalid_handle")
                    specification = source.metadata["model_specification"]
                else:
                    specification = arguments.model_dump()
                path.write_text(json.dumps(specification), encoding="utf-8")
                worker = OwnedModelProcess(policy.handle_ttl_seconds)
                state = await worker.call(
                    {"operation": "models_create", "spec": specification}, policy.timeout_seconds
                )
                if source is not None:
                    state = await worker.call(
                        {
                            "operation": "models_load",
                            "expected_revision": 0,
                            "state_path": str(source.path),
                        },
                        policy.timeout_seconds,
                    )
                    resources.get(source.id, "artifact")
                entry = resources.commit(
                    identity,
                    kind="model",
                    media_type="application/vnd.truthound.model-spec+json",
                    parents=(source.id,) if source else (),
                    metadata={"algorithm": specification["algorithm"]},
                )
                worker.expire_after(entry.expires - resources.clock())
                self.workers[identity] = worker
                self.states[identity] = state
                return envelope(name, data=self._state(entry, state))
            except BaseException:
                if worker:
                    worker.close()
                raise
            finally:
                resources.discard(identity)
        entry = resources.get(arguments.model_id, "model")
        worker = self.workers.get(entry.id)
        if worker is None or worker.closed:
            raise MCPError("invalid_handle")
        if arguments.expected_revision != self.states[entry.id]["revision"]:
            raise MCPError("stale_revision")
        request = {"operation": name, "expected_revision": arguments.expected_revision}
        if name == "models_rule_query":
            request.update(
                action=arguments.action, rule_type=arguments.rule_type, column=arguments.column
            )
        parents = [entry.id]
        data = None
        if hasattr(arguments, "asset_id"):
            data = resources.get(arguments.asset_id, "asset")
            request["asset_path"] = str(data.path)
            parents.append(data.id)
        reference = None
        if hasattr(arguments, "reference_id"):
            reference = resources.get(arguments.reference_id, "asset")
            request.update(reference_path=str(reference.path), columns=arguments.columns)
            parents.append(reference.id)
        output_id = None
        mutation = name in {
            "models_fit",
            "models_partial_fit",
            "models_fit_predict",
            "models_learn_rules",
            "models_detect",
        }
        try:
            if name in {
                "models_predict",
                "models_score",
                "models_rules",
                "models_rule_query",
                "models_fit_predict",
                "models_learn_rules",
                "models_detect",
                "models_diagnostics",
                "models_save",
            }:
                output_id, output = resources.reserve("artifact")
                budget = resources.available_bytes
                if budget <= 0:
                    raise MCPError("resource_exhausted")
                request.update(output_path=str(output), output_budget=budget)
            state = await worker.call(request, policy.timeout_seconds)
            if state["revision"] != arguments.expected_revision + int(mutation):
                worker.close()
                raise MCPError("worker_error")
            self.states[entry.id] = state
            resources.get(entry.id, "model")
            if data:
                resources.get(data.id, "asset")
            if reference:
                resources.get(reference.id, "asset")
            if mutation:
                bound = resources.bind_model_inputs(entry.id, parents[1:])
                worker.expire_after(bound.expires - resources.clock())
            if output_id:
                result = resources.commit(
                    output_id,
                    kind="artifact",
                    media_type="application/vnd.truthound.model+json"
                    if name == "models_save"
                    else "application/vnd.truthound.ml-result+json",
                    parents=tuple(parents),
                    metadata={
                        "model_specification": json.loads(entry.path.read_text(encoding="utf-8"))
                    }
                    if name == "models_save"
                    else None,
                )
                return envelope(
                    name,
                    data=self.session._info(result),
                    provenance=self.session._provenance(data.sha256 if data else None),
                )
            return envelope(name, data=self._state(entry, state))
        except BaseException:
            if mutation or worker.closed:
                worker.close()
                self.workers.pop(entry.id, None)
                self.states.pop(entry.id, None)
                resources.release(entry.id)
            raise
        finally:
            if output_id:
                resources.discard(output_id)

    @staticmethod
    def _state(entry, state):
        return {
            "model_id": entry.id,
            "algorithm": entry.metadata["algorithm"],
            "revision": state["revision"],
            "state": state["state"],
            "trained": state["trained"],
        }
