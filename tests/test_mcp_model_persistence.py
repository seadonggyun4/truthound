"""Native save/load must preserve learned computation, including through MCP."""

import asyncio
import base64
import json

import polars as pl
import pytest

pytestmark = pytest.mark.contract

ALGORITHMS = (
    "ZScoreAnomalyDetector",
    "IQRAnomalyDetector",
    "MADAnomalyDetector",
    "IsolationForestDetector",
    "EnsembleAnomalyDetector",
    "DistributionDriftDetector",
    "FeatureDriftDetector",
    "ConceptDriftDetector",
    "MultivariateDriftDetector",
    "DataProfileRuleLearner",
    "ConstraintMiner",
    "PatternRuleLearner",
)


def specimen(algorithm):
    import truthound.ml as ml

    config = {}
    if algorithm == "IsolationForestDetector":
        config = {"n_estimators": 8, "random_seed": 19}
    elif algorithm == "ConceptDriftDetector":
        config = {"target_column": "target"}
    training = pl.DataFrame(
        {"x": [float(i) for i in range(120)], "target": [i % 2 for i in range(120)]}
    )
    query = training.with_columns((pl.col("x") * 1.5).alias("x"))
    if algorithm == "PatternRuleLearner":
        training = training.with_columns(
            pl.Series("email", [f"person{i}@example.test" for i in range(120)])
        )
        query = query.with_columns(
            pl.Series("email", [f"person{i + 5}@example.test" for i in range(120)])
        )
    return getattr(ml, algorithm)(**config), config, training, query


def semantic(result):
    value = result if isinstance(result, dict) else result.to_dict()
    for key in ("detected_at", "learned_at", "detection_time_ms", "learning_time_ms"):
        value.pop(key, None)
    return value


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_native_model_save_load_retains_trained_predictions(tmp_path, algorithm):
    model, config, training, query = specimen(algorithm)
    model.fit(training.lazy())
    expected = semantic(model.predict(query.lazy()))
    path = tmp_path / "model.json"
    model.save(path)
    assert isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
    restored, _, _, _ = specimen(algorithm)
    restored.load(path)
    assert restored.is_trained
    assert semantic(restored.predict(query.lazy())) == expected
    if hasattr(model, "score"):
        assert restored.score(query.lazy()).to_list() == model.score(query.lazy()).to_list()
    if hasattr(model, "get_rules"):
        assert restored.get_rules() == model.get_rules()


def call(service, name, **arguments):
    response = asyncio.run(service.call_operation_async(name, arguments))
    assert response["execution_status"] == "succeeded", response
    return response["data"]


def artifact(service, identity):
    from truthound.mcp.tables import decode

    chunks, offset = [], 0
    while offset is not None:
        part = call(service, "artifacts_read", resource_id=identity, offset=offset)
        chunks.append(base64.b64decode(part["base64"]))
        offset = part["next_offset"]
    result = json.loads(b"".join(chunks))
    assert result["encoding"] == "tagged-v1"
    result["result"] = decode(result["result"])
    return result


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_mcp_native_saved_model_restores_predictions_and_parent_ownership(tmp_path, algorithm):
    from truthound.mcp.policy import MCPPolicy
    from truthound.mcp.service import MCPService

    direct, config, training, query = specimen(algorithm)
    direct.fit(training.lazy())
    expected = semantic(direct.predict(query.lazy()))
    training.write_parquet(tmp_path / "training.parquet")
    query.write_parquet(tmp_path / "query.parquet")
    service = MCPService(MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True))
    try:
        train = call(service, "assets_capture", path=str(tmp_path / "training.parquet"))[
            "resource_id"
        ]
        query_id = call(service, "assets_capture", path=str(tmp_path / "query.parquet"))[
            "resource_id"
        ]
        model = call(service, "models_create", algorithm=algorithm, config=config)["model_id"]
        call(service, "models_fit", model_id=model, asset_id=train, expected_revision=0)
        saved = call(service, "models_save", model_id=model, expected_revision=1)["resource_id"]
        restored = call(service, "models_restore", resource_id=saved)
        assert restored["model_id"] != model and restored["revision"] == 1
        output = call(
            service,
            "models_predict",
            model_id=restored["model_id"],
            asset_id=query_id,
            expected_revision=1,
        )
        assert semantic(artifact(service, output["resource_id"])["result"]) == expected
        # Releasing the original training input revokes every derived saved
        # state and restored worker; re-entry must not launder its permissions.
        call(service, "resources_release", resource_id=train)
        for identity in (model, restored["model_id"]):
            response = asyncio.run(
                service.call_operation_async(
                    "models_inspect", {"model_id": identity, "expected_revision": 1}
                )
            )
            assert response["execution_status"] == "failed"
            assert response["issues"] == [{"code": "invalid_handle"}]
    finally:
        service.close()


def test_native_model_restores_nondefault_config_and_rejects_wrong_identity(tmp_path):
    from truthound.ml import IQRAnomalyDetector, ZScoreAnomalyDetector
    from truthound.ml.base import ModelLoadError, ModelState

    training = pl.DataFrame({"x": [float(i) for i in range(120)]})
    source = ZScoreAnomalyDetector(contamination=0.2, metadata={"설명": "학습", "tuple": (1, 2)})
    source.fit(training.lazy())
    path = tmp_path / "state.json"
    source.save(path)
    target = ZScoreAnomalyDetector()
    target.load(path)
    assert target.config == source.config
    assert target.score(training.lazy()).equals(source.score(training.lazy()))
    other = IQRAnomalyDetector()
    with pytest.raises(ModelLoadError, match="identity mismatch"):
        other.load(path)
    assert other.state == ModelState.ERROR


@pytest.mark.parametrize("trained", [False, True])
def test_ensemble_edit_preserves_removed_native_detector_and_revocation(tmp_path, trained):
    from truthound.mcp.policy import MCPPolicy
    from truthound.mcp.service import MCPService
    from truthound.ml import EnsembleAnomalyDetector, ZScoreAnomalyDetector

    direct = EnsembleAnomalyDetector()
    child = ZScoreAnomalyDetector(contamination=0.2)
    direct.add_detector(child, weight=2.0)
    frame = pl.DataFrame({"x": [float(i) for i in range(120)]})
    frame.write_parquet(tmp_path / "input.parquet")
    service = MCPService(MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True))
    try:
        asset = call(service, "assets_capture", path=str(tmp_path / "input.parquet"))["resource_id"]
        model = call(service, "models_create", algorithm="EnsembleAnomalyDetector")["model_id"]
        added = call(
            service,
            "models_ensemble_add",
            model_id=model,
            expected_revision=0,
            detector={"algorithm": "ZScoreAnomalyDetector", "config": {"contamination": 0.2}},
            weight=2.0,
        )
        assert added["revision"] == 1 and not added["trained"]
        revision = 1
        if trained:
            direct.fit(frame.lazy())
            call(service, "models_fit", model_id=model, asset_id=asset, expected_revision=revision)
            revision += 1
        removed = direct.remove_detector(0)
        result = call(
            service, "models_ensemble_remove", model_id=model, expected_revision=revision, index=0
        )
        restored = call(service, "models_restore", resource_id=result["resource_id"])
        assert restored["trained"] == removed.is_trained
        assert restored["revision"] == int(trained)
        inspected = call(service, "models_inspect", model_id=model, expected_revision=revision + 1)
        assert inspected["trained"] == direct.is_trained
        if trained:
            scored = call(
                service,
                "models_score",
                model_id=restored["model_id"],
                expected_revision=1,
                asset_id=asset,
            )
            assert (
                artifact(service, scored["resource_id"])["result"]["values"]
                == removed.score(frame.lazy()).to_list()
            )
        call(service, "resources_release", resource_id=model)
        response = asyncio.run(
            service.call_operation_async(
                "models_inspect",
                {"model_id": restored["model_id"], "expected_revision": int(trained)},
            )
        )
        assert response["issues"] == [{"code": "invalid_handle"}]
    finally:
        service.close()


@pytest.mark.parametrize(
    "operation,extra",
    [
        ("models_ensemble_remove", {"index": 0}),
        ("models_ensemble_add", {"detector": {"algorithm": "DataProfileRuleLearner"}}),
    ],
)
def test_ensemble_failed_mutation_invalidates_owned_state(tmp_path, operation, extra):
    from truthound.mcp.policy import MCPPolicy
    from truthound.mcp.service import MCPService

    service = MCPService(MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True))
    try:
        model = call(service, "models_create", algorithm="EnsembleAnomalyDetector")["model_id"]
        result = asyncio.run(
            service.call_operation_async(
                operation, {"model_id": model, "expected_revision": 0, **extra}
            )
        )
        assert result["execution_status"] == "failed"
        assert result["issues"] == [
            {"code": "core_error" if operation.endswith("remove") else "invalid_arguments"}
        ]
        result = asyncio.run(
            service.call_operation_async(
                "models_inspect", {"model_id": model, "expected_revision": 0}
            )
        )
        assert result["issues"] == [{"code": "invalid_handle"}]
    finally:
        service.close()


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_approved_snapshot_survives_session_end_without_retraining(tmp_path, algorithm):
    from truthound.mcp.approval import ApprovalStore
    from truthound.mcp.policy import MCPPolicy
    from truthound.mcp.service import MCPService

    data, control = tmp_path / "data", tmp_path / "control"
    data.mkdir()
    control.mkdir(mode=0o700)
    direct, config, training, query = specimen(algorithm)
    direct.fit(training.lazy())
    expected = semantic(direct.predict(query.lazy()))
    training.write_parquet(data / "train.parquet")
    query.write_parquet(data / "query.parquet")
    policy = MCPPolicy(
        (data,), profile="core", control_directory=control, allow_sensitive_results=True
    )
    service = MCPService(policy)
    try:
        asset = call(service, "assets_capture", path=str(data / "train.parquet"))["resource_id"]
        model = call(service, "models_create", algorithm=algorithm, config=config)["model_id"]
        call(service, "models_fit", model_id=model, asset_id=asset, expected_revision=0)
        saved = call(service, "models_save", model_id=model, expected_revision=1)["resource_id"]
        request = {"resource_id": saved, "idempotency_key": "persist-snapshot-01"}
        pending = asyncio.run(service.call_operation_async("models_persist", request))
        assert pending["issues"] == [{"code": "approval_required"}]
        store = ApprovalStore(control)
        try:
            assert (
                "session_lifetime_transfer"
                in store.review(pending["data"]["plan_id"])["plan"]["effects"]
            )
            store.approve(pending["data"]["plan_id"])
        finally:
            store.close()
        snapshot = call(service, "models_persist", **request)
        assert call(service, "models_persist", **request) == snapshot
    finally:
        service.close()
    service = MCPService(policy)
    try:
        result = asyncio.run(
            service.call_operation_async(
                "models_inspect", {"model_id": model, "expected_revision": 1}
            )
        )
        assert result["issues"] == [{"code": "invalid_handle"}]
        resumed = call(service, "models_resume", snapshot_id=snapshot["snapshot_id"])
        assert resumed["model_id"] != model and resumed["trained"]
        recovered = call(service, "models_resume", idempotency_key="persist-snapshot-01")
        assert recovered["trained"] and recovered["model_id"] != resumed["model_id"]
        call(service, "resources_release", resource_id=recovered["model_id"])
        query_id = call(service, "assets_capture", path=str(data / "query.parquet"))["resource_id"]
        result = call(
            service,
            "models_predict",
            model_id=resumed["model_id"],
            asset_id=query_id,
            expected_revision=1,
        )
        assert semantic(artifact(service, result["resource_id"])["result"]) == expected
        from dataclasses import replace

        service.policy = replace(policy, max_items=policy.max_items + 1)
        denied = asyncio.run(
            service.call_operation_async(
                "models_inspect", {"model_id": resumed["model_id"], "expected_revision": 1}
            )
        )
        assert denied["execution_status"] == "failed"
        denied = asyncio.run(
            service.call_operation_async("models_resume", {"snapshot_id": snapshot["snapshot_id"]})
        )
        assert denied["issues"] == [{"code": "access_denied"}]
    finally:
        service.close()


def test_snapshot_and_receipt_roll_back_together_on_commit_failure(tmp_path, monkeypatch):
    from truthound.mcp.policy import MCPPolicy
    from truthound.mcp.service import MCPService
    from truthound.mcp.session import MCPSession

    data, control = tmp_path / "data", tmp_path / "control"
    data.mkdir()
    control.mkdir(mode=0o700)
    service = MCPService(
        MCPPolicy((data,), profile="core", allow_sensitive_results=True, control_directory=control)
    )
    try:
        session = MCPSession(service)
        service._session = session
        identity, path = session.resources.reserve("artifact")
        path.write_bytes(b"{}")
        session.resources.commit(
            identity,
            kind="artifact",
            media_type="application/vnd.truthound.model+json",
            metadata={
                "model_specification": {"algorithm": "ZScoreAnomalyDetector"},
                "trained": False,
            },
        )
        request = {"resource_id": identity, "idempotency_key": "atomic-snapshot-fault"}
        pending = asyncio.run(service.call_operation_async("models_persist", request))
        assert pending["issues"] == [{"code": "approval_required"}]
        store = session.approvals()
        store.approve(pending["data"]["plan_id"])

        def fail_receipt(*args):
            raise RuntimeError("injected commit boundary failure")

        monkeypatch.setattr(store, "finish_in_transaction", fail_receipt)
        result = asyncio.run(service.call_operation_async("models_persist", request))
        assert result["issues"] == [{"code": "outcome_unknown"}]
        assert store.connection.execute("SELECT COUNT(*) FROM model_snapshots").fetchone()[0] == 0
        assert (
            store.status(pending["data"]["plan_id"], session=session.identity)["state"]
            == "outcome_unknown"
        )
        assert asyncio.run(service.call_operation_async("models_persist", request))["issues"] == [
            {"code": "outcome_unknown"}
        ]
        assert asyncio.run(
            service.call_operation_async(
                "models_resume", {"idempotency_key": request["idempotency_key"]}
            )
        )["issues"] == [{"code": "outcome_unknown"}]
    finally:
        service.close()


def test_incomplete_legacy_forest_is_rejected_instead_of_becoming_ready(tmp_path):
    from truthound.ml.base import ModelLoadError, ModelState

    model, _, training, _ = specimen("IsolationForestDetector")
    model.fit(training.lazy())
    legacy = model._serialize()
    del legacy["trees"]
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    target, _, _, _ = specimen("IsolationForestDetector")
    with pytest.raises(ModelLoadError, match="missing its trained trees"):
        target.load(path)
    assert target.state == ModelState.ERROR


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "import", "value": "os.system"},
        {"type": "dict", "value": [["duplicate", 1], ["duplicate", 2]]},
        {"type": "bytes", "value": "!not-base64!"},
        {"type": "list", "value": "abc"},
        {"type": "dict", "value": [["key", 1, 2]]},
        {"type": "timedelta", "value": [True, 0, 0]},
        {"type": "timedelta", "value": [0, 0, 0, 1]},
        {"type": "Decimal", "value": 0.1},
        float("nan"),
    ],
)
def test_model_state_decoder_rejects_executable_or_ambiguous_data(payload):
    from truthound.ml.serialization import decode

    with pytest.raises((ValueError, TypeError)):
        decode(payload)


@pytest.mark.parametrize(
    "raw",
    [
        '{"format":"truthound-ml","format":"other"}',
        '{"state":NaN}',
        "[]",
    ],
)
def test_model_load_rejects_ambiguous_json_without_ready_state(tmp_path, raw):
    from truthound.ml import ZScoreAnomalyDetector
    from truthound.ml.base import ModelLoadError, ModelState

    path = tmp_path / "corrupt.json"
    path.write_text(raw, encoding="utf-8")
    model = ZScoreAnomalyDetector()
    with pytest.raises(ModelLoadError):
        model.load(path)
    assert model.state == ModelState.ERROR


def test_saved_model_rejects_untrained_stale_foreign_and_changed_state(tmp_path):
    from truthound.mcp.policy import MCPPolicy
    from truthound.mcp.service import MCPService

    policy = MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True)
    service, other = MCPService(policy), MCPService(policy)

    def failed(name, code, **arguments):
        result = asyncio.run(service.call_operation_async(name, arguments))
        assert result["execution_status"] == "failed"
        assert result["issues"] == [{"code": code}]

    try:
        model = call(service, "models_create", algorithm="ZScoreAnomalyDetector", config={})[
            "model_id"
        ]
        failed("models_save", "core_error", model_id=model, expected_revision=0)
        # A rejected read-only save must not destroy the untrained model.
        assert call(service, "models_inspect", model_id=model, expected_revision=0)["revision"] == 0
        pl.DataFrame({"x": list(range(120))}).write_parquet(tmp_path / "input.parquet")
        asset = call(service, "assets_capture", path=str(tmp_path / "input.parquet"))["resource_id"]
        failed("models_restore", "invalid_handle", resource_id=asset)
        call(service, "models_fit", model_id=model, asset_id=asset, expected_revision=0)
        failed("models_save", "stale_revision", model_id=model, expected_revision=0)
        saved = call(service, "models_save", model_id=model, expected_revision=1)["resource_id"]
        foreign = asyncio.run(other.call_operation_async("models_restore", {"resource_id": saved}))
        assert foreign["issues"] == [{"code": "invalid_handle"}]
        entry = service._session.resources.get(saved, "artifact")
        entry.path.write_bytes(b"{}")
        failed("models_restore", "resource_changed", resource_id=saved)
        assert len(service._session._models.workers) == 1
    finally:
        service.close()
        other.close()
