"""Advanced Core algorithms execute natively; resident state never uses pickle."""

import asyncio
import base64
import json

import polars as pl
import pytest

pytest.importorskip("mcp")

from truthound.mcp.model_process import OwnedModelProcess
from truthound.mcp.policy import MCPPolicy
from truthound.mcp.service import MCPService
from truthound.mcp.tables import decode
from truthound.mcp.validator_wire import contracts

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

DETECTORS = (
    "AndersonDarlingDetector",
    "BhattacharyyaDetector",
    "ChiSquareDetector",
    "CramervonMisesDetector",
    "EnergyDetector",
    "HellingerDetector",
    "JensenShannonDetector",
    "KLDivergenceDetector",
    "KSTestDetector",
    "MMDDetector",
    "PSIDetector",
    "TotalVariationDetector",
    "WassersteinDetector",
)


@pytest.fixture
def adapter(tmp_path):
    service = MCPService(MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True))
    yield service
    service.close()


def invoke(adapter, name, **arguments):
    return asyncio.run(adapter.call_operation_async(name, arguments))


def call(adapter, name, **arguments):
    result = invoke(adapter, name, **arguments)
    assert result["execution_status"] == "succeeded", result
    return result["data"]


def artifact(adapter, identity):
    pieces, offset = [], 0
    while offset is not None:
        chunk = call(adapter, "artifacts_read", resource_id=identity, offset=offset, limit=32768)
        pieces.append(base64.b64decode(chunk["base64"]))
        offset = chunk["next_offset"]
    return json.loads(b"".join(pieces))


def frame_asset(adapter, frame):
    return call(
        adapter,
        "assets_inline",
        columns=[
            {"name": name, "dtype": {"kind": str(dtype)}, "values": frame[name].to_list()}
            for name, dtype in frame.schema.items()
        ],
    )["resource_id"]


@pytest.mark.parametrize("algorithm", ["ZScoreAnomalyDetector", "DistributionDriftDetector"])
def test_prediction_result_queries_match_native_and_revoke(adapter, algorithm):
    import truthound.ml as ml

    training = pl.DataFrame({"x": list(range(120))})
    frame = pl.DataFrame({"x": [0, 1, 2, 10000]})
    direct = getattr(ml, algorithm)()
    direct.fit(training.lazy())
    native = direct.predict(frame.lazy())
    asset = frame_asset(adapter, frame)
    model = call(adapter, "models_create", algorithm=algorithm)["model_id"]
    call(
        adapter,
        "models_fit",
        model_id=model,
        asset_id=frame_asset(adapter, training),
        expected_revision=0,
    )
    cases = (
        [
            ("all_scores", None, [s.to_dict() for s in native]),
            ("anomalies", None, [s.to_dict() for s in native.get_anomalies()]),
        ]
        if algorithm == "ZScoreAnomalyDetector"
        else [
            ("drifted_columns", t, native.get_drifted_columns(0.5 if t is None else t))
            for t in (None, 0.0, 1.0, 2.0)
        ]
    )
    for action, threshold, expected in cases:
        result = call(
            adapter,
            "models_result_query",
            model_id=model,
            asset_id=asset,
            expected_revision=1,
            action=action,
            threshold=threshold,
        )
        value = artifact(adapter, result["resource_id"])
        assert value["model_revision"] == 1
        assert decode(value["result"]) == expected
    assert call(adapter, "models_inspect", model_id=model, expected_revision=1)["revision"] == 1
    call(adapter, "resources_release", resource_id=asset)
    assert invoke(adapter, "artifacts_read", resource_id=result["resource_id"])["issues"] == [
        {"code": "invalid_handle"}
    ]


@pytest.mark.parametrize(
    "action,threshold",
    [("all_scores", 0.5), ("anomalies", 0.5), ("drifted_columns", float("inf")), ("execute", None)],
)
def test_prediction_result_query_rejects_invalid_selector(action, threshold):
    from pydantic import ValidationError

    from truthound.mcp.advanced_contracts import ModelResultQuery

    with pytest.raises(ValidationError):
        ModelResultQuery(
            model_id="model_" + "a" * 32,
            asset_id="asset_" + "b" * 32,
            expected_revision=0,
            action=action,
            threshold=threshold,
        )


def test_prediction_result_query_faults_preserve_model(adapter):
    model = call(adapter, "models_create", algorithm="ZScoreAnomalyDetector")["model_id"]
    asset = frame_asset(adapter, pl.DataFrame({"x": list(range(120))}))
    for action, revision, code in (
        ("drifted_columns", 0, "unsupported_operation"),
        ("all_scores", 0, "core_error"),
        ("all_scores", 1, "stale_revision"),
    ):
        failed = invoke(
            adapter,
            "models_result_query",
            model_id=model,
            asset_id=asset,
            expected_revision=revision,
            action=action,
        )
        assert failed["issues"] == [{"code": code}]
    assert not call(adapter, "models_inspect", model_id=model, expected_revision=0)["trained"]


@pytest.mark.parametrize(
    "algorithm", ["DataProfileRuleLearner", "ConstraintMiner", "PatternRuleLearner"]
)
def test_rule_result_operations_preserve_native_semantics_and_revision(adapter, algorithm):
    import truthound.ml as ml

    frame = pl.DataFrame(
        {"x": list(range(120)), "email": [f"p{i}@example.test" for i in range(120)]}
    )
    direct = getattr(ml, algorithm)()
    direct.fit(frame.lazy())
    expected = direct.predict(frame.lazy())
    asset = frame_asset(adapter, frame)
    model = call(adapter, "models_create", algorithm=algorithm)["model_id"]
    call(adapter, "models_fit", model_id=model, asset_id=asset, expected_revision=0)
    for action, selectors, native in (
        ("suite", {}, expected.to_validation_suite()),
        ("validator_specs", {}, [rule.to_validator_spec() for rule in expected]),
        (
            "by_type",
            {"rule_type": "range"},
            [rule.to_dict() for rule in expected.get_rules_by_type("range")],
        ),
        (
            "by_column",
            {"column": "x"},
            [rule.to_dict() for rule in expected.get_rules_for_column("x")],
        ),
        ("by_column", {"column": "missing"}, []),
    ):
        result = call(
            adapter,
            "models_rule_query",
            model_id=model,
            asset_id=asset,
            expected_revision=1,
            action=action,
            **selectors,
        )
        actual = artifact(adapter, result["resource_id"])
        assert actual["model_revision"] == 1
        actual = decode(actual["result"])
        if action == "suite":
            from datetime import datetime

            datetime.fromisoformat(actual.pop("generated_at"))
            native.pop("generated_at")
        assert actual == native
    assert call(adapter, "models_inspect", model_id=model, expected_revision=1)["revision"] == 1
    # All generated rule artifacts depend on both the model and prediction input.
    call(adapter, "resources_release", resource_id=asset)
    denied = invoke(adapter, "artifacts_read", resource_id=result["resource_id"])
    assert denied["issues"] == [{"code": "invalid_handle"}]


@pytest.mark.parametrize(
    "arguments",
    [
        {"action": "by_type"},
        {"action": "by_column"},
        {"action": "suite", "column": "x"},
        {"action": "validator_specs", "rule_type": "range"},
        {"action": "by_type", "rule_type": "range", "column": "x"},
        {"action": "execute"},
    ],
)
def test_rule_query_rejects_ambiguous_selectors(arguments):
    from pydantic import ValidationError

    from truthound.mcp.advanced_contracts import ModelRuleQuery

    with pytest.raises(ValidationError):
        ModelRuleQuery(
            model_id="model_" + "a" * 32,
            asset_id="asset_" + "b" * 32,
            expected_revision=0,
            **arguments,
        )


def test_rule_query_rejects_nonlearner_without_destroying_model(adapter):
    model = call(adapter, "models_create", algorithm="ZScoreAnomalyDetector")["model_id"]
    asset = frame_asset(adapter, pl.DataFrame({"x": list(range(120))}))
    denied = invoke(
        adapter,
        "models_rule_query",
        model_id=model,
        asset_id=asset,
        expected_revision=0,
        action="suite",
    )
    assert denied["issues"] == [{"code": "unsupported_operation"}]
    assert call(adapter, "models_inspect", model_id=model, expected_revision=0)["revision"] == 0


def test_rule_query_untrained_and_stale_fail_without_changing_state(adapter):
    from truthound.ml import DataProfileRuleLearner
    from truthound.ml.base import ModelNotTrainedError

    frame = pl.DataFrame({"x": list(range(120))})
    with pytest.raises(ModelNotTrainedError):
        DataProfileRuleLearner().predict(frame.lazy())
    asset = frame_asset(adapter, frame)
    model = call(adapter, "models_create", algorithm="DataProfileRuleLearner")["model_id"]
    failed = invoke(
        adapter,
        "models_rule_query",
        model_id=model,
        asset_id=asset,
        expected_revision=0,
        action="suite",
    )
    assert failed["issues"] == [{"code": "core_error"}]
    stale = invoke(
        adapter,
        "models_rule_query",
        model_id=model,
        asset_id=asset,
        expected_revision=1,
        action="suite",
    )
    assert stale["issues"] == [{"code": "stale_revision"}]
    assert not call(adapter, "models_inspect", model_id=model, expected_revision=0)["trained"]


@pytest.mark.parametrize(
    "expression,after,error",
    [
        ("*/oops * * * *", "2024-01-01T00:00:00", ValueError),
        ("@daily", "9999-12-31T23:59:59", OverflowError),
    ],
)
def test_schedule_native_runtime_failure_is_not_valid_or_exhausted(
    adapter, expression, after, error
):
    from datetime import datetime

    from truthound.scheduling import CronExpression

    with pytest.raises(error):
        CronExpression.parse(expression).next_n(1, datetime.fromisoformat(after))
    result = invoke(adapter, "schedules_preview", expression=expression, after=after, count=1)
    assert result["execution_status"] == "failed"
    assert result["quality_status"] == "unknown" and result["data"] is None
    assert result["issues"] == [{"code": "core_error"}]


@pytest.mark.parametrize(
    "expression",
    [
        "*/15 * * * *",
        "0 9 * * MON-FRI",
        "0 0 L * *",
        "0 0 15W * *",
        "0 0 ? * 2#3",
        "*/10 * * * * *",
        "0 0 0 1 JAN * 2025",
        "@monthly",
        "0 0 29 FEB *",
        "0 0 0 1 JAN * 2000",
        "invalid",
        "99 * * * *",
    ],
)
@pytest.mark.parametrize("after", ["2024-01-01T00:00:00", "2024-01-01T09:00:00+09:00"])
def test_schedule_preview_matches_native_parser_and_bounded_search(adapter, expression, after):
    from datetime import datetime

    from truthound.scheduling import CronExpression, validate_expression

    start = datetime.fromisoformat(after)
    errors = validate_expression(expression)
    actual = call(adapter, "schedules_preview", expression=expression, after=after, count=3)
    assert actual["errors"] == errors
    assert actual["valid"] is (not errors)
    if errors:
        assert actual["next_times"] == []
        assert actual["matches"] is None and actual["has_seconds"] is None
        assert not actual["search_exhausted"]
    else:
        native = CronExpression.parse(expression)
        expected = [value.isoformat() for value in native.next_n(3, start)]
        assert actual["next_times"] == expected
        assert actual["matches"] == native.matches(start)
        assert actual["has_seconds"] == native.has_seconds
        assert actual["search_exhausted"] == (len(expected) < 3)
    assert actual["returned_count"] == len(actual["next_times"])
    assert actual["requested_count"] == 3


@pytest.mark.parametrize(
    "arguments",
    [
        {"count": 0},
        {"count": 101},
        {"after": "not-a-time"},
        {"expression": "*" * 257},
        {"execute": True},
    ],
)
def test_schedule_preview_rejects_unbounded_or_execution_arguments(adapter, arguments):
    inputs = {"expression": "@daily", "after": "2024-01-01T00:00:00", **arguments}
    response = invoke(adapter, "schedules_preview", **inputs)
    assert response["execution_status"] == "failed"
    assert response["quality_status"] == "unknown"


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_resident_fit_predict_matches_the_same_native_core_algorithm(adapter, algorithm):
    import truthound.ml as ml

    assert set(ALGORITHMS) == set(contracts()["models"])
    config, detectors = {}, None
    if algorithm == "IsolationForestDetector":
        config = {"n_estimators": 8, "random_seed": 19}
    if algorithm == "ConceptDriftDetector":
        config = {"target_column": "target"}
    if algorithm == "EnsembleAnomalyDetector":
        detectors = [{"algorithm": "ZScoreAnomalyDetector"}, {"algorithm": "IQRAnomalyDetector"}]
        direct = ml.EnsembleAnomalyDetector(
            detectors=[ml.ZScoreAnomalyDetector(), ml.IQRAnomalyDetector()]
        )
    else:
        direct = getattr(ml, algorithm)(**config)
    frame = pl.DataFrame(
        {"x": [float(i) for i in range(120)], "target": [i % 2 for i in range(120)]}
    )
    current = pl.DataFrame(
        {"x": [float(i) * 1.5 for i in range(120)], "target": [i % 2 for i in range(120)]}
    )
    if algorithm == "PatternRuleLearner":
        frame = frame.with_columns(
            pl.Series("email", [f"person{i}@example.test" for i in range(120)])
        )
        current = current.with_columns(
            pl.Series("email", [f"person{i + 5}@example.test" for i in range(120)])
        )
    training = frame_asset(adapter, frame)
    query = frame_asset(adapter, current)
    specification = {"algorithm": algorithm, "config": config}
    if detectors is not None:
        specification["detectors"] = detectors
    model = call(adapter, "models_create", **specification)
    assert not model["trained"] and model["revision"] == 0
    fitted = call(
        adapter, "models_fit", model_id=model["model_id"], asset_id=training, expected_revision=0
    )
    assert fitted["trained"] and fitted["revision"] == 1
    direct.fit(frame.lazy())
    expected = direct.predict(current.lazy()).to_dict()
    if algorithm == "PatternRuleLearner":
        assert expected["total_rules"] > 0
    retained = call(
        adapter, "models_predict", model_id=model["model_id"], asset_id=query, expected_revision=1
    )
    result = artifact(adapter, retained["resource_id"])
    assert result["model_algorithm"] == algorithm and result["model_revision"] == 1
    actual = decode(result["result"])
    # These measured wall-clock fields are not algorithm output. No scores,
    # rules, thresholds, statistics or verdicts are normalized or discarded.
    for key in ("detected_at", "learned_at", "detection_time_ms", "learning_time_ms"):
        actual.pop(key, None)
        expected.pop(key, None)
    assert actual == expected
    native_helpers = {
        "ZScoreAnomalyDetector": {"statistics": "get_statistics"},
        "IQRAnomalyDetector": {"statistics": "get_statistics"},
        "MADAnomalyDetector": {"statistics": "get_statistics"},
        "IsolationForestDetector": {"feature_importance": "get_feature_importance"},
        "EnsembleAnomalyDetector": {
            "detector_names": "get_detector_names",
            "detector_weights": "get_detector_weights",
        },
        "FeatureDriftDetector": {"feature_profiles": "get_feature_profiles"},
        "ConceptDriftDetector": {"reference_correlations": "get_reference_correlations"},
        "MultivariateDriftDetector": {"correlation_matrix": "get_correlation_matrix"},
        "DataProfileRuleLearner": {"column_profiles": "get_column_profiles"},
        "PatternRuleLearner": {"detected_patterns": "get_detected_patterns"},
    }
    diagnostics = call(
        adapter, "models_diagnostics", model_id=model["model_id"], expected_revision=1
    )
    values = decode(artifact(adapter, diagnostics["resource_id"])["result"])
    from datetime import datetime

    expected_info = direct.info.to_dict()
    for info in (values["info"], expected_info):
        assert datetime.fromisoformat(info.pop("created_at"))
    assert values["info"] == expected_info
    from dataclasses import asdict

    assert values["config"] == asdict(direct.config)
    assert values["diagnostics"] == {
        key: getattr(direct, method)() for key, method in native_helpers.get(algorithm, {}).items()
    }
    stale = invoke(
        adapter, "models_predict", model_id=model["model_id"], asset_id=query, expected_revision=0
    )
    assert stale["issues"] == [{"code": "stale_revision"}]
    assert call(adapter, "models_inspect", model_id=model["model_id"], expected_revision=1)[
        "trained"
    ]


@pytest.mark.parametrize(
    "method",
    [
        "auto",
        "ks",
        "psi",
        "chi2",
        "js",
        "kl",
        "wasserstein",
        "cvm",
        "anderson",
        "hellinger",
        "bhattacharyya",
        "tv",
        "total_variation",
        "energy",
        "mmd",
    ],
)
def test_drift_every_public_comparison_method_matches_core(adapter, method):
    from truthound.drift import compare

    baseline = pl.DataFrame({"x": [float(i) for i in range(30)]})
    current = pl.DataFrame({"x": [float(i + 10) for i in range(30)]})
    left, right = frame_asset(adapter, baseline), frame_asset(adapter, current)
    report = call(
        adapter,
        "drift_compare",
        baseline_id=left,
        current_id=right,
        columns=["x"],
        method=method,
        threshold=0.1,
        sample_size=20,
    )
    assert (
        decode(artifact(adapter, report["resource_id"]))
        == compare(
            baseline, current, columns=["x"], method=method, threshold=0.1, sample_size=20
        ).to_dict()
    )


@pytest.mark.parametrize("name", DETECTORS)
def test_individual_drift_constructor_settings_and_distinct_columns_match_core(adapter, name):
    import truthound.drift as drift

    assert set(DETECTORS) == set(contracts()["drift_detectors"])
    row = contracts()["drift_detectors"][name]
    settings = {"threshold": 0.2}
    for key, value in (
        ("n_bins", 5),
        ("max_samples", 10),
        ("bandwidth", 2.0),
        ("kernel", "linear"),
    ):
        if key in row["fields"]:
            settings[key] = value
    baseline = pl.DataFrame({"before": [float(i) for i in range(20)]})
    current = pl.DataFrame({"after": [float(i + 10) for i in range(20)]})
    left, right = frame_asset(adapter, baseline), frame_asset(adapter, current)
    result = call(
        adapter,
        "drift_detect",
        detector=name,
        settings=settings,
        baseline_id=left,
        current_id=right,
        baseline_column="before",
        current_column="after",
    )
    expected = (
        getattr(drift, name)(**settings).detect(baseline["before"], current["after"]).to_dict()
    )
    assert decode(artifact(adapter, result["resource_id"])) == expected


def test_models_invalidate_on_training_release_and_process_expires_or_session_closes(adapter):
    training = frame_asset(adapter, pl.DataFrame({"x": [float(i) for i in range(120)]}))
    model = call(adapter, "models_create", algorithm="ZScoreAnomalyDetector")
    call(adapter, "models_fit", model_id=model["model_id"], asset_id=training, expected_revision=0)
    worker = adapter._session._models.workers[model["model_id"]]
    call(adapter, "resources_release", resource_id=training)
    assert worker.closed and worker.process.poll() is not None
    assert invoke(adapter, "models_inspect", model_id=model["model_id"], expected_revision=1)[
        "issues"
    ] == [{"code": "invalid_handle"}]
    model = call(adapter, "models_create", algorithm="ZScoreAnomalyDetector")
    worker = adapter._session._models.workers[model["model_id"]]
    worker.expire_after(0.01)
    worker.process.wait(timeout=3)
    assert invoke(adapter, "models_inspect", model_id=model["model_id"], expected_revision=0)[
        "issues"
    ] == [{"code": "invalid_handle"}]


def test_direct_learning_combined_training_and_explicit_reference_detection(adapter):
    from truthound.ml import (
        DataProfileRuleLearner,
        DistributionDriftDetector,
        ZScoreAnomalyDetector,
    )

    frame = pl.DataFrame({"x": [float(i) for i in range(120)]})
    current = frame.with_columns((pl.col("x") + 120).alias("x"))
    left, right = frame_asset(adapter, frame), frame_asset(adapter, current)
    for algorithm, operation, direct in (
        ("DataProfileRuleLearner", "models_learn_rules", DataProfileRuleLearner()),
        ("ZScoreAnomalyDetector", "models_fit_predict", ZScoreAnomalyDetector()),
        ("DistributionDriftDetector", "models_detect", DistributionDriftDetector()),
    ):
        created = call(adapter, "models_create", algorithm=algorithm)
        arguments = {"model_id": created["model_id"], "asset_id": right, "expected_revision": 0}
        if operation == "models_detect":
            arguments.update(reference_id=left, columns=["x"])
            expected = direct.detect(frame.lazy(), current.lazy(), columns=["x"]).to_dict()
        elif operation == "models_learn_rules":
            expected = direct.learn_rules(current.lazy()).to_dict()
        else:
            expected = direct.fit_predict(current.lazy()).to_dict()
        retained = call(adapter, operation, **arguments)
        actual = decode(artifact(adapter, retained["resource_id"])["result"])
        for key in ("detected_at", "learned_at", "detection_time_ms", "learning_time_ms"):
            actual.pop(key, None)
            expected.pop(key, None)
        assert actual == expected
        assert (
            call(adapter, "models_inspect", model_id=created["model_id"], expected_revision=1)[
                "revision"
            ]
            == 1
        )


def test_model_permissions_cross_session_unknown_constructor_and_online_skip_are_failures(
    adapter, tmp_path
):
    model = call(adapter, "models_create", algorithm="ZScoreAnomalyDetector")
    other = MCPService(adapter.policy)
    try:
        assert invoke(other, "models_inspect", model_id=model["model_id"], expected_revision=0)[
            "issues"
        ] == [{"code": "invalid_handle"}]
    finally:
        other.close()
    assert invoke(adapter, "models_create", algorithm="os.system")["issues"] == [
        {"code": "unknown_model"}
    ]
    assert invoke(
        adapter, "models_create", algorithm="ZScoreAnomalyDetector", config={"unknown": 1}
    )["issues"] == [{"code": "invalid_arguments"}]
    training = frame_asset(adapter, pl.DataFrame({"x": [float(i) for i in range(120)]}))
    result = invoke(
        adapter,
        "models_partial_fit",
        model_id=model["model_id"],
        asset_id=training,
        expected_revision=0,
    )
    assert result["execution_status"] == "failed" and result["quality_status"] == "unknown"


def test_resident_timeout_kills_and_reaps_only_owned_child(tmp_path, monkeypatch):
    import subprocess

    original = subprocess.Popen

    def start_owned(*args, **kwargs):
        return original(
            [args[0][0], "-I", "-c", "import sys,time;sys.stdin.readline();time.sleep(600)"],
            **kwargs,
        )

    monkeypatch.setattr("truthound.mcp.model_process.subprocess.Popen", start_owned)
    worker = OwnedModelProcess(30)
    try:
        with pytest.raises(TimeoutError):
            asyncio.run(worker.call({"operation": "models_create"}, timeout=1))
        assert worker.process.poll() is not None
        from pathlib import Path

        assert not Path(worker.directory.name).exists()
    finally:
        worker.close()


def test_resident_worker_owns_actual_interpreter_and_preserves_environment(monkeypatch):
    import os
    import subprocess
    import sys
    from pathlib import Path

    original = subprocess.Popen

    def start_owned(command, **kwargs):
        program = (
            "import json,os,sys;sys.stdin.readline();"
            "print(json.dumps(dict(ok=True,pid=os.getpid(),prefix=sys.prefix,"
            "isolated=sys.flags.isolated)),flush=True);sys.stdin.read()"
        )
        return original([command[0], "-I", "-c", program], **kwargs)

    monkeypatch.setattr("truthound.mcp.model_process.subprocess.Popen", start_owned)
    monkeypatch.setattr(OwnedModelProcess, "validate_reply", lambda self, reply: reply)
    worker = OwnedModelProcess(30)
    try:
        reply = asyncio.run(worker.call({}, timeout=10))
        assert reply["pid"] == worker.process.pid != os.getpid()
        assert Path(reply["prefix"]).resolve() == Path(sys.prefix).resolve()
        assert reply["isolated"] == 1
    finally:
        worker.close()
    assert worker.process.poll() is not None
    assert not Path(worker.directory.name).exists()


@pytest.mark.parametrize("method", ["ddm", "adwin", "page_hinkley"])
def test_resident_concept_stream_runs_selected_native_detector(adapter, method):
    from truthound.ml import ConceptDriftDetector

    config = {"method": method, "stream_column": "error", "detection_window": 1000}
    reference = pl.DataFrame({"error": [0.0] * 1000})
    current = pl.DataFrame({"error": [1.0] * 1000})
    direct = ConceptDriftDetector(**config)
    direct.fit(reference.lazy())
    expected = direct.predict(current.lazy()).to_dict()
    model = call(adapter, "models_create", algorithm="ConceptDriftDetector", config=config)
    call(
        adapter,
        "models_fit",
        model_id=model["model_id"],
        asset_id=frame_asset(adapter, reference),
        expected_revision=0,
    )
    retained = call(
        adapter,
        "models_predict",
        model_id=model["model_id"],
        asset_id=frame_asset(adapter, current),
        expected_revision=1,
    )
    actual = decode(artifact(adapter, retained["resource_id"])["result"])
    actual.pop("detected_at")
    expected.pop("detected_at")
    assert actual == expected
    assert actual["is_drifted"]
