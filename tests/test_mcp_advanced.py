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
    import sys

    original = subprocess.Popen

    def start_owned(*args, **kwargs):
        return original(
            [sys.executable, "-I", "-c", "import sys,time;sys.stdin.readline();time.sleep(600)"],
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
