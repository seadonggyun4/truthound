"""Owned resident ML process. Only named Core methods; no serialized Python state."""

import contextlib
import json
import os
import sys
from pathlib import Path

from truthound.mcp.encoding import encode
from truthound.mcp.policy import MCPError
from truthound.mcp.resources import BoundedWriter


def detector_spec(model):
    """Describe only admitted constructors; never persist import names or objects."""
    from dataclasses import asdict

    from truthound.mcp.model_wire import validate_spec
    from truthound.ml import EnsembleAnomalyDetector

    value = {"algorithm": type(model).__name__, "config": asdict(model.config), "detectors": None}
    if isinstance(model, EnsembleAnomalyDetector):
        value["detectors"] = [detector_spec(child) for child in model._detectors]
    value = json.loads(json.dumps(value, allow_nan=False))
    validate_spec(value)
    return value


def save_bounded(model, request):
    import tempfile

    with tempfile.TemporaryDirectory(prefix="truthound-model-save-") as directory:
        saved = Path(directory) / "model.json"
        model.save(saved)
        with (
            BoundedWriter(Path(request["output_path"]), request["output_budget"]) as output,
            saved.open("rb") as source,
        ):
            while chunk := source.read(65536):
                output.write(chunk)


def main():
    model, revision = None, 0
    wire = sys.stdout
    with open(os.devnull, "w") as sink:
        while True:
            line = sys.stdin.buffer.readline(131073)
            if not line:
                break
            response = {"ok": False, "code": "worker_error"}
            try:
                detached = {}
                if len(line) > 131072 or not line.endswith(b"\n"):
                    raise MCPError("resource_exhausted")
                request = json.loads(line)
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    operation = request["operation"]
                    if operation == "models_create":
                        if model is not None:
                            raise MCPError("invalid_arguments")
                        from truthound.mcp.model_wire import build

                        model = build(request["spec"])
                        algorithm = request["spec"]["algorithm"]
                    elif model is None:
                        raise MCPError("invalid_handle")
                    else:
                        if request["expected_revision"] != revision:
                            raise MCPError("stale_revision")
                        frame = None
                        if "asset_path" in request:
                            import polars as pl

                            frame = pl.scan_ipc(request["asset_path"], memory_map=False)
                        result = None
                        if operation in {"models_ensemble_add", "models_ensemble_remove"}:
                            from truthound.mcp.model_wire import build, validate_spec
                            from truthound.ml import EnsembleAnomalyDetector
                            from truthound.ml.base import AnomalyDetector

                            if not isinstance(model, EnsembleAnomalyDetector):
                                raise MCPError("unsupported_operation")
                            if operation == "models_ensemble_add":
                                specification = detector_spec(model)
                                specification["detectors"].append(request["detector"])
                                validate_spec(specification)
                                child = build(request["detector"])
                                if not isinstance(child, AnomalyDetector):
                                    raise MCPError("invalid_arguments")
                                model.add_detector(child, weight=request["weight"])
                            else:
                                child = model.remove_detector(request["index"])
                                detached = {
                                    "detached_spec": detector_spec(child),
                                    "detached_trained": child.is_trained,
                                }
                                if child.is_trained:
                                    save_bounded(child, request)
                                else:
                                    with BoundedWriter(
                                        Path(request["output_path"]), request["output_budget"]
                                    ) as output:
                                        output.write(b"{}")
                            revision += 1
                        elif operation == "models_save":
                            save_bounded(model, request)
                        elif operation == "models_load":
                            model.load(request["state_path"])
                            revision += 1
                        elif operation == "models_fit":
                            model.fit(frame)
                            revision += 1
                        elif operation == "models_partial_fit":
                            model.partial_fit(frame)
                            revision += 1
                        elif operation == "models_fit_predict":
                            result = model.fit_predict(frame).to_dict()
                            revision += 1
                        elif operation == "models_learn_rules":
                            from truthound.ml.base import RuleLearner

                            if not isinstance(model, RuleLearner):
                                raise MCPError("unsupported_operation")
                            result = model.learn_rules(frame).to_dict()
                            revision += 1
                        elif operation == "models_detect":
                            from truthound.ml.base import MLDriftDetector

                            if not isinstance(model, MLDriftDetector):
                                raise MCPError("unsupported_operation")
                            reference = pl.scan_ipc(request["reference_path"], memory_map=False)
                            result = model.detect(
                                reference, frame, columns=request["columns"]
                            ).to_dict()
                            revision += 1
                        elif operation == "models_diagnostics":
                            from dataclasses import asdict

                            from truthound.ml import (
                                ConceptDriftDetector,
                                DataProfileRuleLearner,
                                EnsembleAnomalyDetector,
                                FeatureDriftDetector,
                                IsolationForestDetector,
                                MultivariateDriftDetector,
                                PatternRuleLearner,
                                StatisticalAnomalyDetector,
                            )

                            diagnostics = {}
                            if isinstance(model, StatisticalAnomalyDetector):
                                diagnostics["statistics"] = model.get_statistics()
                            if isinstance(model, IsolationForestDetector):
                                diagnostics["feature_importance"] = model.get_feature_importance()
                            if isinstance(model, EnsembleAnomalyDetector):
                                diagnostics.update(
                                    detector_names=model.get_detector_names(),
                                    detector_weights=model.get_detector_weights(),
                                )
                            if isinstance(model, ConceptDriftDetector):
                                diagnostics["reference_correlations"] = (
                                    model.get_reference_correlations()
                                )
                            if isinstance(model, FeatureDriftDetector):
                                diagnostics["feature_profiles"] = model.get_feature_profiles()
                            if isinstance(model, MultivariateDriftDetector):
                                diagnostics["correlation_matrix"] = model.get_correlation_matrix()
                            if isinstance(model, DataProfileRuleLearner):
                                diagnostics["column_profiles"] = model.get_column_profiles()
                            if isinstance(model, PatternRuleLearner):
                                diagnostics["detected_patterns"] = model.get_detected_patterns()
                            result = {
                                "info": model.info.to_dict(),
                                "config": asdict(model.config),
                                "diagnostics": diagnostics,
                            }
                        elif operation == "models_predict":
                            result = model.predict(frame).to_dict()
                        elif operation == "models_score":
                            from truthound.ml.base import AnomalyDetector

                            if not isinstance(model, AnomalyDetector):
                                raise MCPError("unsupported_operation")
                            score = model.score(frame)
                            result = {
                                "name": score.name,
                                "dtype": str(score.dtype),
                                "values": score.to_list(),
                            }
                        elif operation == "models_rules":
                            from truthound.ml.base import RuleLearner

                            if not isinstance(model, RuleLearner):
                                raise MCPError("unsupported_operation")
                            result = [rule.to_dict() for rule in model.get_rules()]
                        elif operation == "models_rule_query":
                            from truthound.ml.base import RuleLearner

                            if not isinstance(model, RuleLearner):
                                raise MCPError("unsupported_operation")
                            prediction = model.predict(frame)
                            action = request["action"]
                            if action == "suite":
                                result = prediction.to_validation_suite()
                            elif action == "by_type":
                                result = [
                                    rule.to_dict()
                                    for rule in prediction.get_rules_by_type(request["rule_type"])
                                ]
                            elif action == "by_column":
                                result = [
                                    rule.to_dict()
                                    for rule in prediction.get_rules_for_column(request["column"])
                                ]
                            elif action == "validator_specs":
                                result = [rule.to_validator_spec() for rule in prediction]
                            else:
                                raise MCPError("invalid_arguments")
                        elif operation != "models_inspect":
                            raise MCPError("unknown_operation")
                        if result is not None:
                            with BoundedWriter(
                                Path(request["output_path"]), request["output_budget"]
                            ) as output:
                                output.write(
                                    json.dumps(
                                        {
                                            "encoding": "tagged-v1",
                                            "model_algorithm": algorithm,
                                            "model_revision": revision,
                                            "result": encode(result),
                                        },
                                        allow_nan=False,
                                    ).encode()
                                )
                    response = {
                        "ok": True,
                        "revision": revision,
                        "state": model.state.value,
                        "trained": model.is_trained,
                        **detached,
                    }
            except Exception as exc:
                response = {
                    "ok": False,
                    "code": exc.code
                    if isinstance(exc, MCPError)
                    else "missing_dependency"
                    if isinstance(exc, ModuleNotFoundError)
                    else "core_error",
                }
            wire.write(json.dumps(response, allow_nan=False) + "\n")
            wire.flush()


if __name__ == "__main__":
    main()
