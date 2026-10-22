"""Algorithm scenarios compare complete canonical Core outcomes, not discovery."""

import asyncio
import base64
import importlib.util
import json
import sys
from pathlib import Path

import polars as pl
import pytest

pytest.importorskip("mcp")

from truthound.mcp.approval import ApprovalStore
from truthound.mcp.operations import _context
from truthound.mcp.policy import MCPPolicy
from truthound.mcp.service import MCPService
from truthound.mcp.tables import decode

_spec = importlib.util.spec_from_file_location(
    "truthound_mcp_validator_scenarios", Path(__file__).with_name("mcp_validator_cases.py")
)
assert _spec is not None and _spec.loader is not None
_cases = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _cases
_spec.loader.exec_module(_cases)
SCENARIOS = _cases.SCENARIOS

pytestmark = pytest.mark.contract


def native_settings(scenario, frame):
    settings = scenario.native_settings
    return settings(frame) if callable(settings) else settings or scenario.settings


def invoke(service, name, **arguments):
    response = asyncio.run(service.call_operation_async(name, arguments))
    assert response["execution_status"] == "succeeded", response.get("issues", response)
    return response["data"]


def read(service, identity):
    parts, offset = [], 0
    while offset is not None:
        chunk = invoke(service, "artifacts_read", resource_id=identity, offset=offset)
        parts.append(base64.b64decode(chunk["base64"]))
        offset = chunk["next_offset"]
    return decode(json.loads(b"".join(parts)))


CASES = [
    (polarity, row)
    for polarity in ("positive", "negative", "error", "caught_error")
    for row in SCENARIOS
    if polarity in {"positive", "negative"} or getattr(row, polarity) is not None
]


@pytest.mark.parametrize(
    ("polarity", "scenario"), CASES, ids=[f"{polarity}-{row.validator}" for polarity, row in CASES]
)
def test_actual_algorithm_matches_core_with_full_canonical_evidence(tmp_path, scenario, polarity):
    from dataclasses import replace

    import truthound as th
    from truthound.validators import get_validator

    if polarity == "negative" and scenario.negative_settings is not None:
        scenario = replace(
            scenario,
            settings=scenario.negative_settings,
            native_settings=scenario.native_negative_settings,
        )
    frame = pl.DataFrame(getattr(scenario, polarity))
    if polarity in {"error", "caught_error"} and scenario.error_overrides is not None:
        scenario = replace(
            scenario,
            native_settings={**native_settings(scenario, frame), **scenario.error_overrides},
            settings={**scenario.settings, **scenario.error_overrides},
        )
    validator = get_validator(scenario.validator)(**native_settings(scenario, frame))
    # This is an independent constructor, not the production wire converter.
    direct = th.check(
        frame,
        validators=[validator],
        result_format="complete",
        context=_context(str(tmp_path / "direct/input.ipc")),
    )
    if polarity == "error":
        assert direct.execution_issues, "A quality violation is not an execution-error fixture"
    elif polarity == "caught_error":
        assert not direct.execution_issues
        assert direct.issues and all(
            issue.issue_type
            in {
                "query_execution_error",
                "expression_execution_error",
                "multi_column_validation_error",
            }
            for issue in direct.issues
        ), "Prove an actual native caught-exception path, not an ordinary negative case"
    else:
        assert not direct.execution_issues, direct.execution_issues
        assert bool(direct.issues) == (polarity == "negative"), (
            "Scenario must exercise a genuine native pass and violation",
            scenario.validator,
            [issue.issue_type for issue in direct.issues],
        )
    response, actual = mcp_run(tmp_path, scenario, frame)
    if polarity == "error":
        assert response["execution_status"] == "failed"
        assert response["issues"] == [{"code": "core_execution_failed"}]
    else:
        assert response["execution_status"] == "succeeded", response
    expected = direct.to_dict()
    for field in (
        "checks",
        "issues",
        "execution_issues",
        "row_count",
        "column_count",
        "result_format",
        "execution_mode",
        "planned_execution_mode",
        "suite_name",
        "source",
    ):
        assert actual[field] == expected[field], field


@pytest.mark.parametrize(
    "scenario", [row for row in SCENARIOS if not row.trusted], ids=lambda row: row.validator
)
def test_actual_constructor_configuration_error_is_not_execution_coverage(tmp_path, scenario):
    from dataclasses import replace

    from truthound.validators import get_validator

    # A real native invariant, not a monkeypatched validate() or omitted
    # required argument. This is reported separately from execution errors.
    native = {**native_settings(scenario, pl.DataFrame(scenario.positive)), "timeout_seconds": 0.0}
    with pytest.raises(ValueError, match="timeout_seconds must be > 0"):
        get_validator(scenario.validator)(**native)
    invalid = replace(scenario, settings={**scenario.settings, "timeout_seconds": 0.0})
    response, artifact = mcp_run(tmp_path, invalid, pl.DataFrame(scenario.positive))
    assert response["execution_status"] == "failed"
    assert response["issues"] == [{"code": "validator_configuration_error"}]
    assert artifact is None


@pytest.mark.parametrize(
    "scenario", [row for row in SCENARIOS if row.trusted], ids=lambda row: row.validator
)
def test_approved_constructor_failure_preserves_uncertain_effects(tmp_path, scenario):
    from dataclasses import replace

    from truthound.validators import get_validator

    with pytest.raises(ValueError, match="timeout_seconds must be > 0"):
        get_validator(scenario.validator)(
            **{**native_settings(scenario, pl.DataFrame(scenario.positive)), "timeout_seconds": 0.0}
        )
    invalid = replace(scenario, settings={**scenario.settings, "timeout_seconds": 0.0})
    response, artifact = mcp_run(tmp_path, invalid, pl.DataFrame(scenario.positive))
    # Installed/trusted code might already have executed before construction
    # failed. A failure reply is not proof of NO_EFFECTS or a retry permit.
    assert response["execution_status"] == "outcome_unknown"
    assert response["issues"] == [{"code": "outcome_unknown"}]
    assert artifact is None


def mcp_run(tmp_path, scenario, frame):
    data = tmp_path / "data"
    data.mkdir()
    path = data / "input.parquet"
    frame.write_parquet(path)
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    service = MCPService(
        MCPPolicy((data,), profile="core", allow_sensitive_results=True, control_directory=control)
    )
    try:
        asset = invoke(service, "assets_capture", path=str(path))["resource_id"]
        references = {"current": asset}
        for name, values in (scenario.frames or {}).items():
            reference_path = data / f"{name}.parquet"
            pl.DataFrame(values).write_parquet(reference_path)
            references[name] = invoke(service, "assets_capture", path=str(reference_path))[
                "resource_id"
            ]

        def wire(value):
            # Test fixture references are not production conversion. Native
            # expected constructors above independently receive actual frames.
            if isinstance(value, dict):
                if set(value) == {"asset_fixture"}:
                    return {"asset_id": references[value["asset_fixture"]]}
                return {key: wire(item) for key, item in value.items()}
            if isinstance(value, list):
                return [wire(item) for item in value]
            return value

        check = {"validator": scenario.validator, "settings": wire(scenario.settings)}
        arguments = {"asset_id": asset, "checks": [check], "result_format": "complete"}
        if scenario.trusted:
            binding = control / "bindings.json"
            binding.write_text(
                json.dumps(
                    {
                        "validators": {
                            "case-template": {
                                "validator": scenario.validator,
                                "settings": check["settings"],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            binding.chmod(0o600)
            check.update(settings={}, binding_id="case-template")
            arguments["idempotency_key"] = "algorithm-template-1"
            pending = asyncio.run(service.call_operation_async("assets_validate", arguments))
            assert pending["issues"] == [{"code": "approval_required"}]
            store = ApprovalStore(control)
            try:
                store.approve(pending["data"]["plan_id"])
            finally:
                store.close()
        response = asyncio.run(service.call_operation_async("assets_validate", arguments))
        if response["execution_status"] == "outcome_unknown":
            replay = asyncio.run(service.call_operation_async("assets_validate", arguments))
            assert replay["execution_status"] == "outcome_unknown"
            assert replay["issues"] == [{"code": "outcome_unknown"}]
            status = invoke(service, "plans_status", plan_id=response["data"]["plan_id"])
            assert status["state"] == "outcome_unknown"
        result = response["data"]
        actual = (
            read(service, result["artifact"]["resource_id"])
            if result and result.get("artifact")
            else None
        )
        return response, actual
    finally:
        service.close()
