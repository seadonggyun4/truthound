"""Full native suite identity and SQL execution errors through the real adapter."""

import inspect
import sqlite3

import polars as pl
import pytest

from tests import test_mcp_suites as suite_helpers
from tests.test_mcp_suites import call, contents, invoke
from truthound.mcp.tables import decode

pytestmark = pytest.mark.contract


@pytest.fixture
def suite_runtime(tmp_path):
    # Share setup/teardown without copying its authorization policy.
    yield from inspect.unwrap(suite_helpers.runtime)(tmp_path)


@pytest.mark.parametrize("parallel", [False, True])
def test_same_display_names_preserve_native_instance_identity_policy_and_details(
    suite_runtime, parallel
):
    from truthound.core import ScanPlanner, ValidationRuntime, build_validation_asset
    from truthound.core.suite import CheckSpec, EvidencePolicy, ValidationSuite
    from truthound.types import ResultFormat, ResultFormatConfig
    from truthound.validators import get_validator

    service, _ = suite_runtime
    values = [-1, -1, 20, 20, 30]
    asset = call(
        service,
        "assets_inline",
        columns=[{"name": "x", "dtype": {"kind": "Int64"}, "values": values}],
    )["resource_id"]
    policy = {
        "result_format": {
            "format": "complete",
            "partial_unexpected_count": 1,
            "include_unexpected_rows": True,
        }
    }
    suite = call(
        service,
        "suites_create",
        spec={
            "checks": [
                {
                    "check_id": "minimum",
                    "check_name": "shared",
                    "validator": "between",
                    "settings": {"columns": ["x"], "min_value": 0},
                },
                {
                    "check_id": "maximum",
                    "check_name": "shared",
                    "validator": "between",
                    "settings": {"columns": ["x"], "max_value": 0},
                },
            ],
            "evidence_policy": policy,
        },
    )["resource_id"]
    actual = decode(
        contents(
            service,
            call(
                service,
                "suites_validate",
                asset_id=asset,
                suite_id=suite,
                parallel=parallel,
                max_workers=2,
            )["artifact"]["resource_id"],
        )
    )
    native_suite = ValidationSuite(
        name="mcp-suite",
        checks=(
            CheckSpec(
                id="minimum",
                name="shared",
                category="distribution",
                factory=lambda: get_validator("between")(columns=["x"], min_value=0),
            ),
            CheckSpec(
                id="maximum",
                name="shared",
                category="distribution",
                factory=lambda: get_validator("between")(columns=["x"], max_value=0),
            ),
        ),
        evidence_policy=EvidencePolicy(
            ResultFormatConfig(
                format=ResultFormat.COMPLETE,
                partial_unexpected_count=1,
                include_unexpected_rows=True,
            )
        ),
    )
    native_asset = build_validation_asset(pl.DataFrame({"x": values}))
    expected = (
        ValidationRuntime()
        .execute(
            asset=native_asset,
            plan=ScanPlanner().plan(
                suite=native_suite, asset=native_asset, parallel=parallel, max_workers=2
            ),
        )
        .to_dict()
    )
    for key in (
        "checks",
        "issues",
        "execution_issues",
        "metadata",
        "result_format",
        "execution_mode",
        "planned_execution_mode",
    ):
        assert actual[key] == expected[key], key
    assert len(actual["checks"]) == 2
    assert [row["metadata"]["_truthound_check_id"] for row in actual["checks"]] == [
        "minimum",
        "maximum",
    ]
    assert [row["issues"][0]["count"] for row in actual["checks"]] == [2, 3]
    assert [len(row["issues"][0]["result"]["unexpected_rows"]) for row in actual["checks"]] == [
        2,
        3,
    ]


@pytest.mark.fault
def test_actual_sql_fallback_failure_publishes_failed_native_artifact_not_success(suite_runtime):
    service, (data, _, _) = suite_runtime
    path = data / "error.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE facts (x TEXT)")
        connection.executemany("INSERT INTO facts VALUES (?)", [("not numeric",)] * 100)
    source = call(service, "sources_open_sqlite", path=str(path), table="facts")["resource_id"]
    failed = invoke(
        service,
        "sources_validate",
        source_id=source,
        checks=[{"validator": "iqr_anomaly", "settings": {"column": "x"}}],
        result_format="complete",
    )
    assert failed["execution_status"] == "failed"
    assert failed["issues"] == [{"code": "core_execution_failed"}]
    native = decode(contents(service, failed["data"]["artifact"]["resource_id"]))
    assert native["row_count"] == 100 and native["execution_issues"]
    assert len(native["checks"]) == 1 and not native["checks"][0]["success"]
    assert native["checks"][0]["metadata"]["_truthound_execution_incomplete"] is True
    assert native["issues"][0]["exception_info"]["raised_exception"] is True
    # The completed failure does not imply the connection itself is unusable.
    assert call(service, "sources_validate_connection", source_id=source)["bytes"] > 0
