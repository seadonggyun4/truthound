"""Advanced profiler composition must match native algorithms, not facade summaries."""

import asyncio
import base64
import json

import polars as pl
import pytest

pytest.importorskip("mcp")

from truthound.mcp.policy import MCPPolicy
from truthound.mcp.service import MCPService
from truthound.mcp.tables import decode

pytestmark = pytest.mark.contract


def call(adapter, operation, **arguments):
    result = asyncio.run(adapter.call_operation_async(operation, arguments))
    assert result["execution_status"] == "succeeded", result
    return result["data"]


def normalize(value):
    if isinstance(value, dict):
        return {
            key: normalize(item)
            for key, item in value.items()
            if key not in {"profiled_at", "profile_duration_ms"}
        }
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value


@pytest.mark.parametrize(
    "generator",
    [
        "SchemaRuleGenerator",
        "StatsRuleGenerator",
        "PatternRuleGenerator",
        "MLRuleGenerator",
    ],
)
@pytest.mark.parametrize("strictness", ["loose", "medium", "strict"])
def test_generated_rules_match_complete_native_result(tmp_path, generator, strictness):
    assert_generated_rules(tmp_path, generator, strictness)


@pytest.mark.parametrize(
    "generator",
    [
        "SchemaRuleGenerator",
        "StatsRuleGenerator",
        "PatternRuleGenerator",
        "MLRuleGenerator",
    ],
)
@pytest.mark.parametrize("strictness", ["loose", "medium", "strict"])
def test_column_rules_match_native_without_table_rules(tmp_path, generator, strictness):
    column = "email" if generator == "PatternRuleGenerator" else "amount"
    assert_generated_rules(tmp_path, generator, strictness, column=column)


def assert_generated_rules(tmp_path, generator, strictness, column=None):
    import truthound.profiler as core

    frame = pl.DataFrame(
        {
            "amount": [float(i) for i in range(30)],
            "email": ["person@example.test"] * 30,
        }
    )
    config = core.ProfilerConfig(include_distributions=True, include_patterns=True)
    profile = core.DataProfiler(
        config=config,
        column_profiler=core.ColumnProfiler(config=config),
        parallel=False,
    ).profile(frame.lazy())
    settings = {"correlation_threshold": 0.8} if generator == "MLRuleGenerator" else {}
    native = getattr(core, generator)(**settings)
    rules = (
        native.generate(profile, core.Strictness(strictness))
        if column is None
        else native.generate_for_column(profile[column], core.Strictness(strictness))
    )
    expected = [rule.to_dict() for rule in rules]
    assert expected, "Exercise real rule generation, not an empty result"
    adapter = MCPService(MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True))
    try:
        asset = call(
            adapter,
            "assets_inline",
            columns=[
                {"name": name, "dtype": {"kind": str(dtype)}, "values": frame[name].to_list()}
                for name, dtype in frame.schema.items()
            ],
        )["resource_id"]
        output = call(
            adapter,
            "profilers_generate_rules",
            asset_id=asset,
            config={"include_distributions": True, "include_patterns": True},
            options={"parallel": False},
            generator={"component": generator, "settings": settings},
            strictness=strictness,
            column=column,
        )["resource_id"]
        parts, offset = [], 0
        while offset is not None:
            chunk = call(adapter, "artifacts_read", resource_id=output, offset=offset)
            parts.append(base64.b64decode(chunk["base64"]))
            offset = chunk["next_offset"]
        assert decode(json.loads(b"".join(parts))) == expected
        if column is not None:
            assert all(rule["columns"] == [column] for rule in expected)
            with pytest.raises(KeyError):
                native.generate_for_column(profile["absent"], core.Strictness(strictness))
            failed = asyncio.run(
                adapter.call_operation_async(
                    "profilers_generate_rules",
                    {
                        "asset_id": asset,
                        "generator": {"component": generator},
                        "column": "absent",
                    },
                )
            )
            assert failed["execution_status"] == "failed"
            assert failed["issues"] == [{"code": "core_error"}]
            assert not failed.get("data"), "No artifact may represent a failed generation"
        call(adapter, "resources_release", resource_id=asset)
        denied = asyncio.run(
            adapter.call_operation_async("artifacts_read", {"resource_id": output})
        )
        assert denied["execution_status"] != "succeeded"
    finally:
        adapter.close()


@pytest.mark.fault
@pytest.mark.parametrize(
    "arguments",
    [
        {"generator": {"component": "os.system"}},
        {"generator": {"component": "SchemaRuleGenerator", "settings": {"typo": True}}},
        {"generator": {"component": "SchemaRuleGenerator"}, "strictness": "unknown"},
    ],
)
def test_rule_generation_rejects_unreviewed_configuration(tmp_path, arguments):
    adapter = MCPService(MCPPolicy((tmp_path,), profile="core"))
    try:
        result = asyncio.run(
            adapter.call_operation_async(
                "profilers_generate_rules",
                {
                    "asset_id": "asset_" + "a" * 32,
                    **arguments,
                },
            )
        )
        assert result["execution_status"] != "succeeded"
        expected_code = (
            "unknown_component"
            if arguments["generator"]["component"] == "os.system"
            else "invalid_arguments"
        )
        assert result["issues"] == [{"code": expected_code}]
    finally:
        adapter.close()


@pytest.mark.parametrize("parallel", [False, True])
def test_full_profiler_configuration_and_selected_components_match_native_core(tmp_path, parallel):
    import truthound.profiler as core

    config = {
        "sample_size": 23,
        "random_seed": 17,
        "include_patterns": True,
        "include_correlations": True,
        "include_distributions": True,
        "top_n_values": 3,
        "pattern_sample_size": 12,
        "correlation_threshold": 0.6,
        "min_pattern_match_ratio": 0.7,
        "n_jobs": 2,
    }
    # DataProfiler's inherited options are distinct from ProfilerConfig in Core.
    # Do not merge them in the adapter and silently alter native semantics.
    options = {
        "parallel": parallel,
        "sample_size": 25,
        "include_patterns": True,
        "include_correlations": True,
        "top_n_values": 4,
        "pattern_sample_size": 14,
    }
    groups = {
        "column_analyzers": [
            ("BasicStatsAnalyzer", {}),
            ("NumericAnalyzer", {}),
            ("StringAnalyzer", {}),
            ("DatetimeAnalyzer", {}),
            ("ValueFrequencyAnalyzer", {}),
            ("PatternAnalyzer", {"min_match_ratio": 0.65}),
        ],
        "table_analyzers": [
            ("DuplicateRowAnalyzer", {}),
            ("MemoryEstimator", {}),
            ("CorrelationAnalyzer", {"threshold": 0.8}),
        ],
        "type_inferrers": [
            ("PhysicalTypeInferrer", {}),
            ("PatternBasedTypeInferrer", {"min_match_ratio": 0.75}),
            (
                "CardinalityTypeInferrer",
                {"categorical_threshold": 0.2, "identifier_threshold": 0.8},
            ),
        ],
    }
    native_groups = {
        group: [getattr(core, name)(**settings) for name, settings in rows]
        for group, rows in groups.items()
    }
    native_config = core.ProfilerConfig(**config)
    profiler = core.DataProfiler(
        config=native_config,
        column_profiler=core.ColumnProfiler(
            config=native_config,
            analyzers=native_groups["column_analyzers"],
            type_inferrers=native_groups["type_inferrers"],
        ),
        table_analyzers=native_groups["table_analyzers"],
        **options,
    )
    # Unique frequencies avoid nondeterministic Core hash-order tie breaking.
    # Keep all 30 input rows and every value/count/ratio in the comparison.
    values = [0] * 10 + [1] * 8 + [2] * 12
    frame = pl.DataFrame(
        {
            "x": [float(i) for i in values],
            "y": [float(i * 2) for i in values],
            "email": [f"person{i}@example.test" for i in values],
        }
    )
    expected = profiler.profile(frame.lazy(), name="typed", source="operator-label").to_dict()
    adapter = MCPService(MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True))
    try:
        discovery = decode(call(adapter, "profilers_catalog")["result"])
        assert all(
            set(discovery["components"][group]) == {n for n, _ in rows}
            for group, rows in groups.items()
        )
        asset = call(
            adapter,
            "assets_inline",
            columns=[
                {"name": name, "dtype": {"kind": str(dtype)}, "values": frame[name].to_list()}
                for name, dtype in frame.schema.items()
            ],
        )["resource_id"]
        output = call(
            adapter,
            "profilers_profile",
            asset_id=asset,
            config=config,
            options=options,
            name="typed",
            source="operator-label",
            **{
                group: [{"component": name, "settings": settings} for name, settings in rows]
                for group, rows in groups.items()
            },
        )["resource_id"]
        parts, offset = [], 0
        while offset is not None:
            chunk = call(adapter, "artifacts_read", resource_id=output, offset=offset)
            parts.append(base64.b64decode(chunk["base64"]))
            offset = chunk["next_offset"]
        actual = decode(json.loads(b"".join(parts)))
        assert normalize(actual) == normalize(expected)
        assert actual["row_count"] == 25  # inherited Core sampling, not config.sample_size
    finally:
        adapter.close()


@pytest.mark.fault
@pytest.mark.parametrize(
    "arguments,code",
    [
        ({"options": {"column_profiler": {}}}, "invalid_arguments"),
        ({"config": {"typo": 1}}, "invalid_arguments"),
        ({"column_analyzers": [{"component": "os.system"}]}, "unknown_component"),
        (
            {"column_analyzers": [{"component": "PatternAnalyzer", "settings": {"patterns": []}}]},
            "operator_configuration_required",
        ),
    ],
)
def test_profiler_rejects_unknown_options_and_untrusted_composition(tmp_path, arguments, code):
    adapter = MCPService(MCPPolicy((tmp_path,), profile="core"))
    try:
        result = asyncio.run(
            adapter.call_operation_async(
                "profilers_profile",
                {
                    "asset_id": "asset_" + "a" * 32,
                    **arguments,
                },
            )
        )
        assert result["issues"] == [{"code": code}]
    finally:
        adapter.close()
