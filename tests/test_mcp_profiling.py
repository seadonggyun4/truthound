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
    "rules",
    [
        None,
        [],
        ["types"],
        ["column_presence"],
        ["nullability"],
        ["column_presence", "types", "types"],
    ],
)
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("change", ["same", "widen", "narrow", "added", "removed"])
def test_schema_compatibility_complete_native_result(tmp_path, rules, strict, change):
    from truthound.profiler.evolution import compatibility as native

    baseline = pl.DataFrame({"x": pl.Series([1], dtype=pl.Int32)})
    current = {
        "same": baseline,
        "widen": baseline.cast({"x": pl.Int64}),
        "narrow": baseline.cast({"x": pl.Int8}),
        "added": baseline.with_columns(pl.lit(1).alias("y")),
        "removed": baseline.rename({"x": "y"}),
    }[change]
    classes = {
        "types": native.TypeCompatibilityChecker,
        "column_presence": native.ColumnPresenceChecker,
        "nullability": native.NullabilityCompatibilityChecker,
    }
    expected = native.SchemaCompatibilityAnalyzer(
        rules=None if rules is None else [classes[name]() for name in rules],
        strict=strict,
    ).analyze(baseline.schema, current.schema)
    service = MCPService(MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True))
    try:
        handles = []
        for index, frame in enumerate((baseline, current)):
            path = tmp_path / f"compatibility-{index}.parquet"
            frame.write_parquet(path)
            handles.append(call(service, "assets_capture", path=str(path))["resource_id"])
        args = dict(baseline_id=handles[0], current_id=handles[1], rules=rules, strict=strict)
        artifact = call(service, "profilers_schema_compatibility", **args)
        assert artifact["media_type"] == "application/vnd.truthound.schema-compatibility+json"
        part = call(service, "artifacts_read", resource_id=artifact["resource_id"])
        assert part["next_offset"] is None
        actual = decode(json.loads(base64.b64decode(part["base64"])))
        assert actual == {
            "is_compatible": expected.is_compatible,
            "level": expected.level.value,
            "issues": [issue.to_dict() for issue in expected.issues],
            "rules_checked": expected.rules_checked,
        }
        call(service, "resources_release", resource_id=handles[int(strict)])
        for operation, arguments in (
            ("artifacts_read", {"resource_id": artifact["resource_id"]}),
            ("profilers_schema_compatibility", args),
        ):
            result = asyncio.run(service.call_operation_async(operation, arguments))
            assert result["execution_status"] == "failed" and not result.get("data")
    finally:
        service.close()


def test_schema_compatibility_rejects_unreviewed_rules(tmp_path):
    service = MCPService(MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True))
    try:
        path = tmp_path / "input.csv"
        path.write_text("x\n1\n", encoding="utf-8")
        identity = call(service, "assets_capture", path=str(path))["resource_id"]
        for rules in (["arbitrary.module"], ["types"] * 17, [{"class": "os.system"}]):
            result = asyncio.run(
                service.call_operation_async(
                    "profilers_schema_compatibility",
                    {
                        "baseline_id": identity,
                        "current_id": identity,
                        "rules": rules,
                    },
                )
            )
            assert result["issues"] == [{"code": "invalid_arguments"}]
            assert result["execution_status"] == "failed" and not result.get("data")
    finally:
        service.close()


@pytest.mark.parametrize(
    "calculator,options",
    [
        (None, {}),
        ({"kind": "levenshtein"}, {"allow_compatible_types": False}),
        ({"kind": "jaro_winkler", "prefix_scale": 0.9}, {}),
        ({"kind": "ngram", "n": 3}, {"similarity_threshold": 0.99}),
        ({"kind": "token"}, {"require_type_match": False}),
        ({"kind": "composite", "calculators": []}, {}),
        (
            {
                "kind": "composite",
                "calculators": [
                    {"calculator": {"kind": "ngram", "n": 2}, "weight": 0.6},
                    {"calculator": {"kind": "jaro_winkler", "prefix_scale": 0.2}, "weight": 0.4},
                ],
            },
            {"high_confidence_threshold": 0.95, "low_confidence_threshold": 0.2},
        ),
    ],
)
def test_column_rename_analysis_matches_native_and_revokes_parent(tmp_path, calculator, options):
    from truthound.profiler.evolution import rename_detector as native

    def build(value):
        if value is None:
            return None
        settings = {key: val for key, val in value.items() if key != "kind"}
        if value["kind"] == "composite" and settings.get("calculators") is not None:
            settings["calculators"] = [
                (build(item["calculator"]), item["weight"]) for item in value["calculators"]
            ]
        classes = {
            "levenshtein": native.LevenshteinSimilarity,
            "token": native.TokenSimilarity,
            "jaro_winkler": native.JaroWinklerSimilarity,
            "ngram": native.NgramSimilarity,
            "composite": native.CompositeSimilarity,
        }
        return classes[value["kind"]](**settings)

    baseline = pl.DataFrame(
        {"customer_name": pl.Series([1], dtype=pl.Int32), "memo": ["a"], "same": [1]}
    )
    current = pl.DataFrame({"customer_names": [1], "unrelated": [1.0], "same": [1]})
    expected = (
        native.ColumnRenameDetector(calculator=build(calculator), **options)
        .detect(
            {name: dtype for name, dtype in current.schema.items() if name not in baseline.schema},
            {name: dtype for name, dtype in baseline.schema.items() if name not in current.schema},
        )
        .to_dict()
    )
    service = MCPService(MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True))
    try:
        handles = []
        for index, frame in enumerate((baseline, current)):
            path = tmp_path / f"schema-{index}.parquet"
            frame.write_parquet(path)
            handles.append(call(service, "assets_capture", path=str(path))["resource_id"])
        args = dict(baseline_id=handles[0], current_id=handles[1], calculator=calculator, **options)
        artifact = call(service, "profilers_column_renames", **args)["resource_id"]
        part = call(service, "artifacts_read", resource_id=artifact)
        assert part["next_offset"] is None
        assert decode(json.loads(base64.b64decode(part["base64"]))) == expected
        call(service, "resources_release", resource_id=handles[1])
        for operation, arguments in (
            ("artifacts_read", {"resource_id": artifact}),
            ("profilers_column_renames", args),
        ):
            denied = asyncio.run(service.call_operation_async(operation, arguments))
            assert denied["execution_status"] == "failed" and not denied.get("data")
    finally:
        service.close()


def test_column_rename_analysis_rejects_unreviewed_calculator_configuration(tmp_path):
    service = MCPService(MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True))
    try:
        path = tmp_path / "input.csv"
        path.write_text("x\n1\n", encoding="utf-8")
        identity = call(service, "assets_capture", path=str(path))["resource_id"]
        for calculator in (
            {"kind": "python", "module": "os"},
            {"kind": "token", "prefix_scale": 0.1},
            {"kind": "jaro_winkler", "prefix_scale": float("inf")},
        ):
            response = asyncio.run(
                service.call_operation_async(
                    "profilers_column_renames",
                    {
                        "baseline_id": identity,
                        "current_id": identity,
                        "calculator": calculator,
                    },
                )
            )
            assert response["execution_status"] == "failed" and not response.get("data")
            assert response["issues"] == [{"code": "invalid_arguments"}]
    finally:
        service.close()


@pytest.mark.parametrize(
    "renames,threshold,released", [(True, 0.8, 0), (False, 0.8, 1), (True, 1.0, 0)]
)
def test_schema_evolution_matches_native_and_requires_both_parents(
    tmp_path, renames, threshold, released
):
    from datetime import datetime

    from truthound.profiler.evolution import SchemaEvolutionDetector

    baseline = pl.DataFrame(
        {
            "account_code": [1],
            "obsolete": [1],
            "widen": pl.Series([1], dtype=pl.Int32),
            "narrow": [1.0],
        }
    )
    current = pl.DataFrame({"account_codes": [1], "new": [1], "widen": [1], "narrow": [1]})
    native = SchemaEvolutionDetector(detect_renames=renames, rename_similarity_threshold=threshold)
    changes = native.detect_changes(current.schema, baseline.schema)
    expected = native.get_change_summary(changes).to_dict()
    assert expected["breaking_changes"] > 0 and expected["type_changes"] == 2
    service = MCPService(MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True))
    try:
        handles = [
            call(
                service,
                "assets_inline",
                columns=[
                    {"name": name, "dtype": {"kind": str(dtype)}, "values": frame[name].to_list()}
                    for name, dtype in frame.schema.items()
                ],
            )["resource_id"]
            for frame in (baseline, current)
        ]
        args = {
            "baseline_id": handles[0],
            "current_id": handles[1],
            "detect_renames": renames,
            "rename_similarity_threshold": threshold,
        }
        artifact = call(service, "profilers_schema_evolution", **args)["resource_id"]
        parts, offset = [], 0
        while offset is not None:
            chunk = call(service, "artifacts_read", resource_id=artifact, offset=offset)
            parts.append(base64.b64decode(chunk["base64"]))
            offset = chunk["next_offset"]
        actual = decode(json.loads(b"".join(parts)))
        # Native set iteration is unordered; compare all change fields as a multiset.
        assert sorted(json.dumps(c, sort_keys=True) for c in actual["changes"]) == sorted(
            json.dumps(c.to_dict(), sort_keys=True) for c in changes
        )
        assert datetime.fromisoformat(actual["summary"].pop("detected_at"))
        expected.pop("detected_at")
        assert actual["summary"] == expected
        call(service, "resources_release", resource_id=handles[released])
        for operation, arguments in (
            ("artifacts_read", {"resource_id": artifact}),
            ("profilers_schema_evolution", args),
        ):
            denied = asyncio.run(service.call_operation_async(operation, arguments))
            assert denied["execution_status"] == "failed" and not denied.get("data")
    finally:
        service.close()


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


@pytest.mark.parametrize(
    "generator",
    ["SchemaRuleGenerator", "StatsRuleGenerator", "PatternRuleGenerator", "MLRuleGenerator"],
)
@pytest.mark.parametrize("strictness", ["loose", "medium", "strict"])
def test_generated_validator_configs_match_native(tmp_path, generator, strictness):
    column = "email" if generator == "PatternRuleGenerator" else "amount"
    assert_generated_rules(tmp_path, generator, strictness, column, "validator_configs")


def assert_generated_rules(tmp_path, generator, strictness, column=None, representation="rules"):
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
    expected = (
        [rule.to_dict() for rule in rules]
        if representation == "rules"
        else [
            {"validator_class": rule.validator_class, "config": rule.to_validator_config()}
            for rule in rules
        ]
    )
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
            representation=representation,
        )["resource_id"]
        parts, offset = [], 0
        while offset is not None:
            chunk = call(adapter, "artifacts_read", resource_id=output, offset=offset)
            parts.append(base64.b64decode(chunk["base64"]))
            offset = chunk["next_offset"]
        assert decode(json.loads(b"".join(parts))) == expected
        if column is not None:
            assert all(rule.columns == (column,) for rule in rules)
            with pytest.raises(KeyError):
                native.generate_for_column(profile["absent"], core.Strictness(strictness))
            failed = asyncio.run(
                adapter.call_operation_async(
                    "profilers_generate_rules",
                    {
                        "asset_id": asset,
                        "generator": {"component": generator},
                        "column": "absent",
                        "representation": representation,
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
        {"generator": {"component": "SchemaRuleGenerator"}, "representation": "execute"},
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
