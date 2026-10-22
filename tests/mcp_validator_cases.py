"""Synthetic algorithm scenarios, independent from production wire conversion.

Every scenario must produce an actual native violation on its negative frame.
Unlisted algorithms are gaps, not successful default-constructor probes.
"""

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class Scenario:
    validator: str
    settings: dict
    positive: dict
    negative: dict
    native_settings: dict | Callable | None = None
    frames: dict | None = None
    trusted: bool = False
    error: dict | None = None
    negative_settings: dict | None = None
    native_negative_settings: dict | None = None
    caught_error: dict | None = None
    error_overrides: dict | None = None


def scenarios():
    cases = []

    def add(
        name,
        settings,
        positive,
        negative,
        native_settings=None,
        frames=None,
        trusted=False,
        error=None,
        negative_settings=None,
        native_negative_settings=None,
        caught_error=None,
    ):
        cases.append(
            Scenario(
                name,
                settings,
                positive,
                negative,
                native_settings,
                frames,
                trusted,
                error,
                negative_settings,
                native_negative_settings,
                caught_error,
            )
        )

    numeric = {"x": [1, 2, 3, 4]}
    repeats = {"x": [1, 1, 1, 2]}
    for name, options, good, bad in (
        ("streaming_null", {}, numeric, {"x": [1, None, None, 4]}),
        ("streaming_completeness", {"min_ratio": 1.0}, numeric, {"x": [1, None, None, 4]}),
        ("streaming_nan", {}, {"x": [1.0, 2.0, 3.0]}, {"x": [1.0, float("nan"), 3.0]}),
        ("streaming_range", {"min_value": 0.0, "max_value": 5.0}, numeric, {"x": [1, 9, 3, -1]}),
        ("streaming_positive", {}, numeric, {"x": [1, -2, 0, 4]}),
        ("streaming_outlier", {}, {"x": [1, 2, 3, 4, 5]}, {"x": [1, 2, 3, 4, 1000]}),
    ):
        # Cross native chunk boundaries rather than testing only construction.
        add(name, {"columns": ["x"], "chunk_size": 2, **options}, good, bad)
    for name, settings in {
        "null": {},
        "not_null": {},
        "completeness_ratio": {"min_ratio": 1.0},
    }.items():
        add(name, {"columns": ["x"], **settings}, numeric, {"x": [1, None, None, None]})
    for name, bad in {"empty_string": "", "whitespace_only": "   ", "default_value": "N/A"}.items():
        config = {"columns": ["x"]}
        if name == "default_value":
            config["default_values"] = [bad]
        add(name, config, {"x": ["valid", "also valid"]}, {"x": ["valid", bad, bad]})
    for name, bad in {
        "nan": float("nan"),
        "not_nan": float("nan"),
        "nan_ratio": float("nan"),
        "infinity": float("inf"),
        "finite": float("inf"),
    }.items():
        add(name, {"columns": ["x"]}, {"x": [1.0, 2.0]}, {"x": [1.0, bad, bad]})
    add(
        "conditional_null",
        {"column": "x", "condition_column": "status", "condition_values": ["needed"]},
        {"x": [1, 2], "status": ["needed", "needed"]},
        {"x": [1, None], "status": ["needed", "needed"]},
    )
    for name, settings in {
        "unique": {},
        "unique_ratio": {"min_ratio": 1.0},
        "approximate_unique_ratio": {"min_ratio": 1.0},
        "distinct_count": {"min_count": 4},
        "approximate_distinct_count": {"min_count": 4},
        "streaming_distinct_count": {"min_count": 4},
    }.items():
        add(name, {"columns": ["x"], **settings}, numeric, repeats)
    add("duplicate", {"subset": ["x"]}, numeric, repeats)
    add("primary_key", {"column": "x"}, numeric, repeats)
    add(
        "compound_key",
        {"columns": ["x", "y"]},
        {"x": [1, 1, 2], "y": [1, 2, 3]},
        {"x": [1, 1, 1], "y": [1, 1, 2]},
    )
    add(
        "column_pair_unique",
        {"column_a": "x", "column_b": "y"},
        {"x": [1, 1, 2], "y": [3, 2, 3]},
        {"x": [1, 1, 1], "y": [1, 1, 2]},
    )
    add(
        "duplicate_within_group",
        {"check_column": "x", "group_by": ["g"]},
        {"x": [1, 2, 3], "g": ["a", "a", "b"]},
        {"x": [1, 1, 1], "g": ["a", "a", "b"]},
    )
    for name in ("unique_within_record", "all_columns_unique_within_record"):
        add(name, {"columns": ["x", "y"]}, {"x": [1, 2], "y": [3, 4]}, {"x": [1, 2], "y": [1, 2]})
    for name, settings in {
        "distinct_values_in_set": {"column": "x", "allowed_values": [1, 2, 3, 4]},
        "distinct_values_equal_set": {"column": "x", "expected_values": [1, 2, 3, 4]},
        "distinct_values_contain_set": {"column": "x", "required_values": [1, 2, 3, 4]},
        "distinct_count_between": {"column": "x", "min_count": 4, "max_count": 4},
    }.items():
        add(
            name,
            settings,
            numeric,
            {"x": [1, 1, 1, 99]} if name == "distinct_values_in_set" else {"x": [1, 1, 1, 2]},
        )
    for name, settings in {
        "between": {"min_value": 0, "max_value": 10},
        "in_set": {"allowed_values": [1, 2, 3, 4]},
        "not_in_set": {"forbidden_values": [-1]},
        "positive": {},
        "non_negative": {},
    }.items():
        add(name, {"columns": ["x"], **settings}, numeric, {"x": [1, -1, -1, -1]})
    add("increasing", {"columns": ["x"]}, numeric, {"x": [4, 3, 2, 1]})
    add("decreasing", {"columns": ["x"]}, {"x": [4, 3, 2, 1]}, numeric)
    for name in (
        "mean_between",
        "median_between",
        "min_between",
        "max_between",
        "sum_between",
        "std_between",
        "variance_between",
    ):
        add(
            name,
            {"columns": ["x"], "min_value": 0, "max_value": 20},
            numeric,
            {"x": [100, 200, 300, 400]},
        )
    add("type", {"columns": ["x"]}, {"x": ["a", "b", "c", "d"]}, {"x": ["a", "1", "2", "3"]})
    for name, (good, bad) in {
        "email": ("alice@example.test", "not-email"),
        "url": ("https://example.test", "not-url"),
        "phone": ("415-555-2671", "not-phone"),
        "ip_address": ("192.0.2.1", "999.999.999.999"),
        "ipv6_address": ("2001:db8::1", "invalid-ipv6"),
        "uuid": ("123e4567-e89b-12d3-a456-426614174000", "invalid-uuid"),
        "alphanumeric": ("abc123", "not valid!"),
        "json_parseable": ('{"a":1}', "not-json"),
    }.items():
        add(name, {"columns": ["x"]}, {"x": [good, good]}, {"x": [bad, bad, bad]})
    add(
        "length",
        {"columns": ["x"], "min_length": 2, "max_length": 5},
        {"x": ["good", "ok"]},
        {"x": ["a", "a", "a"]},
    )
    add(
        "consistent_casing",
        {"columns": ["x"]},
        {"x": ["abc", "def", "ghi"]},
        {"x": ["abc", "DEF", "ghi"]},
    )
    add(
        "json_schema",
        {
            "columns": ["x"],
            "schema": {
                "type": "object",
                "required": ["id"],
                "properties": {"id": {"type": "integer"}},
            },
        },
        {"x": ['{"id":1}', '{"id":2}']},
        {"x": ['{"id":"bad"}', '{"id":"bad"}']},
    )
    add("row_count", {"min_count": 3}, numeric, {"x": [1]})
    add("column_count", {"exact_count": 1}, numeric, {"x": [1], "y": [2]})
    add("column_exists", {"columns": ["x"]}, numeric, {"y": [1]})
    add("column_not_exists", {"columns": ["y"]}, numeric, {"x": [1], "y": [2]})
    add("column_order", {"expected_order": ["x", "y"]}, {"x": [1], "y": [2]}, {"y": [2], "x": [1]})
    add("column_type", {"expected_types": {"x": "int"}}, numeric, {"x": ["bad"]})
    add("table_schema", {"expected_schema": {"x": "Int64"}}, numeric, {"x": ["bad"]})
    add(
        "column_pair",
        {"column_a": "x", "column_b": "y", "relationship": "<"},
        {"x": [1, 2], "y": [3, 4]},
        {"x": [5, 5], "y": [2, 2]},
    )
    add(
        "column_pair_in_set",
        {"column_a": "x", "column_b": "y", "valid_pairs": [[1, 2]]},
        {"x": [1], "y": [2]},
        {"x": [1, 1], "y": [3, 3]},
        native_settings={"column_a": "x", "column_b": "y", "valid_pairs": [(1, 2)]},
    )
    add(
        "column_pair_not_in_set",
        {"column_a": "x", "column_b": "y", "forbidden_pairs": [[1, 2]]},
        {"x": [1], "y": [3]},
        {"x": [1, 1], "y": [2, 2]},
        native_settings={"column_a": "x", "column_b": "y", "forbidden_pairs": [(1, 2)]},
    )
    add(
        "multi_column_unique",
        {"columns": ["x", "y"]},
        {"x": [1, 2], "y": [3, 4]},
        {"x": [1, 1], "y": [2, 2]},
    )
    add(
        "multi_column_sum",
        {"columns": ["x", "y"], "equals_value": 10},
        {"x": [4, 5], "y": [6, 5]},
        {"x": [1, 1], "y": [2, 2]},
    )
    add(
        "multi_column_calculation",
        {"left_column": "x", "right_column": "y", "operator": "+", "equals_value": 10},
        {"x": [4, 5], "y": [6, 5]},
        {"x": [1, 1], "y": [2, 2]},
    )
    for name, settings in {
        "table_row_count_exact": {"expected_rows": 4},
        "table_row_count_range": {"min_rows": 4},
        "table_not_empty": {},
        "table_dimensions": {"min_rows": 4},
        "table_row_column_ratio": {"min_ratio": 3.0},
        "table_memory_size": {"min_size_bytes": 25},
    }.items():
        add(name, settings, numeric, {"x": []} if name == "table_not_empty" else {"x": [1]})
    add("table_column_count", {"expected_count": 1}, numeric, {"x": [1], "y": [2]})
    add("table_required_columns", {"required_columns": ["x"]}, numeric, {"y": [1]})
    add("table_forbidden_columns", {"forbidden_columns": ["y"]}, numeric, {"x": [1], "y": [2]})
    add(
        "table_schema_match",
        {"expected_schema": {"x": {"kind": "Int64"}}},
        numeric,
        {"x": ["bad"]},
        native_settings={"expected_schema": {"x": pl.Int64}},
    )
    add(
        "table_column_types",
        {"column_types": {"x": [{"kind": "Int64"}]}},
        numeric,
        {"x": ["bad"]},
        native_settings={"column_types": {"x": [pl.Int64]}},
    )
    for name, settings in {
        "cardinality": {"column": "x", "min_cardinality": 1.0},
        "uniqueness_ratio": {"column": "x", "min_uniqueness": 1.0},
        "entropy": {"column": "x", "min_entropy": 1.5},
        "value_frequency": {"column": "x", "max_top_frequency": 0.5},
    }.items():
        add(name, settings, numeric, {"x": [1, 1, 1, 1]})
    add(
        "feature_scale",
        {"columns": ["x"], "max_magnitude": 10.0},
        numeric,
        {"x": [100, 200, 300, 400]},
    )
    add(
        "quantile",
        {"columns": ["x"], "quantile": 0.5, "max_value": 10},
        numeric,
        {"x": [100, 200, 300, 400]},
    )
    add("outlier", {"columns": ["x"]}, {"x": list(range(20))}, {"x": list(range(19)) + [1000]})
    add(
        "zscore_outlier",
        {"columns": ["x"], "threshold": 3.0},
        {"x": list(range(20))},
        {"x": list(range(19)) + [1000]},
    )
    add(
        "format",
        {"columns": ["email"]},
        {"email": ["a@example.test"]},
        {"email": ["invalid", "invalid"]},
    )
    pair = {"x": [1, 2, 3], "y": [4, 5, 6]}
    bad_pair = {"x": [7, 7, 7], "y": [1, 1, 1]}
    add("column_comparison", {"columns": ["x", "y"], "operator": "<"}, pair, bad_pair)
    add(
        "column_chain_comparison",
        {"columns": ["x", "y", "z"], "operators": ["<", "<"]},
        {**pair, "z": [8, 9, 10]},
        {**bad_pair, "z": [8, 9, 10]},
    )
    for name, good, bad in (
        ("column_sum", {"x": [4, 4], "y": [6, 6]}, {"x": [1, 1], "y": [2, 2]}),
        ("column_product", {"x": [2, 2], "y": [5, 5]}, {"x": [1, 1], "y": [2, 2]}),
        ("column_difference", {"x": [12, 12], "y": [2, 2]}, {"x": [1, 1], "y": [2, 2]}),
    ):
        add(name, {"columns": ["x", "y"], "expected_value": 10}, good, bad)
    for name, result in (
        ("column_min", [1, 2, 3]),
        ("column_max", [4, 5, 6]),
        ("column_mean", [2.5, 3.5, 4.5]),
    ):
        add(
            name,
            {"columns": ["x", "y"], "result_column": "z"},
            {**pair, "z": result},
            {**pair, "z": [100, 100, 100]},
        )
    add(
        "column_mutual_exclusivity",
        {"columns": ["x", "y"]},
        {"x": [1, None], "y": [None, 2]},
        {"x": [1, 1], "y": [2, 2]},
    )
    add("column_coexistence", {"columns": ["x", "y"]}, pair, {"x": [1, 1], "y": [None, None]})
    add(
        "column_dependency",
        {"condition_column": "x", "condition_value": 1, "required_columns": ["y"]},
        {"x": [1, 1], "y": [2, 2]},
        {"x": [1, 1], "y": [None, None]},
    )
    add(
        "column_implication",
        {
            "antecedent_column": "x",
            "antecedent_value": 1,
            "consequent_column": "y",
            "consequent_value": 2,
        },
        {"x": [1, 1], "y": [2, 2]},
        {"x": [1, 1], "y": [3, 3]},
    )
    for name, column, good, bad in (
        ("latitude", "lat", 37.5, 95.0),
        ("longitude", "lon", 127.0, 185.0),
    ):
        add(name, {"column": column}, {column: [good, good]}, {column: [bad, bad]})
    add(
        "coordinate",
        {"lat_column": "lat", "lon_column": "lon"},
        {"lat": [37.5], "lon": [127.0]},
        {"lat": [95.0], "lon": [185.0]},
    )
    add(
        "coordinate_not_null_island",
        {"lat_column": "lat", "lon_column": "lon"},
        {"lat": [37.5], "lon": [127.0]},
        {"lat": [0.0], "lon": [0.0]},
    )
    add(
        "geo_bounding_box",
        {
            "lat_column": "lat",
            "lon_column": "lon",
            "min_lat": 30.0,
            "max_lat": 40.0,
            "min_lon": 120.0,
            "max_lon": 130.0,
        },
        {"lat": [37.5], "lon": [127.0]},
        {"lat": [45.0], "lon": [140.0]},
    )
    add(
        "geo_distance_from_point",
        {
            "lat_column": "lat",
            "lon_column": "lon",
            "ref_lat": 37.5,
            "ref_lon": 127.0,
            "max_distance": 10.0,
        },
        {"lat": [37.5], "lon": [127.0]},
        {"lat": [0.0], "lon": [0.0]},
    )
    reference = {"x": [1, 2, 3, 4]}
    for name, options, good, bad in (
        (
            "cross_table_aggregate",
            {"column": "x", "aggregate": "sum"},
            reference,
            {"x": [100, 100, 100, 100]},
        ),
        (
            "optimized_cross_table_aggregate",
            {"source_column": "x", "aggregate": "sum"},
            reference,
            {"x": [100, 100, 100, 100]},
        ),
        ("cross_table_distinct_count", {"column": "x"}, reference, {"x": [1, 1, 1, 1]}),
        ("cross_table_row_count", {}, reference, {"x": [1]}),
        ("cross_table_row_count_factor", {"factor": 2.0}, {"x": list(range(8))}, {"x": [1]}),
        ("referential_integrity", {"column": "x"}, reference, {"x": [99, 99, 99]}),
        (
            "streaming_referential_integrity",
            {"column": "x", "chunk_size": 2},
            reference,
            {"x": [99, 99, 99]},
        ),
    ):
        add(
            name,
            {**options, "reference_data": {"asset_fixture": "reference"}},
            good,
            bad,
            native_settings={**options, "reference_data": pl.DataFrame(reference)},
            frames={"reference": reference},
        )
    for name, options, good, bad in (
        ("table_row_count_compare", {}, reference, {"x": [1]}),
        ("table_schema_compare", {}, reference, {"x": ["invalid"]}),
    ):
        add(
            name,
            {**options, "reference_table": {"asset_fixture": "reference"}},
            good,
            bad,
            native_settings={**options, "reference_table": pl.DataFrame(reference).lazy()},
            frames={"reference": reference},
        )
    specification = importlib.util.spec_from_file_location(
        "truthound_additional_algorithm_cases", Path(__file__).with_name("mcp_additional_cases.py")
    )
    assert specification is not None and specification.loader is not None
    additional = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(additional)
    additional.populate(add)
    specification = importlib.util.spec_from_file_location(
        "truthound_remaining_algorithm_cases", Path(__file__).with_name("mcp_remaining_cases.py")
    )
    assert specification is not None and specification.loader is not None
    additional = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(additional)
    additional.populate(add)
    specification = importlib.util.spec_from_file_location(
        "truthound_algorithm_error_cases", Path(__file__).with_name("mcp_error_cases.py")
    )
    assert specification is not None and specification.loader is not None
    errors = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(errors)
    return errors.attach(cases)


SCENARIOS = scenarios()
