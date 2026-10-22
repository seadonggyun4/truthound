"""Explicit natural runtime faults; no mocked validator or constructor failure."""

from dataclasses import replace
from datetime import time

import polars as pl

NUMERIC_INPUT_ERRORS = frozenset(
    [
        "in_set",
        "not_in_set",
        "multi_column_sum",
        "multi_column_calculation",
        "column_dependency",
        "latitude",
        "longitude",
        "coordinate",
        "coordinate_not_null_island",
        "geo_bounding_box",
        "geo_distance_from_point",
        "cross_table_aggregate",
        "optimized_cross_table_aggregate",
        "referential_integrity",
        "streaming_referential_integrity",
        "percentile_anomaly",
        "timeseries_monotonic",
        "multi_column_variance",
        "mean_drift",
        "variance_drift",
        "quantile_drift",
        "range_drift",
        "ks_test",
        "streaming_ks_test",
        "psi",
        "js_divergence",
        "wasserstein_drift",
        "csi",
        "feature_drift",
        "polygon",
        "multi_polygon",
        "polygon_distance",
        "mahalanobis",
        "optimized_mahalanobis",
        "zscore_multivariate",
        "lof",
        "memory_efficient_lof",
        "dbscan_anomaly",
        "isolation_forest",
        "one_class_svm",
        "online_svm",
        "optimized_geo_distance",
        "geo_country",
        "foreign_key",
        "composite_foreign_key",
        "orphan_record",
        "dangling_reference",
        "multi_table_orphan",
        "trend",
        "trend_break",
        "seasonality",
        "seasonal_decomposition",
    ]
)


MISSING_COLUMNS = {
    **dict.fromkeys(
        [
            "conditional_null",
            "duplicate",
            "primary_key",
            "compound_key",
            "column_pair_unique",
            "duplicate_within_group",
            "unique_within_record",
            "distinct_values_in_set",
            "distinct_values_equal_set",
            "distinct_values_contain_set",
            "distinct_count_between",
            "column_pair",
            "column_pair_in_set",
            "column_pair_not_in_set",
            "multi_column_unique",
            "cardinality",
            "uniqueness_ratio",
            "entropy",
            "value_frequency",
            "cross_table_distinct_count",
            "luhn",
            "isbn",
            "credit_card",
            "iban",
            "swift",
            "vat",
            "korean_business_number",
            "korean_rrn",
            "korean_phone",
            "korean_bank_account",
            "japanese_postal_code",
            "japanese_my_number",
            "chinese_id",
            "chinese_uscc",
            "dateutil_parseable",
            "date_order",
            "recent_data",
            "table_freshness",
            "table_data_recency",
            "grouped_recent_data",
            "column_correlation",
            "column_covariance",
            "most_common_value",
            "chi_square",
            "chi_square_drift",
            "kl_divergence",
            "like_pattern",
            "not_like_pattern",
            "elliptic_envelope",
            "pca_anomaly",
            "information_gain",
            "distribution_shape",
        ],
        "x",
    ),
    **dict.fromkeys(
        [
            "timeseries_gap",
            "timeseries_duplicate",
            "timeseries_order",
            "timeseries_completeness",
            "timeseries_value_completeness",
            "timeseries_interval",
            "timeseries_date_range",
            "table_update_frequency",
        ],
        "t",
    ),
    **dict.fromkeys(
        [
            "hierarchy_circular",
            "optimized_hierarchy_circular",
            "hierarchy_depth",
            "self_referential_fk",
        ],
        "id",
    ),
    "cascade_integrity": "fk",
}

# These native consent/erasure branches cast control values to text during
# execution. Nested Arrow values are legal input data but cannot be cast by
# Polars; preserve that real algorithm failure through the MCP result.
NESTED_CONTROL_ERRORS = {
    "gdpr_right_to_erasure": "deleted",
    "ccpa_do_not_sell": "optout",
    "consent_tracking": "consent",
}

DUPLICATE_TARGET_ERRORS = frozenset(
    {
        "empty_string",
        "whitespace_only",
        "nan",
        "not_nan",
        "nan_ratio",
        "infinity",
        "finite",
        "unique",
        "unique_ratio",
        "distinct_count",
        "mean_between",
        "median_between",
        "min_between",
        "max_between",
        "sum_between",
        "std_between",
        "variance_between",
        "type",
        "email",
        "url",
        "phone",
        "ip_address",
        "ipv6_address",
        "uuid",
        "json_parseable",
        "length",
        "quantile",
        "outlier",
        "zscore_outlier",
        "format",
        "regex",
        "regex_list",
        "not_match_regex",
        "not_match_regex_list",
    }
)


MULTI_COLUMN_CAUGHT_ERRORS = frozenset(
    {
        "column_comparison",
        "column_chain_comparison",
        "column_sum",
        "column_product",
        "column_difference",
        "column_min",
        "column_max",
        "column_mean",
        "column_implication",
        "column_ratio",
        "column_percentage",
    }
)

RUNTIME_SETTING_ERRORS = {
    "between": {"min_value": 2**256},
    "json_schema": {"column": "absent"},
    "date_format": {"format": "%Y%Y"},
    "data_retention": {"retention_days": 10**12},
    "streaming_distinct_count": {"precision": 0},
    "distribution": {"n_bins": 0},
    "approximate_unique_ratio": {"min_ratio": 1e308},
    "table_row_count_compare": {"tolerance_ratio": 1e308},
    # Constructors accept this integer; the native bounded chunk iterator
    # rejects it when converting the slice length, before allocating a chunk.
    **dict.fromkeys(
        (
            "streaming_null",
            "streaming_completeness",
            "streaming_nan",
            "streaming_range",
            "streaming_positive",
            "streaming_outlier",
        ),
        {"chunk_size": 2**64},
    ),
}


def attach(cases):
    """Keep every original polarity, configuration, reference and sample count."""
    assert (MULTI_COLUMN_CAUGHT_ERRORS | RUNTIME_SETTING_ERRORS.keys()) <= {
        row.validator for row in cases
    }
    assert len(MISSING_COLUMNS) == 65
    assert not NUMERIC_INPUT_ERRORS.intersection(MISSING_COLUMNS)
    expected = (
        NUMERIC_INPUT_ERRORS
        | {"geo_distance"}
        | set(MISSING_COLUMNS)
        | set(NESTED_CONTROL_ERRORS)
        | DUPLICATE_TARGET_ERRORS
    )
    observed, result = set(), []
    for row in cases:
        if row.validator in {"date_between", "future_date", "past_date"}:
            assert row.error is None and row.caught_error is None
            frame = pl.DataFrame(row.positive)
            frame = frame.with_columns(pl.Series(frame.columns[0], [time(12, 30)] * frame.height))
            # Native datetime selection admits Time, but its calendar-date
            # comparison cannot compare datetime.time to datetime.date.
            result.append(replace(row, error=frame.to_dict(as_series=False)))
            continue
        if row.validator == "all_columns_unique_within_record":
            assert row.error is None and row.caught_error is None
            frame = pl.DataFrame(row.positive)
            frame = frame.with_columns(pl.Series(frame.columns[0], [[1, 2]] * frame.height))
            result.append(
                replace(
                    row, error=frame.to_dict(as_series=False), error_overrides={"type_safe": False}
                )
            )
            continue
        if row.validator in {"approximate_distinct_count", "default_value"}:
            assert row.error is None and row.caught_error is None
            result.append(
                replace(
                    row,
                    error={"x": list(range(1000))}
                    if row.validator == "approximate_distinct_count"
                    else {"x": [1, 2, 3, 4]},
                    error_overrides={"tolerance": 1e308}
                    if row.validator == "approximate_distinct_count"
                    else {"default_values": [2**256]},
                )
            )
            continue
        if row.validator == "column_mutual_exclusivity":
            assert row.error is None and row.caught_error is None
            result.append(
                replace(
                    row,
                    caught_error=row.positive,
                    error_overrides={"columns": list(row.positive) * 2},
                )
            )
            continue
        if row.validator in RUNTIME_SETTING_ERRORS:
            assert row.error is None and row.caught_error is None
            result.append(
                replace(
                    row, error=row.positive, error_overrides=RUNTIME_SETTING_ERRORS[row.validator]
                )
            )
            continue
        if row.validator in MULTI_COLUMN_CAUGHT_ERRORS:
            assert row.error is None and row.caught_error is None
            frame = pl.DataFrame(row.positive).with_columns(
                pl.lit("invalid_numeric_input").alias("x")
            )
            result.append(replace(row, caught_error=frame.to_dict(as_series=False)))
            continue
        if row.validator not in expected:
            result.append(row)
            continue
        assert row.validator not in observed
        assert row.error is None and row.caught_error is None
        observed.add(row.validator)
        frame = pl.DataFrame(row.positive)
        if row.validator in DUPLICATE_TARGET_ERRORS:
            # Native constructors accept repeated targets; actual aggregation
            # then fails on duplicate projection aliases. Do not replace its
            # execution issue with adapter rejection or a successful empty run.
            result.append(
                replace(row, error=row.positive, error_overrides={"columns": frame.columns * 2})
            )
            continue
        if row.validator in NESTED_CONTROL_ERRORS:
            column = NESTED_CONTROL_ERRORS[row.validator]
            frame = frame.with_columns(pl.Series(column, [[1, 2]] * frame.height))
        elif row.validator in MISSING_COLUMNS:
            frame = frame.rename({MISSING_COLUMNS[row.validator]: "__absent_contract_input__"})
        elif row.validator == "geo_distance":
            frame = frame.head(0)
        else:
            numeric = [name for name, dtype in frame.schema.items() if dtype.is_numeric()]
            assert numeric, row.validator
            frame = frame.with_columns(
                [pl.lit("invalid_numeric_input").alias(name) for name in numeric]
            )
        result.append(replace(row, error=frame.to_dict(as_series=False)))
    assert observed == expected and len(observed) == 155
    return result
