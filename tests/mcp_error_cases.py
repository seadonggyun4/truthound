"""Explicit natural runtime faults; no mocked validator or constructor failure."""

from dataclasses import replace

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


def attach(cases):
    """Keep every original polarity, configuration, reference and sample count."""
    assert len(MISSING_COLUMNS) == 65
    assert not NUMERIC_INPUT_ERRORS.intersection(MISSING_COLUMNS)
    expected = NUMERIC_INPUT_ERRORS | {"geo_distance"} | set(MISSING_COLUMNS)
    observed, result = set(), []
    for row in cases:
        if row.validator not in expected:
            result.append(row)
            continue
        assert row.validator not in observed
        assert row.error is None and row.caught_error is None
        observed.add(row.validator)
        frame = pl.DataFrame(row.positive)
        if row.validator in MISSING_COLUMNS:
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
    assert observed == expected and len(observed) == 118
    return result
