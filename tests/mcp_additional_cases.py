"""Independently specified synthetic native algorithm inputs, not runtime resources."""

from datetime import UTC, date, datetime, timedelta

import polars as pl


def populate(add):
    # Synthetic check digits are computed here, independently of Core helpers.
    rrn_prefix = "800101100000"
    rrn = rrn_prefix + str(
        (
            11
            - sum(
                int(d) * w
                for d, w in zip(rrn_prefix, [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5], strict=True)
            )
            % 11
        )
        % 10
    )
    biz_prefix = "123456789"
    biz_total = (
        sum(int(d) * w for d, w in zip(biz_prefix, [1, 3, 7, 1, 3, 7, 1, 3, 5], strict=True))
        + int(biz_prefix[-1]) * 5 // 10
    )
    uscc_prefix = "91110000123456789"
    alphabet = "0123456789ABCDEFGHJKLMNPQRTUWXY"
    uscc = (
        uscc_prefix
        + alphabet[
            -sum(
                alphabet.index(d) * w
                for d, w in zip(
                    uscc_prefix,
                    [1, 3, 9, 27, 19, 26, 16, 17, 20, 29, 25, 13, 8, 24, 10, 30, 28],
                    strict=True,
                )
            )
            % 31
        ]
    )
    for name, value in {
        "luhn": "79927398713",
        "isbn": "9780306406157",
        "credit_card": "4111111111111111",  # published payment sandbox number
        "iban": "GB82WEST12345698765432",
        "swift": "DEUTDEFF",
        "vat": "DE123456789",
        "korean_business_number": biz_prefix + str((-biz_total) % 10),
        "korean_rrn": rrn,
        "korean_phone": "01012345678",
        "korean_bank_account": "1234567890123",
        "japanese_postal_code": "1000001",
        "japanese_my_number": "000000000000",
        "chinese_id": "11010519491231002X",  # standard algorithm example
        "chinese_uscc": uscc,
    }.items():
        add(name, {"column": "x"}, {"x": [value, value]}, {"x": ["invalid", "invalid"]})
    add("range", {"columns": ["age"]}, {"age": [18, 25]}, {"age": [-1, 200]})
    for name in (
        "gdpr_compliance",
        "ccpa_compliance",
        "global_privacy",
        "lgpd_compliance",
        "pipeda_compliance",
        "appi_compliance",
    ):
        add(
            name,
            {},
            {"x": ["ordinary", "ordinary"]},
            {"email": ["synthetic@example.com", "synthetic@example.com"]},
        )
    for name, good, bad in (
        ("gdpr_special_category", {"x": ["ordinary"]}, {"health_diagnosis": ["diabetes"]}),
        ("ccpa_sensitive_info", {"x": ["ordinary"]}, {"ssn": ["123-45-6789"]}),
    ):
        add(name, {}, good, bad)
    add(
        "gdpr_data_minimization",
        {"max_pii_columns": 0},
        {"x": ["ordinary"]},
        {"email": ["synthetic@example.com"]},
    )
    add(
        "gdpr_right_to_erasure",
        {"deletion_flag_column": "deleted"},
        {"deleted": ["no"], "x": ["ordinary"]},
        {"x": ["ordinary"]},
    )
    add(
        "ccpa_consumer_rights",
        {"consumer_id_column": "consumer_id"},
        {
            "consumer_id": ["synthetic"],
            "deletion_request": [None],
            "access_request": [None],
            "optout_sale": ["false"],
            "optout_share": ["false"],
            "limit_sensitive": ["false"],
        },
        {"x": ["ordinary"]},
    )
    add(
        "ccpa_do_not_sell",
        {"optout_column": "optout"},
        {"optout": ["false"], "x": ["ordinary"]},
        {"x": ["ordinary"]},
    )
    add(
        "consent_tracking",
        {"consent_column": "consent", "pii_columns": ["email"]},
        {"consent": ["yes"], "email": ["synthetic@example.com"]},
        {"consent": ["no"], "email": ["synthetic@example.com"]},
    )
    add(
        "data_retention",
        {"date_column": "created", "retention_days": 365, "pii_columns": ["email"]},
        {"created": [datetime(2200, 1, 1)], "email": ["synthetic@example.com"]},
        {"created": [datetime(2000, 1, 1)], "email": ["synthetic@example.com"]},
    )
    regular = [float(i) for i in range(100)]
    outliers = regular[:90] + [10000.0] * 10
    for name in ("iqr_anomaly", "mad_anomaly", "tukey_fences", "grubbs_test"):
        bad = regular[:98] + [10000.0, 20000.0] if name == "grubbs_test" else outliers
        add(
            name,
            {"column": "x", "max_anomaly_ratio": 0.01},
            {"x": regular},
            {"x": bad},
            error={"x": ["not-numeric"] * 100},
        )
    add(
        "percentile_anomaly",
        {
            "column": "x",
            "max_anomaly_ratio": 0.05,
            "lower_percentile": 5.0,
            "upper_percentile": 95.0,
        },
        {"x": [1.0] * 100},
        {"x": regular},
    )
    query_good = {"x": [1, 2], "g": ["a", "a"]}
    query_bad = {"x": [-1, -2], "g": ["a", "a"]}
    for name, options, positive, negative in (
        ("query_returns_rows", {"query": "SELECT * FROM data WHERE x > 0"}, query_good, query_bad),
        (
            "query_returns_no_rows",
            {"query": "SELECT * FROM data WHERE x < 0"},
            query_good,
            query_bad,
        ),
        (
            "query_row_count",
            {"query": "SELECT * FROM data WHERE x > 0", "expected_count": 2},
            query_good,
            query_bad,
        ),
        (
            "query_row_count_ratio",
            {"query": "SELECT * FROM data WHERE x > 0", "min_ratio": 1.0},
            query_good,
            query_bad,
        ),
        (
            "query_returns_single_value",
            {"query": "SELECT SUM(x) AS total FROM data", "expected_value": 3},
            query_good,
            query_bad,
        ),
        (
            "query_aggregate",
            {"query": "SELECT SUM(x) AS total FROM data", "column": "total", "expected_value": 3},
            query_good,
            query_bad,
        ),
        (
            "query_column_unique",
            {"query": "SELECT x FROM data", "column": "x"},
            {"x": [1, 2]},
            {"x": [1, 1]},
        ),
        (
            "query_column_not_null",
            {"query": "SELECT x FROM data", "column": "x"},
            {"x": [1, 2]},
            {"x": [1, None]},
        ),
        (
            "query_column_values",
            {"query": "SELECT x FROM data", "column": "x", "expected_values": [1, 2]},
            query_good,
            query_bad,
        ),
        (
            "query_group_aggregate",
            {
                "query": "SELECT g, SUM(x) AS total FROM data GROUP BY g",
                "group_column": "g",
                "aggregate_column": "total",
                "min_value": 0,
            },
            query_good,
            query_bad,
        ),
        (
            "query_row_count_compare",
            {"query": "SELECT * FROM data WHERE x > 0", "compare_query": "SELECT * FROM data"},
            query_good,
            query_bad,
        ),
        (
            "query_aggregate_compare",
            {
                "query": "SELECT SUM(x) AS total FROM data",
                "compare_query": "SELECT SUM(ABS(x)) AS total FROM data",
                "column": "total",
            },
            query_good,
            query_bad,
        ),
    ):
        add(name, options, positive, negative, trusted=True, caught_error={"absent": [1, 2]})
    expected = {"x": [1, 2]}
    add(
        "query_result_matches",
        {
            "query": "SELECT x FROM data",
            "expected_df": {"asset_fixture": "expected"},
            "check_order": True,
        },
        expected,
        {"x": [2, 1]},
        native_settings={
            "query": "SELECT x FROM data",
            "expected_df": pl.DataFrame(expected),
            "check_order": True,
        },
        frames={"expected": expected},
        trusted=True,
        caught_error={"absent": [1, 2]},
    )
    base = datetime(2024, 1, 1)
    times = [base + timedelta(hours=i) for i in range(24)]
    dates = {"x": [date(2024, 1, 1), date(2024, 1, 2)]}
    add(
        "date_between",
        {"columns": ["x"], "min_date": "2024-01-01", "max_date": "2024-12-31"},
        dates,
        {"x": [date(2020, 1, 1)]},
    )
    add("future_date", {"columns": ["x"]}, dates, {"x": [date(2200, 1, 1)]})
    add("past_date", {"columns": ["x"], "min_date": "2023-01-01"}, dates, {"x": [date(2020, 1, 1)]})
    add(
        "date_format",
        {"columns": ["x"], "format": "%Y-%m-%d"},
        {"x": ["2024-01-01"]},
        {"x": ["not-a-date", "not-a-date"]},
    )
    add(
        "dateutil_parseable",
        {"column": "x", "fuzzy": False},
        {"x": ["2024-01-01"]},
        {"x": ["invalid date", "invalid date"]},
    )
    add(
        "date_order",
        {"first_column": "x", "second_column": "y"},
        {"x": [base], "y": [base + timedelta(days=1)]},
        {"x": [base + timedelta(days=2)], "y": [base]},
    )
    add(
        "timezone",
        {"columns": ["x"], "expected_timezone": "UTC"},
        {"x": [base.replace(tzinfo=UTC)]},
        {"x": [base]},
    )
    reference = base + timedelta(days=1)
    for name, options in (
        ("recent_data", {"column": "x", "max_age_hours": 2}),
        ("table_freshness", {"timestamp_column": "x", "max_age_hours": 2}),
        ("table_data_recency", {"timestamp_column": "x", "max_age_days": 1}),
    ):
        add(
            name,
            {**options, "reference_time": {"value": reference.isoformat()}},
            {"x": [reference - timedelta(minutes=30)]},
            {"x": [base - timedelta(days=5)]},
            native_settings={**options, "reference_time": reference},
        )
    options = {"datetime_column": "x", "group_column": "g", "max_age_hours": 2}
    add(
        "grouped_recent_data",
        {**options, "reference_time": {"value": reference.isoformat()}},
        {"x": [reference - timedelta(minutes=30)], "g": ["a"]},
        {"x": [base - timedelta(days=5)], "g": ["a"]},
        native_settings={**options, "reference_time": reference},
    )
    add(
        "date_part_coverage",
        {"column": "x", "date_part": "hour", "min_coverage": 1.0},
        {"x": times},
        {"x": [times[0], times[-1]]},
        error={"x": ["not-datetime"]},
    )
    for name, settings, positive, negative in (
        (
            "timeseries_gap",
            {"frequency": "1h"},
            times,
            [times[0], times[1], times[8], times[9]],
        ),
        ("timeseries_duplicate", {}, times, times[:3] + [times[2]]),
        ("timeseries_order", {}, times, list(reversed(times))),
        (
            "timeseries_completeness",
            {"frequency": "1h"},
            times,
            [times[0], times[1], times[-1]],
        ),
    ):
        add(name, {"timestamp_column": "t", **settings}, {"t": positive}, {"t": negative})
    add(
        "timeseries_value_completeness",
        {"timestamp_column": "t", "value_column": "x"},
        {"t": times, "x": list(range(24))},
        {"t": times, "x": [None] * 12 + list(range(12))},
    )
    add(
        "timeseries_monotonic",
        {"timestamp_column": "t", "value_column": "x"},
        {"t": times, "x": list(range(24))},
        {"t": times, "x": list(reversed(range(24)))},
    )
    interval = {
        "timestamp_column": "t",
        "min_interval": timedelta(minutes=30),
        "max_interval": timedelta(hours=2),
    }
    add(
        "timeseries_interval",
        {
            **interval,
            "min_interval": {"value": 1800_000_000},
            "max_interval": {"value": 7200_000_000},
        },
        {"t": times},
        {"t": [base + timedelta(hours=4 * i) for i in range(24)]},
        native_settings=interval,
    )
    bounds = {
        "timestamp_column": "t",
        "required_start": base,
        "required_end": times[-1],
        "frequency": "hourly",
    }
    add(
        "timeseries_date_range",
        {
            **bounds,
            "required_start": {"value": base.isoformat()},
            "required_end": {"value": times[-1].isoformat()},
        },
        {"t": times},
        {"t": times[8:12]},
        native_settings=bounds,
    )
    add(
        "table_update_frequency",
        {"timestamp_column": "t", "expected_frequency": "hourly"},
        {"t": times},
        {"t": [times[0], times[1], times[-1]]},
    )
    pair = {"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]}
    inverse = {"x": [1.0, 2.0, 3.0, 4.0], "y": [8.0, 6.0, 4.0, 2.0]}
    add(
        "column_correlation",
        {"column_a": "x", "column_b": "y", "min_correlation": 0.8},
        pair,
        inverse,
    )
    add("column_covariance", {"column_a": "x", "column_b": "y", "min_value": 0.0}, pair, inverse)
    add(
        "multi_column_variance",
        {"columns": ["x", "y"], "max_variance": 20.0},
        pair,
        {"x": [1.0, 100.0, 300.0, 900.0], "y": [1.0, 100.0, 300.0, 900.0]},
    )
    add(
        "most_common_value",
        {"column": "x", "expected_values": [1]},
        {"x": [1, 1, 1, 2]},
        {"x": [2, 2, 2, 1]},
    )
    add(
        "column_ratio",
        {"columns": ["x", "y"], "expected_value": 2.0},
        {"x": [4.0, 8.0], "y": [2.0, 4.0]},
        {"x": [4.0, 4.0], "y": [4.0, 4.0]},
    )
    add(
        "column_percentage",
        {"columns": ["x", "y"], "expected_value": 50.0},
        {"x": [1.0, 2.0], "y": [2.0, 4.0]},
        {"x": [1.0, 1.0], "y": [4.0, 4.0]},
    )
    baseline = {"x": [float(i + 1) for i in range(100)]}
    shifted = {"x": [float(i + 301) for i in range(100)]}
    spread = {"x": [float((i + 1) * 10) for i in range(100)]}
    for name, options, bad in (
        ("mean_drift", {"threshold_abs": 10.0}, shifted),
        ("variance_drift", {"threshold_pct": 20.0}, spread),
        ("quantile_drift", {}, shifted),
        ("range_drift", {}, shifted),
        ("ks_test", {}, shifted),
        ("streaming_ks_test", {"chunk_size": 20}, shifted),
        ("psi", {}, shifted),
        ("js_divergence", {}, shifted),
        ("wasserstein_drift", {"threshold": 10.0}, shifted),
        ("csi", {}, shifted),
    ):
        add(
            name,
            {"column": "x", **options, "reference_data": {"asset_fixture": "reference"}},
            baseline,
            bad,
            native_settings={"column": "x", **options, "reference_data": pl.DataFrame(baseline)},
            frames={"reference": baseline},
        )
    for name, options in (
        ("distribution", {"threshold": 0.1}),
        ("feature_drift", {"columns": ["x"]}),
    ):
        add(
            name,
            {**options, "reference_data": {"asset_fixture": "reference"}},
            baseline,
            shifted,
            native_settings={**options, "reference_data": pl.DataFrame(baseline)},
            frames={"reference": baseline},
        )
    categorical = {"x": ["a"] * 60 + ["b"] * 40}
    different = {"x": ["a"] * 10 + ["b"] * 90}
    for name, options in (
        ("chi_square", {}),
        ("chi_square_drift", {}),
        ("kl_divergence", {"max_divergence": 0.1}),
    ):
        add(
            name,
            {"column": "x", **options, "reference_data": {"asset_fixture": "reference"}},
            categorical,
            different,
            native_settings={"column": "x", **options, "reference_data": pl.DataFrame(categorical)},
            frames={"reference": categorical},
        )
    for name, options, good, bad in (
        ("regex", {"pattern": "^A+$"}, "AAA", "invalid"),
        ("regex_list", {"patterns": ["^A+$", "^B+$"]}, "AAA", "invalid"),
        ("not_match_regex", {"pattern": "^A+$"}, "valid", "AAA"),
        ("not_match_regex_list", {"patterns": ["^A+$", "^B+$"]}, "valid", "AAA"),
        ("like_pattern", {"pattern": "A%", "column": "x"}, "ABC", "invalid"),
        ("not_like_pattern", {"pattern": "A%", "column": "x"}, "valid", "ABC"),
    ):
        add(
            name,
            {"columns": ["x"], **options},
            {"x": [good, good]},
            {"x": [bad, bad]},
            trusted=True,
        )
    expr = {
        "op": "gt",
        "left": {"op": "column", "name": "x"},
        "right": {"op": "literal", "value": 0},
    }
    # Expression validators interpret filter_expr as the predicate for valid rows.
    for name in ("custom_expression", "expression_base"):
        add(
            name,
            {"filter_expr": expr},
            {"x": [1, 2]},
            {"x": [-1, -1]},
            native_settings={"filter_expr": pl.col("x") > 0},
            caught_error={"absent": [1, 2]},
        )
    add(
        "conditional_expression",
        {
            "condition": {
                "op": "gt",
                "left": {"op": "column", "name": "x"},
                "right": {"op": "literal", "value": 0},
            },
            "then_expr": {
                "op": "gt",
                "left": {"op": "column", "name": "y"},
                "right": {"op": "literal", "value": 0},
            },
        },
        {"x": [1, 1], "y": [1, 1]},
        {"x": [1, 1], "y": [-1, -1]},
        native_settings={"condition": pl.col("x") > 0, "then_expr": pl.col("y") > 0},
        error={"absent": [1, 2]},
    )
    add(
        "multi_condition",
        {"conditions": [[expr, "x positive"]]},
        {"x": [1, 2]},
        {"x": [-1, -1]},
        native_settings={"conditions": [(pl.col("x") > 0, "x positive")]},
        error={"absent": [1, 2]},
    )
