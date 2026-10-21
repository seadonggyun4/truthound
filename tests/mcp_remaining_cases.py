"""Synthetic relational, geographic, temporal and ML algorithm specimens."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl


def populate(add):
    condition = {
        "op": "gt",
        "left": {"op": "column", "name": "x"},
        "right": {"op": "literal", "value": 0},
    }
    consequence = {
        "op": "gt",
        "left": {"op": "column", "name": "y"},
        "right": {"op": "literal", "value": 0},
    }
    add(
        "column_consistency",
        {
            "rules": [
                {
                    "when": condition,
                    "then": consequence,
                    "description": "positive x requires positive y",
                }
            ]
        },
        {"x": [1, 2], "y": [1, 2]},
        {"x": [1, 2], "y": [-1, -2]},
        native_settings={
            "rules": [
                {
                    "when": pl.col("x") > 0,
                    "then": pl.col("y") > 0,
                    "description": "positive x requires positive y",
                }
            ]
        },
        error={"x": [1, 2], "absent": [1, 2]},
    )
    wkt = "POLYGON((0 0,0 10,10 10,10 0,0 0))"
    for name in ("polygon", "multi_polygon", "polygon_distance"):
        options = {"lat_column": "lat", "lon_column": "lon"}
        if name == "multi_polygon":
            options["polygons"] = [{"wkt": wkt, "name": "synthetic-square"}]
        else:
            options["polygon_wkt"] = wkt
        if name == "polygon_distance":
            options["max_distance_km"] = 10.0
        add(name, options, {"lat": [5.0], "lon": [5.0]}, {"lat": [20.0], "lon": [20.0]})
    points = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]] * 25)
    rng = np.random.default_rng(4721)
    shifted = np.vstack([rng.normal(0, 0.1, (90, 2)), rng.normal(20, 1, (10, 2))])
    for name in (
        "mahalanobis",
        "optimized_mahalanobis",
        "elliptic_envelope",
        "pca_anomaly",
        "zscore_multivariate",
        "lof",
        "memory_efficient_lof",
        "dbscan_anomaly",
        "isolation_forest",
        "one_class_svm",
        "online_svm",
    ):
        good, bad = points, shifted
        options = {"columns": ["x", "y"], "max_anomaly_ratio": 0.01}
        if name == "dbscan_anomaly":
            options["eps"] = 0.5
        if name == "isolation_forest":
            options["contamination"] = 0.05
        if name == "one_class_svm":
            good, bad = shifted, points
            options["max_anomaly_ratio"] = 0.1
        if name == "online_svm":
            options["n_components"] = 4
        add(
            name,
            options,
            {"x": good[:, 0].tolist(), "y": good[:, 1].tolist()},
            {"x": bad[:, 0].tolist(), "y": bad[:, 1].tolist()},
        )
    for name in ("geo_distance", "optimized_geo_distance"):
        add(
            name,
            {
                "lat1_column": "a",
                "lon1_column": "b",
                "lat2_column": "c",
                "lon2_column": "d",
                "max_distance": 10.0,
            },
            {"a": [0.0], "b": [0.0], "c": [0.0], "d": [0.01]},
            {"a": [0.0], "b": [0.0], "c": [40.0], "d": [40.0]},
        )
    add(
        "geo_country",
        {"lat_column": "lat", "lon_column": "lon", "country": "korea"},
        {"lat": [37.5], "lon": [127.0]},
        {"lat": [0.0], "lon": [0.0]},
    )
    parent = {"id": [1, 2]}
    for name in ("foreign_key", "composite_foreign_key", "orphan_record", "dangling_reference"):
        options = {
            "child_table": "child",
            "child_columns": ["fk"],
            "parent_table": "parent",
            "parent_columns": ["id"],
        }
        good, bad = (
            ({"fk": [1, 2]}, {"fk": [1, 9]})
            if name != "dangling_reference"
            else ({"fk": [1, 2]}, {"fk": [1, 1]})
        )
        if name == "dangling_reference":
            options["min_expected_children"] = 1
            add(
                name,
                {
                    **options,
                    "tables": {
                        "parent": {"asset_fixture": "parent"},
                        "child": {"asset_fixture": "current"},
                    },
                },
                good,
                bad,
                native_settings=lambda frame, options=options: {
                    **options,
                    "tables": {"parent": pl.DataFrame(parent).lazy(), "child": frame.lazy()},
                },
                frames={"parent": parent},
            )
            continue
        add(
            name,
            {**options, "tables": {"parent": {"asset_fixture": "parent"}}},
            good,
            bad,
            native_settings={**options, "tables": {"parent": pl.DataFrame(parent).lazy()}},
            frames={"parent": parent},
        )
    for name in (
        "hierarchy_circular",
        "optimized_hierarchy_circular",
        "hierarchy_depth",
        "self_referential_fk",
    ):
        options = {"table": "nodes", "max_depth": 2 if name == "hierarchy_depth" else 10}
        if name == "self_referential_fk":
            options.update(pk_column="id", fk_column="parent")
        else:
            options.update(id_column="id", parent_column="parent")
        good = {"id": [1, 2, 3], "parent": [None, 1, 1]}
        if name == "hierarchy_depth":
            bad = {"id": [1, 2, 3, 4, 5], "parent": [None, 1, 2, 3, 4]}
        elif name == "self_referential_fk":
            bad = {"id": [1, 2, 3], "parent": [None, 1, 9]}
        else:
            bad = {"id": [1, 2, 3], "parent": [2, 3, 1]}
        add(name, options, good, bad)
    from truthound.validators.referential.base import ForeignKeyRelation
    from truthound.validators.referential.cascade import CascadeAction, CascadeRule

    relation = {
        "child_table": "child",
        "child_columns": ["fk"],
        "parent_table": "parent",
        "parent_columns": ["id"],
    }
    add(
        "cascade_integrity",
        {
            "cascade_rules": [{"relation": relation, "on_delete": "RESTRICT"}],
            "tables": {
                "parent": {"asset_fixture": "parent"},
                "child": {"asset_fixture": "current"},
            },
        },
        {"fk": [1, 2]},
        {"fk": [1, 9]},
        native_settings=lambda frame: {
            "cascade_rules": [
                CascadeRule(ForeignKeyRelation(**relation), on_delete=CascadeAction.RESTRICT)
            ],
            "tables": {"parent": pl.DataFrame(parent).lazy(), "child": frame.lazy()},
        },
        frames={"parent": parent},
    )
    edge = {
        "child_table": "b",
        "child_columns": ["fk"],
        "parent_table": "a",
        "parent_columns": ["id"],
    }
    reverse = {**edge, "child_table": "a", "parent_table": "b"}
    acyclic = {"relations": [edge]}
    cyclic = {"relations": [edge, reverse]}
    add(
        "circular_reference",
        acyclic,
        {"x": [1]},
        {"x": [1]},
        native_settings={"relations": [ForeignKeyRelation(**edge)]},
        negative_settings=cyclic,
        native_negative_settings={
            "relations": [ForeignKeyRelation(**edge), ForeignKeyRelation(**reverse)]
        },
    )
    long_path = [
        edge,
        {**edge, "child_table": "c", "parent_table": "b"},
        {**edge, "child_table": "d", "parent_table": "c"},
    ]
    add(
        "cascade_depth",
        {**acyclic, "max_depth": 2},
        {"x": [1]},
        {"x": [1]},
        native_settings={"relations": [ForeignKeyRelation(**edge)], "max_depth": 2},
        negative_settings={"relations": long_path, "max_depth": 2},
        native_negative_settings={
            "relations": [ForeignKeyRelation(**item) for item in long_path],
            "max_depth": 2,
        },
    )
    add(
        "multi_table_orphan",
        {
            "relations": [relation],
            "tables": {
                "parent": {"asset_fixture": "parent"},
                "child": {"asset_fixture": "current"},
            },
        },
        {"fk": [1, 2]},
        {"fk": [1, 9]},
        native_settings=lambda frame: {
            "relations": [ForeignKeyRelation(**relation)],
            "tables": {"parent": pl.DataFrame(parent).lazy(), "child": frame.lazy()},
        },
        frames={"parent": parent},
    )
    grid = {"x": [float(i) for i in range(100)], "y": [float(i % 2) for i in range(100)]}
    correlated = {"x": grid["x"], "y": grid["x"]}
    add("feature_correlation_matrix", {"columns": ["x", "y"]}, grid, correlated)
    add(
        "target_leakage",
        {"columns": ["x"], "target_column": "y", "check_mutual_info": False},
        grid,
        correlated,
    )
    add(
        "feature_null_impact",
        {"columns": ["x"]},
        {"x": list(range(100))},
        {"x": [None] * 90 + list(range(10))},
    )
    add(
        "information_gain",
        {"column": "x", "target_column": "y", "min_information_gain": 0.5},
        {"x": ["a", "a", "b", "b"] * 25, "y": ["a", "a", "b", "b"] * 25},
        {"x": ["a", "b", "a", "b"] * 25, "y": ["a", "a", "b", "b"] * 25},
    )
    add(
        "distribution_shape",
        {"column": "x", "expected_shape": "uniform"},
        {"x": ["a", "b"] * 50},
        {"x": ["a"] * 99 + ["b"]},
    )
    times = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(100)]
    add(
        "trend",
        {"timestamp_column": "t", "value_column": "x", "expected_direction": "increasing"},
        {"t": times, "x": list(range(100))},
        {"t": times, "x": list(reversed(range(100)))},
    )
    add(
        "trend_break",
        {"timestamp_column": "t", "value_column": "x", "max_breaks": 0},
        {"t": times, "x": [1.0] * 100},
        {"t": times, "x": [1.0] * 50 + [100.0] * 50},
    )
    rng = np.random.default_rng(4721)
    seasonal = [float(np.sin(i * 2 * np.pi / 10)) for i in range(100)]
    noise = rng.standard_normal(100).tolist()
    add(
        "seasonality",
        {"timestamp_column": "t", "value_column": "x", "expected_period": 10},
        {"t": times, "x": seasonal},
        {"t": times, "x": noise},
    )
    add(
        "seasonal_decomposition",
        {"timestamp_column": "t", "value_column": "x", "period": 10},
        {"t": times, "x": seasonal},
        {"t": times, "x": noise},
    )
