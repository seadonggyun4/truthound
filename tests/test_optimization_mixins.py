"""Tests for optimization mixins.

Tests for GraphTraversalMixin, BatchCovarianceMixin, VectorizedGeoMixin,
and LazyAggregationMixin.
"""

import numpy as np
import polars as pl
import pytest

# Graph traversal imports
from truthound.validators.optimization import (
    GraphTraversalMixin,
    TarjanSCC,
    IterativeDFS,
    TopologicalSort,
    CycleInfo,
)

# Covariance imports
from truthound.validators.optimization import (
    BatchCovarianceMixin,
    IncrementalCovariance,
    WoodburyCovariance,
    RobustCovarianceEstimator,
)

# Geo imports
from truthound.validators.optimization import (
    VectorizedGeoMixin,
    SpatialIndexMixin,
    BoundingBox,
    DistanceUnit,
)

# Aggregation imports
from truthound.validators.optimization import (
    LazyAggregationMixin,
    AggregationResult,
    AggregationExpressionBuilder,
)


# =============================================================================
# Graph Traversal Tests
# =============================================================================


class TestIterativeDFS:
    """Tests for IterativeDFS."""

    def test_simple_traversal(self):
        """Test basic DFS traversal."""
        adjacency = {
            "A": ["B", "C"],
            "B": ["D"],
            "C": ["D"],
            "D": [],
        }
        dfs = IterativeDFS(adjacency)
        visited = list(dfs.traverse("A"))

        assert "A" in visited
        assert "B" in visited
        assert "C" in visited
        assert "D" in visited
        assert len(visited) == 4

    def test_preorder_traversal(self):
        """Test preorder traversal order."""
        adjacency = {
            1: [2, 3],
            2: [4],
            3: [],
            4: [],
        }
        dfs = IterativeDFS(adjacency)
        order = list(dfs.traverse(1, order="preorder"))

        # 1 should be first
        assert order[0] == 1
        # 2 and 3 should come after 1
        assert order.index(2) > order.index(1)
        assert order.index(3) > order.index(1)

    def test_find_path(self):
        """Test path finding."""
        adjacency = {
            "A": ["B", "C"],
            "B": ["D", "E"],
            "C": ["F"],
            "D": [],
            "E": ["F"],
            "F": [],
        }
        dfs = IterativeDFS(adjacency)

        path = dfs.find_path("A", "F")
        assert path is not None
        assert path[0] == "A"
        assert path[-1] == "F"

    def test_no_path(self):
        """Test when no path exists."""
        adjacency = {
            "A": ["B"],
            "B": [],
            "C": ["D"],
            "D": [],
        }
        dfs = IterativeDFS(adjacency)

        path = dfs.find_path("A", "D")
        assert path is None

    def test_compute_depths(self):
        """Test depth computation."""
        adjacency = {
            "root": ["child1", "child2"],
            "child1": ["grandchild1"],
            "child2": [],
            "grandchild1": [],
        }
        dfs = IterativeDFS(adjacency)

        depths = dfs.compute_depths(roots=["root"])
        assert depths["root"] == 0
        assert depths["child1"] == 1
        assert depths["child2"] == 1
        assert depths["grandchild1"] == 2


class TestTarjanSCC:
    """Tests for Tarjan's SCC algorithm."""

    def test_find_sccs_no_cycles(self):
        """Test with DAG (no cycles)."""
        adjacency = {
            1: [2],
            2: [3],
            3: [],
        }
        tarjan = TarjanSCC(adjacency)
        sccs = tarjan.find_sccs()

        # Each node is its own SCC
        assert len(sccs) == 3
        for scc in sccs:
            assert len(scc) == 1

    def test_find_sccs_with_cycle(self):
        """Test with cycle."""
        adjacency = {
            1: [2],
            2: [3],
            3: [1],  # Cycle: 1 -> 2 -> 3 -> 1
        }
        tarjan = TarjanSCC(adjacency)
        sccs = tarjan.find_sccs()

        # One SCC with all nodes
        assert len(sccs) == 1
        assert len(sccs[0]) == 3

    def test_find_cycles(self):
        """Test cycle detection."""
        adjacency = {
            "A": ["B"],
            "B": ["C"],
            "C": ["A"],  # Cycle
        }
        tarjan = TarjanSCC(adjacency)
        cycles = tarjan.find_cycles()

        assert len(cycles) == 1
        assert cycles[0].length == 3

    def test_self_loop_detection(self):
        """Test self-loop detection."""
        adjacency = {
            "A": ["A"],  # Self-loop
            "B": [],
        }
        tarjan = TarjanSCC(adjacency)
        cycles = tarjan.find_cycles()

        assert len(cycles) == 1
        assert cycles[0].is_self_loop


class TestTopologicalSort:
    """Tests for topological sorting."""

    def test_simple_sort(self):
        """Test basic topological sort."""
        adjacency = {
            "A": ["B", "C"],
            "B": ["D"],
            "C": ["D"],
            "D": [],
        }
        sorter = TopologicalSort(adjacency)
        order = sorter.sort()

        # A must come before B, C, D
        assert order.index("A") < order.index("B")
        assert order.index("A") < order.index("C")
        assert order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")

    def test_has_cycles(self):
        """Test cycle detection."""
        adjacency = {
            1: [2],
            2: [3],
            3: [1],  # Cycle
        }
        sorter = TopologicalSort(adjacency)

        assert sorter.has_cycles() is True

    def test_no_cycles(self):
        """Test when no cycles exist."""
        adjacency = {
            1: [2],
            2: [3],
            3: [],
        }
        sorter = TopologicalSort(adjacency)

        assert sorter.has_cycles() is False


class TestGraphTraversalMixin:
    """Tests for GraphTraversalMixin."""

    def test_build_adjacency_from_dataframe(self):
        """Test building adjacency list from DataFrame."""

        class TestClass(GraphTraversalMixin):
            pass

        obj = TestClass()
        df = pl.DataFrame({
            "id": [1, 2, 3, 4],
            "parent_id": [None, 1, 1, 2],
        })

        adjacency = obj.build_adjacency_list(df, "id", "parent_id")

        # Verify parent-child relationships (None parents are skipped)
        assert 2 in adjacency.get(1, [])
        assert 3 in adjacency.get(1, [])
        assert 4 in adjacency.get(2, [])

    def test_find_hierarchy_cycles(self):
        """Test hierarchy cycle detection."""

        class TestClass(GraphTraversalMixin):
            pass

        obj = TestClass()
        child_to_parent = {
            1: 2,
            2: 3,
            3: 1,  # Cycle
        }

        cycles = obj.find_hierarchy_cycles(child_to_parent)
        assert len(cycles) >= 1


# =============================================================================
# Covariance Tests
# =============================================================================


class TestIncrementalCovariance:
    """Tests for IncrementalCovariance."""

    def test_single_sample_update(self):
        """Test updating with single samples."""
        cov = IncrementalCovariance(n_features=3)

        samples = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ])

        for sample in samples:
            cov.update(sample)

        assert cov.n_samples == 3
        np.testing.assert_array_almost_equal(
            cov.mean,
            samples.mean(axis=0)
        )

    def test_batch_update(self):
        """Test batch update."""
        cov = IncrementalCovariance(n_features=2)

        batch = np.random.randn(100, 2)
        cov.update_batch(batch)

        assert cov.n_samples == 100
        np.testing.assert_array_almost_equal(
            cov.mean,
            batch.mean(axis=0),
            decimal=10
        )
        np.testing.assert_array_almost_equal(
            cov.covariance,
            np.cov(batch, rowvar=False),
            decimal=10
        )

    def test_incremental_matches_batch(self):
        """Test that incremental matches batch computation."""
        np.random.seed(42)
        data = np.random.randn(1000, 5)

        # Incremental
        inc_cov = IncrementalCovariance(n_features=5)
        for i in range(0, 1000, 100):
            inc_cov.update_batch(data[i : i + 100])

        # Batch
        expected_mean = data.mean(axis=0)
        expected_cov = np.cov(data, rowvar=False)

        np.testing.assert_array_almost_equal(inc_cov.mean, expected_mean, decimal=10)
        np.testing.assert_array_almost_equal(inc_cov.covariance, expected_cov, decimal=10)


class TestWoodburyCovariance:
    """Tests for WoodburyCovariance."""

    def test_from_data(self):
        """Test creating from data."""
        np.random.seed(42)
        data = np.random.randn(100, 3)

        woodbury = WoodburyCovariance.from_data(data)

        assert woodbury.n_samples == 100
        assert woodbury.n_features == 3
        assert woodbury.precision.shape == (3, 3)

    def test_mahalanobis_distance(self):
        """Test Mahalanobis distance computation."""
        np.random.seed(42)
        data = np.random.randn(100, 2)
        woodbury = WoodburyCovariance.from_data(data)

        # Test point
        point = np.array([0.0, 0.0])
        dist = woodbury.mahalanobis(point)

        assert dist >= 0

    def test_mahalanobis_batch(self):
        """Test batch Mahalanobis computation."""
        np.random.seed(42)
        data = np.random.randn(100, 2)
        woodbury = WoodburyCovariance.from_data(data)

        test_points = np.random.randn(10, 2)
        distances = woodbury.mahalanobis_batch(test_points)

        assert len(distances) == 10
        assert all(d >= 0 for d in distances)


class TestBatchCovarianceMixin:
    """Tests for BatchCovarianceMixin."""

    def test_compute_covariance_auto(self):
        """Test automatic strategy selection."""

        class TestClass(BatchCovarianceMixin):
            pass

        obj = TestClass()
        np.random.seed(42)
        data = np.random.randn(500, 3)

        result = obj.compute_covariance_auto(data)

        assert result.mean.shape == (3,)
        assert result.covariance.shape == (3, 3)
        assert result.n_samples == 500

    def test_compute_mahalanobis_distances(self):
        """Test Mahalanobis distance computation."""

        class TestClass(BatchCovarianceMixin):
            pass

        obj = TestClass()
        np.random.seed(42)
        data = np.random.randn(100, 2)

        cov_result = obj.compute_covariance_auto(data)
        distances = obj.compute_mahalanobis_distances(data, cov_result)

        assert len(distances) == 100
        assert all(d >= 0 for d in distances)


# =============================================================================
# Geo Tests
# =============================================================================


class TestVectorizedGeoMixin:
    """Tests for VectorizedGeoMixin."""

    def test_haversine_vectorized_single(self):
        """Test Haversine for single point pair."""

        class TestClass(VectorizedGeoMixin):
            pass

        geo = TestClass()

        # NYC to LA
        nyc_lat, nyc_lon = 40.7128, -74.0060
        la_lat, la_lon = 34.0522, -118.2437

        distance = geo.haversine_vectorized(
            nyc_lat, nyc_lon, la_lat, la_lon,
            unit=DistanceUnit.KILOMETERS
        )

        # Should be approximately 3940 km
        assert 3900 < distance < 4000

    def test_haversine_vectorized_batch(self):
        """Test Haversine for batch of points."""

        class TestClass(VectorizedGeoMixin):
            pass

        geo = TestClass()

        lats1 = np.array([40.7128, 51.5074])  # NYC, London
        lons1 = np.array([-74.0060, -0.1278])
        lats2 = np.array([34.0522, 48.8566])  # LA, Paris
        lons2 = np.array([-118.2437, 2.3522])

        distances = geo.haversine_vectorized(lats1, lons1, lats2, lons2)

        assert len(distances) == 2
        assert all(d > 0 for d in distances)

    def test_pairwise_distances(self):
        """Test pairwise distance matrix."""

        class TestClass(VectorizedGeoMixin):
            pass

        geo = TestClass()

        lats1 = np.array([0.0, 1.0])
        lons1 = np.array([0.0, 0.0])
        lats2 = np.array([0.0, 0.0, 0.0])
        lons2 = np.array([0.0, 1.0, 2.0])

        distances = geo.pairwise_distances(lats1, lons1, lats2, lons2)

        assert distances.shape == (2, 3)
        assert distances[0, 0] == pytest.approx(0.0)  # Same point

    def test_k_nearest_points(self):
        """Test k nearest neighbors."""

        class TestClass(VectorizedGeoMixin):
            pass

        geo = TestClass()

        lats = np.array([0.0, 0.01, 0.02, 1.0, 2.0])
        lons = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        indices, distances = geo.k_nearest_points(0.0, 0.0, lats, lons, k=3)

        assert len(indices) == 3
        assert 0 in indices  # The query point itself
        assert distances[0] == pytest.approx(0.0)

    def test_points_within_radius(self):
        """Test radius search."""

        class TestClass(VectorizedGeoMixin):
            pass

        geo = TestClass()

        lats = np.array([0.0, 0.001, 0.002, 1.0, 2.0])
        lons = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        indices, distances = geo.points_within_radius(
            0.0, 0.0, lats, lons,
            radius=1.0,  # 1 km
            unit=DistanceUnit.KILOMETERS
        )

        # First 3 points are within 1km
        assert len(indices) >= 1

    def test_bearing_vectorized(self):
        """Test bearing calculation."""

        class TestClass(VectorizedGeoMixin):
            pass

        geo = TestClass()

        # Due north
        bearing = geo.bearing_vectorized(0.0, 0.0, 1.0, 0.0)
        assert bearing == pytest.approx(0.0, abs=1.0)

        # Due east
        bearing = geo.bearing_vectorized(0.0, 0.0, 0.0, 1.0)
        assert bearing == pytest.approx(90.0, abs=1.0)


class TestBoundingBox:
    """Tests for BoundingBox."""

    def test_contains(self):
        """Test point containment."""
        bbox = BoundingBox(min_lat=0, max_lat=10, min_lon=0, max_lon=10)

        assert bbox.contains(5, 5) is True
        assert bbox.contains(0, 0) is True
        assert bbox.contains(-1, 5) is False
        assert bbox.contains(5, 11) is False

    def test_contains_vectorized(self):
        """Test vectorized containment."""
        bbox = BoundingBox(min_lat=0, max_lat=10, min_lon=0, max_lon=10)

        lats = np.array([5, -1, 5, 11])
        lons = np.array([5, 5, -1, 5])

        mask = bbox.contains_vectorized(lats, lons)

        assert mask[0] == True  # noqa: E712
        assert mask[1] == False  # noqa: E712
        assert mask[2] == False  # noqa: E712
        assert mask[3] == False  # noqa: E712

    def test_expand(self):
        """Test bounding box expansion."""
        bbox = BoundingBox(min_lat=0, max_lat=10, min_lon=0, max_lon=10)

        expanded = bbox.expand(1.0)

        assert expanded.min_lat == -1.0
        assert expanded.max_lat == 11.0
        assert expanded.min_lon == -1.0
        assert expanded.max_lon == 11.0


# =============================================================================
# Aggregation Tests
# =============================================================================


class TestLazyAggregationMixin:
    """Tests for LazyAggregationMixin."""

    def test_aggregate_lazy(self):
        """Test lazy aggregation."""

        class TestClass(LazyAggregationMixin):
            pass

        obj = TestClass()
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "value": [1, 2, 3, 4],
        }).lazy()

        result = obj.aggregate_lazy(
            df,
            group_by="group",
            agg_exprs=[pl.col("value").sum().alias("total")],
        )

        collected = result.collect()
        assert len(collected) == 2
        assert "total" in collected.columns

    def test_aggregate_with_join(self):
        """Test join with aggregation."""

        class TestClass(LazyAggregationMixin):
            pass

        obj = TestClass()

        orders = pl.DataFrame({
            "order_id": [1, 2, 3],
            "customer": ["A", "B", "C"],
        }).lazy()

        items = pl.DataFrame({
            "order_id": [1, 1, 2, 2, 2, 3],
            "quantity": [5, 3, 2, 1, 4, 6],
        }).lazy()

        result = obj.aggregate_with_join(
            left=orders,
            right=items,
            left_on="order_id",
            right_on="order_id",
            agg_exprs=[pl.col("quantity").sum().alias("total_qty")],
        )

        collected = result.collect()
        assert len(collected) == 3
        assert "total_qty" in collected.columns

    def test_compare_aggregates(self):
        """Test aggregate comparison."""

        class TestClass(LazyAggregationMixin):
            pass

        obj = TestClass()

        source = pl.DataFrame({
            "id": [1, 2, 3],
            "expected_total": [10, 20, 30],
        })

        aggregated = AggregationResult(
            data=pl.DataFrame({
                "id": [1, 2, 3],
                "actual_total": [10, 25, 30],  # 2 has mismatch
            })
        )

        mismatches = obj.compare_aggregates(
            source=source,
            aggregated=aggregated,
            key_column="id",
            source_column="expected_total",
            agg_column="actual_total",
            tolerance=0.0,
        )

        assert len(mismatches) == 1
        assert mismatches["id"][0] == 2

    def test_window_aggregate(self):
        """Test window aggregation."""

        class TestClass(LazyAggregationMixin):
            pass

        obj = TestClass()

        df = pl.DataFrame({
            "group": ["A", "A", "A", "B", "B"],
            "value": [1, 2, 3, 10, 20],
        }).lazy()

        result = obj.window_aggregate(
            df,
            partition_by="group",
            agg_exprs=[pl.col("value").mean().alias("group_mean")],
        ).collect()

        assert "group_mean" in result.columns
        # Group A mean = 2, Group B mean = 15
        a_rows = result.filter(pl.col("group") == "A")
        assert all(v == pytest.approx(2.0) for v in a_rows["group_mean"])

    def test_semi_join_filter(self):
        """Test semi-join filtering."""

        class TestClass(LazyAggregationMixin):
            pass

        obj = TestClass()

        main = pl.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "value": ["a", "b", "c", "d", "e"],
        }).lazy()

        filter_by = pl.DataFrame({
            "id": [2, 4],
        }).lazy()

        # Keep only matching
        result = obj.semi_join_filter(main, filter_by, on="id").collect()
        assert len(result) == 2
        assert set(result["id"].to_list()) == {2, 4}

        # Keep only non-matching (anti join)
        result_anti = obj.semi_join_filter(main, filter_by, on="id", anti=True).collect()
        assert len(result_anti) == 3
        assert set(result_anti["id"].to_list()) == {1, 3, 5}


class TestAggregationExpressionBuilder:
    """Tests for AggregationExpressionBuilder."""

    def test_build_expressions(self):
        """Test building aggregation expressions."""
        builder = AggregationExpressionBuilder()

        exprs = (
            builder
            .sum("qty", alias="total_qty")
            .mean("price", alias="avg_price")
            .count()
            .build()
        )

        assert len(exprs) == 3

    def test_apply_expressions(self):
        """Test applying built expressions."""
        df = pl.DataFrame({
            "group": ["A", "A", "B"],
            "qty": [1, 2, 3],
            "price": [10.0, 20.0, 30.0],
        })

        builder = AggregationExpressionBuilder()
        exprs = (
            builder
            .sum("qty", alias="total_qty")
            .mean("price", alias="avg_price")
            .count()
            .build()
        )

        result = df.group_by("group").agg(exprs)

        assert "total_qty" in result.columns
        assert "avg_price" in result.columns
        assert "count" in result.columns


# =============================================================================
# Integration Tests
# =============================================================================


class TestMixinIntegration:
    """Integration tests for combining mixins."""

    def test_graph_with_polars(self):
        """Test graph traversal with Polars DataFrame."""

        class TestValidator(GraphTraversalMixin):
            pass

        validator = TestValidator()

        df = pl.DataFrame({
            "employee_id": [1, 2, 3, 4, 5],
            "manager_id": [None, 1, 1, 2, 2],
        })

        adjacency = validator.build_adjacency_list(df, "employee_id", "manager_id")
        depths = validator.compute_node_depths(adjacency)

        assert depths[1] == 0  # Root
        assert depths[2] == 1
        assert depths[4] == 2

    def test_geo_with_large_batch(self):
        """Test geo operations with larger dataset."""

        class TestValidator(VectorizedGeoMixin):
            pass

        validator = TestValidator()

        np.random.seed(42)
        n_points = 10000

        lats = np.random.uniform(30, 50, n_points)
        lons = np.random.uniform(-120, -70, n_points)

        # Find all points within 100km of center
        center_lat, center_lon = 40.0, -100.0
        indices, distances = validator.points_within_radius(
            center_lat, center_lon, lats, lons,
            radius=100.0,
            unit=DistanceUnit.KILOMETERS
        )

        assert len(indices) == len(distances)
        assert all(d <= 100.0 for d in distances)
