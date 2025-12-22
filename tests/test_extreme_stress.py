"""Extreme stress tests for production-level validation.

Tests cover:
1. 10M+ row datasets
2. Stock/Crypto tick data patterns
3. High-frequency time series
4. Concurrent processing simulation
5. Memory limits testing
"""

import gc
import random
import time
from datetime import datetime, timedelta

import polars as pl
import pytest

import truthound as th


class TestExtremeScale:
    """Extreme scale tests - 10M+ rows."""

    @pytest.mark.slow
    def test_ten_million_rows(self):
        """Test with 10 million rows."""
        print("\n" + "=" * 60)
        print("10 MILLION ROWS TEST")
        print("=" * 60)

        # Create 10M rows
        start = time.time()
        n_rows = 10_000_000

        df = pl.DataFrame({
            "id": pl.arange(0, n_rows, eager=True),
            "value": pl.Series([random.random() for _ in range(n_rows)]),
            "category": pl.Series([f"cat_{i % 100}" for i in range(n_rows)]),
        })

        creation_time = time.time() - start
        memory_mb = df.estimated_size("mb")
        print(f"Data creation: {creation_time:.2f}s, Memory: {memory_mb:.1f}MB")

        # Test check
        start = time.time()
        report = th.check(df)
        check_time = time.time() - start
        print(f"th.check(): {check_time:.2f}s")

        # Test profile
        start = time.time()
        profile = th.profile(df)
        profile_time = time.time() - start
        print(f"th.profile(): {profile_time:.2f}s")

        # Test learn
        start = time.time()
        schema = th.learn(df)
        learn_time = time.time() - start
        print(f"th.learn(): {learn_time:.2f}s")

        print(f"\nTotal processing time: {check_time + profile_time + learn_time:.2f}s")

        assert report is not None
        assert check_time < 120  # Should complete within 2 minutes
        assert profile_time < 30
        assert learn_time < 30

        del df
        gc.collect()

    @pytest.mark.slow
    def test_wide_dataframe_500_columns(self):
        """Test with 500 columns."""
        print("\n" + "=" * 60)
        print("500 COLUMNS TEST")
        print("=" * 60)

        n_rows = 100_000
        n_cols = 500

        start = time.time()
        data = {f"col_{i}": pl.arange(0, n_rows, eager=True) for i in range(n_cols)}
        df = pl.DataFrame(data)
        creation_time = time.time() - start
        print(f"Data creation: {creation_time:.2f}s")

        start = time.time()
        report = th.check(df)
        check_time = time.time() - start
        print(f"th.check(): {check_time:.2f}s")

        start = time.time()
        schema = th.learn(df)
        learn_time = time.time() - start
        print(f"th.learn(): {learn_time:.2f}s")

        assert report is not None
        assert schema is not None
        assert len(schema.columns) == 500

        del df
        gc.collect()


class TestFinancialTickData:
    """Stock and cryptocurrency tick data tests."""

    def test_stock_ohlcv_data(self):
        """Test stock OHLCV (Open, High, Low, Close, Volume) data."""
        print("\n" + "=" * 60)
        print("STOCK OHLCV DATA TEST (1M ticks)")
        print("=" * 60)

        n_ticks = 1_000_000
        base_price = 50000  # Starting price

        # Simulate realistic price movement
        start = time.time()
        prices = [float(base_price)]
        for _ in range(n_ticks - 1):
            change = random.gauss(0, 100)  # Random walk
            prices.append(max(100.0, prices[-1] + change))

        stock_data = pl.DataFrame({
            "timestamp": pl.datetime_range(
                datetime(2020, 1, 1),
                datetime(2020, 1, 1) + timedelta(seconds=n_ticks),
                interval="1s",
                eager=True,
            ).head(n_ticks),
            "symbol": ["AAPL"] * n_ticks,
            "open": prices,
            "high": [p * random.uniform(1.0, 1.02) for p in prices],
            "low": [p * random.uniform(0.98, 1.0) for p in prices],
            "close": [p * random.uniform(0.99, 1.01) for p in prices],
            "volume": [random.randint(100, 1000000) for _ in range(n_ticks)],
            "vwap": prices,  # Simplified
        })

        creation_time = time.time() - start
        print(f"Data creation: {creation_time:.2f}s")

        # Test all operations
        start = time.time()
        report = th.check(stock_data)
        check_time = time.time() - start
        print(f"th.check(): {check_time:.2f}s")

        start = time.time()
        schema = th.learn(stock_data)
        learn_time = time.time() - start
        print(f"th.learn(): {learn_time:.2f}s")

        start = time.time()
        profile = th.profile(stock_data)
        profile_time = time.time() - start
        print(f"th.profile(): {profile_time:.2f}s")

        assert report is not None
        assert schema is not None
        assert profile is not None

        # Verify schema constraints make sense for financial data
        assert schema.columns["volume"].min_value >= 0
        assert schema.columns["open"].min_value > 0

        del stock_data
        gc.collect()

    def test_crypto_orderbook_data(self):
        """Test cryptocurrency orderbook snapshot data."""
        print("\n" + "=" * 60)
        print("CRYPTO ORDERBOOK DATA TEST")
        print("=" * 60)

        n_snapshots = 500_000

        start = time.time()
        orderbook = pl.DataFrame({
            "timestamp": pl.datetime_range(
                datetime(2024, 1, 1),
                datetime(2024, 1, 1) + timedelta(milliseconds=n_snapshots * 100),
                interval="100ms",
                eager=True,
            ).head(n_snapshots),
            "exchange": random.choices(["binance", "coinbase", "kraken"], k=n_snapshots),
            "symbol": ["BTC-USD"] * n_snapshots,
            "bid_price": [random.uniform(40000, 45000) for _ in range(n_snapshots)],
            "bid_size": [random.uniform(0.01, 10) for _ in range(n_snapshots)],
            "ask_price": [random.uniform(40000, 45000) for _ in range(n_snapshots)],
            "ask_size": [random.uniform(0.01, 10) for _ in range(n_snapshots)],
            "spread": [random.uniform(0.01, 50) for _ in range(n_snapshots)],
            "mid_price": [random.uniform(40000, 45000) for _ in range(n_snapshots)],
        })

        creation_time = time.time() - start
        print(f"Data creation: {creation_time:.2f}s")

        start = time.time()
        report = th.check(orderbook)
        check_time = time.time() - start
        print(f"th.check(): {check_time:.2f}s")

        start = time.time()
        schema = th.learn(orderbook)
        learn_time = time.time() - start
        print(f"th.learn(): {learn_time:.2f}s")

        assert report is not None
        assert schema is not None
        assert check_time < 30

        del orderbook
        gc.collect()

    def test_crypto_trade_stream(self):
        """Test high-frequency cryptocurrency trade data."""
        print("\n" + "=" * 60)
        print("CRYPTO TRADE STREAM TEST (2M trades)")
        print("=" * 60)

        n_trades = 2_000_000

        start = time.time()
        trades = pl.DataFrame({
            "trade_id": pl.arange(0, n_trades, eager=True),
            "timestamp": pl.datetime_range(
                datetime(2024, 1, 1),
                datetime(2024, 1, 1) + timedelta(milliseconds=n_trades * 10),
                interval="10ms",
                eager=True,
            ).head(n_trades),
            "symbol": random.choices(["BTC-USD", "ETH-USD", "SOL-USD"], k=n_trades),
            "side": random.choices(["buy", "sell"], k=n_trades),
            "price": [random.uniform(100, 50000) for _ in range(n_trades)],
            "quantity": [random.uniform(0.0001, 100) for _ in range(n_trades)],
            "quote_quantity": [random.uniform(1, 100000) for _ in range(n_trades)],
        })

        creation_time = time.time() - start
        print(f"Data creation: {creation_time:.2f}s")

        start = time.time()
        report = th.check(trades)
        check_time = time.time() - start
        print(f"th.check(): {check_time:.2f}s")

        start = time.time()
        profile = th.profile(trades)
        profile_time = time.time() - start
        print(f"th.profile(): {profile_time:.2f}s")

        assert report is not None
        assert profile is not None
        assert check_time < 60

        del trades
        gc.collect()


class TestDriftAtScale:
    """Drift detection at extreme scale."""

    def test_drift_5m_vs_5m(self):
        """Test drift detection between two 5M row datasets."""
        print("\n" + "=" * 60)
        print("DRIFT DETECTION: 5M vs 5M rows")
        print("=" * 60)

        n_rows = 5_000_000

        start = time.time()
        baseline = pl.DataFrame({
            "value": pl.Series([random.gauss(100, 10) for _ in range(n_rows)]),
            "category": pl.Series(random.choices(["A", "B", "C"], k=n_rows)),
        })

        current = pl.DataFrame({
            "value": pl.Series([random.gauss(105, 12) for _ in range(n_rows)]),  # Shifted
            "category": pl.Series(random.choices(["A", "B", "C", "D"], k=n_rows)),  # New category
        })
        creation_time = time.time() - start
        print(f"Data creation: {creation_time:.2f}s")

        # Without sampling
        start = time.time()
        drift_full = th.compare(baseline, current)
        full_time = time.time() - start
        print(f"Full comparison: {full_time:.2f}s")

        # With sampling
        start = time.time()
        drift_sampled = th.compare(baseline, current, sample_size=50000)
        sampled_time = time.time() - start
        print(f"Sampled comparison (50k): {sampled_time:.2f}s")

        print(f"Speedup with sampling: {full_time / sampled_time:.1f}x")

        # Both should detect drift
        assert drift_full.has_drift
        assert drift_sampled.has_drift

        del baseline, current
        gc.collect()

    def test_drift_timeseries_pattern(self):
        """Test drift in time series data patterns."""
        print("\n" + "=" * 60)
        print("TIME SERIES DRIFT TEST")
        print("=" * 60)

        n_points = 1_000_000

        # Baseline: stable pattern
        baseline = pl.DataFrame({
            "value": [100.0 + 10.0 * (i % 24) + random.gauss(0, 5) for i in range(n_points)],
            "hour": [i % 24 for i in range(n_points)],
        })

        # Current: pattern shifted (e.g., daylight saving or behavior change)
        current = pl.DataFrame({
            "value": [100.0 + 10.0 * ((i + 3) % 24) + random.gauss(0, 5) for i in range(n_points)],
            "hour": [i % 24 for i in range(n_points)],
        })

        start = time.time()
        drift = th.compare(baseline, current, sample_size=100000)
        elapsed = time.time() - start
        print(f"Drift detection: {elapsed:.2f}s")

        assert drift is not None
        # Note: With periodic patterns, the overall distribution may be similar
        # even if the phase is shifted. The test validates performance, not drift detection.
        assert elapsed < 5  # Should complete quickly with sampling

        del baseline, current
        gc.collect()


class TestRealisticWorkloads:
    """Real-world workload simulations."""

    def test_continuous_monitoring_simulation(self):
        """Simulate continuous data monitoring (100 batches)."""
        print("\n" + "=" * 60)
        print("CONTINUOUS MONITORING SIMULATION (100 batches)")
        print("=" * 60)

        batch_size = 100_000
        n_batches = 100

        # Learn baseline schema
        baseline = pl.DataFrame({
            "metric": [random.gauss(100, 10) for _ in range(batch_size)],
            "status": random.choices(["ok", "warn", "error"], weights=[90, 8, 2], k=batch_size),
        })
        schema = th.learn(baseline)

        total_time = 0
        issues_detected = 0

        for i in range(n_batches):
            # Simulate incoming batch with occasional anomalies
            if i % 20 == 19:  # Every 20th batch has drift
                batch = pl.DataFrame({
                    "metric": [random.gauss(120, 15) for _ in range(batch_size)],  # Shifted
                    "status": random.choices(["ok", "warn", "error"], weights=[70, 20, 10], k=batch_size),
                })
            else:
                batch = pl.DataFrame({
                    "metric": [random.gauss(100, 10) for _ in range(batch_size)],
                    "status": random.choices(["ok", "warn", "error"], weights=[90, 8, 2], k=batch_size),
                })

            start = time.time()
            report = th.check(batch, schema=schema)
            elapsed = time.time() - start
            total_time += elapsed

            if report.has_issues:
                issues_detected += 1

            del batch

        avg_time = total_time / n_batches
        print(f"Processed {n_batches} batches of {batch_size:,} rows each")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average per batch: {avg_time * 1000:.1f}ms")
        print(f"Batches with issues: {issues_detected}")

        # Should process each batch in under 1 second
        assert avg_time < 1.0

        del baseline
        gc.collect()

    def test_multi_table_validation(self):
        """Test validating multiple related tables."""
        print("\n" + "=" * 60)
        print("MULTI-TABLE VALIDATION TEST")
        print("=" * 60)

        # Users table
        users = pl.DataFrame({
            "user_id": pl.arange(0, 100000, eager=True),
            "email": [f"user{i}@example.com" for i in range(100000)],
            "created_at": pl.datetime_range(
                datetime(2020, 1, 1),
                datetime(2024, 1, 1),
                interval="21m",
                eager=True,
            ).head(100000),
        })

        # Orders table
        orders = pl.DataFrame({
            "order_id": pl.arange(0, 1000000, eager=True),
            "user_id": [random.randint(0, 99999) for _ in range(1000000)],
            "amount": [random.uniform(10, 10000) for _ in range(1000000)],
            "status": random.choices(["pending", "completed", "cancelled"], k=1000000),
        })

        # Events table
        events = pl.DataFrame({
            "event_id": pl.arange(0, 5000000, eager=True),
            "user_id": [random.randint(0, 99999) for _ in range(5000000)],
            "event_type": random.choices(["view", "click", "purchase"], k=5000000),
            "timestamp": pl.datetime_range(
                datetime(2024, 1, 1),
                datetime(2024, 1, 1) + timedelta(seconds=5000000),
                interval="1s",
                eager=True,
            ).head(5000000),
        })

        tables = {"users": users, "orders": orders, "events": events}

        total_time = 0
        for name, df in tables.items():
            start = time.time()
            report = th.check(df)
            schema = th.learn(df)
            elapsed = time.time() - start
            total_time += elapsed
            print(f"{name}: {len(df):,} rows, {elapsed:.2f}s")

        print(f"\nTotal validation time: {total_time:.2f}s")

        del users, orders, events
        gc.collect()


class TestMemoryPressure:
    """Memory pressure and stability tests."""

    def test_memory_under_pressure(self):
        """Test behavior under memory pressure."""
        print("\n" + "=" * 60)
        print("MEMORY PRESSURE TEST")
        print("=" * 60)

        # Create and process multiple large datasets
        for i in range(3):
            print(f"\nIteration {i + 1}/3:")

            df = pl.DataFrame({
                "data": pl.arange(0, 3_000_000, eager=True),
                "value": [random.random() for _ in range(3_000_000)],
            })

            start = time.time()
            report = th.check(df)
            profile = th.profile(df)
            schema = th.learn(df)
            elapsed = time.time() - start

            print(f"  Processed 3M rows in {elapsed:.2f}s")

            del df, report, profile, schema
            gc.collect()

        print("\nMemory pressure test completed without OOM")
        assert True

    def test_rapid_fire_operations(self):
        """Test rapid successive operations."""
        print("\n" + "=" * 60)
        print("RAPID FIRE OPERATIONS TEST (1000 ops)")
        print("=" * 60)

        df = pl.DataFrame({
            "value": list(range(10000)),
            "category": random.choices(["A", "B", "C"], k=10000),
        })

        start = time.time()
        for _ in range(1000):
            th.check(df)
        elapsed = time.time() - start

        ops_per_sec = 1000 / elapsed
        print(f"1000 check() operations: {elapsed:.2f}s")
        print(f"Operations per second: {ops_per_sec:.1f}")

        # Should handle at least 100 ops/sec
        assert ops_per_sec > 100

        del df
        gc.collect()


class TestEdgeCasesExtreme:
    """Extreme edge cases."""

    def test_all_same_values(self):
        """Test column where all values are identical."""
        df = pl.DataFrame({
            "constant": [42] * 1_000_000,
            "constant_str": ["same"] * 1_000_000,
        })

        report = th.check(df)
        schema = th.learn(df)

        assert schema is not None
        # Should detect that all values are same
        assert schema.columns["constant"].min_value == schema.columns["constant"].max_value

        del df
        gc.collect()

    def test_high_null_ratio(self):
        """Test data with 99% nulls."""
        n_rows = 1_000_000
        values = [None] * (n_rows - 10000) + list(range(10000))
        random.shuffle(values)

        df = pl.DataFrame({
            "mostly_null": pl.Series(values),
        })

        report = th.check(df)

        # Should detect critical null issue
        assert report.has_issues
        null_issues = [i for i in report.issues if i.issue_type == "null"]
        assert len(null_issues) > 0
        assert any(i.severity.value == "critical" for i in null_issues)

        del df
        gc.collect()

    def test_extreme_cardinality(self):
        """Test extreme cardinality scenarios."""
        n_rows = 1_000_000

        df = pl.DataFrame({
            "all_unique": pl.arange(0, n_rows, eager=True).cast(pl.Utf8),
            "binary": random.choices(["0", "1"], k=n_rows),
        })

        schema = th.learn(df)

        # High cardinality should not have allowed_values
        assert schema.columns["all_unique"].allowed_values is None
        # Binary should have allowed_values
        assert schema.columns["binary"].allowed_values is not None
        assert len(schema.columns["binary"].allowed_values) == 2

        del df
        gc.collect()


# Performance comparison marker
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow (deselect with '-m \"not slow\"')")
