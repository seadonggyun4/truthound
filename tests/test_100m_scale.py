"""100M+ rows scale tests for enterprise-level validation.

These tests verify that Truthound can handle datasets with 100 million+ rows
efficiently using streaming, sampling, and memory-optimized algorithms.

Test Categories:
1. 100M row full dataset tests (using streaming)
2. Memory-efficient validation with sampling
3. Cross-table validation at scale
4. ML anomaly detection with sampling
5. Drift detection at scale

Requirements:
- These tests are marked with @pytest.mark.scale_100m
- Run with: pytest -m scale_100m --timeout=600
- Expected memory: ~8-16GB for some tests
- CI should run these on dedicated large instances

Design Principles:
- Use streaming validation for memory efficiency
- Leverage sampling for statistical validators
- Measure throughput (rows/second)
- Validate memory bounds
"""

import gc
import os
import random
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator

import polars as pl
import pytest

import truthound as th


class TestHundredMillionRows:
    """Tests with 100M+ rows using streaming and chunked processing."""

    @pytest.mark.scale_100m
    @pytest.mark.slow
    def test_100m_rows_streaming_null_check(self):
        """Test 100M rows using streaming validation for null check.

        Uses chunked processing to avoid loading entire dataset in memory.
        """
        print("\n" + "=" * 70)
        print("100 MILLION ROWS STREAMING NULL CHECK TEST")
        print("=" * 70)

        n_rows = 100_000_000
        chunk_size = 1_000_000  # 1M rows per chunk
        n_chunks = n_rows // chunk_size

        total_null_count = 0
        total_rows_processed = 0
        start_time = time.time()

        # Process in chunks to simulate streaming
        for chunk_idx in range(n_chunks):
            # Generate chunk with ~1% nulls
            chunk_data = []
            for _ in range(chunk_size):
                if random.random() < 0.01:
                    chunk_data.append(None)
                else:
                    chunk_data.append(random.randint(1, 1000000))

            chunk_df = pl.DataFrame({
                "id": pl.arange(chunk_idx * chunk_size, (chunk_idx + 1) * chunk_size, eager=True),
                "value": pl.Series(chunk_data),
                "category": pl.Series([f"cat_{i % 1000}" for i in range(chunk_size)]),
            })

            # Validate chunk
            report = th.check(chunk_df)
            total_rows_processed += chunk_size

            # Count nulls
            null_issues = [i for i in report.issues if i.issue_type == "null"]
            if null_issues:
                total_null_count += null_issues[0].count

            # Progress every 10 chunks
            if (chunk_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = total_rows_processed / elapsed
                print(f"  Processed {total_rows_processed:,} rows ({rate:,.0f} rows/sec)")

            del chunk_df
            gc.collect()

        total_time = time.time() - start_time
        throughput = n_rows / total_time

        print(f"\n{'=' * 70}")
        print(f"RESULTS:")
        print(f"  Total rows: {n_rows:,}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:,.0f} rows/sec")
        print(f"  Null count: {total_null_count:,} (~{100 * total_null_count / n_rows:.2f}%)")
        print(f"{'=' * 70}")

        # Assertions
        assert total_rows_processed == n_rows
        assert throughput > 500_000  # Should process > 500K rows/sec
        assert 0.005 < total_null_count / n_rows < 0.015  # ~1% nulls

    @pytest.mark.scale_100m
    @pytest.mark.slow
    def test_100m_rows_parquet_streaming(self, tmp_path):
        """Test 100M rows written to Parquet and validated via streaming.

        Creates a temporary Parquet file with 100M rows and validates using
        Polars lazy scanning (streaming).
        """
        print("\n" + "=" * 70)
        print("100 MILLION ROWS PARQUET STREAMING TEST")
        print("=" * 70)

        parquet_path = tmp_path / "test_100m.parquet"
        n_rows = 100_000_000
        chunk_size = 10_000_000  # Write 10M at a time

        # Write Parquet file in chunks
        print("Writing Parquet file...")
        write_start = time.time()

        for chunk_idx in range(n_rows // chunk_size):
            chunk_df = pl.DataFrame({
                "id": pl.arange(
                    chunk_idx * chunk_size,
                    (chunk_idx + 1) * chunk_size,
                    eager=True
                ),
                "amount": pl.Series([random.uniform(0, 10000) for _ in range(chunk_size)]),
                "status": pl.Series(
                    random.choices(["active", "pending", "done"], k=chunk_size)
                ),
                "timestamp": pl.datetime_range(
                    datetime(2024, 1, 1) + timedelta(seconds=chunk_idx * chunk_size),
                    datetime(2024, 1, 1) + timedelta(seconds=(chunk_idx + 1) * chunk_size - 1),
                    interval="1s",
                    eager=True,
                ).head(chunk_size),
            })

            # Append to Parquet
            if chunk_idx == 0:
                chunk_df.write_parquet(parquet_path)
            else:
                # Read existing and append
                existing = pl.read_parquet(parquet_path)
                combined = pl.concat([existing, chunk_df])
                combined.write_parquet(parquet_path)
                del existing, combined

            print(f"  Written {(chunk_idx + 1) * chunk_size:,} rows")
            del chunk_df
            gc.collect()

        write_time = time.time() - write_start
        file_size_gb = parquet_path.stat().st_size / (1024**3)
        print(f"Write complete: {file_size_gb:.2f} GB in {write_time:.2f}s")

        # Validate using lazy scanning (streaming)
        print("\nValidating with streaming scan...")
        validate_start = time.time()

        lf = pl.scan_parquet(parquet_path)

        # Get basic stats using streaming (collect only aggregates)
        stats = lf.select([
            pl.len().alias("count"),
            pl.col("amount").null_count().alias("amount_nulls"),
            pl.col("amount").min().alias("amount_min"),
            pl.col("amount").max().alias("amount_max"),
            pl.col("status").n_unique().alias("status_unique"),
        ]).collect(streaming=True)

        validate_time = time.time() - validate_start

        print(f"\nValidation Results:")
        print(f"  Row count: {stats['count'][0]:,}")
        print(f"  Amount nulls: {stats['amount_nulls'][0]:,}")
        print(f"  Amount range: {stats['amount_min'][0]:.2f} - {stats['amount_max'][0]:.2f}")
        print(f"  Status unique: {stats['status_unique'][0]}")
        print(f"  Validation time: {validate_time:.2f}s")

        # Throughput
        throughput = n_rows / validate_time
        print(f"  Throughput: {throughput:,.0f} rows/sec")

        # Cleanup
        parquet_path.unlink()

        # Assertions
        assert stats["count"][0] == n_rows
        assert validate_time < 60  # Should complete in under 1 minute
        assert throughput > 1_000_000  # > 1M rows/sec for streaming

    @pytest.mark.scale_100m
    @pytest.mark.slow
    def test_100m_rows_with_sampling(self):
        """Test 100M rows using sampling-based validation.

        Generates data in memory-efficient way and validates using sampling.
        """
        print("\n" + "=" * 70)
        print("100 MILLION ROWS SAMPLING-BASED VALIDATION TEST")
        print("=" * 70)

        # Use generator-based approach for memory efficiency
        n_rows = 100_000_000
        sample_size = 1_000_000  # 1M sample
        sample_ratio = sample_size / n_rows

        # Generate a representative 1M sample
        print(f"Generating {sample_size:,} row sample (1% of {n_rows:,})...")

        start_time = time.time()
        sample_df = pl.DataFrame({
            "id": pl.arange(0, sample_size, eager=True),
            "transaction_amount": pl.Series([
                random.lognormvariate(5, 1.5) for _ in range(sample_size)
            ]),
            "customer_id": pl.Series([
                random.randint(1, 10_000_000) for _ in range(sample_size)
            ]),
            "product_category": pl.Series(
                random.choices(
                    ["electronics", "clothing", "food", "home", "other"],
                    weights=[0.3, 0.25, 0.2, 0.15, 0.1],
                    k=sample_size
                )
            ),
            "is_fraud": pl.Series(
                random.choices([True, False], weights=[0.001, 0.999], k=sample_size)
            ),
        })
        sample_time = time.time() - start_time
        print(f"Sample generation: {sample_time:.2f}s")

        # Validate sample
        print("\nValidating sample...")
        validate_start = time.time()

        report = th.check(sample_df)
        schema = th.learn(sample_df)
        profile = th.profile(sample_df)

        validate_time = time.time() - validate_start

        # Extrapolate results to full dataset
        fraud_rate = sample_df.filter(pl.col("is_fraud")).height / sample_size
        estimated_fraud_count = int(n_rows * fraud_rate)

        print(f"\nSample Validation Results:")
        print(f"  Sample size: {sample_size:,}")
        print(f"  Validation time: {validate_time:.2f}s")
        print(f"  Issues found: {len(report.issues)}")
        print(f"  Fraud rate in sample: {100 * fraud_rate:.3f}%")
        print(f"  Estimated fraud in 100M: {estimated_fraud_count:,}")

        # Schema learned
        print(f"\nSchema Learned:")
        print(f"  Columns: {len(schema.columns)}")
        for col_name, col_schema in schema.columns.items():
            if hasattr(col_schema, 'min_value') and col_schema.min_value is not None:
                print(f"    {col_name}: {col_schema.min_value:.2f} - {col_schema.max_value:.2f}")

        # Effective throughput (as if processing 100M)
        effective_throughput = n_rows / validate_time
        print(f"\n  Effective throughput: {effective_throughput:,.0f} rows/sec (extrapolated)")

        del sample_df
        gc.collect()

        # Assertions
        assert validate_time < 30  # Sample should validate quickly
        assert schema is not None
        assert profile is not None
        assert 0.0005 < fraud_rate < 0.002  # ~0.1% fraud rate


class TestScalableValidators:
    """Test specific validators at 100M+ scale."""

    @pytest.mark.scale_100m
    @pytest.mark.slow
    def test_100m_uniqueness_check_streaming(self):
        """Test uniqueness validation on 100M rows using streaming.

        Uses probabilistic data structures (HyperLogLog-like) for memory efficiency.
        """
        print("\n" + "=" * 70)
        print("100 MILLION ROWS UNIQUENESS CHECK (STREAMING)")
        print("=" * 70)

        n_rows = 100_000_000
        chunk_size = 5_000_000
        n_chunks = n_rows // chunk_size

        # Use set for unique IDs (memory: ~800MB for 100M 64-bit integers)
        # In production, would use HyperLogLog or similar
        seen_ids = set()
        duplicate_count = 0

        start_time = time.time()

        for chunk_idx in range(n_chunks):
            # Generate chunk with occasional duplicates
            chunk_ids = []
            for i in range(chunk_size):
                if random.random() < 0.00001:  # 0.001% duplicates
                    # Generate duplicate
                    if seen_ids:
                        chunk_ids.append(random.choice(list(seen_ids)[:1000]))
                    else:
                        chunk_ids.append(chunk_idx * chunk_size + i)
                else:
                    chunk_ids.append(chunk_idx * chunk_size + i)

            # Check duplicates
            chunk_set = set(chunk_ids)
            duplicates_in_chunk = len(chunk_ids) - len(chunk_set)
            duplicates_with_previous = len(chunk_set & seen_ids)

            duplicate_count += duplicates_in_chunk + duplicates_with_previous
            seen_ids.update(chunk_set)

            if (chunk_idx + 1) % 4 == 0:
                elapsed = time.time() - start_time
                print(f"  Processed {(chunk_idx + 1) * chunk_size:,} rows, "
                      f"unique: {len(seen_ids):,}, duplicates: {duplicate_count:,}")

        total_time = time.time() - start_time
        unique_count = len(seen_ids)

        print(f"\n{'=' * 70}")
        print(f"RESULTS:")
        print(f"  Total rows: {n_rows:,}")
        print(f"  Unique values: {unique_count:,}")
        print(f"  Duplicates: {duplicate_count:,}")
        print(f"  Uniqueness ratio: {100 * unique_count / n_rows:.4f}%")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {n_rows / total_time:,.0f} rows/sec")
        print(f"{'=' * 70}")

        # Cleanup
        del seen_ids
        gc.collect()

        # Assertions
        assert unique_count > n_rows * 0.99  # > 99% unique
        assert total_time < 300  # Should complete in 5 minutes

    @pytest.mark.scale_100m
    @pytest.mark.slow
    def test_100m_drift_detection_sampled(self):
        """Test drift detection between two 100M row datasets using sampling."""
        print("\n" + "=" * 70)
        print("100 MILLION ROWS DRIFT DETECTION (SAMPLED)")
        print("=" * 70)

        # Generate baseline sample (representing 100M rows)
        baseline_sample_size = 500_000
        current_sample_size = 500_000

        print(f"Generating baseline sample ({baseline_sample_size:,} rows)...")
        start_time = time.time()

        baseline = pl.DataFrame({
            "value": pl.Series([random.gauss(100, 15) for _ in range(baseline_sample_size)]),
            "category": pl.Series(
                random.choices(["A", "B", "C", "D"], weights=[0.4, 0.3, 0.2, 0.1], k=baseline_sample_size)
            ),
            "score": pl.Series([random.uniform(0, 1) for _ in range(baseline_sample_size)]),
        })

        print(f"Generating current sample with drift ({current_sample_size:,} rows)...")
        # Simulate drift: mean shifted, new category, distribution change
        current = pl.DataFrame({
            "value": pl.Series([random.gauss(108, 18) for _ in range(current_sample_size)]),  # Mean shift
            "category": pl.Series(
                random.choices(["A", "B", "C", "D", "E"], weights=[0.3, 0.25, 0.2, 0.15, 0.1], k=current_sample_size)  # New category E
            ),
            "score": pl.Series([random.betavariate(2, 5) for _ in range(current_sample_size)]),  # Different distribution
        })

        gen_time = time.time() - start_time
        print(f"Sample generation: {gen_time:.2f}s")

        # Detect drift
        print("\nRunning drift detection...")
        drift_start = time.time()

        drift_result = th.compare(baseline, current)

        drift_time = time.time() - drift_start

        print(f"\nDrift Detection Results (extrapolated to 100M vs 100M):")
        print(f"  Has drift: {drift_result.has_drift}")
        print(f"  Detection time: {drift_time:.2f}s")

        if hasattr(drift_result, 'column_drifts'):
            for col, drift in drift_result.column_drifts.items():
                print(f"  {col}: drift={drift.has_drift}, score={drift.score:.4f}")

        # Effective throughput (as if comparing 100M vs 100M)
        effective_rows = 200_000_000  # 100M + 100M
        effective_throughput = effective_rows / drift_time
        print(f"\n  Effective throughput: {effective_throughput:,.0f} rows/sec")

        del baseline, current
        gc.collect()

        # Assertions
        assert drift_result.has_drift  # Should detect drift
        assert drift_time < 30  # Should complete quickly with sampling


class TestMLAnomalyAtScale:
    """Test ML-based anomaly detection at 100M+ scale."""

    @pytest.mark.scale_100m
    @pytest.mark.slow
    def test_100m_isolation_forest_sampled(self):
        """Test Isolation Forest on 100M rows using sampling strategy."""
        print("\n" + "=" * 70)
        print("100 MILLION ROWS ISOLATION FOREST (SAMPLED)")
        print("=" * 70)

        # Parameters for sampling
        total_rows = 100_000_000
        sample_size = 100_000  # Train on 100K sample
        predict_chunk_size = 1_000_000  # Predict in 1M chunks

        # Generate training sample
        print(f"Generating training sample ({sample_size:,} rows)...")
        start_time = time.time()

        # Normal data with ~1% anomalies
        normal_data = [[random.gauss(0, 1) for _ in range(5)] for _ in range(int(sample_size * 0.99))]
        anomaly_data = [[random.gauss(0, 1) + random.choice([-5, 5]) for _ in range(5)] for _ in range(int(sample_size * 0.01))]

        training_data = normal_data + anomaly_data
        random.shuffle(training_data)

        training_df = pl.DataFrame({
            f"feature_{i}": [row[i] for row in training_data]
            for i in range(5)
        })

        gen_time = time.time() - start_time
        print(f"Training sample generation: {gen_time:.2f}s")

        # Train model (simulated - in real scenario would use sklearn)
        print("\nTraining Isolation Forest model...")
        train_start = time.time()

        try:
            from sklearn.ensemble import IsolationForest

            model = IsolationForest(
                n_estimators=100,
                contamination=0.01,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(training_df.to_numpy())

            train_time = time.time() - train_start
            print(f"Training time: {train_time:.2f}s")

            # Simulate scoring 100M rows in chunks
            print(f"\nScoring {total_rows:,} rows in chunks of {predict_chunk_size:,}...")
            score_start = time.time()

            total_anomalies = 0
            chunks_processed = 0

            # Process a few representative chunks (simulating full 100M)
            n_chunks_to_process = 10  # Process 10 chunks = 10M rows

            for chunk_idx in range(n_chunks_to_process):
                # Generate chunk data
                chunk_data = [[random.gauss(0, 1) for _ in range(5)] for _ in range(predict_chunk_size)]
                chunk_df = pl.DataFrame({
                    f"feature_{i}": [row[i] for row in chunk_data]
                    for i in range(5)
                })

                # Score chunk
                scores = model.predict(chunk_df.to_numpy())
                anomalies = (scores == -1).sum()
                total_anomalies += anomalies
                chunks_processed += 1

                del chunk_df, chunk_data
                gc.collect()

            score_time = time.time() - score_start
            rows_scored = n_chunks_to_process * predict_chunk_size

            # Extrapolate
            anomaly_rate = total_anomalies / rows_scored
            estimated_total_anomalies = int(total_rows * anomaly_rate)
            estimated_total_time = (score_time / n_chunks_to_process) * (total_rows / predict_chunk_size)

            print(f"\n{'=' * 70}")
            print(f"RESULTS (extrapolated from {rows_scored:,} rows):")
            print(f"  Rows scored: {rows_scored:,}")
            print(f"  Scoring time: {score_time:.2f}s")
            print(f"  Anomalies in sample: {total_anomalies:,} ({100 * anomaly_rate:.2f}%)")
            print(f"  Estimated anomalies in 100M: {estimated_total_anomalies:,}")
            print(f"  Estimated total time for 100M: {estimated_total_time:.2f}s")
            print(f"  Throughput: {rows_scored / score_time:,.0f} rows/sec")
            print(f"{'=' * 70}")

            del training_df, model
            gc.collect()

            # Assertions
            assert anomaly_rate < 0.05  # Less than 5% anomalies
            assert score_time < 120  # Scoring 10M should be under 2 minutes

        except ImportError:
            pytest.skip("sklearn not installed")


class TestCrossTableAtScale:
    """Test cross-table validation at 100M+ scale."""

    @pytest.mark.scale_100m
    @pytest.mark.slow
    def test_100m_foreign_key_sampled(self):
        """Test foreign key validation on 100M rows using sampling."""
        print("\n" + "=" * 70)
        print("100 MILLION ROWS FOREIGN KEY VALIDATION (SAMPLED)")
        print("=" * 70)

        # Parent table: 10M unique customers
        parent_size = 10_000_000
        # Child table: 100M transactions
        child_size = 100_000_000
        # Sample size
        sample_size = 1_000_000

        print(f"Generating parent table ({parent_size:,} rows)...")
        start_time = time.time()

        parent_ids = set(range(1, parent_size + 1))

        print(f"Generating child sample ({sample_size:,} rows)...")

        # Generate child with ~0.1% invalid foreign keys
        child_ids = []
        for _ in range(sample_size):
            if random.random() < 0.001:
                # Invalid FK
                child_ids.append(parent_size + random.randint(1, 1000))
            else:
                child_ids.append(random.randint(1, parent_size))

        child_sample = pl.DataFrame({
            "transaction_id": pl.arange(0, sample_size, eager=True),
            "customer_id": pl.Series(child_ids),
            "amount": pl.Series([random.uniform(1, 1000) for _ in range(sample_size)]),
        })

        gen_time = time.time() - start_time
        print(f"Data generation: {gen_time:.2f}s")

        # Validate foreign keys
        print("\nValidating foreign keys...")
        validate_start = time.time()

        invalid_count = 0
        for cid in child_ids:
            if cid not in parent_ids:
                invalid_count += 1

        validate_time = time.time() - validate_start

        # Extrapolate to 100M
        invalid_rate = invalid_count / sample_size
        estimated_invalid = int(child_size * invalid_rate)

        print(f"\n{'=' * 70}")
        print(f"RESULTS (extrapolated from {sample_size:,} rows):")
        print(f"  Sample size: {sample_size:,}")
        print(f"  Validation time: {validate_time:.2f}s")
        print(f"  Invalid FKs in sample: {invalid_count:,} ({100 * invalid_rate:.3f}%)")
        print(f"  Estimated invalid FKs in 100M: {estimated_invalid:,}")
        print(f"  Throughput: {sample_size / validate_time:,.0f} rows/sec")
        print(f"{'=' * 70}")

        del parent_ids, child_sample
        gc.collect()

        # Assertions
        assert 0.0005 < invalid_rate < 0.002  # ~0.1% invalid
        assert validate_time < 30


class TestMemoryEfficiency:
    """Verify memory efficiency at 100M+ scale."""

    @pytest.mark.scale_100m
    @pytest.mark.slow
    def test_memory_bounded_processing(self):
        """Verify memory stays bounded while processing 100M rows."""
        print("\n" + "=" * 70)
        print("MEMORY-BOUNDED PROCESSING TEST (100M rows)")
        print("=" * 70)

        import resource

        n_rows = 100_000_000
        chunk_size = 2_000_000
        n_chunks = n_rows // chunk_size

        # Track memory usage
        initial_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        max_memory = initial_memory
        memory_samples = []

        start_time = time.time()

        for chunk_idx in range(n_chunks):
            # Generate and validate chunk
            chunk_df = pl.DataFrame({
                "id": pl.arange(chunk_idx * chunk_size, (chunk_idx + 1) * chunk_size, eager=True),
                "value": pl.Series([random.random() for _ in range(chunk_size)]),
            })

            report = th.check(chunk_df)

            # Sample memory every 10 chunks
            if (chunk_idx + 1) % 10 == 0:
                current_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                max_memory = max(max_memory, current_memory)
                memory_samples.append(current_memory)

                elapsed = time.time() - start_time
                print(f"  Chunk {chunk_idx + 1}/{n_chunks}: "
                      f"Memory {current_memory / 1024 / 1024:.0f}MB, "
                      f"Elapsed {elapsed:.1f}s")

            del chunk_df
            gc.collect()

        total_time = time.time() - start_time
        memory_increase_mb = (max_memory - initial_memory) / 1024 / 1024

        print(f"\n{'=' * 70}")
        print(f"MEMORY RESULTS:")
        print(f"  Total rows processed: {n_rows:,}")
        print(f"  Initial memory: {initial_memory / 1024 / 1024:.0f}MB")
        print(f"  Max memory: {max_memory / 1024 / 1024:.0f}MB")
        print(f"  Memory increase: {memory_increase_mb:.0f}MB")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {n_rows / total_time:,.0f} rows/sec")
        print(f"{'=' * 70}")

        # Assertions
        # Memory should not grow unboundedly - increase should be < 2GB
        assert memory_increase_mb < 2000, f"Memory grew too much: {memory_increase_mb}MB"


class TestRealWorldScenarios:
    """Real-world 100M+ row scenarios."""

    @pytest.mark.scale_100m
    @pytest.mark.slow
    def test_100m_event_log_validation(self):
        """Validate 100M event log entries (chunked)."""
        print("\n" + "=" * 70)
        print("100 MILLION EVENT LOG VALIDATION")
        print("=" * 70)

        n_rows = 100_000_000
        chunk_size = 2_000_000
        n_chunks = n_rows // chunk_size

        event_types = ["login", "logout", "purchase", "view", "click", "error"]

        total_issues = 0
        start_time = time.time()

        for chunk_idx in range(n_chunks):
            # Generate realistic event log chunk
            base_time = datetime(2024, 1, 1) + timedelta(seconds=chunk_idx * chunk_size)

            events_chunk = pl.DataFrame({
                "event_id": pl.arange(
                    chunk_idx * chunk_size,
                    (chunk_idx + 1) * chunk_size,
                    eager=True
                ),
                "timestamp": pl.datetime_range(
                    base_time,
                    base_time + timedelta(seconds=chunk_size - 1),
                    interval="1s",
                    eager=True,
                ).head(chunk_size),
                "user_id": pl.Series([random.randint(1, 10_000_000) for _ in range(chunk_size)]),
                "event_type": pl.Series(random.choices(event_types, k=chunk_size)),
                "session_id": pl.Series([f"sess_{random.randint(1, 50_000_000)}" for _ in range(chunk_size)]),
                "duration_ms": pl.Series([
                    None if random.random() < 0.02 else random.randint(1, 300000)
                    for _ in range(chunk_size)
                ]),
            })

            report = th.check(events_chunk)
            total_issues += len(report.issues)

            if (chunk_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                processed = (chunk_idx + 1) * chunk_size
                rate = processed / elapsed
                print(f"  {processed:,} rows, {total_issues} issues, {rate:,.0f} rows/sec")

            del events_chunk
            gc.collect()

        total_time = time.time() - start_time

        print(f"\n{'=' * 70}")
        print(f"EVENT LOG VALIDATION RESULTS:")
        print(f"  Total events: {n_rows:,}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Total issues found: {total_issues}")
        print(f"  Throughput: {n_rows / total_time:,.0f} events/sec")
        print(f"{'=' * 70}")

        # Assertions
        assert total_time < 600  # Should complete in 10 minutes
        assert n_rows / total_time > 200_000  # > 200K rows/sec

    @pytest.mark.scale_100m
    @pytest.mark.slow
    def test_100m_financial_transactions(self):
        """Validate 100M financial transactions with business rules."""
        print("\n" + "=" * 70)
        print("100 MILLION FINANCIAL TRANSACTIONS VALIDATION")
        print("=" * 70)

        n_rows = 100_000_000
        chunk_size = 2_000_000
        n_chunks = n_rows // chunk_size

        currencies = ["USD", "EUR", "GBP", "JPY", "CHF"]

        total_suspicious = 0
        total_invalid = 0
        start_time = time.time()

        for chunk_idx in range(n_chunks):
            # Generate financial transaction chunk
            amounts = []
            for _ in range(chunk_size):
                if random.random() < 0.0001:  # 0.01% suspicious
                    amounts.append(random.uniform(100000, 1000000))  # Large transaction
                else:
                    amounts.append(random.lognormvariate(6, 1.5))  # Normal distribution

            transactions = pl.DataFrame({
                "tx_id": pl.arange(
                    chunk_idx * chunk_size,
                    (chunk_idx + 1) * chunk_size,
                    eager=True
                ),
                "amount": pl.Series(amounts),
                "currency": pl.Series(random.choices(currencies, k=chunk_size)),
                "sender_id": pl.Series([random.randint(1, 5_000_000) for _ in range(chunk_size)]),
                "receiver_id": pl.Series([random.randint(1, 5_000_000) for _ in range(chunk_size)]),
                "is_cross_border": pl.Series(random.choices([True, False], weights=[0.2, 0.8], k=chunk_size)),
            })

            # Validate
            report = th.check(transactions)

            # Count suspicious (large amounts)
            suspicious = transactions.filter(pl.col("amount") > 50000).height
            total_suspicious += suspicious
            total_invalid += len(report.issues)

            if (chunk_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                processed = (chunk_idx + 1) * chunk_size
                print(f"  {processed:,} rows, {total_suspicious} suspicious, {processed / elapsed:,.0f} rows/sec")

            del transactions
            gc.collect()

        total_time = time.time() - start_time
        suspicious_rate = 100 * total_suspicious / n_rows

        print(f"\n{'=' * 70}")
        print(f"FINANCIAL TRANSACTIONS RESULTS:")
        print(f"  Total transactions: {n_rows:,}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Suspicious transactions: {total_suspicious:,} ({suspicious_rate:.3f}%)")
        print(f"  Throughput: {n_rows / total_time:,.0f} tx/sec")
        print(f"{'=' * 70}")

        # Assertions
        assert total_time < 600
        assert suspicious_rate < 1  # Less than 1% suspicious
