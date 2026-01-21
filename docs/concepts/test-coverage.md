# Test Coverage

This document provides a comprehensive overview of Truthound's test suite, coverage metrics, and quality assurance practices.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Test Summary](#2-test-summary)
3. [Test Categories](#3-test-categories)
4. [Validator Coverage](#4-validator-coverage)
5. [PII Detection Coverage](#5-pii-detection-coverage)
6. [Performance Benchmarks](#6-performance-benchmarks)
7. [Running Tests](#7-running-tests)

---

## 1. Overview

Truthound maintains comprehensive test coverage to ensure reliability and correctness of all validation features.

### Test Statistics

| Metric | Value |
|--------|-------|
| **Total Tests** | 8,260+ |
| **Test Files** | 340+ |
| **Passing Rate** | 100% |
| **Coverage Target** | 90%+ |

---

## 2. Test Summary

| Test Suite | Description | Status |
|------------|-------------|--------|
| Unit Tests | Core functionality | Pass |
| Stress Tests | Edge cases, data types | Pass |
| Extreme Stress Tests | 10M+ rows, concurrency | Pass |
| Validator Tests | 28 categories, 400+ validators | Pass |
| Stores Tests | Filesystem, S3, GCS, Azure, DB | Pass |
| Reporters Tests | Console, JSON, HTML, Markdown, JUnit | Pass |
| Data Sources Tests | Polars, Pandas, Spark, SQL | Pass |
| Checkpoint Tests | CI/CD, notifications, escalation | Pass |
| Profiler Tests | Auto-profiling, schema evolution | Pass |
| Data Docs Tests | HTML generation, themes, PDF | Pass |
| Plugin Tests | Security, lifecycle, hot reload | Pass |
| ML Module Tests | Anomaly, drift, rule learning | Pass |
| Lineage Tests | Graph, tracker, impact analysis | Pass |
| Realtime Tests | Streaming, incremental validation | Pass |
| Integration Tests | End-to-end workflows | Pass |
| **Total** | **8,260+ test functions** | **All Pass** |

> **Note**: Test counts are dynamic. Run `grep -r "def test_" tests/ | wc -l` for current count.

---

## 3. Test Categories

### 3.1 Unit Tests

Core functionality tests covering individual functions and classes.

```
tests/
├── test_inputs.py           # Input adapter tests
├── test_schema.py           # Schema inference tests
├── test_fingerprint.py      # Data fingerprint tests
├── test_types.py            # Type system tests
└── test_utils.py            # Utility function tests
```

### 3.2 Stress Tests (`test_stress.py`)

Edge case and robustness testing:

- Edge cases (empty data, single row/column)
- All Polars data types (Int8-Int64, Float32/64, String, Boolean, Date, Datetime, Duration, Categorical, List, Struct)
- Real-world patterns (high cardinality, sparse data, time series)
- Malicious inputs (SQL injection patterns, XSS, null bytes, Unicode)
- Memory pressure scenarios

### 3.3 Extreme Stress Tests (`test_extreme_stress.py`)

Large-scale and concurrent operation testing:

- 10M row datasets
- Financial tick data simulation (stock/crypto)
- Mixed type columns
- High duplicate rates
- Wide datasets (100+ columns)
- Concurrent operations

### 3.4 Infrastructure Tests

Tests for storage backends, reporters, and data sources:

```
tests/
├── stores/
│   ├── test_filesystem_store.py
│   ├── test_memory_store.py
│   ├── test_s3_store.py
│   ├── test_gcs_store.py
│   └── test_database_store.py
├── reporters/
│   ├── test_json_reporter.py
│   ├── test_console_reporter.py
│   ├── test_html_reporter.py
│   └── test_markdown_reporter.py
├── datasources/
│   ├── test_polars_source.py
│   ├── test_pandas_source.py
│   ├── test_file_source.py
│   └── test_sql_sources.py
└── mocks/
    ├── cloud_mocks.py
    ├── database_mocks.py
    └── reporter_mocks.py
```

### 3.5 Integration Tests

End-to-end tests covering complete validation workflows:

```
tests/
├── test_check_api.py          # th.check() integration
├── test_compare_api.py        # th.compare() integration
├── test_scan_api.py           # th.scan() integration
├── test_profile_api.py        # th.profile() integration
├── test_cli.py                # CLI integration
└── e2e/
    ├── test_full_pipeline.py
    ├── test_cicd_workflow.py
    └── test_checkpoint.py
```

---

## 4. Validator Coverage

### 4.1 Coverage by Category

| Category | Validators | Coverage |
|----------|------------|----------|
| Schema | 15 | 100% |
| Completeness | 12 | 100% |
| Uniqueness | 17 | 100% |
| Distribution | 15 | 100% |
| String | 19 | 100% |
| Datetime | 10 | 100% |
| Aggregate | 8 | 100% |
| Cross-table | 5 | 100% |
| Multi-column | 21 | 100% |
| Query | 20 | 100% |
| Table | 18 | 100% |
| Geospatial | 13 | 100% |
| Drift | 14 | 100% |
| Anomaly | 18 | 100% |
| Privacy | 16 | 100% |
| Business Rule | 8 | 100% |
| Localization | 9 | 100% |
| ML Feature | 5 | 100% |
| Profiling | 7 | 100% |
| Referential | 14 | 100% |
| Time Series | 14 | 100% |
| Streaming | 12 | 100% |
| Memory | 8 | 100% |
| Optimization | 15 | 100% |
| SDK | 80 | 100% |
| Security | 3 | 100% |
| i18n | 3 | 100% |
| Timeout | - | 100% |

> **Note**: Validator counts include base classes and mixins. Run `grep -r "class.*Validator" src/truthound/validators/ | wc -l` for current count.

### 4.2 Test Structure

Each validator category follows a consistent test structure:

```python
import pytest
import polars as pl
from truthound.validators.completeness import NullCheckValidator

class TestNullCheckValidator:
    """Tests for NullCheckValidator."""

    def test_pass_no_nulls(self):
        """Test passes when no nulls present."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        validator = NullCheckValidator(column="a")
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_fail_with_nulls(self):
        """Test fails when nulls present."""
        df = pl.DataFrame({"a": [1, None, 3]})
        validator = NullCheckValidator(column="a")
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].column == "a"

    def test_threshold_respected(self):
        """Test threshold is respected."""
        df = pl.DataFrame({"a": [1, None, 3, 4, 5]})
        validator = NullCheckValidator(column="a", max_null_ratio=0.25)
        issues = validator.validate(df.lazy())
        assert len(issues) == 0  # 20% nulls < 25% threshold
```

---

## 5. PII Detection Coverage

### 5.1 Pattern Detection Accuracy

| PII Type | Pattern | Confidence |
|----------|---------|------------|
| Email | RFC 5322 compliant | 95% |
| US SSN | `XXX-XX-XXXX` | 98% |
| Phone (International) | ITU-T E.164 | 90% |
| Credit Card | Luhn algorithm validated | 85% |
| Korean RRN | `XXXXXX-XXXXXXX` | 98% |
| Korean Phone | `0XX-XXXX-XXXX` | 90% |
| Korean Bank Account | Bank-specific formats | 80% |
| Korean Passport | `MXXXXXXXX` | 85% |
| Japanese My Number | `XXXX-XXXX-XXXX` | 95% |
| Chinese ID | 18-digit format | 95% |

### 5.2 Regulation Coverage

| Regulation | Patterns | Tests |
|------------|----------|-------|
| GDPR (EU) | 8 | 15 |
| CCPA (California) | 6 | 12 |
| LGPD (Brazil) | 5 | 10 |
| PIPEDA (Canada) | 4 | 8 |
| APPI (Japan) | 3 | 6 |

---

## 6. Performance Benchmarks

### 6.1 Throughput Targets

| Operation | Dataset Size | Target Time | Throughput |
|-----------|--------------|-------------|------------|
| `th.check()` | 10M rows | < 5s | 2.83M rows/sec |
| `th.profile()` | 10M rows | < 0.2s | 66.7M rows/sec |
| `th.learn()` | 10M rows | < 0.3s | 37.0M rows/sec |
| `th.compare()` (sampled) | 5M rows | < 1s | N/A |

### 6.2 Benchmark Tests

```python
class TestPerformance:
    """Performance benchmark tests."""

    @pytest.mark.benchmark
    def test_check_10m_rows(self, large_df):
        """Benchmark th.check() on 10M rows."""
        start = time.time()
        report = th.check(large_df)
        elapsed = time.time() - start

        assert elapsed < 5.0
        assert report.throughput > 2_000_000

    @pytest.mark.benchmark
    def test_drift_5m_rows_sampled(self, baseline_5m, current_5m):
        """Benchmark drift detection with sampling."""
        start = time.time()
        drift = th.compare(baseline_5m, current_5m, sample_size=10000)
        elapsed = time.time() - start

        assert elapsed < 1.0  # 92x speedup with sampling
```

---

## 7. Running Tests

### 7.1 Full Test Suite

```bash
# Run all tests
hatch run test

# Or with pytest directly
pytest tests/

# With coverage
pytest tests/ --cov=truthound --cov-report=html
```

### 7.2 Specific Test Categories

```bash
# Run validator tests
pytest tests/validators/

# Run specific validator category
pytest tests/validators/test_drift_validators.py

# Run stress tests
pytest tests/test_stress.py
pytest tests/test_extreme_stress.py

# Run integration tests
pytest tests/e2e/

# Run performance benchmarks
pytest tests/benchmark/ -m benchmark
```

### 7.3 Test Markers

```bash
# Run only fast tests
pytest -m "not slow"

# Run only benchmark tests
pytest -m benchmark

# Skip optional dependency tests
pytest -m "not optional"
```

### 7.4 Coverage Report

```bash
# Generate HTML coverage report
pytest --cov=truthound --cov-report=html

# View report
open htmlcov/index.html

# Generate XML for CI
pytest --cov=truthound --cov-report=xml
```

### 7.5 CI Configuration

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.11', '3.12']

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install hatch
          hatch env create

      - name: Run tests
        run: hatch run test

      - name: Upload coverage
        uses: codecov/codecov-action@v4
```

---

## See Also

- [Architecture Overview](ARCHITECTURE.md) — System design
- [API Reference](API_REFERENCE.md) — Complete API documentation
- [Examples](EXAMPLES.md) — Usage examples
