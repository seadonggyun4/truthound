# Test Coverage

Truthound maintains comprehensive test coverage to ensure reliability and correctness of all validation features.

---

## Test Summary

| Test Suite | Test Count | Status |
|------------|------------|--------|
| Unit Tests | 39 | Pass |
| Stress Tests | 53 | Pass |
| Extreme Stress Tests | 14 | Pass |
| Validator Tests (P0) | 32 | Pass |
| Validator Tests (P1) | 27 | Pass |
| Validator Tests (P2) | 27 | Pass |
| Drift Validator Tests | 52 | Pass |
| Anomaly Validator Tests | 31 | Pass |
| Multi-column Validator Tests | 43 | Pass |
| Query Validator Tests | 14 | Pass |
| Table Validator Tests | 21 | Pass |
| Geospatial Validator Tests | 26 | Pass |
| Business Rule Validator Tests | 22 | Pass |
| Localization Validator Tests | 28 | Pass |
| ML Feature Validator Tests | 23 | Pass |
| Profiling Validator Tests | 23 | Pass |
| Referential Validator Tests | 28 | Pass |
| Time Series Validator Tests | 30 | Pass |
| Privacy Validator Tests | 46 | Pass |
| Integration Tests | 138 | Pass |
| **Total** | **717** | **All Pass** |

---

## Test Categories

### Stress Tests (`test_stress.py`)

- Edge cases (empty data, single row/column)
- All Polars data types (Int8-Int64, Float32/64, String, Boolean, Date, Datetime, Duration, Categorical, List, Struct)
- Real-world patterns (high cardinality, sparse data, time series)
- Malicious inputs (SQL injection patterns, XSS, null bytes, Unicode)
- Memory pressure scenarios

### Extreme Stress Tests (`test_extreme_stress.py`)

- 10M row datasets
- Financial tick data simulation (stock/crypto)
- Mixed type columns
- High duplicate rates
- Wide datasets (100+ columns)
- Concurrent operations

---

## PII Detection Coverage

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

---

## Running Tests

```bash
# Run all tests
hatch run test

# Run specific test suite
pytest tests/test_stress.py
pytest tests/test_extreme_stress.py

# Run with coverage
pytest --cov=truthound tests/
```

---

[← Back to README](../README.md)
