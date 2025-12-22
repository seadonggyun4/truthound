"""Comprehensive stress tests for Truthound.

Tests cover:
1. Edge cases (empty data, single row, extreme values)
2. Large datasets (1M+ rows)
3. Various data types (datetime, nested, Korean text)
4. Real-world data patterns (public data, banking, user data)
5. Malicious inputs (SQL injection, special characters)
6. Memory and performance stress
"""

import gc
import random
import string
import time
from datetime import datetime, timedelta

import polars as pl
import pytest

import truthound as th


class TestEdgeCases:
    """Edge case tests - empty, minimal, and extreme data."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pl.DataFrame({"col": []})

        # Should not crash
        report = th.check(empty_df)
        assert report is not None

    def test_single_row(self):
        """Test handling of single row data."""
        single = pl.DataFrame({"id": [1], "name": ["test"]})

        report = th.check(single)
        assert report is not None

    def test_single_column(self):
        """Test handling of single column data."""
        single_col = pl.DataFrame({"only_col": list(range(100))})

        report = th.check(single_col)
        schema = th.learn(single_col)

        assert len(schema.columns) == 1

    def test_all_nulls(self):
        """Test column with all null values."""
        all_nulls = pl.DataFrame({
            "id": [1, 2, 3],
            "empty": [None, None, None],
        })

        report = th.check(all_nulls)
        assert report.has_issues
        # Should detect critical null issue
        null_issues = [i for i in report.issues if i.issue_type == "null"]
        assert any(i.severity.value == "critical" for i in null_issues)

    def test_extreme_numeric_values(self):
        """Test extreme numeric values."""
        extreme = pl.DataFrame({
            "tiny": [1e-300, 1e-299, 1e-298],
            "huge": [1e300, 1e299, 1e298],
            "mixed": [-1e100, 0, 1e100],
        })

        report = th.check(extreme)
        schema = th.learn(extreme)
        profile = th.profile(extreme)

        assert profile is not None
        assert schema is not None

    def test_integer_overflow_boundary(self):
        """Test values near integer overflow boundaries."""
        boundary = pl.DataFrame({
            "int64_max": [9223372036854775807, 9223372036854775806],
            "int64_min": [-9223372036854775808, -9223372036854775807],
        })

        report = th.check(boundary)
        schema = th.learn(boundary)

        assert schema is not None

    def test_empty_strings(self):
        """Test columns with empty strings."""
        empty_strings = pl.DataFrame({
            "name": ["", "", "valid", ""],
            "code": ["A", "", "", "B"],
        })

        report = th.check(empty_strings)
        profile = th.profile(empty_strings)

        assert profile is not None

    def test_whitespace_only_strings(self):
        """Test columns with whitespace-only strings."""
        whitespace = pl.DataFrame({
            "spaces": ["   ", "  ", "valid", "\t\t"],
            "mixed": [" ", "text", "  \n  ", "ok"],
        })

        report = th.check(whitespace)
        assert report is not None

    def test_unicode_edge_cases(self):
        """Test various Unicode edge cases."""
        unicode_df = pl.DataFrame({
            "emoji": ["😀", "🎉", "🚀", "💯"],
            "rtl": ["مرحبا", "שלום", "hello", "世界"],
            "combining": ["é", "é", "ñ", "ü"],  # Different Unicode normalizations
            "zero_width": ["a\u200bb", "c\u200dd", "normal", "text"],
        })

        report = th.check(unicode_df)
        schema = th.learn(unicode_df)

        assert schema is not None

    def test_very_long_strings(self):
        """Test handling of very long strings."""
        long_string = "x" * 100000
        long_strings = pl.DataFrame({
            "normal": ["short", "medium length", "a bit longer"],
            "very_long": [long_string, long_string[:50000], "short"],
        })

        report = th.check(long_strings)
        profile = th.profile(long_strings)

        assert profile is not None


class TestDataTypes:
    """Test various data types and combinations."""

    def test_datetime_types(self):
        """Test datetime handling."""
        now = datetime.now()
        dates = pl.DataFrame({
            "timestamp": [now, now - timedelta(days=1), now - timedelta(days=365)],
            "date_only": pl.Series([now.date(), (now - timedelta(days=1)).date(),
                                    (now - timedelta(days=365)).date()]),
        })

        report = th.check(dates)
        schema = th.learn(dates)

        assert schema is not None

    def test_boolean_type(self):
        """Test boolean columns."""
        bools = pl.DataFrame({
            "flag": [True, False, True, False, True],
            "all_true": [True, True, True, True, True],
            "all_false": [False, False, False, False, False],
        })

        report = th.check(bools)
        schema = th.learn(bools)

        assert schema is not None

    def test_mixed_numeric_types(self):
        """Test mixed numeric types."""
        mixed = pl.DataFrame({
            "integers": [1, 2, 3, 4, 5],
            "floats": [1.1, 2.2, 3.3, 4.4, 5.5],
            "mixed_precision": [1.0, 2.5, 3, 4.123456789, 5],
        })

        report = th.check(mixed)
        schema = th.learn(mixed)

        assert schema is not None

    def test_categorical_high_cardinality(self):
        """Test categorical with very high cardinality."""
        high_card = pl.DataFrame({
            "unique_ids": [f"id_{i}" for i in range(1000)],
            "values": list(range(1000)),
        })

        report = th.check(high_card)
        schema = th.learn(high_card)

        # Should not treat high cardinality as categorical
        assert schema.columns["unique_ids"].allowed_values is None

    def test_korean_text(self):
        """Test Korean text handling."""
        korean = pl.DataFrame({
            "name": ["홍길동", "김철수", "이영희", "박지민"],
            "address": ["서울특별시 강남구", "부산광역시 해운대구", "대구광역시 중구", "인천광역시 남동구"],
            "phone": ["010-1234-5678", "010-9876-5432", "010-1111-2222", "010-3333-4444"],
        })

        report = th.check(korean)
        schema = th.learn(korean)
        profile = th.profile(korean)

        assert schema is not None
        assert profile is not None

    def test_mixed_language(self):
        """Test mixed language content."""
        mixed_lang = pl.DataFrame({
            "content": ["Hello 世界", "안녕하세요 World", "Bonjour 你好", "こんにちは Hi"],
            "code": ["EN-KO", "KO-EN", "FR-CN", "JP-EN"],
        })

        report = th.check(mixed_lang)
        schema = th.learn(mixed_lang)

        assert schema is not None


class TestRealWorldPatterns:
    """Test real-world data patterns."""

    def test_public_data_format(self):
        """Test Korean public data format (공공데이터)."""
        public_data = pl.DataFrame({
            "관리번호": ["2024-001", "2024-002", "2024-003"],
            "시도": ["서울특별시", "부산광역시", "대구광역시"],
            "시군구": ["강남구", "해운대구", "중구"],
            "상호명": ["(주)테스트", "테스트상회", "테스트마트"],
            "업종코드": ["123456", "234567", "345678"],
            "인허가일자": ["2024-01-15", "2024-02-20", "2024-03-25"],
            "폐업일자": [None, None, "2024-06-30"],
            "좌표X": [127.0276, 129.0756, 128.6014],
            "좌표Y": [37.4979, 35.1628, 35.8714],
        })

        report = th.check(public_data)
        schema = th.learn(public_data)
        pii = th.scan(public_data)

        assert schema is not None
        # Should detect some patterns

    def test_banking_transaction_data(self):
        """Test banking/financial transaction patterns."""
        transactions = pl.DataFrame({
            "transaction_id": [f"TXN{i:010d}" for i in range(100)],
            "account_number": [f"123-456-{i:06d}" for i in range(100)],
            "amount": [random.uniform(-10000, 50000) for _ in range(100)],
            "currency": ["KRW"] * 80 + ["USD"] * 15 + ["EUR"] * 5,
            "timestamp": [datetime.now() - timedelta(hours=i) for i in range(100)],
            "merchant": [f"Merchant_{i % 20}" for i in range(100)],
            "category": random.choices(["식비", "교통", "쇼핑", "문화", "기타"], k=100),
            "is_fraud": [False] * 95 + [True] * 5,
        })

        report = th.check(transactions)
        schema = th.learn(transactions)
        pii = th.scan(transactions)

        assert schema is not None
        # Should detect account numbers as PII
        assert pii.has_pii

    def test_user_profile_data(self):
        """Test user profile data patterns."""
        users = pl.DataFrame({
            "user_id": list(range(1, 101)),
            "username": [f"user_{i}" for i in range(100)],
            "email": [f"user{i}@example.com" for i in range(100)],
            "phone": [f"010-{random.randint(1000,9999)}-{random.randint(1000,9999)}" for _ in range(100)],
            "age": [random.randint(18, 80) for _ in range(100)],
            "gender": random.choices(["M", "F", "Other"], k=100),
            "signup_date": [datetime.now() - timedelta(days=random.randint(0, 365)) for _ in range(100)],
            "last_login": [datetime.now() - timedelta(hours=random.randint(0, 720)) for _ in range(100)],
            "subscription": random.choices(["free", "basic", "premium"], k=100),
            "country": random.choices(["KR", "US", "JP", "CN"], weights=[70, 15, 10, 5], k=100),
        })

        report = th.check(users)
        schema = th.learn(users)
        pii = th.scan(users)

        assert pii.has_pii
        # Should detect email, phone as PII

    def test_ecommerce_order_data(self):
        """Test e-commerce order data."""
        orders = pl.DataFrame({
            "order_id": [f"ORD-{2024}-{i:06d}" for i in range(200)],
            "customer_id": [f"CUST-{random.randint(1, 50):05d}" for _ in range(200)],
            "product_id": [f"PROD-{random.randint(1, 100):04d}" for _ in range(200)],
            "quantity": [random.randint(1, 10) for _ in range(200)],
            "unit_price": [random.uniform(1000, 100000) for _ in range(200)],
            "discount_rate": [random.choice([0.0, 0.05, 0.1, 0.15, 0.2]) for _ in range(200)],
            "shipping_fee": [float(random.choice([0, 2500, 3000, 5000])) for _ in range(200)],
            "order_status": random.choices(["pending", "confirmed", "shipped", "delivered", "cancelled"], k=200),
            "payment_method": random.choices(["card", "bank", "kakao", "naver"], k=200),
        })

        report = th.check(orders)
        schema = th.learn(orders)

        assert schema is not None

    def test_log_data_pattern(self):
        """Test application log data patterns."""
        log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        logs = pl.DataFrame({
            "timestamp": [datetime.now() - timedelta(seconds=i) for i in range(500)],
            "level": random.choices(log_levels, weights=[10, 60, 20, 8, 2], k=500),
            "service": random.choices(["api", "worker", "scheduler", "db"], k=500),
            "message": [f"Log message {i}" for i in range(500)],
            "request_id": [f"req-{random.randint(10000, 99999)}" for _ in range(500)],
            "duration_ms": [random.uniform(0.1, 5000) for _ in range(500)],
            "status_code": random.choices([200, 201, 400, 401, 404, 500], weights=[70, 10, 5, 5, 5, 5], k=500),
        })

        report = th.check(logs)
        schema = th.learn(logs)

        assert schema is not None


class TestMaliciousInputs:
    """Test handling of potentially malicious inputs."""

    def test_sql_injection_strings(self):
        """Test SQL injection patterns in data."""
        sql_injection = pl.DataFrame({
            "user_input": [
                "'; DROP TABLE users; --",
                "1' OR '1'='1",
                "admin'--",
                "1; DELETE FROM orders",
                "normal input",
            ],
            "id": [1, 2, 3, 4, 5],
        })

        # Should not crash, just process as strings
        report = th.check(sql_injection)
        schema = th.learn(sql_injection)

        assert schema is not None
        assert len(schema.columns) == 2

    def test_xss_patterns(self):
        """Test XSS patterns in data."""
        xss_data = pl.DataFrame({
            "content": [
                "<script>alert('xss')</script>",
                "<img src=x onerror=alert(1)>",
                "javascript:alert(document.cookie)",
                "<svg onload=alert(1)>",
                "Normal text content",
            ],
        })

        report = th.check(xss_data)
        schema = th.learn(xss_data)

        assert schema is not None

    def test_path_traversal_strings(self):
        """Test path traversal patterns."""
        path_traversal = pl.DataFrame({
            "filename": [
                "../../../etc/passwd",
                "..\\..\\windows\\system32",
                "/etc/shadow",
                "normal_file.txt",
                "document.pdf",
            ],
        })

        report = th.check(path_traversal)
        assert report is not None

    def test_special_characters(self):
        """Test special and control characters."""
        special = pl.DataFrame({
            "with_null": ["hello\x00world", "test\x00", "normal"],
            "with_control": ["text\x01\x02\x03", "more\x07bell", "clean"],
            "with_escape": ["line1\nline2", "tab\there", "normal"],
        })

        report = th.check(special)
        schema = th.learn(special)

        assert schema is not None

    def test_extremely_nested_json_string(self):
        """Test extremely nested JSON-like strings."""
        nested = '{"a":' * 100 + '"value"' + '}' * 100
        json_strings = pl.DataFrame({
            "json_col": [nested, '{"simple": "json"}', '[]'],
        })

        report = th.check(json_strings)
        assert report is not None


class TestDriftDetection:
    """Comprehensive drift detection tests."""

    def test_gradual_drift(self):
        """Test detection of gradual distribution drift."""
        baseline = pl.DataFrame({
            "value": [random.gauss(100, 10) for _ in range(1000)],
        })

        # Slightly shifted
        current = pl.DataFrame({
            "value": [random.gauss(105, 10) for _ in range(1000)],
        })

        drift = th.compare(baseline, current)
        assert drift is not None
        # Should detect some drift

    def test_sudden_drift(self):
        """Test detection of sudden distribution change."""
        baseline = pl.DataFrame({
            "value": [random.gauss(100, 10) for _ in range(1000)],
        })

        # Completely different distribution
        current = pl.DataFrame({
            "value": [random.gauss(500, 50) for _ in range(1000)],
        })

        drift = th.compare(baseline, current)
        assert drift.has_drift
        assert drift.has_high_drift

    def test_categorical_drift(self):
        """Test categorical distribution drift."""
        baseline = pl.DataFrame({
            "category": random.choices(["A", "B", "C"], weights=[50, 30, 20], k=1000),
        })

        # Different distribution
        current = pl.DataFrame({
            "category": random.choices(["A", "B", "C", "D"], weights=[20, 20, 30, 30], k=1000),
        })

        drift = th.compare(baseline, current)
        assert drift.has_drift

    def test_new_category_detection(self):
        """Test detection of new categories."""
        baseline = pl.DataFrame({
            "status": ["active", "inactive", "pending"] * 100,
        })

        current = pl.DataFrame({
            "status": ["active", "inactive", "pending", "suspended", "deleted"] * 60,
        })

        drift = th.compare(baseline, current)
        assert drift.has_drift

    def test_multi_column_drift(self):
        """Test drift across multiple columns."""
        baseline = pl.DataFrame({
            "numeric": [random.gauss(0, 1) for _ in range(1000)],
            "category": random.choices(["X", "Y", "Z"], k=1000),
            "stable": list(range(1000)),
        })

        current = pl.DataFrame({
            "numeric": [random.gauss(2, 1) for _ in range(1000)],  # Shifted
            "category": random.choices(["X", "Y", "W"], k=1000),  # New category
            "stable": list(range(1000)),  # No change
        })

        drift = th.compare(baseline, current)
        drifted = drift.get_drifted_columns()

        assert "numeric" in drifted
        assert "stable" not in drifted


class TestLargeDatasets:
    """Performance tests with large datasets."""

    def test_million_rows(self):
        """Test with 1 million rows."""
        start = time.time()

        large_df = pl.DataFrame({
            "id": list(range(1_000_000)),
            "value": [random.random() for _ in range(1_000_000)],
            "category": random.choices(["A", "B", "C", "D", "E"], k=1_000_000),
        })

        creation_time = time.time() - start
        print(f"\nData creation: {creation_time:.2f}s")

        # Test check
        start = time.time()
        report = th.check(large_df)
        check_time = time.time() - start
        print(f"th.check(): {check_time:.2f}s")

        # Test profile
        start = time.time()
        profile = th.profile(large_df)
        profile_time = time.time() - start
        print(f"th.profile(): {profile_time:.2f}s")

        # Test learn
        start = time.time()
        schema = th.learn(large_df)
        learn_time = time.time() - start
        print(f"th.learn(): {learn_time:.2f}s")

        assert report is not None
        assert profile is not None
        assert schema is not None

        # Performance assertions (should complete in reasonable time)
        assert check_time < 30  # 30 seconds max
        assert profile_time < 30
        assert learn_time < 30

    def test_many_columns(self):
        """Test with many columns (100+)."""
        data = {f"col_{i}": list(range(10000)) for i in range(100)}
        wide_df = pl.DataFrame(data)

        start = time.time()
        report = th.check(wide_df)
        elapsed = time.time() - start

        print(f"\n100 columns check: {elapsed:.2f}s")

        assert report is not None
        assert elapsed < 60

    def test_drift_with_sampling(self):
        """Test drift detection with sampling on large data."""
        baseline = pl.DataFrame({
            "value": [random.gauss(0, 1) for _ in range(1_000_000)],
        })

        current = pl.DataFrame({
            "value": [random.gauss(1, 1) for _ in range(1_000_000)],
        })

        start = time.time()
        drift = th.compare(baseline, current, sample_size=10000)
        elapsed = time.time() - start

        print(f"\nDrift with sampling: {elapsed:.2f}s")

        assert drift.has_drift
        assert elapsed < 10  # Should be fast with sampling


class TestMemoryStress:
    """Memory stress tests."""

    def test_memory_cleanup(self):
        """Test that memory is properly cleaned up."""
        gc.collect()

        for i in range(5):
            # Create large dataframe
            df = pl.DataFrame({
                "data": list(range(500_000)),
            })

            # Process it
            th.check(df)
            th.profile(df)

            # Delete reference
            del df
            gc.collect()

        # If we got here without OOM, test passes
        assert True

    def test_repeated_operations(self):
        """Test repeated operations don't leak memory."""
        df = pl.DataFrame({
            "value": list(range(10000)),
            "category": random.choices(["A", "B", "C"], k=10000),
        })

        for i in range(20):
            th.check(df)
            th.profile(df)
            th.learn(df)

            if i % 5 == 0:
                gc.collect()

        assert True


class TestCLIIntegration:
    """Test CLI command integration."""

    def test_all_cli_commands_exist(self):
        """Verify all CLI commands are accessible."""
        from truthound.cli import app

        # Get all registered commands
        commands = list(app.registered_commands)
        command_names = [cmd.name for cmd in commands]

        expected = ["learn", "check", "scan", "mask", "profile", "compare"]
        for cmd in expected:
            assert cmd in command_names, f"Missing CLI command: {cmd}"


class TestSchemaValidation:
    """Schema validation stress tests."""

    def test_schema_with_all_violations(self):
        """Test schema detecting multiple violation types at once."""
        # Learn from baseline
        baseline = pl.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
            "score": [85, 90, 78, 92, 88],
            "status": ["active", "active", "inactive", "active", "inactive"],
        })
        schema = th.learn(baseline)

        # Create data with many violations
        bad_data = pl.DataFrame({
            "id": [1, 2, 3, None, 5],  # Unexpected null
            "name": ["A", "B", "C", "D", "E"],  # Shorter strings
            "score": [85, 150, -10, 92, 88],  # Out of range
            "status": ["active", "deleted", "inactive", "new", "inactive"],  # Invalid values
            "extra": [1, 2, 3, 4, 5],  # Extra column
        })

        report = th.check(bad_data, schema=schema)

        # Should detect multiple issues
        assert report.has_issues
        assert len(report.issues) >= 3

    def test_schema_evolution(self):
        """Test schema handling as data evolves."""
        # Initial schema
        v1 = pl.DataFrame({
            "id": [1, 2, 3],
            "value": [10, 20, 30],
        })
        schema_v1 = th.learn(v1)

        # Data evolved - new column added
        v2 = pl.DataFrame({
            "id": [4, 5, 6],
            "value": [40, 50, 60],
            "new_field": ["a", "b", "c"],
        })

        report = th.check(v2, schema=schema_v1)

        # Should detect extra column
        extra_issues = [i for i in report.issues if i.issue_type == "extra_column"]
        assert len(extra_issues) > 0
