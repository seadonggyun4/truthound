"""Tests for auto schema caching system."""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from truthound.cache import (
    SchemaCache,
    get_data_fingerprint,
    get_or_learn_schema,
    get_source_key,
)
from truthound.schema import learn


class TestFingerprinting:
    """Tests for data fingerprinting functions."""

    def test_fingerprint_file(self, tmp_path: Path):
        """Test fingerprint generation for files."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("a,b\n1,2\n3,4")

        fp1 = get_data_fingerprint(str(test_file))
        fp2 = get_data_fingerprint(str(test_file))

        assert fp1 == fp2
        assert len(fp1) == 16  # SHA256 truncated to 16 chars

    def test_fingerprint_changes_on_modification(self, tmp_path: Path):
        """Test fingerprint changes when file is modified."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("a,b\n1,2\n3,4")
        fp1 = get_data_fingerprint(str(test_file))

        # Modify file
        test_file.write_text("a,b\n1,2\n3,4\n5,6")
        fp2 = get_data_fingerprint(str(test_file))

        assert fp1 != fp2

    def test_fingerprint_dict(self):
        """Test fingerprint generation for dict."""
        data = {"a": [1, 2, 3], "b": [4, 5, 6]}

        fp1 = get_data_fingerprint(data)
        fp2 = get_data_fingerprint(data)

        assert fp1 == fp2

    def test_fingerprint_dataframe(self):
        """Test fingerprint generation for DataFrame."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        fp1 = get_data_fingerprint(df)
        fp2 = get_data_fingerprint(df)

        assert fp1 == fp2

    def test_source_key_file(self, tmp_path: Path):
        """Test source key for files uses absolute path."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("a,b\n1,2")

        key = get_source_key(str(test_file))

        assert key == str(test_file.absolute())

    def test_source_key_dict(self):
        """Test source key for dicts."""
        data = {"col_a": [1, 2], "col_b": [3, 4]}

        key = get_source_key(data)

        assert "dict" in key
        assert "col_a" in key
        assert "col_b" in key


class TestSchemaCache:
    """Tests for SchemaCache class."""

    def test_cache_save_and_retrieve(self, tmp_path: Path):
        """Test saving and retrieving schema from cache."""
        cache = SchemaCache(cache_dir=tmp_path / ".truthound")
        data = {"value": [1, 2, 3, 4, 5]}

        # Learn and save schema
        schema = learn(data)
        saved_path = cache.save_schema(data, schema)

        assert saved_path.exists()
        assert saved_path.suffix == ".yaml"

        # Retrieve from cache
        retrieved = cache.get_schema(data)

        assert retrieved is not None
        assert len(retrieved.columns) == 1
        assert "value" in retrieved.columns

    def test_cache_returns_none_for_uncached(self, tmp_path: Path):
        """Test returns None for data not in cache."""
        cache = SchemaCache(cache_dir=tmp_path / ".truthound")
        data = {"value": [1, 2, 3]}

        result = cache.get_schema(data)

        assert result is None

    def test_cache_invalidates_on_data_change(self, tmp_path: Path):
        """Test cache invalidation when data changes."""
        cache = SchemaCache(cache_dir=tmp_path / ".truthound")

        # Save with initial data
        data1 = {"value": [1, 2, 3]}
        schema = learn(data1)
        cache.save_schema(data1, schema)

        # Different data with same keys but different size
        data2 = {"value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

        # Should not return cached schema (fingerprint differs)
        result = cache.get_schema(data2)

        assert result is None

    def test_cache_list_cached(self, tmp_path: Path):
        """Test listing cached schemas."""
        cache = SchemaCache(cache_dir=tmp_path / ".truthound")

        # Cache multiple schemas
        data1 = {"a": [1, 2, 3]}
        data2 = {"b": [4, 5, 6]}

        cache.save_schema(data1, learn(data1))
        cache.save_schema(data2, learn(data2))

        cached = cache.list_cached()

        assert len(cached) == 2

    def test_cache_clear(self, tmp_path: Path):
        """Test clearing cache."""
        cache = SchemaCache(cache_dir=tmp_path / ".truthound")
        data = {"value": [1, 2, 3]}

        cache.save_schema(data, learn(data))
        assert len(cache.list_cached()) == 1

        cache.clear()

        assert len(cache.list_cached()) == 0
        assert cache.get_schema(data) is None


class TestGetOrLearnSchema:
    """Tests for get_or_learn_schema function."""

    def test_learns_new_schema(self, tmp_path: Path, monkeypatch):
        """Test learning new schema when not cached."""
        # Use isolated cache directory
        cache_dir = tmp_path / ".truthound"
        monkeypatch.setattr(
            "truthound.cache.get_cache_dir",
            lambda base_path=None: cache_dir
        )
        # Reset global cache
        import truthound.cache
        truthound.cache._global_cache = None

        data = {"value": list(range(100))}

        schema, was_cached = get_or_learn_schema(data)

        assert was_cached is False
        assert len(schema.columns) == 1

    def test_returns_cached_schema(self, tmp_path: Path, monkeypatch):
        """Test returning cached schema on second call."""
        cache_dir = tmp_path / ".truthound"
        monkeypatch.setattr(
            "truthound.cache.get_cache_dir",
            lambda base_path=None: cache_dir
        )
        import truthound.cache
        truthound.cache._global_cache = None

        data = {"value": list(range(100))}

        # First call - learns
        schema1, was_cached1 = get_or_learn_schema(data)
        # Second call - retrieves from cache
        schema2, was_cached2 = get_or_learn_schema(data)

        assert was_cached1 is False
        assert was_cached2 is True
        # schema.columns is a dict keyed by column name
        assert list(schema1.columns.keys()) == list(schema2.columns.keys())

    def test_force_learn_bypasses_cache(self, tmp_path: Path, monkeypatch):
        """Test force_learn parameter."""
        cache_dir = tmp_path / ".truthound"
        monkeypatch.setattr(
            "truthound.cache.get_cache_dir",
            lambda base_path=None: cache_dir
        )
        import truthound.cache
        truthound.cache._global_cache = None

        data = {"value": list(range(100))}

        # Cache schema
        get_or_learn_schema(data)

        # Force re-learn
        schema, was_cached = get_or_learn_schema(data, force_learn=True)

        assert was_cached is False


class TestFileCaching:
    """Tests for file-based caching."""

    def test_cache_file_schema(self, tmp_path: Path, monkeypatch):
        """Test caching schema for file input."""
        cache_dir = tmp_path / ".truthound"
        monkeypatch.setattr(
            "truthound.cache.get_cache_dir",
            lambda base_path=None: cache_dir
        )
        import truthound.cache
        truthound.cache._global_cache = None

        # Create test file
        test_file = tmp_path / "data.csv"
        pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]}).write_csv(test_file)

        # Learn and cache
        schema1, cached1 = get_or_learn_schema(str(test_file))

        # Retrieve from cache
        schema2, cached2 = get_or_learn_schema(str(test_file))

        assert cached1 is False
        assert cached2 is True
        assert len(schema1.columns) == 2

    def test_cache_invalidates_on_file_change(self, tmp_path: Path, monkeypatch):
        """Test cache invalidates when file changes."""
        cache_dir = tmp_path / ".truthound"
        monkeypatch.setattr(
            "truthound.cache.get_cache_dir",
            lambda base_path=None: cache_dir
        )
        import truthound.cache
        truthound.cache._global_cache = None

        test_file = tmp_path / "data.csv"
        pl.DataFrame({"id": [1, 2, 3]}).write_csv(test_file)

        # Cache schema
        get_or_learn_schema(str(test_file))

        # Modify file
        pl.DataFrame({"id": [1, 2, 3, 4, 5]}).write_csv(test_file)

        # Should re-learn (cache invalidated)
        schema, cached = get_or_learn_schema(str(test_file))

        assert cached is False
