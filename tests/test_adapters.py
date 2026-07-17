"""Tests for input adapters."""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from truthound.adapters import to_lazyframe


class TestToLazyframe:
    """Tests for the to_lazyframe function."""

    def test_from_dict(self):
        """Test conversion from dictionary."""
        data = {"a": [1, 2, 3], "b": ["x", "y", "z"]}
        lf = to_lazyframe(data)

        assert isinstance(lf, pl.LazyFrame)
        df = lf.collect()
        assert len(df) == 3
        assert df.columns == ["a", "b"]

    def test_from_polars_dataframe(self):
        """Test conversion from Polars DataFrame."""
        df = pl.DataFrame({"col": [1, 2, 3]})
        lf = to_lazyframe(df)

        assert isinstance(lf, pl.LazyFrame)
        assert lf.collect().equals(df)

    def test_from_polars_lazyframe(self):
        """Test passthrough of Polars LazyFrame."""
        original_lf = pl.DataFrame({"col": [1, 2, 3]}).lazy()
        lf = to_lazyframe(original_lf)

        assert isinstance(lf, pl.LazyFrame)
        assert lf is original_lf

    def test_from_csv_file(self):
        """Test loading from CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b\n1,x\n2,y\n3,z\n")
            f.flush()

            lf = to_lazyframe(f.name)
            assert isinstance(lf, pl.LazyFrame)

            df = lf.collect()
            assert len(df) == 3
            assert df.columns == ["a", "b"]

            Path(f.name).unlink()

    def test_from_json_file(self):
        """Test loading from JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]')
            f.flush()

            lf = to_lazyframe(f.name)
            assert isinstance(lf, pl.LazyFrame)

            df = lf.collect()
            assert len(df) == 2

            Path(f.name).unlink()

    def test_from_top_level_json_object(self):
        """A complete JSON object is represented as one record."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(
                '{"asset_refs": [{"id": "asset-1"}], '
                '"metadata": {"branch": "draft"}, "active": true}'
            )
            f.flush()

            df = to_lazyframe(f.name).collect()

            assert df.height == 1
            assert df["active"].to_list() == [True]
            assert df["metadata"].to_list() == [{"branch": "draft"}]
            assert df["asset_refs"].to_list() == [[{"id": "asset-1"}]]

            Path(f.name).unlink()

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            ("42", 42),
            ('"ready"', "ready"),
            ("true", True),
            ("null", None),
        ],
    )
    def test_from_top_level_json_scalar(self, payload, expected):
        """Legal scalar JSON documents remain inspectable as one record."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(payload)
            f.flush()

            df = to_lazyframe(f.name).collect()

            assert df.height == 1
            assert df.columns == ["value"]
            assert df["value"].to_list() == [expected]

            Path(f.name).unlink()

    def test_from_ndjson_file_uses_line_delimited_contract(self):
        """NDJSON remains a distinct line-delimited input format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ndjson", delete=False) as f:
            f.write('{"id": 1}\n{"id": 2}\n')
            f.flush()

            df = to_lazyframe(f.name).collect()

            assert df["id"].to_list() == [1, 2]

            Path(f.name).unlink()

    def test_malformed_json_document_is_rejected(self):
        """A malformed .json file must not be reinterpreted as NDJSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"id": 1}\n{"id": 2}\n')
            f.flush()

            with pytest.raises(ValueError):
                to_lazyframe(f.name)

            Path(f.name).unlink()

    def test_from_parquet_file(self):
        """Test loading from Parquet file."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
            df.write_parquet(f.name)

            lf = to_lazyframe(f.name)
            assert isinstance(lf, pl.LazyFrame)

            loaded_df = lf.collect()
            assert loaded_df.equals(df)

            Path(f.name).unlink()

    def test_file_not_found(self):
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError):
            to_lazyframe("/nonexistent/file.csv")

    def test_unsupported_extension(self):
        """Test error handling for unsupported file extension."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"test")
            f.flush()

            with pytest.raises(ValueError, match="Unsupported file extension"):
                to_lazyframe(f.name)

            Path(f.name).unlink()

    def test_unsupported_type(self):
        """Test error handling for unsupported input type."""
        with pytest.raises(ValueError, match="Unsupported input type"):
            to_lazyframe(12345)


class TestPandasIntegration:
    """Tests for pandas DataFrame integration."""

    def test_from_pandas_dataframe(self):
        """Test conversion from pandas DataFrame."""
        pytest.importorskip("pandas")
        import pandas as pd

        pdf = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        lf = to_lazyframe(pdf)

        assert isinstance(lf, pl.LazyFrame)
        df = lf.collect()
        assert len(df) == 3
        assert df.columns == ["a", "b"]
