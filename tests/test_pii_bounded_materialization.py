"""PII sampling semantics must not require full-frame materialization."""
import polars as pl
import pytest

import truthound as th
from truthound.scanners import scan_pii


def forbid_large_collections(monkeypatch):
    original = pl.LazyFrame.collect

    def bounded(frame, *args, **kwargs):
        result = original(frame, *args, **kwargs)
        assert result.height <= 1000, "PII materialized more than its sample bound"
        return result

    monkeypatch.setattr(pl.LazyFrame, "collect", bounded)


def test_public_scan_keeps_total_rows_without_full_frame(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.parquet"
    pl.DataFrame({"email": [None] * 1500 + ["person@example.com"] * 2000,
                  "numeric": list(range(3500))}).write_parquet(path, row_group_size=256)
    forbid_large_collections(monkeypatch)
    result = th.scan(str(path))
    assert result.row_count == 3500
    assert result.findings[0]["column"] == "email"
    assert result.findings[0]["count"] == 2000


def test_sampling_uses_first_non_null_values_not_first_rows(monkeypatch):
    frame = pl.DataFrame({"email": [None] * 1500 + ["person@example.com"] * 1000 + ["ordinary"] * 1000})
    forbid_large_collections(monkeypatch)
    findings = scan_pii(frame.lazy())
    assert len(findings) == 1
    assert findings[0]["count"] == 2000


@pytest.mark.parametrize("frame", [pl.DataFrame({"value": [None] * 2500}, schema={"value": pl.String}), pl.DataFrame({"value": list(range(2500))}), pl.DataFrame(schema={"value": pl.String})])
def test_empty_null_and_numeric_inputs_have_no_findings(frame, monkeypatch):
    forbid_large_collections(monkeypatch)
    assert scan_pii(frame.lazy()) == []


def test_parquet_reader_is_batched_and_matches_lazy_result(tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    frame = pl.DataFrame({"email": [None] * 1500 + ["person@example.com"] * 500 + ["ordinary"] * 500 + ["person@example.com"] * 2000})
    expected = scan_pii(frame.lazy())
    path = tmp_path / "synthetic.parquet"
    frame.write_parquet(path, row_group_size=200)
    original = pq.ParquetFile.iter_batches
    sizes = []

    def batches(file, *args, **kwargs):
        assert kwargs["batch_size"] == 1024
        assert kwargs["use_threads"] is False
        for batch in original(file, *args, **kwargs):
            sizes.append(batch.num_rows)
            yield batch

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", batches)
    result = th.scan(str(path))
    assert result.row_count == frame.height
    assert result.findings == expected
    assert len(sizes) > 1 and max(sizes) <= 1024


def test_parquet_read_error_is_not_a_clean_pii_result(tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    path = tmp_path / "synthetic.parquet"
    pl.DataFrame({"value": ["ordinary"]}).write_parquet(path)

    def failed(*args, **kwargs):
        raise OSError("synthetic reader failure")

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", failed)
    with pytest.raises(OSError, match="synthetic reader failure"):
        th.scan(str(path))
