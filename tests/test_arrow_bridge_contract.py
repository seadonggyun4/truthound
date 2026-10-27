"""Actual Arrow/Dask conversions must honor selected bounds and policies."""

import dask.dataframe as dd
import pandas as pd
import polars as pl
import pytest

from truthound.execution.distributed.arrow_bridge import ArrowBridge, ArrowBridgeConfig


def test_configuration_is_captured_and_index_selection_reaches_generic_conversion():
    config = ArrowBridgeConfig(preserve_index=True)
    bridge = ArrowBridge(config)
    config.preserve_index = False
    frame = pd.DataFrame({"x": [1, 2]}, index=pd.Index([7, 9], name="row_id"))
    table = bridge.to_arrow(frame)
    assert table.column("row_id").to_pylist() == [7, 9]
    assert bridge.config.preserve_index is True


def test_null_error_and_memory_limits_apply_to_already_native_inputs():
    with pytest.raises(ValueError, match="null"):
        ArrowBridge(ArrowBridgeConfig(null_handling="error")).to_polars(
            pl.DataFrame({"x": [1, None]})
        )
    with pytest.raises(MemoryError):
        ArrowBridge(ArrowBridgeConfig(max_memory_bytes=1)).to_arrow(pl.DataFrame({"x": [1, 2]}))


def test_native_dask_chunks_obey_explicit_row_bound_and_preserve_index():
    frame = pd.DataFrame({"x": range(9)}, index=pd.Index(range(10, 19), name="row_id"))
    source = dd.from_pandas(frame, npartitions=2)
    bridge = ArrowBridge(ArrowBridgeConfig(preserve_index=True))
    chunks = list(bridge.dask_to_polars_chunked(source, chunk_size=2))
    assert all(len(chunk) <= 2 for chunk in chunks)
    assert pl.concat(chunks)["row_id"].to_list() == list(range(10, 19))


@pytest.mark.parametrize(
    "settings", [{"batch_size": 0}, {"max_memory_bytes": -1}, {"null_handling": "unknown"}]
)
def test_invalid_conversion_settings_fail_before_native_work(settings):
    with pytest.raises(ValueError):
        ArrowBridgeConfig(**settings)


def test_explicit_sentinels_and_temporal_casts_are_lossless_and_preserve_field_metadata():
    import pyarrow as pa

    table = pa.Table.from_arrays(
        [pa.array([1, None], type=pa.int64())],
        schema=pa.schema([pa.field("x", pa.int64(), metadata={b"unit": b"items"})]),
    )
    bridge = ArrowBridge(ArrowBridgeConfig(null_handling="sentinel", sentinel_values={"x": -1}))
    result = bridge.to_arrow(table)
    assert result.column("x").to_pylist() == [1, -1]
    assert result.schema.field("x").metadata == {b"unit": b"items"}
    with pytest.raises(ValueError, match="sentinel"):
        ArrowBridge(ArrowBridgeConfig(null_handling="sentinel")).to_arrow(table)
    temporal = pa.table({"time": pa.array([1001], type=pa.timestamp("ns"))})
    with pytest.raises(pa.ArrowInvalid):
        ArrowBridge().to_arrow(temporal)
    assert ArrowBridge(ArrowBridgeConfig(coerce_temporal=False)).to_arrow(temporal).equals(temporal)


def test_native_arrow_conversion_does_not_require_optional_pandas(monkeypatch):
    import sys

    import pyarrow as pa

    table = pa.table({"x": [1, 2]})
    monkeypatch.setitem(sys.modules, "pandas", None)
    assert ArrowBridge().to_arrow(table).equals(table)
