"""Type metadata must not smuggle Enum values through summary-only responses."""

import asyncio
import json
import sys
from types import SimpleNamespace

import polars as pl
import pytest

pytest.importorskip("mcp")
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from truthound.mcp.projection import project

pytestmark = pytest.mark.fault
MARKER = "synthetic_private_category_9f31"


@pytest.mark.parametrize(
    ("dtype", "family"),
    [
        (f"Enum(categories=['{MARKER}'])", "Enum"),
        (f"List(Enum(categories=['{MARKER}']))", "List"),
        (f"Struct({{'nested': Enum(categories=['{MARKER}'])}})", "Struct"),
        ("untrusted-private-type", "Unknown"),
        ("Int64", "Int64"),
        (f"Datetime(time_unit='us', time_zone='{MARKER}')", "Datetime"),
    ],
)
def test_profile_dtype_is_only_an_allowlisted_family(dtype, family):
    result = SimpleNamespace(
        row_count=1,
        column_count=1,
        columns=[dict(name="kind", dtype=dtype, null_pct=0, unique_pct=100)],
    )
    data, truncated = project("profile", result, 100)
    assert not truncated
    assert data["columns"][0]["dtype"] == family
    assert MARKER not in json.dumps(data)


def test_stdio_enum_parquet_profile_does_not_return_category_values(tmp_path):
    source = tmp_path / "input.parquet"
    pl.DataFrame({"kind": pl.Series([MARKER], dtype=pl.Enum([MARKER]))}).write_parquet(source)

    async def scenario():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-I", "-m", "truthound.mcp", "serve", "--allow-dir", str(tmp_path)],
        )
        async with stdio_client(params) as (read, write), ClientSession(read, write) as client:
            await client.initialize()
            result = await client.call_tool("profile_file", {"path": str(source)})
            assert not result.isError
            assert result.structuredContent["data"]["columns"][0]["dtype"] == "Enum"
            assert MARKER not in result.model_dump_json()

    asyncio.run(scenario())
