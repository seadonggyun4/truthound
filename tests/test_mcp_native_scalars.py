"""Native numerical results keep exact values across optional NumPy scalar types."""

import math

import numpy as np
import pytest

from truthound.mcp.encoding import encode
from truthound.mcp.policy import MCPError
from truthound.mcp.tables import decode

pytestmark = pytest.mark.contract


def test_native_numpy_counts_float_stats_and_binary_values_roundtrip_without_repr():
    values = [
        np.int64(-(2**63)),
        np.uint64(2**64 - 1),
        np.bool_(True),
        np.float32(0.1),
        np.float64(float("inf")),
        np.bytes_(b"\0\xff"),
        np.str_("local"),
    ]
    for value in values:
        native = value.item()
        assert encode(value) == encode(native)
        assert decode(encode(value)) == native
    assert math.isnan(decode(encode(np.float32(float("nan")))))
    with pytest.raises(MCPError, match="unsupported_value"):
        encode(np.array([object()], dtype=object))
    with pytest.raises(MCPError, match="unsupported_value"):
        encode(np.complex128(1 + 2j))
