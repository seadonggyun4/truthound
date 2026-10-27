"""Native adaptive/chunked codecs selected by an approved fixed factory alias."""

import math

from truthound.mcp.governance_worker import selected
from truthound.mcp.policy import MCPError


def number(value):
    if type(value) not in {int, float} or not math.isfinite(value):
        raise MCPError("unsupported_value")
    return value


def execute_codec(owner, args, budget):
    from truthound.stores.compression.adaptive import DataCharacteristics, DataType
    from truthound.stores.compression.base import CompressionAlgorithm

    owner.guard()
    kind, command = args["codec_kind"], args["command"]
    data = selected(owner.payloads, args["payload_id"])
    codec = selected(getattr(owner.runtime, kind), args["codec_id"])
    if command == "codec_analyze":
        analysis = codec.analyze(data)
        if not isinstance(analysis, DataCharacteristics) or not isinstance(
            analysis.detected_type, DataType
        ):
            raise MCPError("unsupported_value")
        owner.guard()
        algorithms = {value.value for value in CompressionAlgorithm}
        return {
            "analysis": {
                "size": number(analysis.size),
                "entropy": number(analysis.entropy),
                "repetition_score": number(analysis.repetition_score),
                "pattern_score": number(analysis.pattern_score),
                "compressibility_estimate": number(analysis.compressibility_estimate),
                "detected_type": analysis.detected_type.name,
                "sample_ratios": {
                    key: number(value)
                    for key, value in analysis.sample_compression_ratios.items()
                    if key in algorithms
                },
                "sample_errors": [key for key in analysis.sample_errors if key in algorithms],
                "analysis_budget_exhausted": analysis.analysis_budget_exhausted is True,
            }
        }
    if command == "codec_recommend":
        profiles = codec.get_recommendation(data)
        owner.guard()
        if any(not isinstance(value.algorithm, CompressionAlgorithm) for value in profiles):
            raise MCPError("unsupported_value")
        return {
            "profiles": [
                {
                    "algorithm": value.algorithm.value,
                    "score": number(value.score),
                    "expected_ratio": number(value.expected_ratio),
                    "expected_speed_mbps": number(value.expected_speed_mbps),
                }
                for value in profiles
            ]
        }
    maximum = min(budget, 16 * 1024 * 1024) - sum(map(len, owner.payloads.values()))
    if maximum < 0:
        raise MCPError("resource_exhausted")
    if command == "codec_range":
        result = (
            codec.decompress_range(data, args["start"], args["end"], max_output_bytes=maximum)
            if kind == "chunked_compression"
            else codec.decrypt_with_header(
                data, start=args["start"], end=args["end"], max_output_bytes=maximum
            )
        )
    elif kind == "adaptive":
        result = (
            codec.compress(data)
            if command == "codec_process"
            else codec.decompress(data, max_output_bytes=maximum)
        )
    elif kind == "chunked_compression":
        result = (
            codec.compress_with_header(data)
            if command == "codec_process"
            else codec.decompress(data, max_output_bytes=maximum)
        )
    else:
        result = (
            codec.encrypt_with_header(data)
            if command == "codec_process"
            else codec.decrypt_with_header(data, max_output_bytes=maximum)
        )
    owner.guard()
    return owner.retain(result, budget)
