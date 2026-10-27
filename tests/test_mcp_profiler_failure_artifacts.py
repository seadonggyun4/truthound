"""A failed native profiling component must not become a successful resource receipt."""

import asyncio

import pytest

from tests.test_mcp_sources import call, read
from truthound.mcp.policy import MCPPolicy
from truthound.mcp.service import MCPService

pytestmark = pytest.mark.fault


@pytest.mark.parametrize("operation", ["profilers_profile", "profilers_generate_rules"])
def test_native_partial_artifact_is_retained_as_failed_and_never_generates_rules(
    tmp_path, monkeypatch, operation
):
    from truthound.mcp import session_worker
    from truthound.profiler.column_profiler import BasicStatsAnalyzer
    from truthound.profiler.generators.schema_rules import SchemaRuleGenerator

    service = MCPService(MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True))
    try:
        asset = call(
            service,
            "assets_inline",
            columns=[{"name": "x", "dtype": {"kind": "Int64"}, "values": [1, None]}],
        )["resource_id"]
        native = service._execute_request
        diagnostics = []

        async def execute(request, directory):
            # Fault-inject native execution in-process; exercise the real worker and parent protocol.
            if request["operation"].startswith("profilers_"):
                try:
                    return session_worker.execute(request)
                except Exception as error:
                    import traceback

                    diagnostics.append(
                        (
                            type(error).__name__,
                            [
                                (frame.name, frame.lineno)
                                for frame in traceback.extract_tb(error.__traceback__)
                            ],
                        )
                    )
                    raise
            return await native(request, directory)

        def failed(*args, **kwargs):
            raise RuntimeError("synthetic-private-profiler-query")

        def forbidden(*args, **kwargs):
            raise AssertionError("Generator must not consume an incomplete profile")

        monkeypatch.setattr(service, "_execute_request", execute)
        monkeypatch.setattr(BasicStatsAnalyzer, "analyze", failed)
        monkeypatch.setattr(SchemaRuleGenerator, "generate", forbidden)
        arguments = {"asset_id": asset, "column_analyzers": [{"component": "BasicStatsAnalyzer"}]}
        if operation == "profilers_generate_rules":
            arguments["generator"] = {"component": "SchemaRuleGenerator"}
        response = asyncio.run(service.call_operation_async(operation, arguments))
        assert response["execution_status"] == "failed" and response["issues"] == [
            {"code": "native_profile_incomplete"}
        ], diagnostics
        assert response["data"]["media_type"] == "application/vnd.truthound.advanced-profile+json"
        identity = response["data"]["resource_id"]
        profile = read(service, identity)
        assert (
            profile["complete"] is False
            and profile["columns"][0]["execution_issues"][0]["error_type"] == "RuntimeError"
        )
        assert "synthetic-private-profiler-query" not in str(profile)
        call(service, "resources_release", resource_id=asset)
        assert asyncio.run(
            service.call_operation_async("artifacts_read", {"resource_id": identity})
        )["issues"] == [{"code": "invalid_handle"}]
    finally:
        service.close()


@pytest.mark.parametrize("operation", ["profilers_profile", "profilers_generate_rules"])
def test_actual_native_sampled_profile_metadata_matches_retained_result(tmp_path, operation):
    service = MCPService(MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True))
    try:
        asset = call(
            service,
            "assets_inline",
            columns=[{"name": "x", "dtype": {"kind": "Int64"}, "values": list(range(10))}],
        )["resource_id"]
        arguments = {"asset_id": asset, "config": {"sample_size": 2, "random_seed": 42}}
        if operation == "profilers_generate_rules":
            arguments["generator"] = {"component": "SchemaRuleGenerator"}
        info = call(service, operation, **arguments)
        assert info["row_count"] == 2 and info["column_count"] == 1
        if operation == "profilers_profile":
            assert read(service, info["resource_id"])["row_count"] == 2
    finally:
        service.close()
