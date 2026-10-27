"""Installed custom profiler compositions, failure artifacts and fresh-owner restore."""

import json
import sys
from types import SimpleNamespace

import pytest

from tests import test_mcp_plugin_bindings as plugins
from tests.test_mcp_brokers import approved
from truthound.mcp.approval import ApprovalStore

runtime = plugins.runtime
pytestmark = pytest.mark.contract


def prepare(runtime, monkeypatch):
    from truthound.mcp import model_process

    adapter, control, _, _, _ = runtime
    monkeypatch.setattr(model_process, "sys", SimpleNamespace(
        executable=adapter._worker_argv()[0], platform=sys.platform,
        _base_executable=sys._base_executable,
    ))
    document = json.loads((control / "bindings.json").read_text())
    document["plugins"]["fixture"]["capabilities"]["profiler-factory"] = "profiler-factory"
    document["profiler_services"] = {"local": "profiler-factory"}
    (control / "bindings.json").write_text(json.dumps(document))
    return reopen(runtime, "profiler-open")


def reopen(runtime, key):
    return approved(runtime, "profiler_services_open", binding_id="local", idempotency_key=key)["data"]["resource_id"]


def test_bound_schema_only_analyzers_preserve_strict_verdicts_and_input_ownership(runtime, monkeypatch):
    runtime[2].with_name("schema-only-mode").write_text("enabled")
    service = prepare(runtime, monkeypatch)
    current = plugins.invoke(runtime[0], "assets_inline", columns=[
        {"name": "x", "dtype": {"kind": "Int64"}, "values": [1]},
    ])["data"]["resource_id"]
    _, state = call(runtime, service, "inspect")
    assert not state["profilers"] and state["schema_analyzers"] == ["failure", "strict", "warning"]
    _, strict = call(runtime, service, "schema_compare", analyzer_id="strict", baseline_id=runtime[4], current_id=current)
    assert strict["is_compatible"] is False and strict["level"] == "none"
    _, warning = call(runtime, service, "schema_compare", key="warning-schema-comparison",
                   analyzer_id="warning", baseline_id=runtime[4], current_id=current)
    assert warning["is_compatible"] is True and len(warning["issues"]) == 1
    assert runtime[2].read_text().count("schema-rule-check") == 2
    worker = runtime[0]._session._administration["profiler_services"].workers[service]
    plugins.invoke(runtime[0], "resources_release", resource_id=current)
    assert worker.closed and worker.process.poll() is not None


def test_bound_failed_schema_rule_cannot_create_a_compatible_result(runtime, monkeypatch):
    service = prepare(runtime, monkeypatch)
    request = dict(binding_id="local", service_id=service, command="schema_compare",
                   analyzer_id="failure", baseline_id=runtime[4], current_id=runtime[4],
                   idempotency_key="failed-bound-schema-check")
    pending = plugins.invoke(runtime[0], "profiler_services_execute", **request)
    store = ApprovalStore(runtime[1])
    try:
        store.approve(pending["data"]["plan_id"])
    finally:
        store.close()
    result = plugins.invoke(runtime[0], "profiler_services_execute", **request)
    assert result["execution_status"] == "outcome_unknown", result
    assert result["data"] is None and "synthetic-private-schema-rule" not in str(result)
    assert runtime[2].read_text().count("schema-rule-failed") == 1


def call(runtime, service, command, key=None, **arguments):
    request = dict(binding_id="local", service_id=service, command=command,
                   idempotency_key=key or "profiler-" + command, **arguments)
    pending = plugins.invoke(runtime[0], "profiler_services_execute", **request)
    assert pending["issues"] == [{"code": "approval_required"}], pending
    store = ApprovalStore(runtime[1])
    try:
        store.approve(pending["data"]["plan_id"])
    finally:
        store.close()
    response = plugins.invoke(runtime[0], "profiler_services_execute", **request)
    assert response.get("data"), response
    payload = plugins.source_artifact(runtime[0], response["data"]["resource_id"])
    return response, payload


def asset(runtime):
    adapter = runtime[0]
    result = plugins.invoke(adapter, "assets_inline", columns=[{"name": "code", "dtype": {"kind": "String"}, "values": ["A1", "A2", "A3"]}])
    assert result["execution_status"] == "succeeded", result
    return result["data"]["resource_id"]


def test_native_custom_composition_save_restore_execute_and_close(runtime, monkeypatch):
    service = prepare(runtime, monkeypatch)
    source = asset(runtime)
    _, profiled = call(runtime, service, "profile", profiler_id="custom", asset_id=source)
    profile_id = profiled["profile_id"]
    assert profiled["profile"]["row_count"] == 2
    column = profiled["profile"]["columns"][0]
    assert column["detected_patterns"][0]["pattern"] == "custom-code"
    assert "custom-analyze" in runtime[2].read_text() and "custom-infer" in runtime[2].read_text()
    exported, raw_profile = call(runtime, service, "profile_export", profile_id=profile_id)
    assert raw_profile == profiled["profile"]
    rejected = plugins.invoke(runtime[0], "profiler_services_execute", binding_id="local", service_id=service,
                              command="profile_import", artifact_id=exported["data"]["resource_id"], idempotency_key="cycle-import")
    assert rejected["issues"] == [{"code": "invalid_arguments"}]
    importer = reopen(runtime, "profile-import-owner")
    _, imported = call(runtime, importer, "profile_import", artifact_id=exported["data"]["resource_id"])
    assert imported["profile"] == profiled["profile"]
    call(runtime, service, "profile_save", profile_id=profile_id, slot_id="local")
    _, generated = call(runtime, service, "generate_suite", profile_id=profile_id, generator_id="schema")
    suite_id = generated["suite_id"]
    call(runtime, service, "suite_save", suite_id=suite_id, slot_id="local")
    response, execution = call(runtime, service, "suite_execute", suite_id=suite_id, executor_id="native", asset_id=source)
    assert response["execution_status"] == "succeeded", response
    assert execution["execution"]["executed_rules"] == len(generated["suite"]["rules"])
    assert execution["execution"]["skipped_rules"] == 0 and not execution["error_types"]
    plugins.invoke(runtime[0], "resources_release", resource_id=service)
    assert "profiler-close" in runtime[2].read_text()
    fresh = reopen(runtime, "profiler-fresh-open")
    _, restored = call(runtime, fresh, "profile_restore", key="fresh-profile-restore", slot_id="local")
    assert restored["profile"] == profiled["profile"]
    _, suite = call(runtime, fresh, "suite_restore", key="fresh-suite-restore", slot_id="local")
    assert suite["suite"] == generated["suite"]
    call(runtime, fresh, "suite_execute", key="fresh-suite-execute", suite_id=suite["suite_id"], executor_id="native", asset_id=source, dry_run=True)
    call(runtime, fresh, "slot_delete", key="delete-profile-slot", slot_kind="profile", slot_id="local")
    call(runtime, fresh, "slot_delete", key="delete-suite-slot", slot_kind="suite", slot_id="local")
    assert not runtime[2].with_name("operator-profile.json").exists()
    assert not runtime[2].with_name("operator-suite.json").exists()


def test_partial_custom_profile_stays_failed_after_save_restore_and_cannot_generate(runtime, monkeypatch):
    runtime[2].with_name("profiler-failed").touch()
    service = prepare(runtime, monkeypatch)
    source = asset(runtime)
    response, partial = call(runtime, service, "profile", profiler_id="custom", asset_id=source)
    assert response["execution_status"] == "failed" and response["issues"] == [{"code": "native_profile_incomplete"}]
    assert partial["profile"]["complete"] is False
    before = runtime[2].read_text().count("custom-analyze")
    replay = plugins.invoke(runtime[0], "profiler_services_execute", binding_id="local", service_id=service,
                            command="profile", idempotency_key="profiler-profile", profiler_id="custom", asset_id=source)
    assert replay == response and runtime[2].read_text().count("custom-analyze") == before

    assert "synthetic-private-profile-query" not in json.dumps(partial)
    call(runtime, service, "profile_save", profile_id=partial["profile_id"], slot_id="local")
    response, blocked = call(runtime, service, "generate_suite", profile_id=partial["profile_id"], generator_id="schema")
    assert response["execution_status"] == "failed" and "suite_id" not in blocked
    plugins.invoke(runtime[0], "resources_release", resource_id=service)
    fresh = reopen(runtime, "partial-fresh-open")
    response, restored = call(runtime, fresh, "profile_restore", slot_id="local")
    assert response["execution_status"] == "failed" and restored["profile"] == partial["profile"]
    plugins.invoke(runtime[0], "resources_release", resource_id=source)
    stale = plugins.invoke(runtime[0], "artifacts_read", resource_id=response["data"]["resource_id"])
    # This new owner read an explicitly approved fixed slot, independent of the released input.
    assert stale["execution_status"] == "succeeded"


def test_unconvertible_generated_suite_is_failed_artifact_not_successful_zero_rules(runtime, monkeypatch):
    service = prepare(runtime, monkeypatch)
    source = asset(runtime)
    _, profile = call(runtime, service, "profile", profiler_id="custom", asset_id=source)
    _, suite = call(runtime, service, "generate_suite", profile_id=profile["profile_id"], generator_id="missing")
    response, result = call(runtime, service, "suite_execute", suite_id=suite["suite_id"], executor_id="native", asset_id=source)
    assert response["execution_status"] == "failed" and response["issues"] == [{"code": "native_suite_incomplete"}]
    assert result["execution"]["success"] is False and result["execution"]["skipped_rules"] == 1
    assert result["error_types"] == ["ValueError"]
    assert "operator-profile.json" not in json.dumps(result)


def test_native_incremental_reuses_restored_fingerprint_and_refreshes_changed_values(runtime, monkeypatch):
    service = prepare(runtime, monkeypatch)
    inputs = []
    for values in ([1, 2, 3], [100, 200, 300]):
        result = plugins.invoke(runtime[0], "assets_inline", columns=[{"name": "x", "dtype": {"kind": "Int64"}, "values": values}])
        assert result["execution_status"] == "succeeded"
        inputs.append(result["data"]["resource_id"])
    _, baseline = call(runtime, service, "incremental_profile", incremental_id="native", asset_id=inputs[0])
    call(runtime, service, "profile_save", profile_id=baseline["profile_id"], slot_id="local")
    plugins.invoke(runtime[0], "resources_release", resource_id=service)
    fresh = reopen(runtime, "incremental-fresh-owner")
    _, restored = call(runtime, fresh, "profile_restore", slot_id="local")
    _, unchanged = call(runtime, fresh, "incremental_profile", key="incremental-unchanged", incremental_id="native", asset_id=inputs[0], previous_profile_id=restored["profile_id"])
    assert unchanged["incremental"]["skipped_columns"] == ["x"]
    _, changed = call(runtime, fresh, "incremental_profile", key="incremental-changed", incremental_id="native", asset_id=inputs[1], previous_profile_id=restored["profile_id"])
    assert changed["incremental"]["profiled_columns"] == ["x"]
    assert changed["profile"]["columns"][0]["distribution"]["mean"] == 200


@pytest.mark.parametrize("failed", [False, True])
def test_native_streamed_source_profiles_and_closes_each_source_iterator(runtime, monkeypatch, failed):
    import polars as pl

    from tests.test_mcp_profiling import normalize
    from truthound.profiler.streaming import StreamingProfiler

    if failed:
        runtime[2].with_name("profiler-stream-failed").touch()
    service = prepare(runtime, monkeypatch)
    response, result = call(runtime, service, "stream_profile", streaming_id="native", stream_id="local", progress_id="local")
    if failed:
        assert response["issues"] == [{"code": "native_profile_incomplete"}]
        assert result["profile"]["columns"][0]["execution_issues"][0]["error_type"] == "RuntimeError"
        assert "synthetic-private-chunk-query" not in json.dumps(result)
    else:
        direct = StreamingProfiler(chunk_size=1).profile_dataframe(pl.DataFrame({"x": [1, 2]}))
        assert normalize(result["profile"]["columns"]) == normalize(direct.to_dict()["columns"])
    marker = runtime[2].read_text()
    assert marker.count("chunk-source-open") == marker.count("chunk-source-close") == marker.count("chunk-iterator-close") == 1
    assert marker.count("chunk-progress") == 2
    response, fresh = call(runtime, service, "stream_profile", key="stream-current-run", streaming_id="native", stream_id="local")
    assert response["execution_status"] == "succeeded" and fresh["profile"]["row_count"] == 2
    assert "complete" not in fresh["profile"]


def test_native_profile_cache_persists_reopens_and_zero_ttl_expires(runtime, monkeypatch):
    service = prepare(runtime, monkeypatch)
    source = asset(runtime)
    _, profile = call(runtime, service, "cache_get_or_compute", cache_id="local", cache_key_id="local", profiler_id="custom", asset_id=source)
    assert profile["found"]
    before = runtime[2].read_text().count("custom-analyze")
    _, cached = call(runtime, service, "cache_get_or_compute", key="cache-compute-again", cache_id="local", cache_key_id="local", profiler_id="custom", asset_id=source)
    assert cached["profile"] == profile["profile"] and runtime[2].read_text().count("custom-analyze") == before
    plugins.invoke(runtime[0], "resources_release", resource_id=service)
    fresh = reopen(runtime, "cache-fresh-open")
    _, restored = call(runtime, fresh, "cache_get", cache_id="local", cache_key_id="local")
    assert restored["profile"] == profile["profile"]
    _, stats = call(runtime, fresh, "cache_stats", cache_id="local")
    assert stats["stats"]["file_count"] == 1
    assert "private-profile-cache" not in json.dumps(stats)
    assert "internal-private-cache-key" not in json.dumps(restored)
    call(runtime, fresh, "cache_set", cache_id="local", cache_key_id="local", profile_id=restored["profile_id"], ttl_seconds=0)
    _, expired = call(runtime, fresh, "cache_get", key="cache-after-zero-ttl", cache_id="local", cache_key_id="local")
    assert expired == {"found": False}
    call(runtime, fresh, "cache_set", key="cache-pattern-seed", cache_id="local", cache_key_id="local", profile_id=restored["profile_id"])
    _, invalidated = call(runtime, fresh, "cache_invalidate_pattern", cache_id="local", pattern_id="selected")
    assert invalidated == {"invalidated": 1}
    call(runtime, fresh, "cache_clear", cache_id="local")


def test_native_cache_corruption_keeps_bytes_and_failed_intent_until_explicit_invalidation(runtime, monkeypatch):
    service = prepare(runtime, monkeypatch)
    source = asset(runtime)
    call(runtime, service, "cache_get_or_compute", cache_id="local", cache_key_id="local", profiler_id="custom", asset_id=source)
    path = next(runtime[2].with_name("private-profile-cache").glob("*.json.gz"))
    path.write_bytes(b"synthetic-corrupt-native-cache")
    request = dict(binding_id="local", service_id=service, command="cache_get", cache_id="local", cache_key_id="local", idempotency_key="cache-corrupt-read")
    pending = plugins.invoke(runtime[0], "profiler_services_execute", **request)
    store = ApprovalStore(runtime[1])
    try:
        store.approve(pending["data"]["plan_id"])
    finally:
        store.close()
    failed = plugins.invoke(runtime[0], "profiler_services_execute", **request)
    assert failed["issues"] == [{"code": "outcome_unknown"}] and not failed["data"]
    assert path.read_bytes() == b"synthetic-corrupt-native-cache"
    assert "private-profile-cache" not in json.dumps(failed)
    fresh = reopen(runtime, "cache-repair-owner")
    _, repaired = call(runtime, fresh, "cache_invalidate", cache_id="local", cache_key_id="local")
    assert repaired == {"invalidated": True} and not path.exists()


def test_native_generated_suite_isolated_deadline_uses_serialized_binding_guard(runtime, monkeypatch):
    service = prepare(runtime, monkeypatch)
    source = asset(runtime)
    _, profile = call(runtime, service, "profile", profiler_id="custom", asset_id=source)
    _, suite = call(runtime, service, "generate_suite", profile_id=profile["profile_id"], generator_id="schema")
    response, complete = call(runtime, service, "suite_execute", suite_id=suite["suite_id"], executor_id="native", asset_id=source, timeout_seconds=5)
    assert response["execution_status"] == "succeeded", response
    assert complete["execution"]["executed_rules"] == len(suite["suite"]["rules"])
    assert not complete["error_types"]
    response, timeout = call(runtime, service, "suite_execute", key="suite-short-deadline", suite_id=suite["suite_id"], executor_id="native", asset_id=source, timeout_seconds=0.001)
    assert response["issues"] == [{"code": "native_suite_incomplete"}]
    assert timeout["execution"]["failed_rules"] == len(suite["suite"]["rules"])
    assert set(timeout["error_types"]) == {"TimeoutError"}


def test_installed_quality_scoring_uses_changed_labels_and_redacts_native_rule(runtime, monkeypatch):
    service = prepare(runtime, monkeypatch)
    assets = []
    for labels in ([True, False], [False, True]):
        assets.append(plugins.invoke(runtime[0], 'assets_inline', columns=[
            {'name': 'x', 'dtype': {'kind': 'Int64'}, 'values': [1, -1]},
            {'name': 'valid', 'dtype': {'kind': 'Boolean'}, 'values': labels},
        ])['data']['resource_id'])
    _, first = call(runtime, service, 'quality_score', scorer_id='sampling', rule_ids=['positive'], asset_id=assets[0], label_column='valid')
    _, second = call(runtime, service, 'quality_compare', scorer_id='sampling', rule_ids=['positive'], asset_id=assets[1], label_column='valid')
    assert first['scores'][0]['metrics']['f1_score'] == 1
    assert second['scores'][0]['metrics']['f1_score'] == 0
    assert first['labels_provided'] is True and first['scores'][0]['rule_id'] == 'positive'
    assert 'private-quality-rule' not in str(first)
    _, cleared = call(runtime, service, 'quality_cache_clear', scorer_id='sampling')
    assert cleared == {'cleared': True}
    plugins.invoke(runtime[0], 'resources_release', resource_id=service)
    restored = reopen(runtime, 'quality-owner-reopened')
    _, result = call(runtime, restored, 'quality_score', key='quality-after-restart', scorer_id='sampling', rule_ids=['positive'], asset_id=assets[1], label_column='valid')
    assert result['scores'][0]['metrics']['f1_score'] == 0


def test_installed_quality_history_survives_owner_restart(runtime, monkeypatch):
    service = prepare(runtime, monkeypatch)
    asset = plugins.invoke(runtime[0], 'assets_inline', columns=[
        {'name': 'x', 'dtype': {'kind': 'Int64'}, 'values': [1, -1]},
        {'name': 'valid', 'dtype': {'kind': 'Boolean'}, 'values': [True, False]},
    ])['data']['resource_id']
    _, recorded = call(runtime, service, 'quality_record', trend_id='local', scorer_id='sampling', rule_ids=['positive'], asset_id=asset, label_column='valid')
    assert recorded['scores'][0]['metrics']['f1_score'] == 1
    plugins.invoke(runtime[0], 'resources_release', resource_id=service)
    reopened = reopen(runtime, 'quality-history-reopened')
    _, history = call(runtime, reopened, 'quality_history', trend_id='local', rule_ids=['positive'])
    assert len(history['history']) == 1
    assert history['history'][0]['metrics']['confusion_matrix']['true_positives'] == 1
    assert history['history'][0]['metrics']['quality_level'] == 'excellent'
    assert 'private-quality' not in str(history) and 'notes' not in str(history)
    _, trend = call(runtime, reopened, 'quality_trend', trend_id='local', rule_ids=['positive'])
    assert trend['trend']['trend'] == 'insufficient_data'


@pytest.mark.parametrize('sampler_id', ['head', 'parallel', 'sketch'])
def test_installed_sampling_retains_exact_snapshot_for_profile_and_release(runtime, monkeypatch, sampler_id):
    service = prepare(runtime, monkeypatch)
    _, sampled = call(runtime, service, 'sampling_sample', sampler_id=sampler_id, asset_id=runtime[4])
    identity = sampled['sample_id']
    assert sampled['metrics']['sample_size'] == 1
    _, page = call(runtime, service, 'sample_get', sample_id=identity)
    assert page['returned_rows'] == 1 and page['truncated'] is False
    if sampler_id == 'sketch':
        column = page['columns'][0]
        _, sketch = call(runtime, service, 'sample_sketch_query', sample_id=identity, column=column, sketch_value=page['rows'][0][column])
        assert sketch['estimated_distinct'] == 1 and sketch['estimated_frequency'] == 1
    _, profiled = call(runtime, service, 'sample_profile', sample_id=identity, profiler_id='custom')
    assert profiled['profile']['row_count'] == 1
    _, released = call(runtime, service, 'sample_release', sample_id=identity)
    assert released == {'released': True}
    _, state = call(runtime, service, 'inspect')
    assert state['samples'] == [] and state['samplers'] == ['head', 'parallel', 'sketch']


def test_installed_native_type_feedback_persists_selected_model(runtime, monkeypatch):
    service = prepare(runtime, monkeypatch)
    asset = plugins.invoke(runtime[0], 'assets_inline', columns=[
        {'name': 'x', 'dtype': {'kind': 'Int64'}, 'values': [1, 2]},
    ])['data']['resource_id']
    _, first = call(runtime, service, 'inference_feedback', inferrer_id='local', column='x', true_type='integer', inference_context_id='selected', asset_id=asset)
    assert first == {'learning_enabled': True, 'pending_feedback': 1}
    _, trained = call(runtime, service, 'inference_feedback', key='feedback-trained', inferrer_id='local', column='x', true_type='integer', inference_context_id='selected', asset_id=asset)
    assert trained['pending_feedback'] == 0
    plugins.invoke(runtime[0], 'resources_release', resource_id=service)
    reopened = reopen(runtime, 'inference-model-reopened')
    _, inferred = call(runtime, reopened, 'inference_infer', inferrer_id='local', column='x', asset_id=asset)
    assert inferred['inferred_type'] == 'integer' and inferred['confidence'] == 1
    assert 'private-native-table' not in str(inferred) and 'private-inference-model' not in str(inferred)
    _, table = call(runtime, reopened, 'inference_table', inferrer_id='local', asset_id=asset)
    assert table['columns'][0]['inferred_type'] == 'integer'


def test_installed_profile_visualization_exports_html_and_saves_native_digest(runtime, monkeypatch):
    import base64
    import hashlib
    service = prepare(runtime, monkeypatch)
    _, profile = call(runtime, service, 'profile', profiler_id='custom', asset_id=runtime[4])
    response = approved(runtime, 'profiler_services_execute', binding_id='local', service_id=service,
                        command='visualization_export', visualizer_id='svg', profile_id=profile['profile_id'],
                        idempotency_key='native-visualization-export')
    content, offset = [], 0
    while offset is not None:
        result = plugins.invoke(runtime[0], 'artifacts_read', resource_id=response['data']['resource_id'], offset=offset)
        assert result['execution_status'] == 'succeeded'
        content.append(base64.b64decode(result['data']['base64']))
        offset = result['data']['next_offset']
    rendered = b''.join(content)
    assert rendered.startswith(b'<!DOCTYPE html>')
    assert b'<svg' in rendered and b'<script src=' not in rendered
    _, saved = call(runtime, service, 'visualization_save', visualizer_id='svg', visualization_slot_id='local', profile_id=profile['profile_id'])
    persisted = runtime[2].with_name('private-native-profile.html').read_bytes()
    assert saved['bytes'] == len(persisted)
    assert saved['sha256'] == hashlib.sha256(persisted).hexdigest()
    assert 'private-native-profile' not in str(saved)


def test_installed_distributed_profiler_uses_native_global_statistics(runtime, monkeypatch):
    service = prepare(runtime, monkeypatch)
    _, result = call(runtime, service, 'profile', profiler_id='distributed', asset_id=runtime[4])
    profile = result['profile']
    assert profile['row_count'] == 2 and profile.get('complete', True) is True
    assert profile['columns'][0]['distinct_count'] == 2
    identity = result['profile_id']
    _, saved = call(runtime, service, 'profile_save', profile_id=identity, slot_id='local')
    assert saved
    plugins.invoke(runtime[0], 'resources_release', resource_id=service)
    assert 'profiler-close' in runtime[2].read_text()


def test_installed_native_classifier_preserves_named_features(runtime, monkeypatch):
    service = prepare(runtime, monkeypatch)
    source = plugins.invoke(runtime[0], "assets_inline", columns=[
        {"name": "email", "dtype": {"kind": "String"}, "values": ["person@example.test"] * 10},
    ])["data"]["resource_id"]
    _, result = call(runtime, service, "inference_infer", inferrer_id="native-rules", asset_id=source, column="email")
    assert result["inferred_type"] == "email" and result["confidence"] > 0
    result = plugins.invoke(runtime[0], "resources_release", resource_id=service)
    assert result["execution_status"] == "succeeded"
