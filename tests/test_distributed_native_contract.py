"""Actual SDK regressions, isolated from shared cluster and JVM lifetimes."""

import json

import psutil
import pytest

from tests.verification_support import OwnedProbe
from truthound.mcp.tables import decode

pytestmark = pytest.mark.contract


def test_real_owned_ray_startup_uses_registered_alive_node_before_native_work(tmp_path):
    runtime = OwnedProbe("ray", tmp_path / "ray-node-lookup.log", guard_node_lookup=True)
    try:
        ready = runtime.receive()
        assert ready["ready"] is True and ready["node_lookup_queries"] >= 1
        result = decode(runtime.call("basics"))
        assert result["rows"] == 3 and result["stats"]["count"] == 2
        assert result["stats"]["mean"] == 2
    finally:
        runtime.close()
    assert runtime.process.returncode == 0


def test_real_spark_sdk_command_builder_starts_native_engine_without_shell(tmp_path):
    runtime = OwnedProbe("spark", tmp_path / "spark-sdk-builder.log", direct_spark_launcher=True)
    try:
        ready = runtime.receive()
        assert ready["ready"] is True and ready["sdk_launcher_uses"] == 1
        result = decode(runtime.call("basics"))
        assert result["rows"] == 3 and result["stats"]["count"] == 2
        assert result["stats"]["mean"] == 2
    finally:
        runtime.close()
    assert runtime.process.returncode == 0


def test_native_probe_utf8_log_does_not_corrupt_its_separate_protocol(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    probe = Path(__file__).resolve().parents[1] / "verification/native_distributed_probe.py"
    program = (
        "import importlib.util,sys; "
        "spec=importlib.util.spec_from_file_location('native_probe',sys.argv[1]); "
        "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m); "
        "assert sys.stdout.encoding.lower()=='cp1252'; "
        "wire=m.redirect_native_output(sys.argv[2]); "
        "assert sys.platform!='win32' or __import__('_winapi').GetStdHandle(-11)==__import__('msvcrt').get_osfhandle(1); "
        "assert sys.platform!='win32' or __import__('_winapi').GetStdHandle(-12)==__import__('msvcrt').get_osfhandle(2); "
        "print('native progress \\u2714',flush=True); "
        "print('native error \\u03bb',file=sys.stderr,flush=True); "
        "wire.write('protocol-only\\n');wire.close()"
    )
    log = tmp_path / "native.log"
    result = subprocess.run(
        [sys.executable, "-c", program, str(probe), str(log)], capture_output=True,
        env={**os.environ, "PYTHONIOENCODING": "cp1252"}, check=True,
    )
    assert result.stdout.splitlines() == [b"protocol-only"] and result.stderr == b""
    assert log.read_bytes() == "native progress ✔\nnative error λ\n".encode("utf-8")


@pytest.mark.parametrize("empty", [True, False])
def test_bounded_protocol_failure_records_structure_without_reply_content(tmp_path, empty):
    from concurrent.futures import ThreadPoolExecutor
    from io import BytesIO
    from types import SimpleNamespace

    secret = b"synthetic-private-truncated-native-reply"
    reply = b"" if empty else secret
    probe = object.__new__(OwnedProbe)
    probe.directory = SimpleNamespace(name=str(tmp_path))
    probe.log_path = tmp_path / "driver.log"
    probe.process = SimpleNamespace(stdout=BytesIO(reply), poll=lambda: None)
    probe.capture = lambda: None
    with ThreadPoolExecutor(max_workers=1) as probe.pool:
        with pytest.raises(RuntimeError, match="bounded protocol reply") as error:
            probe.receive()
    note = "\n".join(error.value.__notes__)
    assert secret.decode() not in note and str(tmp_path) not in note
    observed = json.loads(note.split(": ", 1)[1])["protocol_reply"]
    assert observed["bytes"] == len(reply)
    assert observed["empty"] is empty and observed["newline"] is False
    assert 0 <= observed["elapsed_seconds"] < 30


def test_early_pipe_eof_retains_natural_driver_exit_after_owned_cleanup(tmp_path, monkeypatch):
    import subprocess
    import sys

    from tests.verification_support import _load

    runner = _load("native_distributed_runner")
    original = subprocess.Popen
    program = "import os,time;os.close(1);time.sleep(0.2);os._exit(7)"

    def launch(_argv, **kwargs):
        if any(str(value).endswith("native_distributed_probe.py") for value in _argv):
            return original([sys.executable, "-I", "-c", program], **kwargs)
        return original(_argv, **kwargs)

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    runtime = runner.OwnedProbe("ray", tmp_path / "driver.log")
    with pytest.raises(RuntimeError, match="bounded protocol reply") as error:
        try:
            runtime.receive()
        finally:
            runtime.close()
    records = [json.loads(note.split(": ", 1)[1]) for note in error.value.__notes__]
    assert records[0]["protocol_reply"]["empty"] is True
    assert records[-1] == {"driver_exit_code": 7, "driver_killed_after_close_deadline": False}


def test_native_startup_diagnostic_keeps_only_closed_markers(tmp_path):
    import json
    from types import SimpleNamespace

    root = tmp_path / "owned"
    logs = root / "th-native-test" / "session_test" / "logs"
    logs.mkdir(parents=True)
    secret = "synthetic-private-native-log-value"
    (logs / "gcs_server.out").write_text("GcsServer server started " + secret + "\nFinished registering node. " + secret)
    (logs / "raylet.out").write_text("Raylet of id, " + secret + " started. Raylet consists of node_manager and object_manager.")
    (logs / "unreviewed.log").write_text("No module named " + secret)
    driver = tmp_path / "driver.log"
    driver.write_text("native-probe-time:ray-start:1.25\nnative-probe-time:ray-start:nan\nnative-probe-time:private-stage:2.0\nnative-probe:ray-start\nnative-probe:spark-gateway-start\nRunning Spark version " + secret + "\nnative-probe:" + secret + "\n")
    probe = object.__new__(OwnedProbe)
    probe.directory = SimpleNamespace(name=str(root))
    probe.log_path = driver
    probe.process = SimpleNamespace(poll=lambda: None)
    probe.current_command = secret
    note = probe._diagnostic_note()
    assert secret not in note and str(tmp_path) not in note
    value = json.loads(note.split(": ", 1)[1])
    assert value["phases"] == ["ray-start", "spark-gateway-start"]
    assert value["phase_elapsed_seconds"] == {"ray-start": 1.25}
    assert value["native_startup"]["spark_markers"]["spark_context_started"] is True
    assert value["native_startup"]["log_files"] == 2
    assert value["native_startup"]["markers"]["raylet_started"] is True
    assert value["native_startup"]["markers"]["node_registered"] is True
    assert value["native_startup"]["markers"]["gcs_started"] is True
    assert value["native_startup"]["markers"]["module_missing"] is False



def test_ray_fatal_evidence_keeps_closed_cause_markers_and_reviewed_source_locations(tmp_path):
    from types import SimpleNamespace

    logs = tmp_path / "th-native-test/session_test/logs"
    logs.mkdir(parents=True)
    secret = "synthetic-private-path-and-payload"
    (logs / "dashboard_agent.err").write_text(
        "Permission denied " + secret + "\n[synthetic E 12 34] agent_manager.cc:77: one Ray agent failed " + secret
    )
    driver = tmp_path / "driver.log"
    driver.write_text(
        "[synthetic F 12 34] core_worker.cc:123: Failed to register worker " + secret
        + "\n[synthetic E 12 34] private_component.cc:456: " + secret
        + "\n[synthetic I 12 34] raylet_client.cc:999: " + secret
        + "\n[synthetic W 12 34] raylet_ipc_client.cc:72: Exiting because the Raylet IPC connection failed and the local Raylet is dead. " + secret
    )
    probe = object.__new__(OwnedProbe)
    probe.directory = SimpleNamespace(name=str(tmp_path))
    probe.log_path = driver
    report = probe._native_startup_state()
    assert secret not in json.dumps(report) and str(tmp_path) not in json.dumps(report)
    assert report["ray_failure_markers"]["worker_registration_failed"]
    assert report["ray_failure_markers"]["agent_failed"]
    assert report["ray_failure_markers"]["permission_denied"]
    assert report["ray_failure_markers"]["local_raylet_dead"]
    assert not report["ray_failure_markers"]["address_in_use"]
    assert report["native_source_locations"] == [
        {"file": "agent_manager.cc", "line": 77}, {"file": "core_worker.cc", "line": 123},
        {"file": "raylet_ipc_client.cc", "line": 72},
    ]


def test_native_ray_local_checkpoint_destination_resolves_as_one_directory(tmp_path):
    from pyarrow.fs import FileType, LocalFileSystem
    from ray.data.datasource.path_util import _resolve_paths_and_filesystem

    target = tmp_path / "owned checkpoint destination"
    target.mkdir()
    paths, filesystem = _resolve_paths_and_filesystem(target.as_posix(), LocalFileSystem())
    assert len(paths) == 1 and filesystem.get_file_info(paths[0]).type == FileType.Directory


def test_native_failure_diagnostic_preserves_causes_without_messages_or_paths(tmp_path):
    import json

    from tests.verification_support import _load

    probe = _load("native_distributed_probe")
    try:
        try:
            raise OSError(13, "synthetic-private-query HADOOP_HOME", str(tmp_path / "private-data"))
        except OSError as inner:
            raise RuntimeError("synthetic-private-shutdown") from inner
    except RuntimeError as outer:
        report = probe.failure_diagnostic(outer)
    encoded = json.dumps(report)
    assert "synthetic-private" not in encoded and str(tmp_path) not in encoded
    assert report["error_type"] == "RuntimeError"
    assert report["exception_chain"][0]["error_type"] == "PermissionError"
    assert report["known_causes"]["windows_hadoop_native_missing"]
    assert report["frames"] and report["exception_chain"][0]["frames"]
    for item in [report, *report["exception_chain"]]:
        assert all("/" not in frame["file"] and "\\" not in frame["file"] for frame in item["frames"])

    class UnavailableNativeMessage(RuntimeError):
        def __str__(self):
            raise RuntimeError("synthetic-private-closed-JVM")

    unavailable = probe.failure_diagnostic(UnavailableNativeMessage())
    assert unavailable["message_unavailable"]
    assert "synthetic-private" not in json.dumps(unavailable)


def test_windows_native_scratch_is_removed_only_after_its_owner_closes(tmp_path, monkeypatch):
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    from tests.verification_support import _load

    probe = _load("native_distributed_probe")
    with tempfile.TemporaryDirectory(dir=tmp_path) as owner:
        with monkeypatch.context() as patch:
            patch.chdir(owner)
            patch.setattr(probe, "os", SimpleNamespace(name="nt"))
            with probe.runtime_directory("native-sdk-") as directory:
                scratch = Path(directory)
                assert scratch.is_relative_to(Path(owner))
                (scratch / "native.log").write_text("owned synthetic log")
            assert scratch.is_dir()
        assert scratch.is_dir()
    assert not Path(owner).exists() and not scratch.exists()


@pytest.mark.parametrize("backend", ["ray", "spark"])
def test_null_statistics_and_sampling_use_observed_rows(tmp_path, backend, record_property):
    runtime = OwnedProbe(backend, tmp_path / (backend + ".log"))
    try:
        assert runtime.receive()["ready"] is True
        record_property("native_startup", json.dumps(runtime.startup_metrics, sort_keys=True))
        basic = decode(runtime.call("basics"))
        sample = decode(runtime.call("sample"))
        assert (basic["rows"], basic["nulls"], basic["sum"]) == (3, 1, 4)
        assert (basic["stats"]["count"], sample["rows"]) == (2, 2)
        assert basic["stats"]["std"] == pytest.approx(2**0.5)
        assert decode(runtime.call("polars")) == {"rows": 3, "columns": ["x", "g"]}
        aggregates = decode(runtime.call("aggregations"))
        record_property("native_aggregations", json.dumps(runtime.last_command_metrics, sort_keys=True))
        assert aggregates["sum"] == 4 and aggregates["mean"] == 2
        assert aggregates["std"] == pytest.approx(2**0.5)
        assert aggregates["var"] == pytest.approx(2)
        assert aggregates["median"] == 2
        assert aggregates["min"] == 1 and aggregates["max"] == 3
        assert aggregates["count"] == 2 and aggregates["null_count"] == 1
        assert aggregates["count_distinct"] == basic["distinct"]
        assert aggregates["first"] == 1 and aggregates["last"] == 3
        values = decode(runtime.call("values"))
        assert values["counts"] == {"a": 2, "b": 1}
        assert {row["value"]: row["count"] for row in values["nullable_counts"]} == {
            1: 1,
            None: 1,
            3: 1,
        }
        assert values["group_context_unchanged"] is True
        assert values["duplicates"] == 1 and values["pair_duplicates"] == 0
        # Spark's documented approximate rank quantile differs from Core's
        # linear Polars interpolation; preserve that native contract.
        assert values["quantiles"] == ([1, 1, 3] if backend == "spark" else [1, 2, 3])
        assert decode(runtime.call("partition")) == {"value": 3, "rows": 3, "errors": 0}
        plan_result = decode(runtime.call("aggregation_plan"))
        assert len(plan_result["snapshots"]) == 3
        assert plan_result["snapshots"][-1] == plan_result["final"]
        groups = {row["keys"]["g"]: row["aggregations"] for row in plan_result["final"]["groups"]}
        assert groups == {
            "a": {"*_count": 2, "x_count": 2, "x_mean": 2, "x_std": 1},
            "b": {"*_count": 1, "x_count": 0, "x_mean": 0, "x_std": 0},
        }
        native_spec = decode(runtime.call("aggregation_spec"))
        assert {row["keys"]["g"]: row["aggregations"] for row in native_spec["groups"]} == groups

    finally:
        runtime.close()
    for pid, created in runtime.owned.items():
        try:
            child = psutil.Process(pid)
            assert child.create_time() != created or child.status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            pass


def test_ray_native_task_limit_actor_execution_and_block_target_are_applied(tmp_path, record_property):
    runtime = OwnedProbe("ray", tmp_path / "ray-configuration.log")
    try:
        assert runtime.receive()["ready"] is True
        record_property("native_startup", json.dumps(runtime.startup_metrics, sort_keys=True))
        result = decode(runtime.call("ray_configuration"))
        assert result["task_rows"] == result["actor_rows"] == result["calls"] == 4
        assert result["peak"] == 1 and result["all_actor"] is True
        assert result["target"] == 1024 * 1024 and result["global_unchanged"] is True
    finally:
        runtime.close()


@pytest.mark.parametrize("backend", ["ray", "spark"])
def test_native_row_cache_honors_disabled_cache_and_explicit_invalidation(tmp_path, backend, record_property):
    runtime = OwnedProbe(backend, tmp_path / (backend + "-cache.log"))
    try:
        assert runtime.receive()["ready"] is True
        record_property("native_startup", json.dumps(runtime.startup_metrics, sort_keys=True))
        assert decode(runtime.call("cache_configuration")) == {
            "uncached_calls": 2,
            "cached_calls": 1,
            "cleared_calls": 2,
            "rows": [3] * 5,
        }
    finally:
        runtime.close()


@pytest.mark.parametrize("backend", ["ray", "spark"])
def test_native_initial_partition_defaults_and_partition_scope(tmp_path, backend):
    runtime = OwnedProbe(backend, tmp_path / (backend + "-partitions.log"))
    try:
        assert runtime.receive()["ready"] is True
        result = decode(runtime.call("partition_configuration"))
        assert {
            key: result[key] for key in ("initial", "explicit", "rows", "captured", "root")
        } == {
            "initial": 2,
            "explicit": 1,
            "rows": 3,
            "captured": 2,
            "root": 2,
        }
        partitions = result["separate"]["partitions"]
        assert len(partitions) == 2
        assert sum(row["result"]["*_count"] for row in partitions) == 3
        assert sum(row["result"]["x_sum"] for row in partitions) == 4
        assert [row["result"] for row in result["empty"]["partitions"]] == [
            {"*_count": 0, "x_sum": 0}
        ] * 2
    finally:
        runtime.close()


def test_native_spark_false_flags_override_previous_true_and_capture_configuration(tmp_path):
    runtime = OwnedProbe("spark", tmp_path / "spark-flags.log")
    try:
        assert runtime.receive()["ready"] is True
        assert decode(runtime.call("spark_false_flags")) == {
            "arrow": "false",
            "adaptive": "false",
            "captured": {},
        }
    finally:
        runtime.close()


@pytest.mark.parametrize("backend", ["ray", "spark"])
def test_native_eager_mode_materializes_with_owned_spark_cache_cleanup(tmp_path, backend):
    runtime = OwnedProbe(backend, tmp_path / (backend + "-eager.log"))
    try:
        assert runtime.receive()["ready"] is True
        result = decode(runtime.call("eager_configuration"))
        assert result["materialized"] is True
        assert result["rows"] == (3 if backend == "ray" else 2)
        if backend == "spark":
            assert result["owned_cleaned"] is result["borrowed_retained"] is True
            assert result["borrowed_rows"] == 2
    finally:
        runtime.close()


@pytest.mark.parametrize("backend", ["ray", "spark"])
def test_native_partition_strategy_keeps_key_groups_ranges_and_round_robin_rows(tmp_path, backend):
    runtime = OwnedProbe(backend, tmp_path / (backend + "-strategies.log"))
    try:
        assert runtime.receive()["ready"] is True
        for strategy in ("column", "row_range", "round_robin", "custom"):
            result = decode(runtime.call("partition_strategy_" + strategy))
            assert result["count"] == len(result["partitions"]) == 3
            assert result["context_unchanged"] is True
            groups = [row["result"]["groups"] for row in result["partitions"]]
            assert sum(group["aggregations"]["*_count"] for part in groups for group in part) == 12
            if strategy == "column":
                assert all(
                    sum(any(group["keys"]["g"] == key for group in part) for part in groups) == 1
                    for key in ("a", "b")
                )
            elif strategy == "row_range":
                bounds = sorted(
                    (
                        min(group["aggregations"]["x_minmax"]["min"] for group in part),
                        max(group["aggregations"]["x_minmax"]["max"] for group in part),
                    )
                    for part in groups
                    if part
                )
                assert all(
                    left[1] <= right[0] for left, right in zip(bounds, bounds[1:], strict=False)
                )
            elif strategy == "round_robin":
                assert result["single_rows"] == 12
                assert [
                    sum(group["aggregations"]["*_count"] for group in part) for part in groups
                ] == [4, 4, 4]
    finally:
        runtime.close()


@pytest.mark.parametrize("backend", ["ray", "spark"])
def test_native_checkpoint_enabled_creates_real_snapshot(tmp_path, backend):
    runtime = OwnedProbe(backend, tmp_path / (backend + "-checkpoint.log"))
    try:
        ready = runtime.receive()
        assert ready["ready"] is True
        assert ready["owner_pid"] == runtime.process.pid
        if backend == "spark":
            import os

            if os.name == "nt":
                prerequisites = decode(runtime.call("checkpoint_prerequisites"))
                assert all(prerequisites.values()), prerequisites
        result = decode(runtime.call("checkpoint_configuration"))
        assert result["rows"] == 3
        if backend == "ray":
            assert result == {"rows": 3, "restored_rows": 3, "sum": 4, "committed": True}
        else:
            assert result["checkpointed"] is result["files"] is True
    finally:
        runtime.close()


@pytest.mark.parametrize("backend", ["ray", "spark"])
def test_native_streaming_arrow_batches_and_early_close_preserve_runtime(tmp_path, backend):
    runtime = OwnedProbe(backend, tmp_path / (backend + "-streaming.log"))
    try:
        assert runtime.receive()["ready"] is True
        assert decode(runtime.call("streaming_configuration")) == {
            "first": 1,
            "sizes": [1, 1, 1],
            "rows": 3,
            "columns": ["x", "g"],
            "nulls": 1,
            "after_close_rows": 3,
            "memory_blocked": True,
        }
    finally:
        runtime.close()


def test_native_ray_empty_checkpoint_preserves_schema(tmp_path):
    runtime = OwnedProbe("ray", tmp_path / "ray-empty-checkpoint.log")
    try:
        assert runtime.receive()["ready"] is True
        assert decode(runtime.call("empty_checkpoint")) == {
            "rows": 0,
            "columns": ["x", "g"],
            "committed": True,
        }
    finally:
        runtime.close()


def test_native_spark_manual_arrow_fallback_enforces_transfer_policy(tmp_path):
    runtime = OwnedProbe("spark", tmp_path / "spark-manual-arrow.log")
    try:
        assert runtime.receive()["ready"] is True
        assert decode(runtime.call("manual_arrow_policy")) == {
            "sizes": [1, 1, 1], "invalid": [True, True, True], "blocked": True,
        }
    finally:
        runtime.close()


@pytest.mark.parametrize("backend", ["ray", "spark"])
def test_native_arrow_bridge_applies_selected_policy_without_session_mutation(tmp_path, backend):
    runtime = OwnedProbe(backend, tmp_path / (backend + "-arrow-bridge.log"))
    try:
        assert runtime.receive()["ready"] is True
        assert decode(runtime.call("arrow_bridge_policy")) == {
            "rows": 3, "sizes": [1, 1, 1], "null_rejected": True, "session_unchanged": True,
        }
    finally:
        runtime.close()


@pytest.mark.parametrize("fail", [False, True])
def test_spark_gateway_observer_restores_launcher_and_preserves_failure(monkeypatch, fail):
    import sys
    from types import SimpleNamespace

    from tests.verification_support import _load

    calls, phases = [], []
    failure = RuntimeError("synthetic-private-gateway-error")
    result = object()

    def launch(*args, **kwargs):
        calls.append((args, kwargs))
        if fail:
            raise failure
        return result

    context = type("SyntheticContext", (), {"__module__": "synthetic_spark_context"})
    module = SimpleNamespace(launch_gateway=launch)
    monkeypatch.setitem(sys.modules, context.__module__, module)
    probe = _load("native_distributed_probe")
    try:
        with probe.spark_gateway_progress(context, phases.append):
            assert module.launch_gateway("configuration", popen_kwargs={"stdin": 1}) is result
    except RuntimeError as error:
        assert fail and error is failure
    else:
        assert not fail
    assert calls == [(("configuration",), {"popen_kwargs": {"stdin": 1}})]
    assert module.launch_gateway is launch
    assert phases == (["spark-gateway-start"] if fail else ["spark-gateway-start", "spark-gateway-started"])


def test_startup_watchdog_classifies_only_reviewed_sdk_frames():
    from types import SimpleNamespace

    from tests.verification_support import _load

    def frame(module, function, parent=None):
        return SimpleNamespace(f_globals={"__name__": module, "private": "private-frame-value"}, f_code=SimpleNamespace(co_name=function), f_back=parent)

    probe = _load("native_distributed_probe")
    sdk = frame("ray._private.services", "start_api_server")
    assert probe.blocked_startup_phase(frame("private-module", "private-function", sdk)) == "ray-api-server-wait"
    assert probe.blocked_startup_phase(frame("private-module", "start_api_server")) is None
    assert probe.blocked_startup_phase(frame("pyspark.java_gateway", "launch_gateway")) == "spark-gateway-wait"


def test_installed_startup_snapshot_excludes_private_frames_paths_and_locals():
    from types import SimpleNamespace

    from tests.verification_support import _load

    probe = _load("native_distributed_probe")
    sdk = SimpleNamespace(
        f_globals={"__name__": "py4j.java_gateway"},
        f_code=SimpleNamespace(co_filename="/private/secret/java_gateway.py", co_name="send_command"),
        f_lineno=12, f_back=None, f_locals={"secret": "private-argument"},
    )
    private = SimpleNamespace(
        f_globals={"__name__": "private_operator"},
        f_code=SimpleNamespace(co_filename="/private/operator.py", co_name="secret_method"),
        f_lineno=42, f_back=sdk,
    )
    result = probe.startup_snapshot(private)
    assert result == {"phase": "spark-jvm-rpc-wait", "sdk_wait_frames": [
        {"package": "py4j", "file": "java_gateway.py", "function": "send_command", "line": 12},
    ]}
    assert "private" not in json.dumps(result) and "secret" not in json.dumps(result)


def test_spark_start_failure_never_formats_gateway_backed_exception(monkeypatch, tmp_path, capsys):
    from types import SimpleNamespace

    from py4j.protocol import Py4JJavaError
    from pyspark import SparkContext
    from pyspark.sql import SparkSession

    from tests.verification_support import _load

    formatted = []

    class GatewayClient:
        def send_command(self, command):
            formatted.append(True)
            raise AssertionError("A stopped JVM cannot format this exception")

    error = Py4JJavaError("synthetic private SDK detail", SimpleNamespace(
        _target_id="synthetic-owned-error", _gateway_client=GatewayClient(),
    ))

    class Builder:
        def master(self, *args):
            return self

        appName = config = master

        def getOrCreate(self):
            raise error

    monkeypatch.setattr(SparkSession, "builder", Builder())
    monkeypatch.setattr(SparkContext, "_active_spark_context", None)
    probe = _load("native_distributed_probe")
    with pytest.raises(Py4JJavaError) as caught, probe.engine("spark", tmp_path):
        pytest.fail("A failed SDK must not yield an engine")
    assert caught.value is error
    assert formatted == []
    assert "synthetic private SDK detail" not in capsys.readouterr().out


def test_owned_sdk_process_samples_omit_names_ids_arguments_and_reused_pids():
    from contextlib import nullcontext
    from types import SimpleNamespace

    from verification.native_distributed_runner import owned_process_snapshot

    def forbidden():
        pytest.fail("Process arguments must never be read")

    process = SimpleNamespace(
        oneshot=nullcontext, create_time=lambda: 1,
        name=lambda: "synthetic-private-process-name",
        cmdline=forbidden, pid=123456,
        cpu_times=lambda: SimpleNamespace(user=2, system=3),
        memory_info=lambda: SimpleNamespace(rss=4096), num_threads=lambda: 2,
    )
    assert owned_process_snapshot(process, 2, 20) is None
    report = owned_process_snapshot(process, 1, 20)
    assert report["role"] == "other" and report["last_cpu_seconds"] == 5
    assert report["last_rss_bytes"] == 4096 and report["last_threads"] == 2
    assert "synthetic-private" not in json.dumps(report) and "123456" not in json.dumps(report)


def test_timed_stack_capture_runs_in_python_thread_with_gil_and_retains_trace(tmp_path, monkeypatch):
    import ctypes
    import threading
    from types import SimpleNamespace

    from tests.verification_support import _load

    probe = _load("native_distributed_probe")
    original = probe.faulthandler.dump_traceback
    done = threading.Event()
    observed = []

    def capture(**kwargs):
        observed.append(ctypes.pythonapi.PyGILState_Check())
        original(**kwargs)
        done.set()

    def forbidden(*args, **kwargs):
        pytest.fail("Timed capture must not use the asynchronous native watchdog")

    monkeypatch.setattr(probe.faulthandler, "dump_traceback", capture)
    monkeypatch.setattr(probe.faulthandler, "dump_traceback_later", forbidden)
    log = tmp_path / "stack.log"
    with log.open("wb") as stream:
        monkeypatch.setattr(probe, "sys", SimpleNamespace(stderr=stream))
        with probe.stack_watchdog(delay=0.01):
            assert done.wait(timeout=2)
    assert observed == [1]
    assert b"capture" in log.read_bytes() and b"Thread" in log.read_bytes()


@pytest.mark.parametrize("failure", [False, True])
def test_installed_startup_watchdog_cancelled_on_success_and_failure(monkeypatch, failure):
    from tests.verification_support import _load

    probe = _load("native_distributed_probe")
    events = []

    class Timer:
        def __init__(self, delay, callback):
            assert delay == 20

        def start(self):
            events.append("start")

        def cancel(self):
            events.append("cancel")

        def join(self, timeout):
            events.append(("join", timeout))

    monkeypatch.setattr(probe.threading, "Timer", Timer)
    try:
        with probe.startup_watchdog(lambda payload: events.append(payload)) as cancel:
            if failure:
                raise ValueError("synthetic startup failure")
            cancel()
    except ValueError:
        assert failure
    assert events == (["start", "cancel", ("join", 1)] if failure
                      else ["start", "cancel", "cancel", ("join", 1)])


def test_native_wait_stack_exports_sdk_structure_without_paths_or_payload(tmp_path):
    import subprocess
    import sys
    from types import SimpleNamespace

    log = tmp_path / "owned-sdk.log"
    secret = "synthetic-private-wait-stack-value"
    source = "/" + secret + "/site-packages/ray/_private/node.py"
    code = "import faulthandler,sys;faulthandler.dump_traceback(file=sys.stderr)"
    with log.open("wb") as stream:
        subprocess.run([sys.executable, "-c", "exec(compile(" + repr(code) + "," + repr(source) + ",'exec'))"], stderr=stream, check=True)
    with log.open("a") as stream:
        stream.write('  File "/private/site-packages/ray/../private.py", line 1 in private_function\n')
        stream.write("private SDK error: " + secret + "\n")
    probe = object.__new__(OwnedProbe)
    probe.directory = SimpleNamespace(name=str(tmp_path))
    probe.log_path = log
    probe.process = SimpleNamespace(poll=lambda: None)
    report = probe._diagnostic_report()
    assert report["sdk_wait_frames"] == [{"package": "ray", "file": "node.py", "line": 1, "function": "<module>"}]
    encoded = json.dumps(report)
    assert secret not in encoded and str(tmp_path) not in encoded and "private_function" not in encoded


def test_failed_spark_initialization_unregisters_owned_callback_before_gateway_exists(monkeypatch):
    from tests.verification_support import _load

    probe = _load("native_distributed_probe")
    callback, events = object(), []
    monkeypatch.setattr(probe.atexit, "unregister", events.append)
    assert probe.close_owned_spark_gateway({"gateway": None, "exit_callbacks": [callback]}) is False
    assert events == [callback]


def test_owned_direct_jvm_cleanup_uses_its_process_handle_without_taskkill(monkeypatch):
    import subprocess
    from types import SimpleNamespace

    from tests.verification_support import _load

    probe = _load("native_distributed_probe")
    events = []

    def wait(timeout):
        events.append(("wait", timeout))
        if events.count(("wait", timeout)) == 1:
            raise subprocess.TimeoutExpired("synthetic-java", timeout)

    process = SimpleNamespace(
        _truthound_owned_local_jvm=True,
        stdin=SimpleNamespace(closed=False, close=lambda: events.append("stdin-close")),
        wait=wait, kill=lambda: events.append("handle-kill"),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("A direct JVM must not launch a PID-based taskkill")

    monkeypatch.setattr(probe.subprocess, "run", forbidden)
    gateway = SimpleNamespace(proc=process, shutdown=lambda: events.append("shutdown"))
    assert probe.close_owned_spark_gateway({"gateway": gateway, "exit_callbacks": []}) is True
    assert events == ["shutdown", "stdin-close", ("wait", 1), "handle-kill", ("wait", 1)]


def test_owned_spark_gateway_cancels_only_its_callback_and_reaps_after_rpc_failure(monkeypatch):
    from types import SimpleNamespace

    from tests.verification_support import _load

    probe = _load("native_distributed_probe")
    events = []
    callback = object()
    failure = RuntimeError("synthetic-private-gateway-close")

    def shutdown():
        events.append("shutdown")
        raise failure

    stdin = SimpleNamespace(closed=False, close=lambda: events.append("stdin-closed"))
    process = SimpleNamespace(stdin=stdin, wait=lambda timeout: events.append(("wait", timeout)))
    owner = {"gateway": SimpleNamespace(proc=process, shutdown=shutdown), "exit_callbacks": [callback]}
    monkeypatch.setattr(probe.atexit, "unregister", lambda item: events.append(("unregister", item)))
    with pytest.raises(RuntimeError) as error:
        probe.close_owned_spark_gateway(owner)
    assert error.value is failure
    assert events == [("unregister", callback), "shutdown", "stdin-closed", ("wait", 1)]
    events.clear()
    borrowed = {"gateway": SimpleNamespace(proc=None, shutdown=shutdown), "exit_callbacks": [callback]}
    assert probe.close_owned_spark_gateway(borrowed) is False
    assert events == []


def test_spark_launcher_observer_retains_only_its_sdk_exit_callback(monkeypatch):
    import sys
    from types import SimpleNamespace

    from tests.verification_support import _load

    registered = []
    exit_module = SimpleNamespace(register=lambda callback: registered.append(callback) or callback)
    sdk = SimpleNamespace(atexit=exit_module)
    monkeypatch.setitem(sys.modules, "pyspark.java_gateway", sdk)

    def killChild():
        pass

    killChild.__module__ = "pyspark.java_gateway"

    def unrelated():
        pass

    gateway = object()

    def launch():
        sdk.atexit.register(killChild)
        sdk.atexit.register(unrelated)
        return gateway

    launch.__module__ = "pyspark.java_gateway"
    context = type("SyntheticContext", (), {"__module__": "synthetic_owned_spark_context"})
    context_module = SimpleNamespace(launch_gateway=launch)
    monkeypatch.setitem(sys.modules, context.__module__, context_module)
    probe = _load("native_distributed_probe")
    with probe.spark_gateway_progress(context, lambda label: None) as owner:
        assert context_module.launch_gateway() is gateway
    assert owner == {"gateway": gateway, "exit_callbacks": [killChild]}
    assert registered == [killChild, unrelated]
    assert sdk.atexit is exit_module and context_module.launch_gateway is launch


def test_real_owned_spark_gateway_is_reaped_before_driver_exit(tmp_path):
    log = tmp_path / "owned-spark-gateway.log"
    runtime = OwnedProbe("spark", log)
    launcher = None
    try:
        assert runtime.receive()["ready"] is True
        owner = decode(runtime.call("spark_gateway_owner"))
        assert owner["alive"] is True
        launcher = (owner["pid"], psutil.Process(owner["pid"]).create_time())
    finally:
        runtime.close()
    assert "native-probe:spark-gateway-reaped" in log.read_text(encoding="utf-8").splitlines()
    try:
        process = psutil.Process(launcher[0])
        assert process.create_time() != launcher[1] or process.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass


@pytest.mark.parametrize("returncode", [0, 5])
def test_owned_windows_spark_tree_kill_is_synchronous_and_cleanup_failure_stays_visible(monkeypatch, returncode):
    from types import SimpleNamespace

    from tests.verification_support import _load

    probe = _load("native_distributed_probe")
    calls, waits = [], []

    def wait(timeout):
        waits.append(timeout)
        if len(waits) == 1:
            raise probe.subprocess.TimeoutExpired("owned-launcher", timeout)
        return 0

    process = SimpleNamespace(pid=12345, stdin=SimpleNamespace(closed=True), poll=lambda: None, wait=wait)
    owner = {"gateway": SimpleNamespace(proc=process, shutdown=lambda: None), "exit_callbacks": []}
    monkeypatch.setattr(probe, "os", SimpleNamespace(name="nt", environ={"SYSTEMROOT": "C:\\Windows"}))

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(probe.subprocess, "run", run)
    if returncode:
        with pytest.raises(RuntimeError, match="Owned Spark launcher cleanup failed"):
            probe.close_owned_spark_gateway(owner)
    else:
        assert probe.close_owned_spark_gateway(owner) is True
    assert len(calls) == 1
    argv, options = calls[0]
    assert argv[1:] == ["/f", "/t", "/pid", "12345"]
    assert argv[0].endswith("System32\\taskkill.exe")
    assert options["timeout"] == 2 and options.get("shell", False) is False
    assert waits == ([1] if returncode else [1, 1])


def test_ray_default_aggregations_reuse_stats_but_preserve_invalidation_and_custom_ddof(tmp_path):
    runtime = OwnedProbe("ray", tmp_path / "ray-aggregation-cache.log")
    try:
        assert runtime.receive()["ready"] is True
        result = decode(runtime.call("aggregation_cache_contract"))
        assert result["stats_native_aggregations"] == 1
        assert result["means"] == [2.0, 2.0] and result["sums"] == [4.0, 4.0]
        assert result["cached_calls"] == {"mean": 0, "sum": 1, "std": 0}
        assert result["population"] == {"population": 1.0} and result["custom_std_calls"] == 1
        assert result["invalidated"] == 2.0 and result["invalidated_calls"] == 1
        assert result["disabled"] == [2.0, 2.0] and result["disabled_calls"] == 2
    finally:
        runtime.close()


def test_ray_batched_statistics_preserve_empty_null_nan_and_literal_column(tmp_path):
    runtime = OwnedProbe("ray", tmp_path / "ray-statistics-shapes.log")
    try:
        assert runtime.receive()["ready"] is True
        native_nan_mean = decode(runtime.call("statistics_nan_mean"))
        native_nan_std = decode(runtime.call("statistics_nan_std"))
        for command, count, nulls, mean, std, minimum, maximum in [
            ("statistics_null", 0, 2, None, None, None, None),
            ("statistics_empty", 0, 0, None, None, None, None),
            ("statistics_single", 1, 0, 7.0, "nan", 7.0, 7.0),
            ("statistics_nan", 2, 1, native_nan_mean, native_nan_std, 1.0, 1.0),
            ("statistics_literal", 2, 1, 2.0, pytest.approx(2**0.5), 1.0, 3.0),
        ]:
            assert decode(runtime.call(command)) == {
                "count": count, "null_count": nulls, "mean": mean,
                "std": std, "min": minimum, "max": maximum,
            }
    finally:
        runtime.close()
