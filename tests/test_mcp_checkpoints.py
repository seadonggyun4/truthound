"""Real installed factories, native Checkpoint parity, approval and lifecycle."""

import json
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests import test_mcp_plugin_bindings as plugins
from truthound.mcp.approval import ApprovalStore

pytestmark = pytest.mark.contract
runtime = plugins.runtime
invoke = plugins.invoke


def prepare(runtime, monkeypatch, *, approve=True, asynchronous=False):
    from truthound.mcp import model_process

    adapter, control, marker, _, _ = runtime
    monkeypatch.setattr(
        model_process,
        "sys",
        SimpleNamespace(
            executable=adapter._worker_argv()[0],
            platform=sys.platform,
            _base_executable=sys._base_executable,
        ),
    )
    path = control / "bindings.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    capability = "async-checkpoint-factory" if asynchronous else "checkpoint-factory"
    operation = "checkpoints_run_async" if asynchronous else "checkpoints_run"
    config["plugins"]["fixture"]["capabilities"][capability] = capability
    config["checkpoints"] = {"daily": capability}
    path.write_text(json.dumps(config), encoding="utf-8")
    request = {"binding_id": "daily", "idempotency_key": "native-checkpoint"}
    pending = invoke(adapter, operation, **request)
    assert pending["issues"] == [{"code": "approval_required"}], pending
    assert "operator_configured_native_action_retries" in pending["data"]["effects"]
    assert not marker.exists()
    if approve:
        store = ApprovalStore(control)
        try:
            store.approve(pending["data"]["plan_id"])
        finally:
            store.close()
    return request


@pytest.mark.parametrize("fail", [False, True])
def test_native_async_checkpoint_parity(runtime, monkeypatch, tmp_path, fail):
    import asyncio

    from truthound.checkpoint import CheckpointResult
    from truthound.checkpoint.async_checkpoint import AsyncCheckpoint
    from truthound.datasources.sql.sqlite import SQLiteDataSource, SQLiteDataSourceConfig

    adapter, _, marker, _, _ = runtime
    (tmp_path / "checkpoint-settings.json").write_text(json.dumps({"fail": fail}), encoding="utf-8")
    request = prepare(runtime, monkeypatch, asynchronous=True)
    response = invoke(adapter, "checkpoints_run_async", **request)
    assert response["execution_status"] == "succeeded", response
    result = plugins.source_artifact(adapter, response["data"]["resource_id"])

    async def direct_run():
        with SQLiteDataSource(
            database=str(tmp_path / "operator.db"),
            table="items",
            config=SQLiteDataSourceConfig(name="approved-source"),
        ) as source:
            async with AsyncCheckpoint(
                name="approved-checkpoint", data_source=source, validators=["null"]
            ) as checkpoint:
                return await checkpoint.run_async()

    direct = asyncio.run(direct_run())
    native = CheckpointResult.from_dict(result)
    assert native.validation_run is not None and direct.validation_run is not None
    for key in ("checks", "issues", "execution_issues", "row_count", "column_count"):
        assert native.validation_run.to_dict()[key] == direct.validation_run.to_dict()[key]
    if not fail:
        assert native.status == direct.status
    assert result["action_results"][0]["status"] == ("error" if fail else "success")
    effects = marker.read_text(encoding="utf-8")
    assert effects.splitlines()[-4:] == [
        "async-checkpoint-close",
        "source-close",
        "checkpoint-close",
        "teardown",
    ]
    assert invoke(adapter, "checkpoints_run_async", **request) == response
    assert marker.read_text(encoding="utf-8") == effects


@pytest.mark.parametrize("fault", ["wrong", "block"])
def test_async_checkpoint_unknown_never_retries(runtime, monkeypatch, tmp_path, fault):
    adapter, _, marker, _, _ = runtime
    (tmp_path / "checkpoint-settings.json").write_text(json.dumps({fault: True}), encoding="utf-8")
    if fault == "block":
        adapter.policy = replace(adapter.policy, timeout_seconds=2)
    request = prepare(runtime, monkeypatch, asynchronous=True)
    first = invoke(adapter, "checkpoints_run_async", **request)
    assert first["issues"] == [{"code": "outcome_unknown"}], first
    effects = marker.read_bytes() if marker.exists() else b""
    assert invoke(adapter, "checkpoints_run_async", **request)["issues"] == first["issues"]
    assert (marker.read_bytes() if marker.exists() else b"") == effects


def test_native_checkpoint_result_and_action_cleanup(runtime, monkeypatch, tmp_path):
    from truthound.checkpoint import Checkpoint, CheckpointResult
    from truthound.datasources.sql.sqlite import SQLiteDataSource, SQLiteDataSourceConfig

    adapter, _, marker, _, _ = runtime
    request = prepare(runtime, monkeypatch)
    response = invoke(adapter, "checkpoints_run", **request)
    assert response["execution_status"] == "succeeded", response
    result = plugins.source_artifact(adapter, response["data"]["resource_id"])
    with SQLiteDataSource(
        database=str(tmp_path / "operator.db"),
        table="items",
        config=SQLiteDataSourceConfig(name="approved-source"),
    ) as source:
        direct = Checkpoint(
            name="approved-checkpoint", data_source=source, validators=["null"]
        ).run()
    native = CheckpointResult.from_dict(result)
    assert native.status == direct.status
    assert native.validation_run is not None and direct.validation_run is not None
    for key in ("checks", "issues", "execution_issues", "row_count", "column_count"):
        assert native.validation_run.to_dict()[key] == direct.validation_run.to_dict()[key]
    assert result["action_results"][0]["status"] == "success"
    assert result["action_results"][0]["details"] == {"status": direct.status.value}
    effects = marker.read_text(encoding="utf-8")
    assert effects.splitlines()[-3:] == ["source-close", "checkpoint-close", "teardown"]
    assert effects.count("checkpoint-action") == 1
    assert invoke(adapter, "checkpoints_run", **request) == response
    assert marker.read_text(encoding="utf-8") == effects


@pytest.mark.parametrize("change", ["binding", "package", "policy", "release"])
def test_checkpoint_artifact_authority_is_revocable(runtime, monkeypatch, change):
    adapter, control, _, code, _ = runtime
    request = prepare(runtime, monkeypatch)
    response = invoke(adapter, "checkpoints_run", **request)
    assert response["execution_status"] == "succeeded", response
    identity = response["data"]["resource_id"]
    if change == "binding":
        (control / "bindings.json").write_text("{}", encoding="utf-8")
    elif change == "package":
        code.write_text(code.read_text(encoding="utf-8") + "\n# altered\n", encoding="utf-8")
    elif change == "policy":
        adapter.policy = replace(adapter.policy, max_items=99)
    else:
        assert (
            invoke(adapter, "resources_release", resource_id=identity)["execution_status"]
            == "succeeded"
        )
    assert invoke(adapter, "artifacts_read", resource_id=identity)["execution_status"] == "failed"
    assert invoke(adapter, "checkpoints_run", **request)["execution_status"] == "failed"


def test_checkpoint_approval_and_closed_input(runtime, monkeypatch):
    adapter, _, marker, _, _ = runtime
    request = prepare(runtime, monkeypatch, approve=False)
    assert invoke(adapter, "checkpoints_run", **request)["issues"] == [
        {"code": "approval_required"}
    ]
    for extra in (
        {"factory": "os.system"},
        {"context": {}},
        {"actions": []},
        {"data_source": "private"},
    ):
        assert invoke(adapter, "checkpoints_run", **request, **extra)["issues"] == [
            {"code": "invalid_arguments"}
        ]
    assert not marker.exists()


@pytest.mark.parametrize("router", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_native_actions_router_failure_and_retry(runtime, monkeypatch, tmp_path, router, fail):
    adapter, _, marker, _, _ = runtime
    (tmp_path / "checkpoint-settings.json").write_text(
        json.dumps({"router": router, "fail": fail, "retries": 1}), encoding="utf-8"
    )
    request = prepare(runtime, monkeypatch)
    response = invoke(adapter, "checkpoints_run", **request)
    assert response["execution_status"] == "succeeded", response
    result = plugins.source_artifact(adapter, response["data"]["resource_id"])
    assert result["action_results"][0]["status"] == ("error" if fail else "success")
    if fail:
        assert result["status"] == "error"
        assert result["action_results"][0]["error"] == "synthetic-action-error"
    effects = marker.read_text(encoding="utf-8")
    assert effects.count("checkpoint-action") == (2 if fail else 1)
    assert invoke(adapter, "checkpoints_run", **request) == response
    assert marker.read_text(encoding="utf-8") == effects


@pytest.mark.parametrize("fault", ["wrong", "block"])
def test_uncertain_checkpoint_is_not_reexecuted(runtime, monkeypatch, tmp_path, fault):
    from truthound.mcp import checkpoints

    adapter, _, marker, _, _ = runtime
    (tmp_path / "checkpoint-settings.json").write_text(json.dumps({fault: True}), encoding="utf-8")
    if fault == "block":
        adapter.policy = replace(adapter.policy, timeout_seconds=2)
    request = prepare(runtime, monkeypatch)
    created = []
    original = checkpoints.CheckpointProcess

    def track(ttl):
        worker = original(ttl)
        created.append(worker)
        return worker

    monkeypatch.setattr(checkpoints, "CheckpointProcess", track)
    for _ in range(2):
        result = invoke(adapter, "checkpoints_run", **request)
        assert result["issues"] == [{"code": "outcome_unknown"}], result
    assert len(created) == 1 and created[0].closed
    assert created[0].process.poll() is not None
    if marker.exists():
        assert "checkpoint-action" not in marker.read_text(encoding="utf-8")


@pytest.mark.parametrize("signal", ["eof", "extra-command"])
def test_parent_pipe_allows_native_child_launch_and_revokes_worker(signal):
    command = (
        "import sys,subprocess,time; "
        "from truthound.mcp.parent_pipe import watch_parent; "
        "watch_parent(sys.stdin.fileno()); "
        "subprocess.run([sys.executable,'-c','print(123,flush=True)'],check=True,timeout=5); "
        "time.sleep(30)"
    )
    process = subprocess.Popen(
        [sys.executable, "-I", "-c", command],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # Communicate the revocation after the inherited-pipe child completes.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(process.stdout.readline)
            try:
                assert future.result(timeout=10).strip() == b"123"
            except BaseException:
                process.kill()
                raise
        if signal == "extra-command":
            process.stdin.write(b"x")
            process.stdin.flush()
        else:
            process.stdin.close()
        assert process.wait(timeout=5) == 75
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream:
                stream.close()


@pytest.mark.parametrize("signal", ["eof", "extra-command"])
def test_windows_pipe_wait_does_not_issue_blocking_read(signal):
    from truthound.mcp.parent_pipe import wait_windows_pipe

    calls, pauses = [], []

    def peek(handle):
        assert handle == 42
        calls.append(handle)
        if len(calls) == 1:
            return (0, 0)
        if signal == "eof":
            raise BrokenPipeError
        return (1, 0)

    wait_windows_pipe(42, peek, pauses.append)
    assert calls == [42, 42] and pauses == [0.05]


def test_resident_parent_watch_does_not_consume_commands_and_reaps():
    command = (
        "import sys,time; "
        "from truthound.mcp.parent_pipe import watch_resident_parent; "
        "watch_resident_parent(sys.stdin.fileno()); "
        "print(sys.stdin.readline().strip(),flush=True); time.sleep(30)"
    )
    process = subprocess.Popen(
        [sys.executable, "-I", "-c", command],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        import concurrent.futures

        process.stdin.write(b"command\n")
        process.stdin.flush()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(process.stdout.readline)
            try:
                assert future.result(timeout=10).strip() == b"command"
            except BaseException:
                process.kill()
                raise
        process.stdin.close()
        assert process.wait(timeout=5) == 75
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream:
                stream.close()
