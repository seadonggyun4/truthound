"""Keep the real SDK argv protocol, configuration and scoped ownership intact."""

import threading
from types import SimpleNamespace

import pytest

from truthound.execution.distributed import _spark_startup as startup

pytestmark = pytest.mark.contract


def wire(args, prefix=b"\0\r\n"):
    return prefix + b"\0".join(arg.encode() for arg in args) + b"\0"


def test_sdk_nul_arguments_preserve_unicode_spaces_quotes_and_environment(tmp_path):
    java = str(tmp_path / "java")
    args = [java, "org.apache.spark.deploy.SparkSubmit", "--conf",
            'spark.app.name= 한글 & "literal" $(unchanged) ', "", "pyspark-shell"]
    observed, environment = startup._decode_command(
        wire(["env", "PYTHONPATH=/synthetic path", *args]), {"original": "retained"}, java,
    )
    assert observed == args
    assert environment == {"original": "retained", "PYTHONPATH": "/synthetic path"}


@pytest.mark.parametrize("bad", ["framing", "encoding", "program", "class", "resource", "oversized"])
def test_invalid_sdk_output_never_becomes_an_executable_command(tmp_path, bad):
    java = str(tmp_path / "java")
    args = [java, "org.apache.spark.deploy.SparkSubmit", "pyspark-shell"]
    if bad == "program":
        args[0] = str(tmp_path / "unapproved-program")
    elif bad == "class":
        args[1] = "unapproved-class"
    elif bad == "resource":
        args[2] = "unapproved-resource"
    raw = wire(args)
    if bad == "framing":
        raw = raw[1:]
    elif bad == "encoding":
        raw = b"\0\n\xff\0"
    elif bad == "oversized":
        raw = b"\0\n" + b"x" * (1024 * 1024) + b"\0"
    with pytest.raises(RuntimeError) as caught:
        startup._decode_command(raw, {}, java)
    assert str(tmp_path) not in str(caught.value)


def test_sdk_hook_restores_after_builder_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(startup, "_stock_windows_scripts", lambda home: True)
    from pyspark import java_gateway

    original = java_gateway.Popen
    failure = RuntimeError("synthetic builder failure")

    def fail(*args):
        raise failure

    monkeypatch.setattr(startup, "_gateway_command", fail)
    command = [str(tmp_path / "bin/spark-submit.cmd"), "--conf", "spark.master=local[1]"]
    with pytest.raises(RuntimeError) as caught, startup.local_gateway_launcher(force=True):
        java_gateway.Popen(command, env={})
    assert caught.value is failure and java_gateway.Popen is original


def test_other_threads_delegate_and_real_sdk_process_options_are_preserved(monkeypatch, tmp_path):
    monkeypatch.setattr(startup, "_stock_windows_scripts", lambda home: True)
    from pyspark import java_gateway

    command = [str(tmp_path / "bin/spark-submit.cmd"), "--conf", "spark.master=local[1]"]
    generated = ["sdk-java", "sdk-generated-arguments"]
    calls, builds = [], []

    def popen(args, **kwargs):
        calls.append((args, kwargs))
        return "owned-process"

    def build(args, environment, remaining):
        assert 0 < remaining <= 30
        builds.append(args)
        return generated, {"sdk": "environment"}

    monkeypatch.setattr(java_gateway, "Popen", popen)
    monkeypatch.setattr(startup, "_gateway_command", build)
    with startup.local_gateway_launcher(force=True) as state:
        assert java_gateway.Popen(command, env={}, stdin=123) == "owned-process"
        thread = threading.Thread(target=lambda: java_gateway.Popen(command, env={}, stdin=456))
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert state["launches"] == 1
    assert builds == [command]
    assert calls == [(generated, {"env": {"sdk": "environment"}, "stdin": 123}),
                     (command, {"env": {}, "stdin": 456})]
    assert java_gateway.Popen is popen


@pytest.mark.parametrize("mode", ["external", "overridden", "remote", "home", "conf", "script", "popen"])
def test_operator_configuration_and_external_clusters_retain_sdk_launch(monkeypatch, tmp_path, mode):
    from pyspark import java_gateway

    command = [str(tmp_path / "bin/spark-submit.cmd"), "--conf", "spark.master=local[1]"]
    environment = {}
    options = {}
    if mode == "external":
        command[-1] = "spark.master=spark://synthetic-cluster:7077"
    elif mode == "overridden":
        command += ["--master", "spark://synthetic-cluster:7077"]
    elif mode == "remote":
        command += ["--conf", "spark.remote=sc://synthetic-cluster:15002"]
    elif mode == "home":
        environment["SPARK_HOME"] = "operator-owned"
    elif mode == "conf":
        environment["SPARK_CONF_DIR"] = "operator-owned"
    elif mode == "script":
        (tmp_path / "conf").mkdir()
        (tmp_path / "conf/spark-env.cmd").write_text("operator-owned")
    else:
        options["creationflags"] = 123
    calls = []
    monkeypatch.setattr(java_gateway, "Popen", lambda args, **kwargs: calls.append((args, kwargs)))

    def forbidden(*args):
        pytest.fail("The operator's SDK path must remain unchanged")

    monkeypatch.setattr(startup, "_gateway_command", forbidden)
    with startup.local_gateway_launcher(force=True):
        java_gateway.Popen(command, env=environment, **options)
    assert calls == [(command, {"env": environment, **options})]


@pytest.mark.parametrize("platform,version", [("linux", "4.2.0"), ("win32", "4.1.0")])
def test_other_platforms_and_sdk_versions_keep_original_popen(monkeypatch, platform, version):
    import pyspark
    from pyspark import java_gateway

    original = java_gateway.Popen
    monkeypatch.setattr(startup, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(pyspark, "__version__", version)
    with startup.local_gateway_launcher():
        assert java_gateway.Popen is original


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_execution_without_optional_spark_keeps_original_backend(monkeypatch, platform):
    import sys

    monkeypatch.setattr(startup, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setitem(sys.modules, "pyspark", None)
    with startup.local_gateway_launcher() as state:
        assert state["launches"] == 0


def test_nested_owned_startup_reuses_builder_and_original_deadline(monkeypatch, tmp_path):
    monkeypatch.setattr(startup, "_stock_windows_scripts", lambda home: True)
    from pyspark import java_gateway

    command = [str(tmp_path / "bin/spark-submit.cmd"), "--conf", "spark.master=local[1]"]
    calls = []
    def original(args, **kwargs):
        return "owned"
    monkeypatch.setattr(java_gateway, "Popen", original)

    def build(args, environment, remaining):
        calls.append(remaining)
        return ["java"], environment

    monkeypatch.setattr(startup, "_gateway_command", build)
    with startup.local_gateway_launcher(force=True) as outer:
        with startup.local_gateway_launcher(force=True) as inner:
            assert inner is outer
            assert java_gateway.Popen(command, env={}) == "owned"
        assert outer["launches"] == 1
    assert len(calls) == 1 and 0 < calls[0] <= 30
    assert java_gateway.Popen is original


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("resident", [False, True])
def test_resident_unregisters_only_callback_for_its_exact_process_and_reaps_failed_init(monkeypatch, tmp_path, fail, resident):
    monkeypatch.setattr(startup, "_stock_windows_scripts", lambda home: True)
    from pyspark import java_gateway

    events = []
    process = SimpleNamespace(
        poll=lambda: None, kill=lambda: events.append("kill"),
        wait=lambda timeout: events.append(("wait", timeout)),
    )

    def callback_for(owned):
        def killChild():
            return owned

        killChild.__module__ = "pyspark.java_gateway"
        return killChild

    ours, borrowed = callback_for(process), callback_for(object())
    original_exit = SimpleNamespace(
        register=lambda callback: events.append(("register", callback)),
        unregister=lambda callback: events.append(("unregister", callback)),
    )
    monkeypatch.setattr(java_gateway, "atexit", original_exit)
    monkeypatch.setattr(java_gateway, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(startup, "_gateway_command", lambda args, env, remaining: (["java"], env))
    command = [str(tmp_path / "bin/spark-submit.cmd"), "--conf", "spark.master=local[1]"]
    try:
        with startup.local_gateway_launcher(force=True, resident=resident) as state:
            java_gateway.Popen(command, env={})
            java_gateway.atexit.register(ours)
            java_gateway.atexit.register(borrowed)
            if fail:
                raise ValueError("synthetic SDK init failure")
    except ValueError:
        assert fail
    expected = [("register", ours), ("register", borrowed)]
    if fail:
        expected += ["kill", ("wait", 1)]
    if resident or fail:
        expected += [("unregister", ours)]
    assert events == expected and state["cancelled_callbacks"] == int(resident or fail)
    assert java_gateway.atexit is original_exit
    assert not hasattr(startup._ACTIVE, "context")


def test_installed_sdk_scripts_are_verified_but_modified_script_delegates(tmp_path):
    from pathlib import Path

    import pyspark

    home = Path(pyspark.__file__).parent
    assert startup._stock_windows_scripts(home)
    (tmp_path / "bin").mkdir()
    for name in startup._STOCK_SCRIPTS:
        (tmp_path / "bin" / name).write_bytes((home / "bin" / name).read_bytes())
    assert startup._stock_windows_scripts(tmp_path)
    (tmp_path / "bin/spark-submit.cmd").write_bytes(b"operator-defined startup")
    assert not startup._stock_windows_scripts(tmp_path)


def test_exhausted_original_startup_budget_cannot_launch_a_late_jvm(monkeypatch, tmp_path):
    from pyspark import java_gateway

    ticks = iter([100.0, 100.0, 131.0])
    monkeypatch.setattr(startup, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    monkeypatch.setattr(startup, "_stock_windows_scripts", lambda home: True)
    monkeypatch.setattr(startup, "_gateway_command", lambda args, env, remaining: (["java"], env))

    def forbidden(*args, **kwargs):
        pytest.fail("No JVM may start after the original deadline")

    monkeypatch.setattr(java_gateway, "Popen", forbidden)
    command = [str(tmp_path / "bin/spark-submit.cmd"), "--conf", "spark.master=local[1]"]
    with pytest.raises(TimeoutError), startup.local_gateway_launcher(force=True):
        java_gateway.Popen(command, env={})
    assert java_gateway.Popen is forbidden and not hasattr(startup._ACTIVE, "context")
