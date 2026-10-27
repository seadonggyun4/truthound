"""Scoped owned-local Spark startup through the real SDK command builder."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

_STARTUP_LOCK = threading.RLock()
_ACTIVE = threading.local()
_MAX_COMMAND_BYTES = 1024 * 1024
_STOCK_SCRIPTS = {
    "find-spark-home.cmd": "c3b3a92bc39a6bb2a9e52b9724f0e33a3072abb4696e56d8166417e760be1fa1",
    "load-spark-env.cmd": "5d57c722a10dbb050fa118e1c410caecfd1e3e8d537313e4a63142db288fe161",
    "spark-class2.cmd": "fac2af5af5d8bf24acd68eb73e3fffad3b79016aab25f3716deeee0b3321792a",
    "spark-submit.cmd": "0ef22b5c64a23bda5b5e165e50826145415707f6a6e2f7364df92ef9f84f4420",
    "spark-submit2.cmd": "a0dfdebecef1691b9c275df4c27dbf09a40495035d3c59c4547d962937ede627",
}


def _stock_windows_scripts(home):
    """Delegate modified/operator SDK scripts; never modify their installation."""
    try:
        for name, expected in _STOCK_SCRIPTS.items():
            path = home / "bin" / name
            if path.is_symlink():
                return False
            with path.open("rb") as stream:
                raw = stream.read(32769)
            if len(raw) > 32768 or hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() != expected:
                return False
    except OSError:
        return False
    return True


def _decode_command(raw, environment, java):
    """Decode the SDK's NUL protocol without shell parsing or whitespace loss."""
    prefix = b"\0\r\n" if raw.startswith(b"\0\r\n") else b"\0\n"
    if len(raw) > _MAX_COMMAND_BYTES or not raw.startswith(prefix) or not raw.endswith(b"\0"):
        raise RuntimeError("Invalid Spark SDK launcher protocol")
    try:
        args = raw[len(prefix):-1].decode("utf-8").split("\0")
    except UnicodeError:
        raise RuntimeError("Invalid Spark SDK launcher encoding") from None
    child_environment = dict(environment)
    if args and args[0] == "env":
        args.pop(0)
        while args and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", args[0], re.DOTALL):
            key, value = args.pop(0).split("=", 1)
            child_environment[key] = value
    if (not args or Path(args[0]).resolve() != Path(java).resolve()
            or "org.apache.spark.deploy.SparkSubmit" not in args or "pyspark-shell" not in args):
        raise RuntimeError("Unexpected Spark SDK launcher command")
    return args, child_environment


def _gateway_command(command, environment, remaining):
    home = Path(command[0]).resolve().parents[1]
    jars = list((home / "jars").glob("spark-launcher_2.13-4.2.*.jar"))
    if len(jars) != 1:
        raise RuntimeError("Spark SDK launcher JAR is not uniquely installed")
    selected = environment.get("JAVA_HOME")
    java = str(Path(selected) / "bin" / ("java.exe" if os.name == "nt" else "java")) if selected else shutil.which("java", path=environment.get("PATH"))
    if not java or not Path(java).is_file():
        raise RuntimeError("Spark SDK Java executable is unavailable")
    if remaining <= 0:
        raise TimeoutError("Spark local startup exceeded its deadline")
    child_environment = dict(environment)
    child_environment.update(
        SPARK_HOME=str(home), SPARK_CONF_DIR=str(home / "conf"),
        SPARK_ENV_LOADED="1", SPARK_SCALA_VERSION="2.13", PYTHONHASHSEED="0",
    )
    child_environment.pop("SHELL", None)
    builder_environment = dict(child_environment, SHELL="truthound-sdk-nul-protocol")
    # Main explicitly supports NUL output when SHELL is set, including Windows.
    # A single real SDK JAR avoids searching every Spark JAR for the builder.
    with tempfile.TemporaryFile() as output:
        try:
            result = subprocess.run(
                [java, "-Xmx128m", "-Dfile.encoding=UTF-8", "-cp", str(jars[0]),
                 "org.apache.spark.launcher.Main", "org.apache.spark.deploy.SparkSubmit",
                 *command[1:]],
                env=builder_environment, stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.DEVNULL, timeout=remaining, check=False,
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError("Spark local startup exceeded its deadline") from None
        if result.returncode:
            raise RuntimeError("Spark SDK command builder failed")
        output.seek(0)
        raw = output.read(_MAX_COMMAND_BYTES + 1)
    return _decode_command(raw, child_environment, java)


@contextmanager
def local_gateway_launcher(*, force=False, progress=None, resident=False):
    """Adapt owned Windows 4.2 local launches; keep SDK auth/readiness/cleanup."""
    state = {"launches": 0, "cancelled_callbacks": 0}
    if not (force or sys.platform == "win32"):
        yield state
        return
    try:
        import pyspark
        from pyspark import java_gateway
    except ImportError:
        yield state
        return
    if not pyspark.__version__.startswith("4.2."):
        yield state
        return
    active = getattr(_ACTIVE, "context", None)
    if active is not None:
        previous = active["progress"]
        if progress is not None:
            active["progress"] = progress
        if resident:
            active["resident"] = True
        try:
            yield active["state"]
        finally:
            active["progress"] = previous
        return
    with _STARTUP_LOCK:
        owner = threading.get_ident()
        original = java_gateway.Popen
        original_exit = java_gateway.atexit
        processes, callbacks = [], []
        deadline = time.monotonic() + 30
        active = {"state": state, "progress": progress, "resident": resident}
        _ACTIVE.context = active
        failed = False

        def launch(command, **kwargs):
            if threading.get_ident() != owner or not isinstance(command, (list, tuple)) or not command:
                return original(command, **kwargs)
            supported = {"stdin", "env"}
            if force and os.name != "nt":
                supported.add("preexec_fn")  # The SDK's stock Unix SIGINT setup.
            if kwargs.keys() - supported:
                return original(command, **kwargs)
            environment = kwargs.get("env", {})
            home = Path(command[0]).resolve().parents[1]
            pairs = list(zip(command[1:], command[2:], strict=False))
            masters = [value.split("=", 1)[1] for flag, value in pairs
                       if flag == "--conf" and value.startswith("spark.master=")]
            masters += [value for flag, value in pairs if flag == "--master"]
            local = masters and all(re.fullmatch(r"local(?:\[(?:\*|[1-9][0-9]*)(?:,[1-9][0-9]*)?\])?", value) for value in masters)
            remote = "--remote" in command or any(value.startswith("spark.remote=") or value == "spark.api.mode=connect" for value in command)
            if (not local or remote
                    or Path(command[0]).name not in {"spark-submit", "spark-submit.cmd"}
                    or environment.get("SPARK_HOME") or environment.get("SPARK_CONF_DIR")
                    or environment.get("SPARK_ENV_LOADED") or environment.get("SPARK_PREPEND_CLASSES")
                    or environment.get("SPARK_PRINT_LAUNCH_COMMAND")
                    or (home / "conf" / "spark-env.cmd").exists()
                    or (home / "conf" / "spark-env.sh").exists()
                    or (home / "conf" / "spark-defaults.conf").exists()
                    or not _stock_windows_scripts(home)):
                return original(command, **kwargs)
            callback = active["progress"]
            if callback:
                callback("spark-launcher-build")
            args, child_environment = _gateway_command(command, environment, deadline - time.monotonic())
            if time.monotonic() >= deadline:
                raise TimeoutError("Spark local startup exceeded its deadline")
            if callback:
                callback("spark-launcher-built")
            result = original(args, **{**kwargs, "env": child_environment})
            if isinstance(result, subprocess.Popen):
                result._truthound_owned_local_jvm = True
            processes.append(result)
            state["launches"] += 1
            return result

        class ResidentExitObserver:
            def register(self, callback, *args, **kwargs):
                result = original_exit.register(callback, *args, **kwargs)
                if (threading.get_ident() == owner and not args and not kwargs
                        and getattr(callback, "__module__", None) == "pyspark.java_gateway"
                        and getattr(callback, "__name__", None) == "killChild"):
                    for cell in getattr(callback, "__closure__", None) or ():
                        try:
                            process = cell.cell_contents
                        except ValueError:
                            continue
                        if any(process is owned for owned in processes):
                            callbacks.append(callback)
                            break
                return result

            def __getattr__(self, name):
                return getattr(original_exit, name)

        java_gateway.Popen = launch
        java_gateway.atexit = ResidentExitObserver()
        try:
            yield state
        except BaseException:
            failed = True
            # These are exact Popen handles created here, never borrowed PIDs.
            # Reap failed initialization before the worker can orphan its JVM.
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=1)
            raise
        finally:
            java_gateway.Popen = original
            java_gateway.atexit = original_exit
            try:
                if active["resident"] or failed:
                    for callback in callbacks:
                        original_exit.unregister(callback)
                        state["cancelled_callbacks"] += 1
            finally:
                processes.clear()
                del _ACTIVE.context
