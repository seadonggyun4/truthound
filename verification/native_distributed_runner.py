"""Bounded local SDK probe runner; reap only its recorded process tree."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path

import psutil

COMMAND_LABELS = frozenset(
    {
        "basics",
        "sample",
        "polars",
        "aggregations",
        "values",
        "aggregation_plan",
        "aggregation_spec",
        "cache_configuration",
        "aggregation_cache_contract",
        "ray_configuration",
        "spark_gateway_owner",
        "statistics_null", "statistics_empty", "statistics_single", "statistics_nan", "statistics_literal",
        "statistics_nan_mean", "statistics_nan_std",
    }
)


def environment(home=None):
    result = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "PATH",
            "SYSTEMROOT",
            "WINDIR",
            "TEMP",
            "TMP",
            "PROCESSOR_ARCHITECTURE",
            "PROCESSOR_ARCHITEW6432",
            "HADOOP_HOME",
        }
    }
    result.update(
        POLARS_MAX_THREADS="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        RAY_USAGE_STATS_ENABLED="0",
        RAY_ADDRESS="local",
        SPARK_LOCAL_IP="127.0.0.1",
        SPARK_LOCAL_HOSTNAME="localhost",
        PYSPARK_PYTHON=sys.executable,
    )
    if sys.platform == "darwin":
        result["JAVA_HOME"] = subprocess.check_output(
            ["/usr/libexec/java_home", "-v", "17"], text=True
        ).strip()
    else:
        selected = next(
            (
                os.environ[key]
                for key in ("JAVA_HOME_17_X64", "JAVA_HOME_21_X64", "JAVA_HOME")
                if os.environ.get(key)
            ),
            None,
        )
        if selected:
            result["JAVA_HOME"] = selected
    if home is not None:
        result.update(HOME=str(home), USERPROFILE=str(home))
    return result


class OwnedProbe:
    def __init__(self, backend, log_path, *, guard_node_lookup=False):
        self.log_path = Path(log_path)
        self.directory = tempfile.TemporaryDirectory(prefix="truthound-native-driver-")
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.owned = {}
        self.ownership_lock = threading.Lock()
        self.capture_count = 0
        self.capture_wall = 0.0
        self.capture_cpu = 0.0
        self.stop_monitor = threading.Event()
        self.closed = False
        native_environment = environment(self.directory.name)
        executable = sys.executable
        if sys.platform == "win32" and executable != sys._base_executable:
            native_environment["__PYVENV_LAUNCHER__"] = executable
            executable = sys._base_executable
        try:
            self.process = subprocess.Popen(
                [
                    executable,
                    "-I",
                    str(Path(__file__).with_name("native_distributed_probe.py")),
                    backend,
                    "--log-path",
                    str(log_path),
                    *(["--guard-node-lookup"] if guard_node_lookup else []),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=self.directory.name,
                env=native_environment,
            )
            self.created = psutil.Process(self.process.pid).create_time()
        except BaseException:
            if hasattr(self, "process"):
                if self.process.poll() is None:
                    self.process.kill()
                    self.process.wait(timeout=5)
                for stream in (self.process.stdin, self.process.stdout):
                    stream.close()
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.directory.cleanup()
            raise
        self.monitor = threading.Thread(target=self._monitor, daemon=True)
        self.monitor.start()

    def _monitor(self):
        while not self.stop_monitor.wait(0.05):
            self.capture()

    def capture(self):
        started, cpu = time.monotonic(), time.thread_time()
        try:
            root = psutil.Process(self.process.pid)
            if root.create_time() == self.created:
                for child in root.children(recursive=True):
                    try:
                        created = child.create_time()
                        with self.ownership_lock:
                            self.owned[child.pid] = created
                    except (psutil.Error, OSError):
                        continue
        except (psutil.Error, OSError):
            pass
        finally:
            with self.ownership_lock:
                self.capture_count += 1
                self.capture_wall += time.monotonic() - started
                self.capture_cpu += time.thread_time() - cpu

    def _native_startup_state(self):
        markers = {
            "gcs_started": (b"GcsServer server started",),
            "gcs_connection_failed": (b"Failed to connect to GCS",),
            "raylet_started": (b"started. Raylet consists of node_manager and object_manager",),
            "node_registered": (b"Finished registering node.",),
            "port_bind_failed": (
                b"Address already in use",
                b"Only one usage of each socket address",
            ),
            "module_missing": (b"No module named",),
            "worker_registration_timeout": (b"not registered within the timeout",),
        }
        observed = {key: False for key in markers}
        count = 0
        root = Path(self.directory.name)
        for logs in root.glob("th-native-*/session_*/logs"):
            if logs.parent.is_symlink() or not logs.resolve().is_relative_to(root.resolve()):
                continue
            for name in (
                "gcs_server.out",
                "gcs_server.err",
                "raylet.out",
                "raylet.err",
                "dashboard.log",
                "monitor.log",
            ):
                path = logs / name
                if path.is_symlink() or not path.is_file():
                    continue
                count += 1
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 65536))
                    tail = stream.read(65536)
                for key, tokens in markers.items():
                    observed[key] |= any(token in tail for token in tokens)
        driver_markers = {
            "jvm_started": b"Using Spark's default log4j profile",
            "spark_context_started": b"Running Spark version",
            "driver_service_started": b"Successfully started service 'sparkDriver'",
            "block_manager_started": b"Initialized BlockManager",
            "gateway_exited": b"JAVA_GATEWAY_EXITED",
            "hadoop_native_missing": b"UnsatisfiedLinkError",
            "port_bind_failed": b"failed after 16 retries",
        }
        driver = b""
        if self.log_path.is_file() and not self.log_path.is_symlink():
            with self.log_path.open("rb") as stream:
                stream.seek(max(0, self.log_path.stat().st_size - 65536))
                driver = stream.read(65536)
        return {
            "log_files": count,
            "markers": observed,
            "spark_markers": {key: token in driver for key, token in driver_markers.items()},
        }

    def _diagnostic_report(self):
        allowed = {"core-import", "core-imported"} | {
            backend + "-" + stage
            for backend in ("ray", "spark")
            for stage in (
                "import",
                "imported",
                "start",
                "started",
                "gateway-start",
                "gateway-started",
                "gateway-reaped",
                "dataset",
                "frame",
                "ready",
            )
        }
        allowed |= {
            "startup-unclassified-wait",
            "ray-api-server-wait",
            "ray-gcs-wait",
            "ray-node-wait",
            "spark-gateway-wait",
            "spark-jvm-rpc-wait",
        }
        phases = []
        elapsed = {}
        sdk_frames = []
        if self.log_path.is_file():
            with self.log_path.open("rb") as stream:
                stream.seek(max(0, self.log_path.stat().st_size - 65536))
                for line in stream.read(65536).decode("utf-8", errors="replace").splitlines():
                    match = re.fullmatch(
                        r'  File "[^"\n]*[/\\]site-packages[/\\](ray|pyspark|py4j)[/\\]([^"\n]+\.py)", line ([0-9]{1,7}) in ([A-Za-z0-9_<>]{1,128})',
                        line,
                    )
                    if match and len(sdk_frames) < 16:
                        package, relative, number, function = match.groups()
                        parts = re.split(r"[/\\]", relative)
                        if ".." not in parts and re.fullmatch(r"[A-Za-z0-9_]+\.py", parts[-1]):
                            sdk_frames.append(
                                {
                                    "package": package,
                                    "file": parts[-1],
                                    "line": int(number),
                                    "function": function,
                                }
                            )
                    if line.startswith("native-probe:") and line[13:] in allowed:
                        phases.append(line[13:])
                    if line.startswith("native-probe-time:"):
                        parts = line.split(":")
                        if len(parts) == 3 and parts[1] in allowed:
                            try:
                                value = float(parts[2])
                            except ValueError:
                                continue
                            if math.isfinite(value) and 0 <= value <= 3600:
                                elapsed[parts[1]] = value
        try:
            startup = self._native_startup_state()
        except OSError:
            startup = {"unavailable": True}
        return {
            "phases": phases,
            "active_command": self.current_command
            if getattr(self, "current_command", None) in COMMAND_LABELS
            else None,
            "protocol_reply": getattr(self, "protocol_reply", None),
            "phase_elapsed_seconds": elapsed,
            "sdk_wait_frames": sdk_frames,
            "returncode": self.process.poll(),
            "native_startup": startup,
            "capture_count": getattr(self, "capture_count", 0),
            "capture_wall_seconds": getattr(self, "capture_wall", 0.0),
            "capture_cpu_seconds": getattr(self, "capture_cpu", 0.0),
        }

    def _diagnostic_note(self):
        return "Native probe state: " + json.dumps(self._diagnostic_report())

    def receive(self):
        started = time.monotonic()
        try:
            line = self.pool.submit(self.process.stdout.readline, 1024 * 1024 + 1).result(
                timeout=30
            )
        except TimeoutError as error:
            # Only closed synthetic stage labels may leave the owned SDK log.
            self.reply_failure = error
            error.add_note(self._diagnostic_note())
            raise
        self.protocol_reply = {
            "bytes": len(line),
            "empty": not line,
            "newline": line.endswith(b"\n"),
            "elapsed_seconds": time.monotonic() - started,
        }
        self.capture()
        if not line.endswith(b"\n") or len(line) > 1024 * 1024:
            error = RuntimeError("Native runtime did not return a bounded protocol reply")
            self.reply_failure = error
            error.add_note(self._diagnostic_note())
            raise error
        value = json.loads(line)
        if value.get("ready") is True:
            self.startup_metrics = self._diagnostic_report()
        if value.get("ok") is not True:
            raise RuntimeError("Native runtime failed: " + json.dumps(value, sort_keys=True))
        return value

    def call(self, command):
        self.process.stdin.write(json.dumps({"command": command}).encode() + b"\n")
        self.process.stdin.flush()
        self.current_command = (
            command if type(command) is str and command in COMMAND_LABELS else "other"
        )
        started = time.monotonic()
        try:
            return self.receive()["value"]
        finally:
            self.last_command_metrics = {
                "command": self.current_command,
                "elapsed_seconds": time.monotonic() - started,
            }

    def close(self):
        if self.closed:
            return
        self.closed = True
        forced_driver_exit = False
        self.capture()
        try:
            if self.process.poll() is None:
                with suppress(OSError):
                    self.process.stdin.write(b'{"close":true}\n')
                    self.process.stdin.flush()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.capture()
                    forced_driver_exit = True
                    self.process.kill()
                    self.process.wait(timeout=5)
            self.stop_monitor.set()
            self.monitor.join(timeout=2)
            with self.ownership_lock:
                owned = list(self.owned.items())
            errors = []
            for pid, created in reversed(owned):
                try:
                    child = psutil.Process(pid)
                    if child.create_time() != created or child.status() == psutil.STATUS_ZOMBIE:
                        continue
                    child.terminate()
                    try:
                        child.wait(timeout=1)
                    except psutil.TimeoutExpired:
                        if child.create_time() == created:
                            child.kill()
                            child.wait(timeout=5)
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    pass
                except (psutil.Error, OSError) as error:
                    errors.append(error)
            if errors:
                raise ExceptionGroup("Native probe cleanup failed", errors)
        finally:
            failure = getattr(self, "reply_failure", None)
            if failure is not None:
                failure.add_note(
                    "Native probe termination: "
                    + json.dumps(
                        {
                            "driver_exit_code": self.process.poll(),
                            "driver_killed_after_close_deadline": forced_driver_exit,
                        }
                    )
                )
            self.stop_monitor.set()
            self.monitor.join(timeout=2)
            for stream in (self.process.stdin, self.process.stdout):
                with suppress(OSError):
                    stream.close()
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.directory.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("ray", "spark"))
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument(
        "--commands",
        nargs="+",
        choices=(
            "basics",
            "regex",
            "sample",
            "range",
            "polars",
            "aggregations",
            "values",
            "partition",
            "invalid_regex",
        ),
        default=["basics"],
    )
    args = parser.parse_args()
    runtime = OwnedProbe(args.backend, args.log_path)
    try:
        ready = runtime.receive()
        print(json.dumps({"ready": ready}), flush=True)
        for command in args.commands:
            print(json.dumps({"command": command, "value": runtime.call(command)}), flush=True)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
