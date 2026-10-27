"""Bounded local SDK probe runner; reap only its recorded process tree."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path

import psutil


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
    def __init__(self, backend, log_path):
        self.directory = tempfile.TemporaryDirectory(prefix="truthound-native-driver-")
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.owned = {}
        self.ownership_lock = threading.Lock()
        self.stop_monitor = threading.Event()
        self.closed = False
        try:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    str(Path(__file__).with_name("native_distributed_probe.py")),
                    backend,
                    "--log-path",
                    str(log_path),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=self.directory.name,
                env=environment(self.directory.name),
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

    def receive(self):
        line = self.pool.submit(self.process.stdout.readline, 1024 * 1024 + 1).result(timeout=30)
        self.capture()
        if not line.endswith(b"\n") or len(line) > 1024 * 1024:
            raise RuntimeError("Native runtime did not return a bounded protocol reply")
        value = json.loads(line)
        if value.get("ok") is not True:
            raise RuntimeError("Native runtime failed: " + json.dumps(value, sort_keys=True))
        return value

    def call(self, command):
        self.process.stdin.write(json.dumps({"command": command}).encode() + b"\n")
        self.process.stdin.flush()
        return self.receive()["value"]

    def close(self):
        if self.closed:
            return
        self.closed = True
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
