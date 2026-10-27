"""Managed native Streamlit dashboard process with explicit readiness and cleanup."""

from __future__ import annotations

import http.client
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path

from truthound.profiler.dashboard.config import DashboardConfig, DashboardTheme


class DashboardServer:
    """Own one native dashboard process; config is data, never generated Python."""

    dependency = "streamlit"

    def __init__(
        self, config: DashboardConfig | None = None, *, python_executable: str | None = None
    ):
        self.config = config or DashboardConfig()
        self.python_executable = python_executable or sys.executable
        self._process = None
        self._directory = None

    def start(self, *, timeout_seconds: float = 10):
        if self._process is not None and self._process.poll() is None:
            return self.status()
        if not 0 < timeout_seconds <= 30:
            raise ValueError("Dashboard startup timeout must be in (0, 30]")
        if not 1 <= self.config.port <= 65535:
            raise ValueError("Dashboard port must be in 1..65535")
        if (
            self.python_executable == sys.executable
            and importlib.util.find_spec(self.dependency) is None
        ):
            raise ImportError(f"{self.dependency} is required for the native dashboard")
        self.close()
        # Never report readiness from an unrelated service already on this port.
        listeners = []
        try:
            for family, socktype, protocol, _, address in socket.getaddrinfo(
                self.config.host, self.config.port, type=socket.SOCK_STREAM
            ):
                listener = socket.socket(family, socktype, protocol)
                listeners.append(listener)
                listener.bind(address)
        finally:
            for listener in listeners:
                listener.close()
        try:
            argv, directory = self._prepare_launch()
            options = (
                {"start_new_session": True}
                if os.name != "nt"
                else {
                    "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP,
                }
            )
            self._process = subprocess.Popen(
                argv,
                cwd=directory,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **options,
            )
            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    raise RuntimeError("Native dashboard process exited during startup")
                if self._ready():
                    return self.status()
                time.sleep(0.05)
            raise TimeoutError("Native dashboard did not become ready")
        except BaseException:
            self.close()
            raise

    def _prepare_launch(self):
        directory = tempfile.TemporaryDirectory(prefix="truthound-dashboard-")
        self._directory = directory
        path = Path(directory.name) / "configuration.json"
        payload = asdict(self.config)
        payload["theme"] = (
            self.config.theme.value
            if isinstance(self.config.theme, DashboardTheme)
            else self.config.theme
        )
        path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        path.chmod(0o600)
        return [
            self.python_executable,
            str(Path(__file__).resolve()),
            "--serve",
            str(path),
        ], directory.name

    def _ready(self):
        connection = http.client.HTTPConnection(self.config.host, self.config.port, timeout=0.2)
        try:
            connection.request("GET", "/_stcore/health")
            response = connection.getresponse()
            return response.status == 200 and response.read(16) == b"ok"
        except (OSError, http.client.HTTPException):
            return False
        finally:
            connection.close()

    def status(self):
        process = self._process
        running = process is not None and process.poll() is None
        host = "[" + self.config.host + "]" if ":" in self.config.host else self.config.host
        return {
            "running": running,
            "ready": running and self._ready(),
            "url": f"http://{host}:{self.config.port}" if running else None,
            "exit_code": process.poll() if process is not None else None,
        }

    def stop(self):
        process = self._process
        if process is not None and process.poll() is None:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait(timeout=5)
        if process is not None and process.stdin:
            process.stdin.close()
        return self.status()

    def wait(self):
        if self._process is None:
            raise RuntimeError("Dashboard has not started")
        return self._process.wait()

    def close(self):
        self.stop()
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def render(configuration_path):
    from truthound.profiler.dashboard.app import DashboardApp

    payload = json.loads(Path(configuration_path).read_text(encoding="utf-8"))
    payload["theme"] = DashboardTheme(payload["theme"])
    DashboardApp(DashboardConfig(**payload)).run()


if __name__ == "__main__":
    if sys.argv[1] == "--serve":
        from streamlit.web import cli

        from truthound._process_supervision import watch_resident_parent

        watch_resident_parent(sys.stdin.fileno(), own_process_group=True)
        configuration = Path(sys.argv[2])
        settings = json.loads(configuration.read_text(encoding="utf-8"))
        sys.argv = [
            "streamlit",
            "run",
            str(Path(__file__).resolve()),
            "--server.address",
            settings["host"],
            "--server.port",
            str(settings["port"]),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
            "--",
            str(configuration),
        ]
        cli.main()
    else:
        render(sys.argv[1])
