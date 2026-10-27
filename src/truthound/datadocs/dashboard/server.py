"""Owned Reflex backend and compiled frontend on one native HTTP listener."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

from truthound.datadocs.dashboard.app import DashboardConfig
from truthound.profiler.dashboard.server import DashboardServer


class ReflexDashboardServer(DashboardServer):
    """Serve an operator-prepared Reflex project without installing at start.

    ``prepare`` is an explicit local build operation requiring Node/npm. The
    project contains private profile data and is never a distribution artifact.
    """

    dependency = "reflex"

    def __init__(self, project_directory, config=None, *, python_executable=None):
        selected = config or DashboardConfig()
        if selected.profile_path is not None:
            selected = replace(
                selected, profile_path=str(Path(selected.profile_path).expanduser().resolve())
            )
        super().__init__(selected, python_executable=python_executable)
        self.project_directory = Path(project_directory).resolve()

    def prepare(self, *, timeout_seconds=300):
        if self.status()["running"]:
            raise RuntimeError("Stop the dashboard before preparing a project")
        if not 0 < timeout_seconds <= 600:
            raise ValueError("Preparation timeout must be in (0, 600]")
        profile_digest = self._profile_digest()
        # Never overwrite an existing project or unrelated user files.
        self.project_directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        configuration = self.project_directory / "configuration.json"
        configuration.write_text(json.dumps(asdict(self.config), allow_nan=False), encoding="utf-8")
        configuration.chmod(0o600)
        process = subprocess.Popen(
            [self.python_executable, "-m", __name__, "--prepare"],
            cwd=self.project_directory,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=os.name != "nt",
        )
        # Reuse the native process-tree stop path on timeout/interruption.
        self._process = process
        try:
            process.wait(timeout=timeout_seconds)
            if process.returncode != 0:
                raise RuntimeError("Native Reflex preparation failed")
        except BaseException:
            self.stop()
            raise
        finally:
            if process.stdin:
                process.stdin.close()
            self._process = None
        digest = hashlib.sha256(configuration.read_bytes()).hexdigest()
        if self._profile_digest() != profile_digest:
            raise ValueError("Profile changed during dashboard preparation")
        runtime = json.loads((self.project_directory / "prepared-runtime.json").read_text())
        (self.project_directory / "prepared.json").write_text(
            json.dumps(
                {
                    "config_sha256": digest,
                    "profile_sha256": profile_digest,
                    **runtime,
                }
            ),
            encoding="utf-8",
        )

    def _profile_digest(self):
        if self.config.profile_path is None:
            return None
        return hashlib.sha256(Path(self.config.profile_path).read_bytes()).hexdigest()

    def _prepare_launch(self):
        configuration = self.project_directory / "configuration.json"
        prepared = json.loads((self.project_directory / "prepared.json").read_text())
        digest = hashlib.sha256(configuration.read_bytes()).hexdigest()
        expected = json.dumps(asdict(self.config), allow_nan=False)
        if prepared["config_sha256"] != digest or configuration.read_text() != expected:
            raise ValueError("Prepared dashboard configuration has changed")
        if prepared["profile_sha256"] != self._profile_digest():
            raise ValueError("Prepared dashboard profile has changed")
        if not (self.project_directory / ".web" / "build" / "client" / "index.html").is_file():
            raise ValueError("Dashboard frontend is not prepared")
        return [self.python_executable, "-m", __name__, "--serve"], str(self.project_directory)

    def _ready(self):
        connection = http.client.HTTPConnection(self.config.host, self.config.port, timeout=0.2)
        try:
            connection.request("GET", "/ping")
            response = connection.getresponse()
            if response.status != 200 or response.read(16) != b'"pong"':
                return False
            connection.request("GET", "/")
            frontend = connection.getresponse()
            return frontend.status == 200 and frontend.getheader("content-type", "").startswith(
                "text/html"
            )
        except (OSError, http.client.HTTPException):
            return False
        finally:
            connection.close()


def _configure():
    root = Path.cwd()
    os.environ["REFLEX_DIR"] = str(root / ".reflex")
    os.environ["REFLEX_USE_NPM"] = "true"
    os.environ["NPM_CONFIG_REGISTRY"] = "https://registry.npmjs.org"
    os.environ.setdefault("NPM_CONFIG_CACHE", str(root / ".npm-cache"))
    os.environ["REFLEX_ENV_MODE"] = "prod"
    os.environ["REFLEX_COMPILE_EXECUTOR"] = "thread"
    sys.path.insert(0, str(root))


def _prepare():
    _configure()
    root = Path.cwd()
    (root / "rxconfig.py").write_text(
        """import json
from pathlib import Path
import reflex as rx
settings = json.loads(Path("configuration.json").read_text())
host = settings["host"]
address = "[" + host + "]" if ":" in host else host
url = "http://" + address + ":" + str(settings["port"])
config = rx.Config(app_name="truthound_dashboard", telemetry_enabled=False,
    backend_host=host, backend_port=settings["port"], frontend_port=settings["port"],
    api_url=url, deploy_url=url, cors_allowed_origins=[url],
    plugins=[], disable_plugins=["reflex.plugins.sitemap.SitemapPlugin"],
    show_built_with_reflex=False, db_url=None)
""",
        encoding="utf-8",
    )
    module = root / "truthound_dashboard"
    module.mkdir()
    (module / "__init__.py").write_text("", encoding="utf-8")
    (module / "truthound_dashboard.py").write_text(
        """import json
from pathlib import Path
from truthound.datadocs.dashboard.app import DashboardApp, DashboardConfig
dashboard = DashboardApp(DashboardConfig(**json.loads(Path("configuration.json").read_text())))
app = dashboard._create_app()
""",
        encoding="utf-8",
    )
    from reflex.utils.build import setup_frontend_prod
    from reflex.utils.frontend_skeleton import initialize_web_directory
    from reflex.utils.prerequisites import get_compiled_app

    initialize_web_directory()
    get_compiled_app(prerender_routes=True, use_rich=False)
    setup_frontend_prod(root)
    from importlib.metadata import version

    (root / "prepared-runtime.json").write_text(
        json.dumps({"reflex_version": version("reflex")}), encoding="utf-8"
    )


def _serve():
    _configure()
    from importlib.metadata import version

    prepared = json.loads((Path.cwd() / "prepared.json").read_text())
    if prepared["reflex_version"] != version("reflex"):
        raise RuntimeError("Prepared Reflex runtime version has changed")
    import uvicorn
    from reflex.environment import environment
    from truthound_dashboard.truthound_dashboard import app, dashboard

    environment.REFLEX_SKIP_COMPILE.set(True)
    environment.REFLEX_MOUNT_FRONTEND_COMPILED_APP.set(True)
    uvicorn.run(
        app(),
        host=dashboard.config.host,
        port=dashboard.config.port,
        log_level="debug" if dashboard.config.debug else "error",
    )


if __name__ == "__main__":
    from truthound._process_supervision import watch_resident_parent

    watch_resident_parent(sys.stdin.fileno(), own_process_group=True)
    if sys.argv[1:] == ["--prepare"]:
        _prepare()
    elif sys.argv[1:] == ["--serve"]:
        _serve()
    else:
        raise SystemExit(2)
