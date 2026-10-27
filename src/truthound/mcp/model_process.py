"""Bounded owned resident worker, killed and reaped on timeout, EOF or expiry."""

import json
import os
import subprocess
import sys
import tempfile
import threading
from contextlib import suppress

import psutil

from truthound.mcp.policy import MCPError


class OwnedModelProcess:
    worker_module = "truthound.mcp.model_worker"
    max_reply_bytes = 65536

    def __init__(self, ttl):
        self.directory = tempfile.TemporaryDirectory(prefix="truthound-mcp-model-")
        self._timer = None
        self._closed = False
        self._close_lock = threading.Lock()
        self._descendants = {}
        self._descendants_lock = threading.Lock()
        self._descendants_stop = threading.Event()
        environment = {
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
        environment.update(POLARS_MAX_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        executable = sys.executable
        if sys.platform == "win32" and executable != sys._base_executable:
            # Match multiprocessing's Windows venv launch: own the interpreter,
            # not the redirector whose termination can precede its child's exit.
            environment["__PYVENV_LAUNCHER__"] = executable
            executable = sys._base_executable
        try:
            self.process = subprocess.Popen(
                [executable, "-I", "-m", self.worker_module],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=self.directory.name,
                env=environment,
            )
            self._process_created = psutil.Process(self.process.pid).create_time()
        except BaseException:
            if hasattr(self, "process"):
                if self.process.poll() is None:
                    self.process.kill()
                self.process.wait(timeout=5)
            self.directory.cleanup()
            raise
        self._descendants_thread = threading.Thread(target=self._watch_descendants, daemon=True)
        self._descendants_thread.start()
        self.expire_after(ttl)

    def _capture_descendants(self):
        try:
            parent = psutil.Process(self.process.pid)
            if parent.create_time() != self._process_created:
                return
            for child in parent.children(recursive=True):
                try:
                    created = child.create_time()
                    with self._descendants_lock:
                        self._descendants[child.pid] = (child, created)
                except (psutil.Error, OSError):
                    continue
        except (psutil.Error, OSError):
            pass

    def _watch_descendants(self):
        while not self._descendants_stop.wait(0.05):
            self._capture_descendants()

    def expire_after(self, seconds):
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(max(0, seconds), self.close)
        self._timer.daemon = True
        self._timer.start()

    @property
    def closed(self):
        return self._closed or self.process.poll() is not None

    def _request_shutdown(self):
        """Optional bounded native cleanup, with descendants already captured."""

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            if self._timer:
                self._timer.cancel()
            try:
                self._capture_descendants()
                # Snapshot descendants before a graceful shutdown can reparent
                # them. Provider cleanup is bounded; the captured tree is reaped
                # even if the parent exits before its descendants do.
                with suppress(Exception):
                    self._request_shutdown()
                self._capture_descendants()
                self._descendants_stop.set()
                self._descendants_thread.join(timeout=1)
                with self._descendants_lock:
                    recorded = list(self._descendants.values())
                children = []
                for child, created in reversed(recorded):
                    with suppress(psutil.Error, OSError):
                        if child.create_time() == created:
                            child.kill()
                            children.append(child)
                if self.process.poll() is None:
                    self.process.kill()
                self.process.wait(timeout=5)
                if children:
                    psutil.wait_procs(children, timeout=5)
            finally:
                self._descendants_stop.set()
                for stream in (self.process.stdin, self.process.stdout):
                    if stream:
                        stream.close()
                self.directory.cleanup()

    def _exchange(self, request):
        if self.closed:
            raise MCPError("invalid_handle")
        self.process.stdin.write(json.dumps(request, allow_nan=False).encode() + b"\n")
        self.process.stdin.flush()
        response = self.process.stdout.readline(self.max_reply_bytes + 1)
        if len(response) > self.max_reply_bytes or not response.endswith(b"\n"):
            raise MCPError("worker_error")
        result = json.loads(response)
        if result.get("ok") is not True:
            raise MCPError(result.get("code", "worker_error"))
        return self.validate_reply(result)

    def validate_reply(self, result):
        extra = {"detached_spec", "detached_trained"} if "detached_spec" in result else set()
        if (
            set(result) != {"ok", "revision", "state", "trained"} | extra
            or type(result["revision"]) is not int
            or type(result["trained"]) is not bool
        ):
            raise MCPError("worker_error")
        if extra:
            from truthound.mcp.advanced_contracts import ModelSpec

            ModelSpec.model_validate(result["detached_spec"])
            if type(result["detached_trained"]) is not bool:
                raise MCPError("worker_error")
        return result

    async def call(self, request, timeout):
        import anyio

        try:
            with anyio.fail_after(timeout):
                return await anyio.to_thread.run_sync(
                    self._exchange, request, abandon_on_cancel=True
                )
        except MCPError as exc:
            if exc.code == "worker_error":
                self.close()
            raise
        except BaseException:
            # A protocol failure or cancellation invalidates resident state.
            # Never continue with a model whose mutation outcome is uncertain.
            self.close()
            raise
