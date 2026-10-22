"""Bounded owned resident worker, killed and reaped on timeout, EOF or expiry."""

import json
import os
import subprocess
import sys
import tempfile
import threading

from truthound.mcp.policy import MCPError


class OwnedModelProcess:
    worker_module = "truthound.mcp.model_worker"
    max_reply_bytes = 65536

    def __init__(self, ttl):
        self.directory = tempfile.TemporaryDirectory(prefix="truthound-mcp-model-")
        self._timer = None
        self._closed = False
        self._close_lock = threading.Lock()
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
        except BaseException:
            self.directory.cleanup()
            raise
        self.expire_after(ttl)

    def expire_after(self, seconds):
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(max(0, seconds), self.close)
        self._timer.daemon = True
        self._timer.start()

    @property
    def closed(self):
        return self._closed or self.process.poll() is not None

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            if self._timer:
                self._timer.cancel()
            try:
                if self.process.poll() is None:
                    self.process.kill()
                self.process.wait(timeout=5)
            finally:
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
        if (
            set(result) != {"ok", "revision", "state", "trained"}
            or type(result["revision"]) is not int
            or type(result["trained"]) is not bool
        ):
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
