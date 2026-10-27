"""Owned loopback HTTP receiver for native OpenLineage verification."""

import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace


@contextmanager
def receiver():
    state = SimpleNamespace(events=[], status=200, delay=0.0, entered=threading.Event())

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            size = int(self.headers.get("Content-Length", "0"))
            if size > 1024 * 1024:
                self.send_error(413)
                return
            state.events.append(json.loads(self.rfile.read(size)))
            state.entered.set()
            time.sleep(state.delay)
            self.send_response(state.status)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.endpoint = f"http://127.0.0.1:{server.server_port}/api/v1/lineage"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError("Owned receiver did not stop")
