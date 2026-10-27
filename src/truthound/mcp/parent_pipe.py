"""One-shot worker ownership without a pending Windows pipe read."""

import os
import threading


def wait_windows_pipe(handle, peek, pause):
    # A pending synchronous ReadFile can block another operation on the same
    # inherited pipe (including child runtime handle initialization). Poll
    # availability instead; no competing read is active after request parsing.
    while True:
        try:
            if peek(handle)[0]:
                return
        except OSError:
            return
        pause(0.05)


def watch_parent(fd):
    if os.name == "nt":
        import _winapi
        import msvcrt

        handle = msvcrt.get_osfhandle(fd)

        def wait():
            wait_windows_pipe(handle, _winapi.PeekNamedPipe, threading.Event().wait)
    else:

        def wait():
            os.read(fd, 1)

    def supervise():
        try:
            wait()
        finally:
            os._exit(75)

    threading.Thread(target=supervise, daemon=True).start()


# Shared with native process owners, including the managed dashboard server.
from truthound._process_supervision import watch_resident_parent  # noqa: E402,F401
