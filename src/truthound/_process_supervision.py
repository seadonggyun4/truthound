"""Parent disconnect supervision for resident native child processes."""

import os
import threading


def watch_resident_parent(fd, *, own_process_group=False):
    """Watch only disconnect, without consuming a resident worker's commands."""
    if os.name == "nt":
        import _winapi
        import msvcrt

        handle = msvcrt.get_osfhandle(fd)

        def disconnected():
            try:
                _winapi.PeekNamedPipe(handle)
                return False
            except OSError:
                return True
    else:
        import select

        poller = select.poll()
        poller.register(fd, select.POLLHUP | select.POLLERR | select.POLLNVAL)

        def disconnected():
            return bool(poller.poll(0))

    def supervise():
        pause = threading.Event()
        while not pause.wait(0.05):
            if disconnected():
                if own_process_group:
                    if os.name == "nt":
                        import subprocess

                        subprocess.run(
                            ["taskkill", "/PID", str(os.getpid()), "/T", "/F"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=False,
                        )
                    elif os.getpgrp() == os.getpid():
                        import signal

                        os.killpg(os.getpid(), signal.SIGKILL)
                os._exit(75)

    threading.Thread(target=supervise, daemon=True).start()


def watch_process_lease(connection, *, own_process_group=False):
    """A private parent-owned pipe fences a native timeout child after parent death."""
    import contextlib

    def supervise():
        while True:
            try:
                if not connection.poll(0.05):
                    continue
                connection.recv_bytes()
            except (EOFError, OSError):
                if own_process_group:
                    if os.name == "nt":
                        import subprocess

                        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
                            subprocess.run(
                                ["taskkill", "/PID", str(os.getpid()), "/T", "/F"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                check=False, timeout=2,
                            )
                    elif os.getpgrp() == os.getpid():
                        import signal

                        os.killpg(os.getpid(), signal.SIGKILL)
                os._exit(75)

    threading.Thread(target=supervise, daemon=True).start()
