"""Separate an owned CLI worker's protocol from native standard descriptors."""

import os
import sys


def native_worker_wire():
    """Permanently isolate this worker; SDK atexit output stays isolated too."""
    sys.stdout.flush()
    sys.stderr.flush()
    wire = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1, encoding="utf-8", newline="\n")
    try:
        # Windows standard-stream dup() is an inheritable exception to PEP 446.
        os.set_inheritable(wire.fileno(), False)
        descriptor = os.open(os.devnull, os.O_WRONLY | getattr(os, "O_BINARY", 0))
        try:
            os.dup2(descriptor, 1)
            os.dup2(descriptor, 2)
        finally:
            os.close(descriptor)
        if os.name == "nt":
            import ctypes
            import msvcrt
            from ctypes import wintypes

            # CRT descriptors and Win32 standard handles are separate APIs.
            # Newly loaded SDK DLLs and their subprocesses must see the sink.
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            set_standard = kernel.SetStdHandle
            set_standard.argtypes = (wintypes.DWORD, wintypes.HANDLE)
            set_standard.restype = wintypes.BOOL
            for selector, fd in ((-11, 1), (-12, 2)):
                if not set_standard(selector, msvcrt.get_osfhandle(fd)):
                    raise ctypes.WinError(ctypes.get_last_error())
        # A Windows pipe can start as cp1252. Even discarded Python progress
        # must accept SDK Unicode without raising before reaching the sink.
        for stream in (sys.stdout, sys.stderr):
            stream.reconfigure(encoding="utf-8", errors="strict", newline="\n")
    except BaseException:
        wire.close()
        raise
    return wire


class CancellablePipeReader:
    """Read bounded protocol frames without a pending blocking pipe read."""

    def __init__(self, stream):
        import threading

        self.stream = stream
        self.stopped = threading.Event()
        self.lock = threading.Lock()
        self.pending = bytearray()

    def stop(self):
        self.stopped.set()

    def readline(self, limit):
        with self.lock:
            if self.stopped.is_set():
                return b""
            fd = os.dup(self.stream.fileno())
            try:
                os.set_inheritable(fd, False)
                while not self.stopped.is_set():
                    end = self.pending.find(b"\n", 0, limit)
                    size = end + 1 if end >= 0 else min(len(self.pending), limit)
                    if end >= 0 or size == limit:
                        result = bytes(self.pending[:size])
                        del self.pending[:size]
                        return result
                    if os.name == "nt":
                        import _winapi
                        import msvcrt

                        try:
                            # A zero-size peek returns (available, left), without a buffer.
                            available = _winapi.PeekNamedPipe(msvcrt.get_osfhandle(fd), 0)[0]
                        except OSError as error:
                            if error.winerror not in (109, 232):
                                raise
                            break
                    else:
                        import select

                        available = limit - len(self.pending) if select.select([fd], [], [], 0)[0] else 0
                    if available:
                        chunk = os.read(fd, min(available, limit - len(self.pending)))
                        if not chunk:
                            break
                        self.pending.extend(chunk)
                    else:
                        self.stopped.wait(0.01)
                result = bytes(self.pending)
                self.pending.clear()
                return result
            finally:
                os.close(fd)
