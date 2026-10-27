"""Bounded instance-owned write queue with explicit uncertain-write reconciliation."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import RLock
from uuid import uuid4

from truthound.stores.base import StoreWriteError


class WriteBehind:
    def __init__(self, store, max_pending=1000):
        if type(max_pending) is not int or max_pending < 1:
            raise ValueError("max_pending must be positive")
        self.store = store
        self.max_pending = max_pending
        self.lock = RLock()
        self.queue = []
        self.future = None
        self.executor = None
        self.closed = False
        self.completed = 0
        self.guard = None

    def submit(self, item):
        snapshot = deepcopy(item)
        identity = getattr(snapshot, "id", None)
        if not isinstance(identity, str) or not identity:
            raise ValueError("Write-behind requires a stable nonempty item.id")
        with self.lock:
            if self.closed:
                raise StoreWriteError("Write-behind is closed")
            if self.queue and self.queue[0]["state"] == "unknown":
                raise StoreWriteError("Write outcome requires reconciliation")
            if len(self.queue) >= self.max_pending:
                raise StoreWriteError("Write-behind queue is full")
            self.queue.append(
                {"id": identity, "item": snapshot, "state": "pending", "token": uuid4().hex}
            )
            self._start()
        return identity

    def _start(self):
        if self.future is None or self.future.done():
            if self.executor is None:
                self.executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="truthound-cache-write"
                )
            self.future = self.executor.submit(self._drain)

    def _drain(self):
        while True:
            with self.lock:
                if not self.queue or self.queue[0]["state"] == "unknown":
                    self.future = None
                    return
                entry = self.queue[0]
                entry["state"] = "writing"
            try:
                if self.guard is not None:
                    self.guard()
                identity = self.store.save(entry["item"])
                if identity != entry["id"]:
                    raise StoreWriteError("Native write changed its record identity")
            except BaseException:
                with self.lock:
                    entry["state"] = "unknown"
                    self.future = None
                return
            with self.lock:
                self.queue.pop(0)
                self.completed += 1

    def flush(self):
        while True:
            with self.lock:
                if self.queue and self.queue[0]["state"] == "unknown":
                    raise StoreWriteError("Write outcome requires reconciliation")
                if not self.queue:
                    return self.completed
                self._start()
                future = self.future
            future.result()

    def resolve(self, token, action, *, accept_duplicate_risk=False):
        if action not in {"confirm", "discard", "retry"}:
            raise ValueError("Unknown reconciliation action")
        if accept_duplicate_risk != (action == "retry"):
            raise ValueError("Retry requires explicit duplicate-risk acceptance")
        future = self.future
        if future is not None:
            future.result()
        with self.lock:
            if (
                self.closed
                or not self.queue
                or self.queue[0]["state"] != "unknown"
                or self.queue[0]["token"] != token
            ):
                raise StoreWriteError("No reconcilable uncertain write")
            identity = self.queue[0]["id"]
            if action == "retry":
                self.queue[0]["state"] = "pending"
            else:
                self.queue.pop(0)
            if self.queue:
                self._start()
            return identity

    def stats(self):
        with self.lock:
            return {
                "pending": len(self.queue),
                "completed": self.completed,
                "outcome_unknown": bool(self.queue and self.queue[0]["state"] == "unknown"),
                "closed": self.closed,
                "uncertain_token": self.queue[0]["token"]
                if self.queue and self.queue[0]["state"] == "unknown"
                else None,
            }

    def close(self):
        try:
            self.flush()
        except BaseException:
            # Keep the queue available for explicit reconciliation and retry.
            if self.executor is not None:
                self.executor.shutdown(wait=True)
                self.executor = None
            raise
        else:
            with self.lock:
                self.closed = True
            if self.executor is not None:
                self.executor.shutdown(wait=True)
                self.executor = None
