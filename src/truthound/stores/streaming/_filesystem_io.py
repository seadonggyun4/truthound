"""Filesystem boundaries and atomic publication shared by native stream owners."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


# Keep native stream imports compatible while sharing the ordinary-store IO core.
from truthound.stores._filesystem_io import (
    component as component,
)
from truthound.stores._filesystem_io import (
    contained_path as contained_path,
)
from truthound.stores._filesystem_io import (
    publish as publish,
)


def run_lease_path(root: Path, run_id: str) -> Path:
    """Keep the locking inode outside a run directory that can be deleted."""
    return root.absolute() / ("_stream-" + hashlib.sha256(run_id.encode()).hexdigest())


class StreamFileLease:
    """A real native lock released by explicit close or operating-system process loss."""

    def __init__(self, path: Path):
        from truthound.stores.concurrency.locks import (
            LockMode,
            NoOpLockStrategy,
            get_default_lock_strategy,
        )
        from truthound.stores.concurrency.manager import LockTimeout

        self.strategy = get_default_lock_strategy()
        if isinstance(self.strategy, NoOpLockStrategy):
            raise RuntimeError("Native stream ownership requires a real file locking provider")
        try:
            self.handle = self.strategy.acquire(
                path.absolute(), LockMode.EXCLUSIVE, timeout=0, blocking=False
            )
        except LockTimeout:
            raise RuntimeError("A native stream writer already owns this run") from None

    def close(self):
        if self.handle is not None:
            self.strategy.release(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
