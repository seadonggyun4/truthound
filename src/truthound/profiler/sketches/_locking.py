"""Lock helpers for sketch data structures."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def acquire_ordered_locks(*locks: Any) -> Iterator[None]:
    """Acquire locks once in a deterministic order.

    Deduplicating identical lock objects avoids self-deadlock when the same
    sketch is merged with itself, and deterministic ordering prevents lock
    inversion when two sketches are merged concurrently in opposite orders.
    """

    ordered: list[Any] = []
    seen: set[int] = set()

    for lock in sorted(locks, key=id):
        lock_id = id(lock)
        if lock_id in seen:
            continue
        seen.add(lock_id)
        ordered.append(lock)

    for lock in ordered:
        lock.acquire()

    try:
        yield
    finally:
        for lock in reversed(ordered):
            lock.release()
