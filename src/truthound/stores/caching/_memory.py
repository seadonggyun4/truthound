"""Bounded retained-value estimates, not a reservation of process or provider heaps."""

import sys
from dataclasses import fields, is_dataclass


def estimate(value, limit):
    seen, pending, total = set(), [iter((value,))], 0
    while pending and total <= limit:
        try:
            item = next(pending[-1])
        except StopIteration:
            pending.pop()
            continue
        identity = id(item)
        if identity in seen:
            continue
        seen.add(identity)
        total += sys.getsizeof(item)
        if isinstance(item, dict):
            pending.append(iter(item.keys()))
            pending.append(iter(item.values()))
        elif isinstance(item, (list, tuple, set, frozenset)):
            pending.append(iter(item))
        elif is_dataclass(item) and not isinstance(item, type):
            pending.append(
                map(lambda field, record=item: getattr(record, field.name), fields(item))
            )
    return total


def make_room(cache, key, size):
    budget = cache.config.max_memory_mb * 1024 * 1024
    if size > budget:
        raise MemoryError("Value exceeds cache memory estimate budget")
    while True:
        old = cache._cache.get(key)
        entry = getattr(old, "entry", old)
        old_size = entry.size_bytes if entry is not None else 0
        if cache._total_memory_bytes - old_size + size <= budget:
            return
        if not cache._evict(1):
            raise MemoryError("Cannot evict enough cache entries")
