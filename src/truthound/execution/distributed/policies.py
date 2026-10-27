"""Cooperative deadlines and explicit retry declarations for native operations.

A deadline rejects late results; it does not kill borrowed runtimes or interrupt
arbitrary native code. Only an explicit RetryableExecutionError permits replay.
"""

from __future__ import annotations

import time
from contextvars import ContextVar
from functools import wraps

_deadline = ContextVar("truthound_distributed_deadline", default=None)


class RetryableExecutionError(RuntimeError):
    """The caller declares that the failed computation is safe to repeat.

    Do not use this for a write or callback whose outcome is uncertain. Native
    frameworks can wrap worker errors; replay requires this type to propagate.
    """


class DistributedOperationTimeout(TimeoutError):
    """A native operation exceeded its cooperative result deadline."""


def check_deadline():
    deadline = _deadline.get()
    if deadline is not None and time.monotonic() >= deadline:
        raise DistributedOperationTimeout("Distributed operation deadline exceeded")


def operation(function):
    """Share a deadline across nested native calls; never cache late results."""

    @wraps(function)
    def run(self, *args, **kwargs):
        inherited = _deadline.get()
        token = None
        if inherited is None:
            token = _deadline.set(time.monotonic() + self.config.timeout_seconds)
        try:
            for attempt in range(1 if inherited is not None else self.config.retry_count + 1):
                check_deadline()
                try:
                    result = function(self, *args, **kwargs)
                    check_deadline()
                    return result
                except RetryableExecutionError:
                    self.clear_cache()
                    if inherited is not None or attempt == self.config.retry_count:
                        raise
        except BaseException:
            self.clear_cache()
            raise
        finally:
            if token is not None:
                _deadline.reset(token)

    return run


def checked_batches(engine, iterator):
    """Validate each yield and close native iterators even on early consumer exit."""
    deadline = _deadline.get()
    if deadline is None:
        deadline = time.monotonic() + engine.config.timeout_seconds
    iterator = iter(iterator)
    failure = None
    try:
        limit = transfer_memory_limit(engine)
        while True:
            token = _deadline.set(deadline)
            try:
                check_deadline()
                try:
                    batch = next(iterator)
                except StopIteration:
                    return
                check_deadline()
                if batch.nbytes > limit:
                    raise MemoryError("Arrow batch exceeds the selected transfer memory fraction")
            finally:
                _deadline.reset(token)
            yield batch
    except BaseException as error:
        failure = error
        raise
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            try:
                close()
            except BaseException as cleanup:
                if failure is not None:
                    raise BaseExceptionGroup(
                        "Arrow iteration and cleanup failed", [failure, cleanup]
                    ) from None
                raise


def batch_size(engine, value):
    selected = engine.config.arrow_batch_size if value is None else value
    if type(selected) is not int or selected < 1:
        raise ValueError("Arrow batch size must be a positive integer")
    return selected


def transfer_memory_limit(engine):
    """A cooperative transfer bound, not a worker or process heap allocator."""
    import psutil

    return int(psutil.virtual_memory().available * engine.config.memory_fraction)
