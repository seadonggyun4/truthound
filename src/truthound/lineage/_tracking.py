"""Transactional tracker edits over the existing graph persistence contract."""

import hashlib
import json
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from inspect import signature
from pathlib import Path

from truthound.lineage.base import LineageGraph


def fingerprint(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Lineage persistence does not follow symlinks")
    if not path.exists():
        return None
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Lineage persistence exceeds 16 MiB")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def writer_lock(path):
    from truthound.stores.concurrency.locks import (
        LockMode,
        NoOpLockStrategy,
        get_default_lock_strategy,
    )

    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    strategy = get_default_lock_strategy()
    if isinstance(strategy, NoOpLockStrategy):
        raise RuntimeError("Lineage persistence requires a native lock")
    handle = strategy.acquire(path.with_name(path.name + ".lock"), LockMode.EXCLUSIVE, timeout=5)
    try:
        yield
    finally:
        strategy.release(handle)


def sample_values(value):
    """Capture explicitly supplied finite JSON rows, never read a source asset."""
    from truthound.stores._filesystem_io import validate_values

    if not isinstance(value, list) or len(value) > 100:
        raise ValueError("Lineage samples require at most 100 explicit rows")
    if any(
        not isinstance(row, dict) or any(not isinstance(key, str) for key in row) for row in value
    ):
        raise ValueError("Lineage sample rows require string column keys")
    validate_values(value)
    try:
        payload = json.dumps(value, allow_nan=False, ensure_ascii=False).encode("utf-8")
    except (TypeError, RecursionError) as error:
        raise ValueError("Lineage samples must be finite JSON values") from error
    if len(payload) > 64 * 1024:
        raise ValueError("Lineage samples exceed 64 KiB")
    return json.loads(payload)


def graph_edit(method):
    parameters = signature(method)

    @wraps(method)
    def edit(self, *args, **kwargs):
        with self._mutation_lock:
            if self._persistence_failed:
                raise RuntimeError("Reopen tracker after an unconfirmed persistence failure")
            if (
                method.__name__
                in {"track_source", "track_transformation", "track_validation", "track_output"}
                and "samples" in kwargs
            ):
                kwargs = dict(kwargs)
                supplied = kwargs.pop("samples")
                if self._config.store_samples and supplied is not None:
                    kwargs["samples"] = sample_values(supplied)
            bound = parameters.bind(self, *args, **kwargs)
            for source in bound.arguments.get("sources", []):
                self._graph.get_node(source)
            candidate = object.__new__(type(self))
            candidate.__dict__ = self.__dict__.copy()
            candidate._graph = LineageGraph.from_dict(deepcopy(self._graph.to_dict()), self._config)
            candidate._history = list(self._history)
            result = method(candidate, *args, **kwargs)
            if self._config.persist_path is not None:
                publish_tracker_graph(self, candidate._graph, self._config.persist_path)
            self._graph = candidate._graph
            self._history = candidate._history
            return result

    return edit


def publish_tracker_graph(tracker, graph, path):
    """Share the same version fence for ordinary edits and explicit primary saves."""
    if tracker._persistence_failed:
        raise RuntimeError("Reopen tracker after an unconfirmed persistence failure")
    path = Path(path)
    content = json.dumps(graph.to_dict(), indent=2, allow_nan=False).encode("utf-8")
    if len(content) > 16 * 1024 * 1024:
        raise ValueError("Lineage persistence exceeds 16 MiB")
    with writer_lock(path):
        if fingerprint(path) != tracker._persistence_version:
            raise RuntimeError("Lineage snapshot changed; reopen before writing")
        from truthound.stores._filesystem_io import publish

        try:
            publish(path, content)
            confirmed = fingerprint(path)
            if confirmed != hashlib.sha256(content).hexdigest():
                raise RuntimeError("Lineage publication could not be confirmed")
            tracker._persistence_version = confirmed
        except Exception:
            tracker._persistence_failed = True
            raise
