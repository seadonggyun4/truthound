"""Instance-local memory stores with bounded updates and coherent snapshots."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, replace
from datetime import datetime
from functools import wraps
from threading import RLock
from typing import Any

from truthound.stores.base import (
    ExpectationStore,
    StoreConfig,
    StoreNotFoundError,
    StoreQuery,
    ValidationStore,
)
from truthound.stores.expectations import ExpectationSuite
from truthound.stores.results import ValidationResult


@dataclass
class MemoryConfig(StoreConfig):
    """Memory capacity (zero is unlimited) and nested-value snapshot behavior."""

    max_items: int = 0
    deep_copy: bool = True

    def __post_init__(self):
        self.validate()

    def validate(self):
        if type(self.max_items) is not int or self.max_items < 0:
            raise ValueError("max_items must be a non-negative integer")
        if type(self.deep_copy) is not bool:
            raise ValueError("deep_copy must be boolean")
        if not isinstance(self.metadata, dict):
            raise ValueError("Memory store metadata must be a mapping")
        if self.serialization_format != "json" or self.compression not in {None, "none"}:
            raise ValueError("Memory stores retain canonical mappings, without byte codecs")


def _configuration(max_items, deep_copy, kwargs, config):
    if type(max_items) is not int or max_items < 0 or type(deep_copy) is not bool:
        raise ValueError("Invalid memory capacity or copy mode")
    if config is not None:
        if not isinstance(config, MemoryConfig):
            raise TypeError("config must be MemoryConfig")
        if max_items != 0 or deep_copy is not True or kwargs:
            raise ValueError("Select a config object or individual memory settings")
        return replace(deepcopy(config))
    known = {field.name for field in fields(MemoryConfig)}
    if set(kwargs) - known:
        raise TypeError("Unknown memory store setting")
    return MemoryConfig(max_items=max_items, deep_copy=deep_copy, **kwargs)


def _locked(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            self._config.validate()
            self._guard()
            result = method(self, *args, **kwargs)
            self._guard()
            return result

    return call


def _page(ids, query):
    if (
        type(query.offset) is not int
        or query.offset < 0
        or (query.limit is not None and (type(query.limit) is not int or query.limit < 0))
    ):
        raise ValueError("Invalid store query pagination")
    return (
        ids[query.offset :]
        if query.limit is None
        else ids[query.offset : query.offset + query.limit]
    )


class _MemoryState:
    def _setup(self):
        self._data = {}
        self._metadata = {}
        self._lock = RLock()
        self._io_guard = None

    def set_io_guard(self, guard):
        if guard is not None and not callable(guard):
            raise TypeError("Memory store guard must be callable")
        self._io_guard = guard

    def _guard(self):
        if self._io_guard is not None:
            self._io_guard()

    def _do_initialize(self):
        pass

    def _snapshot(self, item, *, metadata_key):
        data = item.to_dict()
        if self._config.metadata:
            data[metadata_key] = {**deepcopy(self._config.metadata), **data.get(metadata_key, {})}
        return deepcopy(data) if self._config.deep_copy else data

    @staticmethod
    def _meta(data):
        return {
            "data_asset": data.get("data_asset"),
            "run_time": data.get("run_time", data.get("updated_at")),
            "status": data.get("status"),
            "tags": data.get("tags", {}),
        }

    def _put(self, identity, data):
        if not isinstance(identity, str) or not identity:
            raise ValueError("Memory record identity must be a non-empty string")
        self._guard()
        # Updating an existing ID consumes no additional capacity.
        if (
            identity not in self._data
            and self._config.max_items
            and len(self._data) >= self._config.max_items
        ):
            oldest = min(
                self._data, key=lambda key: (self._meta(self._data[key])["run_time"] or "", key)
            )
            del self._data[oldest]
            self._metadata.pop(oldest, None)
        self._data[identity] = data
        self._metadata[identity] = self._meta(data)
        return identity

    @_locked
    def exists(self, item_id: str) -> bool:
        self.initialize()
        return item_id in self._data

    @_locked
    def delete(self, item_id: str) -> bool:
        self.initialize()
        if item_id not in self._data:
            return False
        del self._data[item_id]
        self._metadata.pop(item_id, None)
        return True

    @_locked
    def list_ids(self, query: StoreQuery | None = None) -> list[str]:
        self.initialize()
        if query is None:
            return list(self._data)
        if query.order_by not in {"run_time", "data_asset", "status"}:
            raise ValueError("Unsupported memory store ordering")
        selected = [
            (key, self._meta(data))
            for key, data in self._data.items()
            if query.matches(self._meta(data))
        ]

        def order(row):
            value = row[1][query.order_by]
            return (
                datetime.fromisoformat(value)
                if query.order_by == "run_time" and value
                else value or "",
                row[0],
            )

        selected.sort(key=order, reverse=not query.ascending)
        return _page([identity for identity, _ in selected], query)

    @_locked
    def query(self, query: StoreQuery):
        return [self.get(identity) for identity in self.list_ids(query)]

    @_locked
    def clear_all(self) -> int:
        count = len(self._data)
        self._data.clear()
        self._metadata.clear()
        return count


class MemoryStore(_MemoryState, ValidationStore[MemoryConfig]):
    """Validation records live for the instance lifetime, without persistence."""

    def __init__(
        self,
        max_items: int = 0,
        deep_copy: bool = True,
        *,
        config: MemoryConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(_configuration(max_items, deep_copy, kwargs, config))
        self._setup()

    @classmethod
    def _default_config(cls):
        return MemoryConfig()

    @_locked
    def save(self, item: ValidationResult) -> str:
        self.initialize()
        return self._put(item.run_id, self._snapshot(item, metadata_key="metadata"))

    @_locked
    def get(self, item_id: str) -> ValidationResult:
        self.initialize()
        if item_id not in self._data:
            raise StoreNotFoundError("ValidationResult", item_id)
        data = deepcopy(self._data[item_id]) if self._config.deep_copy else self._data[item_id]
        return ValidationResult.from_dict(data)


class MemoryExpectationStore(_MemoryState, ExpectationStore[MemoryConfig]):
    """Named suites with the same capacity, query and nested-value semantics."""

    def __init__(
        self,
        deep_copy: bool = True,
        *,
        max_items: int = 0,
        config: MemoryConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(_configuration(max_items, deep_copy, kwargs, config))
        self._setup()

    @classmethod
    def _default_config(cls):
        return MemoryConfig()

    @_locked
    def save(self, item: ExpectationSuite) -> str:
        self.initialize()
        return self._put(item.name, self._snapshot(item, metadata_key="meta"))

    @_locked
    def get(self, item_id: str) -> ExpectationSuite:
        self.initialize()
        if item_id not in self._data:
            raise StoreNotFoundError("ExpectationSuite", item_id)
        data = deepcopy(self._data[item_id]) if self._config.deep_copy else self._data[item_id]
        return ExpectationSuite.from_dict(data)
