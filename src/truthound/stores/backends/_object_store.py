"""Shared Core object-store record/index state; provider IO stays in backends."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime
from threading import RLock
from typing import TypeVar

from truthound.stores._filesystem_io import component
from truthound.stores.backends.memory import _page
from truthound.stores.base import (
    StoreConfig,
    StoreConnectionError,
    StoreNotFoundError,
    StoreReadError,
    StoreWriteError,
    ValidationStore,
)
from truthound.stores.compression import get_compressor
from truthound.stores.results import ValidationResult

ConfigT = TypeVar("ConfigT", bound=StoreConfig)


class ObjectValidationStore(ValidationStore[ConfigT]):
    _provider_errors = ()

    def __init__(self, config):
        super().__init__(config)
        self._client = None
        self._cleanup_uncertain = False
        self._index = {}
        self._index_etag = None
        self._needs_reconcile = False
        self._io_guard = None
        self._lock = RLock()

    @property
    def _errors(self):
        return self._provider_errors + (OSError, ValueError, TypeError, KeyError, RuntimeError)

    def set_io_guard(self, guard):
        if guard is not None and not callable(guard):
            raise TypeError("Object store guard must be callable")
        self._io_guard = guard

    def _guard(self):
        if self._cleanup_uncertain:
            raise StoreConnectionError(
                "Object store", "Client cleanup is unconfirmed; retry close before new work"
            )
        self._config.validate()
        if self._io_guard is not None:
            self._io_guard()

    def _do_initialize(self):
        with self._lock:
            try:
                self._connect()
                self._load_index()
                scanned = self._scan()
                if set(scanned) != set(self._index):
                    raise ValueError("Object store inventory and index differ")
                for identity, meta in self._index.items():
                    current = scanned[identity]
                    if any(
                        meta.get(key) != current[key]
                        for key in ("data_asset", "run_time", "status", "key", "tags")
                    ) or any(
                        key in meta and meta[key] != current[key] for key in ("sha256", "etag")
                    ):
                        raise ValueError("Object store indexed record differs")
                self._index = scanned
            except self._errors as error:
                self._needs_reconcile = True
                raise StoreReadError(
                    "Object store initialization requires explicit index reconciliation"
                ) from error

    def _index_key(self):
        return self._config.get_full_prefix() + "_index.json"

    def _load_index(self):
        try:
            content, etag = self._read_bytes(self._index_key())
        except self._provider_errors as error:
            if not self._missing(error):
                raise
            self._index, self._index_etag = {}, None
            return
        loaded = json.loads(content)
        if not isinstance(loaded, dict) or any(
            not isinstance(meta, dict) for meta in loaded.values()
        ):
            raise ValueError("Invalid Object store index")
        legacy_key = getattr(self, "_legacy_key_field", None)
        for identity, meta in loaded.items():
            self._get_key(identity)
            if legacy_key is not None and "key" not in meta and legacy_key in meta:
                meta["key"] = meta.pop(legacy_key)
        self._index, self._index_etag = loaded, etag

    def _suffix(self):
        suffix = ".yaml" if self._config.serialization_format == "yaml" else ".json"
        if self._config.use_compression:
            suffix += {"gzip": ".gz", "zstd": ".zst"}.get(
                self._config.compression, "." + self._config.compression
            )
        return suffix

    def _get_key(self, item_id):
        component(item_id)
        return self._config.get_full_prefix() + "results/" + item_id + self._suffix()

    def _serialize(self, data):
        if self._config.serialization_format == "yaml":
            import yaml

            content = yaml.safe_dump(data, allow_unicode=True).encode()
        else:
            content = json.dumps(data, default=str, allow_nan=False).encode()
        self._guard()
        if self._config.use_compression:
            content = get_compressor(self._config.compression).compress(content)
        self._guard()
        if len(content) > self._config.max_object_bytes:
            raise ValueError("Object store record exceeds configured bound")
        return content

    def _deserialize(self, content):
        if self._config.use_compression:
            content = get_compressor(self._config.compression).decompress(content)
        if len(content) > self._config.max_object_bytes:
            raise ValueError("Decoded Object store record exceeds configured bound")
        if self._config.serialization_format == "yaml":
            import yaml

            try:
                data = yaml.safe_load(content)
            except yaml.YAMLError as error:
                raise ValueError("Invalid Object store YAML record") from error
        else:
            data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("Object store record must contain a mapping")
        return data

    def _read_record(self, identity, *, etag=None):
        content, observed = self._read_bytes(self._get_key(identity), etag=etag)
        item = ValidationResult.from_dict(self._deserialize(content))
        if item.run_id != identity:
            raise ValueError("Object store record identity differs")
        return item, self._meta(item, content, observed)

    def _meta(self, item, content, etag):
        return {
            "data_asset": item.data_asset,
            "run_time": item.run_time.isoformat(),
            "status": item.status.value,
            "tags": deepcopy(item.tags),
            "key": self._get_key(item.run_id),
            "sha256": hashlib.sha256(content).hexdigest(),
            "etag": etag,
        }

    def _scan(self):
        prefix = self._config.get_full_prefix() + "results/"
        scanned = {}
        keys = iter(self._keys(prefix))
        while True:
            self._guard()
            key = next(keys, None)
            self._guard()
            if key is None:
                break
            if not key.startswith(prefix) or not key.endswith(self._suffix()):
                raise ValueError("Unexpected object record")
            identity = key[len(prefix) : -len(self._suffix())]
            if self._get_key(identity) != key or identity in scanned:
                raise ValueError("Ambiguous object record identity")
            _, meta = self._read_record(identity)
            scanned[identity] = meta
        return scanned

    def _check_index(self):
        self._guard()
        try:
            observed = self._head(self._index_key(), etag=self._index_etag)
        except self._provider_errors as error:
            if self._index_etag is None and self._missing(error):
                return
            raise
        if self._index_etag is None or observed != self._index_etag:
            raise RuntimeError("Object index receipt changed")

    def _save_index(self, candidate):
        content = json.dumps(candidate, sort_keys=True, allow_nan=False).encode()
        if len(content) > self._config.max_object_bytes:
            raise ValueError("Object index exceeds configured bound")
        self._guard()
        etag = self._put(self._index_key(), content, etag=self._index_etag, is_index=True)
        self._guard()
        self._index, self._index_etag = candidate, etag

    def _mutable(self):
        if self._needs_reconcile:
            raise StoreWriteError("Object store requires explicit index reconciliation")
        self.initialize()
        self._check_index()

    def _record_metadata_defaults(self):
        return self._config.metadata

    def save(self, item):
        with self._lock:
            identity = item.run_id
            key = self._get_key(identity)
            try:
                self._mutable()
                data = item.to_dict()
                data["metadata"] = {
                    **deepcopy(self._record_metadata_defaults()),
                    **data.get("metadata", {}),
                }
                captured = ValidationResult.from_dict(data)
                content = self._serialize(data)
                previous = self._index.get(identity)
                self._guard()
                etag = self._put(
                    key, content, etag=previous["etag"] if previous else None, item=captured
                )
                self._guard()
                self._save_index({**self._index, identity: self._meta(captured, content, etag)})
                return identity
            except self._errors as error:
                self._needs_reconcile = True
                raise StoreWriteError(
                    "Object store write is unconfirmed; explicit reconciliation is required"
                ) from error

    def get(self, item_id):
        with self._lock:
            self._get_key(item_id)
            self.initialize()
            try:
                item, meta = self._read_record(item_id)
                if item_id not in self._index or meta != self._index[item_id]:
                    raise ValueError("Object store indexed record changed")
                return item
            except self._provider_errors as error:
                if self._missing(error):
                    if item_id in self._index:
                        self._needs_reconcile = True
                    raise StoreNotFoundError("ValidationResult", item_id) from None
                self._needs_reconcile = True
                raise StoreReadError("Object store record read failed") from error
            except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
                self._needs_reconcile = True
                raise StoreReadError(
                    "Object store record integrity requires reconciliation"
                ) from error

    def exists(self, item_id):
        with self._lock:
            key = self._get_key(item_id)
            self.initialize()
            self._guard()
            try:
                self._head(key)
                self._guard()
                return True
            except self._provider_errors as error:
                if self._missing(error):
                    if item_id in self._index:
                        self._needs_reconcile = True
                    return False
                raise StoreReadError("Object store existence check failed") from error

    def delete(self, item_id):
        with self._lock:
            key = self._get_key(item_id)
            try:
                self._mutable()
                if item_id not in self._index:
                    return False
                self._guard()
                self._delete(key, self._index[item_id]["etag"])
                self._guard()
                candidate = dict(self._index)
                del candidate[item_id]
                self._save_index(candidate)
                return True
            except self._errors as error:
                self._needs_reconcile = True
                raise StoreWriteError(
                    "Object store deletion is unconfirmed; explicit reconciliation is required"
                ) from error

    def list_ids(self, query=None):
        with self._lock:
            self.initialize()
            self._guard()
            if self._needs_reconcile:
                raise StoreReadError("Object store index requires reconciliation")
            if query is None:
                return list(self._index)
            if query.order_by not in {"run_time", "data_asset", "status"}:
                raise ValueError("Unsupported object store ordering")
            selected = [
                (identity, meta) for identity, meta in self._index.items() if query.matches(meta)
            ]
            selected.sort(
                key=lambda row: (
                    datetime.fromisoformat(row[1][query.order_by])
                    if query.order_by == "run_time"
                    else row[1][query.order_by],
                    row[0],
                ),
                reverse=not query.ascending,
            )
            return _page([identity for identity, _ in selected], query)

    def query(self, query):
        with self._lock:
            try:
                return [self.get(identity) for identity in self.list_ids(query)]
            except StoreNotFoundError as error:
                self._needs_reconcile = True
                raise StoreReadError("Object store query references a missing record") from error

    def rebuild_index(self):
        with self._lock:
            try:
                self._connect()
                self._guard()
                try:
                    self._index_etag = self._head(self._index_key())
                except self._provider_errors as error:
                    if not self._missing(error):
                        raise
                    self._index_etag = None
                scanned = self._scan()
                self._check_index()
                self._save_index(scanned)
            except self._errors as error:
                self._needs_reconcile = True
                raise StoreWriteError("Object store index reconciliation is unconfirmed") from error
            self._needs_reconcile = False
            self._initialized = True
            return len(scanned)
