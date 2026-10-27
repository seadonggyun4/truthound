"""Local single-writer stores with atomic files and explicit index reconciliation."""

from __future__ import annotations

import hashlib
import json
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, fields, replace
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import quote

from yaml import YAMLError

from truthound.stores._filesystem_io import (
    component,
    contained_path,
    publish,
    read_json,
    read_yaml,
    validate_values,
)
from truthound.stores.base import (
    ExpectationStore,
    StoreConfig,
    StoreNotFoundError,
    StoreQuery,
    StoreReadError,
    StoreWriteError,
    ValidationStore,
)
from truthound.stores.compression.base import CompressionError
from truthound.stores.compression.providers import get_compressor
from truthound.stores.expectations import ExpectationSuite
from truthound.stores.results import ValidationResult


@dataclass
class FileSystemConfig(StoreConfig):
    """Pinned format, codec, directory and metadata settings for local records."""

    base_path: str = ".truthound/store"
    file_extension: str = ".json"
    create_dirs: bool = True
    pretty_print: bool = True
    use_compression: bool = False

    def __post_init__(self):
        if self.serialization_format not in {"json", "yaml"}:
            raise ValueError("Filesystem records support JSON or safe YAML")
        if (
            not isinstance(self.file_extension, str)
            or not self.file_extension.startswith(".")
            or any(token in self.file_extension for token in ("/", "\\", "\0"))
        ):
            raise ValueError("Invalid filesystem record suffix")
        for value in (self.create_dirs, self.pretty_print, self.use_compression):
            if type(value) is not bool:
                raise ValueError("Filesystem mode flags must be boolean")
        if not isinstance(self.metadata, dict):
            raise ValueError("Store metadata must be a mapping")
        for value in (self.namespace, self.prefix):
            if (
                not isinstance(value, str)
                or "\0" in value
                or "\\" in value
                or Path(value).is_absolute()
                or bool(PureWindowsPath(value).drive)
                or any(part in {".", ".."} for part in value.split("/") if part)
            ):
                raise ValueError("Store namespace and prefix must be relative paths")
        if self.compression is not None:
            self.compression = get_compressor(self.compression).algorithm.value
            self.use_compression = self.compression != "none"
        elif self.use_compression:
            self.compression = "gzip"

    def get_full_path(self) -> Path:
        path = Path(self.base_path)
        for part in (self.namespace, self.prefix):
            if part:
                path = path / part
        return path


def _configuration(base_path, namespace, prefix, options, compression=None):
    known = {item.name for item in fields(FileSystemConfig)}
    unknown = set(options) - known
    if unknown:
        raise TypeError("Unknown filesystem store setting: " + ", ".join(sorted(unknown)))
    options = dict(options)
    if "serialization_format" not in options and options.get("file_extension") in {".yaml", ".yml"}:
        options["serialization_format"] = "yaml"
    if "file_extension" not in options and options.get("serialization_format") == "yaml":
        options["file_extension"] = ".yaml"
    if compression is not None:
        if type(compression) is bool:
            options["use_compression"] = compression or options.get("use_compression", False)
        elif isinstance(compression, str):
            options["compression"] = compression
        else:
            raise ValueError("Invalid filesystem compression setting")
    return FileSystemConfig(base_path=base_path, namespace=namespace, prefix=prefix, **options)


def _captured_config(
    config, base_path, namespace, prefix, kwargs, *, expected_prefix, compression=False
):
    if not isinstance(config, FileSystemConfig):
        raise TypeError("config must be FileSystemConfig")
    if (
        base_path != ".truthound/store"
        or namespace != "default"
        or prefix != expected_prefix
        or compression is not False
        or kwargs
    ):
        raise ValueError("Select a config object or individual filesystem settings")
    return replace(deepcopy(config))


class _FileSystemCodec:
    def set_io_guard(self, guard):
        if guard is not None and not callable(guard):
            raise TypeError("Store IO guard must be callable")
        self._io_guard = guard

    def _guard(self):
        guard = getattr(self, "_io_guard", None)
        if guard is not None:
            guard()

    def _prepare_path(self):
        self._guard()
        path = self._config.get_full_path()
        if self._config.create_dirs:
            path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            raise StoreReadError("Selected store directory does not exist")
        self._guard()
        return path

    def _suffix(self):
        suffix = self._config.file_extension
        if self._config.use_compression:
            suffix += {"gzip": ".gz", "zstd": ".zst"}.get(
                self._config.compression, "." + self._config.compression
            )
        return suffix

    def _serialize(self, data):
        self._guard()
        validate_values(data)
        if self._config.serialization_format == "yaml":
            import yaml

            content = yaml.safe_dump(data, allow_unicode=True, sort_keys=False).encode()
        else:
            content = json.dumps(
                data, indent=2 if self._config.pretty_print else None, default=str, allow_nan=False
            ).encode()
        if self._config.use_compression:
            content = get_compressor(self._config.compression).compress(content)
        self._guard()
        return content

    def _deserialize(self, content):
        self._guard()
        if self._config.use_compression:
            content = get_compressor(self._config.compression).decompress(content)
        if self._config.serialization_format == "yaml":
            data = read_yaml(content)
        else:
            data = read_json(content)
        validate_values(data)
        if not isinstance(data, dict):
            raise ValueError("Stored record must be a mapping")
        self._guard()
        return data

    def _publish(self, path, content):
        self._guard()
        publish(path, content)
        self._guard()

    def _record_paths(self):
        path = self._prepare_path()
        return sorted(
            (
                entry
                for entry in path.iterdir()
                if entry.name.endswith(self._suffix())
                and (entry.name != "_index.json" or not getattr(self, "_uses_index", False))
            ),
            key=lambda entry: entry.name,
        )


def _page(items, query):
    if query is None:
        return items
    if (
        type(query.offset) is not int
        or query.offset < 0
        or (query.limit is not None and (type(query.limit) is not int or query.limit < 0))
    ):
        raise ValueError("Invalid store query pagination")
    return (
        items[query.offset :]
        if query.limit is None
        else items[query.offset : query.offset + query.limit]
    )


class FileSystemStore(_FileSystemCodec, ValidationStore[FileSystemConfig]):
    """Atomic local records and checked index; use concurrent stores for shared writers.

    Record and index publication are separate steps. Any unconfirmed write fences
    mutation until explicit rebuild_index; a fresh owner detects inconsistent
    records/index rather than silently discarding recovery evidence.
    """

    _uses_index = True

    def __init__(
        self,
        base_path: str = ".truthound/store",
        namespace: str = "default",
        prefix: str = "validations",
        compression: bool | str = False,
        config: FileSystemConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            _captured_config(
                config,
                base_path,
                namespace,
                prefix,
                kwargs,
                expected_prefix="validations",
                compression=compression,
            )
            if config is not None
            else _configuration(base_path, namespace, prefix, kwargs, compression)
        )
        self._index = {}
        self._index_path = None
        self._needs_reconcile = False

    @classmethod
    def _default_config(cls):
        return FileSystemConfig()

    def _get_file_path(self, item_id):
        component(item_id)
        if item_id == "_index":
            raise ValueError("Reserved record identity")
        return contained_path(self._config.get_full_path(), item_id + self._suffix())

    def _read_record(self, path, expected=None):
        self._guard()
        try:
            if path.is_symlink() or not path.is_file():
                raise ValueError("Invalid record file")
            content = path.read_bytes()
            item = ValidationResult.from_dict(self._deserialize(content))
            if expected is not None and item.run_id != expected:
                raise ValueError("Record identity mismatch")
            if self._get_file_path(item.run_id).name != path.name:
                raise ValueError("Record name mismatch")
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            CompressionError,
            YAMLError,
            RecursionError,
        ) as error:
            self._needs_reconcile = True
            raise StoreReadError("Failed to read validation record") from error
        self._guard()
        return item, content

    @staticmethod
    def _entry(item, path, content):
        return {
            "data_asset": item.data_asset,
            "run_time": item.run_time.isoformat(),
            "status": item.status.value,
            "file": path.name,
            "tags": deepcopy(item.tags),
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    def _scan_index(self):
        index = {}
        for path in self._record_paths():
            self._guard()
            item, content = self._read_record(path)
            if item.run_id in index:
                raise StoreReadError("Duplicate stored validation identity")
            index[item.run_id] = self._entry(item, path, content)
        return index

    def _do_initialize(self):
        self._index_path = self._prepare_path() / "_index.json"
        observed = self._scan_index()
        if self._index_path.exists() or self._index_path.is_symlink():
            try:
                if self._index_path.is_symlink():
                    raise ValueError("Index cannot be a symbolic link")
                saved = read_json(self._index_path.read_bytes())
                if not isinstance(saved, dict):
                    raise ValueError("Invalid index mapping")
                if set(saved) != set(observed):
                    raise ValueError("Index records mismatch")
                for identity, metadata in saved.items():
                    if not isinstance(metadata, dict) or any(
                        metadata.get(key) != value
                        for key, value in observed[identity].items()
                        if key != "sha256" or "sha256" in metadata
                    ):
                        raise ValueError("Index metadata mismatch")
            except Exception as error:
                self._needs_reconcile = True
                raise StoreReadError("Store index requires explicit reconciliation") from error
        elif observed:
            self._needs_reconcile = True
            raise StoreReadError("Missing store index requires explicit reconciliation")
        self._index = observed
        self._needs_reconcile = False

    def _save_index(self):
        if self._index_path is None:
            raise StoreWriteError("Store index is not initialized")
        self._publish(self._index_path, json.dumps(self._index, indent=2, allow_nan=False).encode())

    def _require_mutation(self):
        self.initialize()
        self._guard()
        if self._needs_reconcile:
            raise StoreWriteError("Store requires explicit reconciliation before mutation")

    def save(self, item: ValidationResult) -> str:
        path = self._get_file_path(item.run_id)
        self._require_mutation()
        before = self._index
        attempted = False
        try:
            data = item.to_dict()
            data["metadata"] = {**deepcopy(self._config.metadata), **data.get("metadata", {})}
            content = self._serialize(data)
            stored = ValidationResult.from_dict(self._deserialize(content))
            attempted = True
            self._publish(path, content)
            self._index = {**before, item.run_id: self._entry(stored, path, content)}
            self._save_index()
        except BaseException as error:
            self._index = before
            if attempted:
                self._needs_reconcile = True
            if isinstance(error, (OSError, ValueError, TypeError, CompressionError, YAMLError)):
                raise StoreWriteError("Failed to publish validation record/index") from error
            raise
        return item.run_id

    def get(self, item_id: str) -> ValidationResult:
        path = self._get_file_path(item_id)
        self.initialize()
        self._guard()
        if not path.exists():
            raise StoreNotFoundError("ValidationResult", item_id)
        item, content = self._read_record(path, item_id)
        expected = self._index.get(item_id, {}).get("sha256")
        if expected is not None and expected != hashlib.sha256(content).hexdigest():
            self._needs_reconcile = True
            raise StoreReadError("Record/index mismatch requires explicit reconciliation")
        return item

    def exists(self, item_id: str) -> bool:
        path = self._get_file_path(item_id)
        self.initialize()
        self._guard()
        return path.is_file() and not path.is_symlink()

    def delete(self, item_id: str) -> bool:
        path = self._get_file_path(item_id)
        self._require_mutation()
        before = self._index
        existed = path.exists()
        try:
            self._guard()
            path.unlink(missing_ok=True)
            self._guard()
            self._index = {key: value for key, value in before.items() if key != item_id}
            self._save_index()
        except BaseException as error:
            self._index = before
            self._needs_reconcile = True
            if isinstance(error, OSError):
                raise StoreWriteError("Failed to delete validation record/index") from error
            raise
        return existed

    def list_ids(self, query: StoreQuery | None = None) -> list[str]:
        self.initialize()
        self._guard()
        if self._needs_reconcile:
            raise StoreReadError("Store index requires explicit reconciliation")
        if query is None:
            return list(self._index)
        if query.order_by not in {"run_time", "data_asset", "status"}:
            raise ValueError("Unsupported validation store ordering")
        selected = [(key, value) for key, value in self._index.items() if query.matches(value)]

        def order(row):
            value = row[1][query.order_by]
            return (
                datetime.fromisoformat(value) if query.order_by == "run_time" else value,
                row[0],
            )

        selected.sort(key=order, reverse=not query.ascending)
        return _page([key for key, _ in selected], query)

    def query(self, query: StoreQuery) -> list[ValidationResult]:
        return [self.get(identity) for identity in self.list_ids(query)]

    def rebuild_index(self) -> int:
        """Validate every record, then atomically replace the index explicitly."""
        self._index_path = self._prepare_path() / "_index.json"
        try:
            rebuilt = self._scan_index()
        except BaseException:
            self._needs_reconcile = True
            raise
        before = self._index
        self._index = rebuilt
        try:
            self._save_index()
        except BaseException:
            self._index = before
            self._needs_reconcile = True
            raise
        self._needs_reconcile = False
        self._initialized = True
        return len(rebuilt)


class FileSystemExpectationStore(_FileSystemCodec, ExpectationStore[FileSystemConfig]):
    """Safe named suites with the same native format/codec and atomic IO contract."""

    def __init__(
        self,
        base_path: str = ".truthound/store",
        namespace: str = "default",
        prefix: str = "expectations",
        config: FileSystemConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            _captured_config(
                config, base_path, namespace, prefix, kwargs, expected_prefix="expectations"
            )
            if config is not None
            else _configuration(base_path, namespace, prefix, kwargs)
        )

    @classmethod
    def _default_config(cls):
        return FileSystemConfig(prefix="expectations")

    def _do_initialize(self):
        self._prepare_path()

    def _get_file_path(self, name):
        if not isinstance(name, str) or not name or "\0" in name:
            raise ValueError("Invalid expectation suite name")
        return contained_path(self._config.get_full_path(), quote(name, safe="") + self._suffix())

    def _read_suite(self, path, expected=None):
        self._guard()
        try:
            if path.is_symlink() or not path.is_file():
                raise ValueError("Invalid suite file")
            item = ExpectationSuite.from_dict(self._deserialize(path.read_bytes()))
            if expected is not None and item.name != expected:
                raise ValueError("Suite identity mismatch")
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            CompressionError,
            YAMLError,
            RecursionError,
        ) as error:
            raise StoreReadError("Failed to read expectation suite") from error
        self._guard()
        return item

    def _existing_path(self, name):
        path = self._get_file_path(name)
        if path.exists():
            return path
        legacy_name = name.replace("/", "_").replace("\\", "_")
        legacy = contained_path(self._config.get_full_path(), legacy_name + self._suffix())
        if legacy != path and legacy.exists():
            return legacy
        raise StoreNotFoundError("ExpectationSuite", name)

    def save(self, item: ExpectationSuite) -> str:
        path = self._get_file_path(item.name)
        self.initialize()
        with suppress(StoreNotFoundError):
            path = self._existing_path(item.name)
        try:
            if path.exists() and self._read_suite(path).name != item.name:
                raise StoreWriteError("Legacy suite filename collision requires migration")
            data = item.to_dict()
            data["meta"] = {**deepcopy(self._config.metadata), **data.get("meta", {})}
            self._publish(path, self._serialize(data))
        except (OSError, ValueError, TypeError, CompressionError, YAMLError) as error:
            raise StoreWriteError("Failed to publish expectation suite") from error
        return item.name

    def get(self, item_id: str) -> ExpectationSuite:
        self.initialize()
        return self._read_suite(self._existing_path(item_id), item_id)

    def exists(self, item_id: str) -> bool:
        try:
            self.get(item_id)
        except StoreNotFoundError:
            return False
        return True

    def delete(self, item_id: str) -> bool:
        self.initialize()
        try:
            path = self._existing_path(item_id)
        except StoreNotFoundError:
            return False
        self._read_suite(path, item_id)
        self._guard()
        try:
            path.unlink()
        except OSError as error:
            raise StoreWriteError("Failed to delete expectation suite") from error
        self._guard()
        return True

    def list_ids(self, query: StoreQuery | None = None) -> list[str]:
        self.initialize()
        rows = {}
        for path in self._record_paths():
            item = self._read_suite(path)
            if item.name in rows:
                raise StoreReadError("Duplicate stored suite identity requires migration")
            if query is None or query.matches(
                {"data_asset": item.data_asset, "run_time": item.updated_at, "tags": item.tags}
            ):
                rows[item.name] = item
        if query is None:
            return sorted(rows)
        if query.order_by not in {"name", "run_time", "data_asset"}:
            raise ValueError("Unsupported expectation store ordering")
        field = "updated_at" if query.order_by == "run_time" else query.order_by
        ordered = sorted(
            rows,
            key=lambda name: (getattr(rows[name], field) or "", name),
            reverse=not query.ascending,
        )
        return _page(ordered, query)

    def query(self, query: StoreQuery) -> list[ExpectationSuite]:
        return [self.get(name) for name in self.list_ids(query)]
