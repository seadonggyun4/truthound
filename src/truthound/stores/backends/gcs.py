"""Google Cloud Storage with native generation receipts and shared Core state."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from typing import Any

from truthound.stores.backends._object_store import ObjectValidationStore
from truthound.stores.base import StoreConfig, StoreConnectionError
from truthound.stores.compression import get_compressor

try:
    from google.api_core.exceptions import GoogleAPIError
    from google.cloud import storage
    from google.cloud.exceptions import NotFound
    from google.resumable_media.common import DataCorruption, InvalidResponse

    HAS_GCS = True
except ImportError:
    storage = None
    NotFound = GoogleAPIError = DataCorruption = InvalidResponse = Exception
    HAS_GCS = False


def _require_gcs():
    if not HAS_GCS:
        raise ImportError(
            "google-cloud-storage is required. Install with: pip install truthound[gcs]"
        )


@dataclass
class GCSConfig(StoreConfig):
    bucket: str = ""
    prefix: str = "truthound/"
    project: str | None = None
    credentials_path: str | None = field(default=None, repr=False)
    use_compression: bool = True
    max_object_bytes: int = 128 * 1024 * 1024

    def __post_init__(self):
        self.validate()

    def validate(self):
        if not isinstance(self.bucket, str) or not self.bucket:
            raise ValueError("GCS bucket must be a non-empty string")
        for value in (self.prefix, self.namespace):
            if (
                not isinstance(value, str)
                or "\\" in value
                or value.startswith("/")
                or any(part in {".", ".."} for part in value.split("/"))
            ):
                raise ValueError("GCS prefix and namespace must be relative object paths")
        if type(self.use_compression) is not bool or not isinstance(self.metadata, dict):
            raise ValueError("Invalid GCS compression flag or metadata")
        if self.serialization_format not in {"json", "yaml"}:
            raise ValueError("GCS records require JSON or safe YAML")
        if type(self.max_object_bytes) is not int or self.max_object_bytes < 1:
            raise ValueError("max_object_bytes must be a positive integer")
        if self.compression is not None:
            self.compression = get_compressor(self.compression).algorithm.value
            self.use_compression = self.compression != "none"
        elif self.use_compression:
            self.compression = "gzip"

    def get_full_prefix(self):
        parts = [part.strip("/") for part in (self.prefix, self.namespace) if part.strip("/")]
        return "/".join(parts) + ("/" if parts else "")


class GCSStore(ObjectValidationStore[GCSConfig]):
    """Persist canonical validation records with conditional generation checks."""

    _provider_errors = (GoogleAPIError, DataCorruption, InvalidResponse)
    _legacy_key_field = "blob_name"

    def __init__(
        self,
        bucket: str = "",
        prefix: str = "truthound/",
        project: str | None = None,
        credentials_path: str | None = None,
        namespace: str = "default",
        compression: bool | str = True,
        *,
        config: GCSConfig | None = None,
        **kwargs: Any,
    ):
        _require_gcs()
        if config is not None:
            if not isinstance(config, GCSConfig):
                raise TypeError("config must be GCSConfig")
            if (
                bucket
                or prefix != "truthound/"
                or project is not None
                or credentials_path is not None
                or namespace != "default"
                or compression is not True
                or kwargs
            ):
                raise ValueError("Select a config object or individual GCS settings")
            selected = replace(deepcopy(config))
        else:
            if set(kwargs) - {item.name for item in fields(GCSConfig)}:
                raise TypeError("Unknown GCS store setting")
            options = deepcopy(kwargs)
            if type(compression) is bool:
                options.setdefault("use_compression", compression)
            elif isinstance(compression, str):
                options["compression"] = compression
            else:
                raise ValueError("Invalid GCS compression setting")
            selected = GCSConfig(
                bucket=bucket,
                prefix=prefix,
                project=project,
                credentials_path=credentials_path,
                namespace=namespace,
                **options,
            )
        super().__init__(selected)
        self._bucket = None

    @classmethod
    def _default_config(cls):
        raise ValueError("GCSStore requires an explicit bucket configuration")

    @staticmethod
    def _missing(error):
        return isinstance(error, NotFound)

    @staticmethod
    def _generation(blob):
        generation = blob.generation
        if (
            generation is None
            or isinstance(generation, bool)
            or not str(generation).isdigit()
            or int(generation) < 1
        ):
            raise RuntimeError("GCS has no confirmed generation receipt")
        return str(generation)

    def _connect(self):
        self._guard()
        if self._bucket is not None:
            return
        if self._client is not None:
            raise StoreConnectionError("GCS", "Previous client cleanup is incomplete")
        options = {"project": self._config.project} if self._config.project else {}
        client = (
            storage.Client.from_service_account_json(self._config.credentials_path, **options)
            if self._config.credentials_path
            else storage.Client(**options)
        )
        self._client = client
        try:
            bucket = client.bucket(self._config.bucket)
            self._guard()
            if not bucket.exists():
                raise StoreConnectionError("GCS", "Bucket not found")
            self._guard()
        except BaseException as original:
            try:
                self.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "GCS connection and cleanup failed", [original, cleanup]
                ) from None
            if isinstance(original, GoogleAPIError):
                raise StoreConnectionError("GCS", "Connection failed") from None
            raise
        self._bucket = bucket

    def _get_blob_name(self, item_id):
        return self._get_key(item_id)

    def _head(self, key, *, etag=None):
        blob = self._bucket.blob(key)
        self._guard()
        blob.reload(**({"if_generation_match": int(etag)} if etag else {}))
        self._guard()
        return self._generation(blob)

    def _read_bytes(self, key, *, etag=None):
        blob = self._bucket.blob(key)
        self._guard()
        blob.reload(**({"if_generation_match": int(etag)} if etag else {}))
        observed = self._generation(blob)
        if blob.size is None or blob.size > self._config.max_object_bytes:
            raise ValueError("GCS object exceeds configured bound")
        self._guard()
        # GCS/requests otherwise transparently decodes Content-Encoding.
        # Core owns exactly one codec reversal and hashes stored bytes.
        content = blob.download_as_bytes(raw_download=True, if_generation_match=int(observed))
        if len(content) != blob.size or len(content) > self._config.max_object_bytes:
            raise ValueError("GCS object length differs from its receipt")
        self._guard()
        return content, observed

    def _put(self, key, content, *, etag=None, is_index=False, item=None):
        blob = self._bucket.blob(key)
        content_type = (
            "application/json"
            if is_index or self._config.serialization_format == "json"
            else "application/yaml"
        )
        if not is_index and self._config.use_compression and self._config.compression == "gzip":
            blob.content_encoding = "gzip"
        self._guard()
        blob.upload_from_string(
            content, content_type=content_type, if_generation_match=int(etag) if etag else 0
        )
        self._guard()
        return self._generation(blob)

    def _delete(self, key, etag):
        self._guard()
        self._bucket.blob(key).delete(if_generation_match=int(etag))
        self._guard()

    def _keys(self, prefix):
        for blob in self._bucket.list_blobs(prefix=prefix):
            yield blob.name

    def close(self):
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                except BaseException:
                    self._cleanup_uncertain = True
                    self._initialized = False
                    raise
                self._client = None
            self._cleanup_uncertain = False
            self._bucket = None
            self._initialized = False
