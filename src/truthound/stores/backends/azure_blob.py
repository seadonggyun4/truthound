"""Azure Blob storage with conditional receipts and shared Core index state."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from typing import Any
from urllib.parse import quote

from truthound.stores.backends._object_store import ObjectValidationStore
from truthound.stores.base import StoreConfig, StoreConnectionError, StoreReadError, StoreWriteError
from truthound.stores.compression import get_compressor

try:
    from azure.core import MatchConditions
    from azure.core.exceptions import AzureError, ResourceNotFoundError
    from azure.storage.blob import BlobServiceClient, ContentSettings, StandardBlobTier

    HAS_AZURE = True
except ImportError:
    MatchConditions = BlobServiceClient = ContentSettings = StandardBlobTier = None
    AzureError = ResourceNotFoundError = Exception
    HAS_AZURE = False

TIERS = {"Hot", "Cool", "Cold", "Archive"}


def _require_azure():
    if not HAS_AZURE:
        raise ImportError(
            "azure-storage-blob is required. Install with: pip install truthound[azure]"
        )


@dataclass
class AzureBlobConfig(StoreConfig):
    container: str = ""
    prefix: str = "truthound/"
    connection_string: str | None = field(default=None, repr=False)
    account_url: str | None = field(default=None, repr=False)
    account_name: str | None = field(default=None, repr=False)
    account_key: str | None = field(default=None, repr=False)
    sas_token: str | None = field(default=None, repr=False)
    use_compression: bool = True
    content_type: str = "application/json"
    access_tier: str | None = None
    metadata: dict[str, str] = field(default_factory=dict, repr=False)
    record_metadata: dict[str, Any] = field(default_factory=dict, repr=False)
    max_object_bytes: int = 128 * 1024 * 1024
    anonymous: bool = False

    def __post_init__(self):
        self.validate()

    def validate(self):
        if not isinstance(self.container, str) or not self.container:
            raise ValueError("Azure container must be a non-empty string")
        for value in (self.prefix, self.namespace):
            if (
                not isinstance(value, str)
                or "\\" in value
                or value.startswith("/")
                or any(part in {".", ".."} for part in value.split("/"))
            ):
                raise ValueError("Azure prefix and namespace must be relative object paths")
        if type(self.use_compression) is not bool or type(self.anonymous) is not bool:
            raise ValueError("Azure compression and anonymous flags must be boolean")
        if self.serialization_format not in {"json", "yaml"}:
            raise ValueError("Azure records require JSON or safe YAML")
        if not isinstance(self.metadata, dict) or any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or not key.isascii()
            or not value.isascii()
            for key, value in self.metadata.items()
        ):
            raise ValueError("Azure blob metadata requires ASCII string keys and values")
        if not isinstance(self.record_metadata, dict):
            raise ValueError("Azure record_metadata must be a mapping")
        if not isinstance(self.content_type, str) or not self.content_type:
            raise ValueError("Azure content_type must be a non-empty string")
        if self.access_tier is not None and self.access_tier not in TIERS:
            raise ValueError("Unsupported Azure access tier")
        if type(self.max_object_bytes) is not int or self.max_object_bytes < 1:
            raise ValueError("max_object_bytes must be a positive integer")
        for name in (
            "connection_string",
            "account_url",
            "account_name",
            "account_key",
            "sas_token",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError("Azure connection settings must be non-empty strings")
        if (
            sum(
                value is not None
                for value in (self.connection_string, self.account_url, self.account_name)
            )
            > 1
        ):
            raise ValueError("Select one Azure connection source")
        if (
            self.account_key is not None
            and self.sas_token is not None
            or self.connection_string is not None
            and (self.account_key is not None or self.sas_token is not None)
            or self.anonymous
            and (
                self.connection_string is not None
                or self.account_key is not None
                or self.sas_token is not None
            )
        ):
            raise ValueError("Azure authentication settings conflict")
        if self.compression is not None:
            self.compression = get_compressor(self.compression).algorithm.value
            self.use_compression = self.compression != "none"
        elif self.use_compression:
            self.compression = "gzip"

    def get_full_prefix(self):
        parts = [part.strip("/") for part in (self.prefix, self.namespace) if part.strip("/")]
        return "/".join(parts) + ("/" if parts else "")


class AzureBlobStore(ObjectValidationStore[AzureBlobConfig]):
    """Native Azure object lifetime, ETag conditions and explicit index recovery."""

    _provider_errors = (AzureError,)
    _legacy_key_field = "blob_name"

    def __init__(
        self,
        container: str = "",
        prefix: str = "truthound/",
        connection_string: str | None = None,
        account_url: str | None = None,
        account_name: str | None = None,
        account_key: str | None = None,
        sas_token: str | None = None,
        namespace: str = "default",
        compression: bool | str = True,
        access_tier: str | None = None,
        *,
        config: AzureBlobConfig | None = None,
        **kwargs: Any,
    ):
        _require_azure()
        if config is not None:
            if not isinstance(config, AzureBlobConfig):
                raise TypeError("config must be AzureBlobConfig")
            if (
                container
                or prefix != "truthound/"
                or any(
                    value is not None
                    for value in (
                        connection_string,
                        account_url,
                        account_name,
                        account_key,
                        sas_token,
                        access_tier,
                    )
                )
                or namespace != "default"
                or compression is not True
                or kwargs
            ):
                raise ValueError("Select a config object or individual Azure settings")
            selected = replace(deepcopy(config))
        else:
            if set(kwargs) - {item.name for item in fields(AzureBlobConfig)}:
                raise TypeError("Unknown Azure store setting")
            options = deepcopy(kwargs)
            if type(compression) is bool:
                options.setdefault("use_compression", compression)
            elif isinstance(compression, str):
                options["compression"] = compression
            else:
                raise ValueError("Invalid Azure compression setting")
            selected = AzureBlobConfig(
                container=container,
                prefix=prefix,
                connection_string=connection_string,
                account_url=account_url,
                account_name=account_name,
                account_key=account_key,
                sas_token=sas_token,
                namespace=namespace,
                access_tier=access_tier,
                **options,
            )
        super().__init__(selected)
        self._container_client = None
        self._credential = None
        self._cleanup_uncertain = False

    @classmethod
    def _default_config(cls):
        raise ValueError("AzureBlobStore requires explicit container configuration")

    @staticmethod
    def _missing(error):
        return isinstance(error, ResourceNotFoundError)

    def _guard(self):
        if self._cleanup_uncertain:
            raise StoreConnectionError(
                "Azure Blob", "Client cleanup is unconfirmed; retry close before new work"
            )
        super()._guard()

    def _connect(self):
        self._guard()
        if self._container_client is not None:
            return
        if self._client is not None or self._credential is not None:
            raise StoreConnectionError("Azure Blob", "Previous client cleanup is incomplete")
        try:
            if self._config.connection_string:
                self._client = BlobServiceClient.from_connection_string(
                    self._config.connection_string
                )
            else:
                url = self._config.account_url or (
                    f"https://{self._config.account_name}.blob.core.windows.net"
                    if self._config.account_name
                    else None
                )
                if url is None:
                    raise StoreConnectionError(
                        "Azure Blob", "An account URL, name or connection string is required"
                    )
                credential = self._config.sas_token or self._config.account_key
                if credential is None and not self._config.anonymous:
                    try:
                        from azure.identity import DefaultAzureCredential
                    except ImportError:
                        raise ImportError(
                            "Default Azure credentials require azure-identity; anonymous access must be selected explicitly"
                        ) from None
                    self._credential = credential = DefaultAzureCredential()
                self._client = BlobServiceClient(account_url=url, credential=credential)
            container = self._client.get_container_client(self._config.container)
            self._guard()
            if not container.exists():
                raise StoreConnectionError("Azure Blob", "Container not found")
            self._guard()
        except BaseException as original:
            try:
                self.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "Azure connection and cleanup failed", [original, cleanup]
                ) from None
            if isinstance(original, AzureError):
                raise StoreConnectionError("Azure Blob", "Connection failed") from None
            raise
        self._container_client = container

    def _record_metadata_defaults(self):
        return self._config.record_metadata

    def _get_blob_name(self, item_id):
        return self._get_key(item_id)

    @staticmethod
    def _etag(properties):
        value = properties.get("etag")
        if not isinstance(value, str) or not value:
            raise RuntimeError("Azure has no confirmed ETag receipt")
        return value

    @staticmethod
    def _condition(etag):
        return {"etag": etag, "match_condition": MatchConditions.IfNotModified} if etag else {}

    def _properties(self, key, *, etag=None):
        self._guard()
        result = self._container_client.get_blob_client(key).get_blob_properties(
            **self._condition(etag)
        )
        self._guard()
        return result

    def _head(self, key, *, etag=None):
        return self._etag(self._properties(key, etag=etag))

    def _read_bytes(self, key, *, etag=None):
        properties = self._properties(key, etag=etag)
        observed = self._etag(properties)
        size = properties.get("size")
        if type(size) is not int or size < 0 or size > self._config.max_object_bytes:
            raise ValueError("Azure object exceeds configured bound")
        self._guard()
        downloader = self._container_client.get_blob_client(key).download_blob(
            decompress=False, **self._condition(observed)
        )
        content = downloader.readall()
        self._guard()
        if len(content) != size or len(content) > self._config.max_object_bytes:
            raise ValueError("Azure object length differs from its receipt")
        return content, observed

    def _put(self, key, content, *, etag=None, is_index=False, item=None):
        blob = self._container_client.get_blob_client(key)
        content_type = (
            "application/json"
            if is_index
            else "application/yaml"
            if self._config.content_type == "application/json"
            and self._config.serialization_format == "yaml"
            else self._config.content_type
        )
        encoding = (
            "gzip"
            if not is_index and self._config.use_compression and self._config.compression == "gzip"
            else None
        )
        options = {
            "overwrite": etag is not None,
            "content_settings": ContentSettings(
                content_type=content_type, content_encoding=encoding
            ),
            **self._condition(etag),
        }
        if not is_index:
            options["metadata"] = {
                "data_asset": quote(item.data_asset, safe=""),
                "run_time": item.run_time.isoformat(),
                "status": item.status.value,
                **self._config.metadata,
            }
            if self._config.access_tier is not None:
                options["standard_blob_tier"] = StandardBlobTier(self._config.access_tier)
        self._guard()
        response = blob.upload_blob(content, **options)
        self._guard()
        return self._etag(response)

    def _delete(self, key, etag):
        observed = []
        self._guard()
        self._container_client.get_blob_client(key).delete_blob(
            **self._condition(etag),
            raw_response_hook=lambda response: observed.append(response.http_response.status_code),
        )
        self._guard()
        if observed != [202]:
            raise RuntimeError("Azure deletion has no confirmed HTTP receipt")

    def _keys(self, prefix):
        for blob in self._container_client.list_blobs(name_starts_with=prefix):
            yield blob.name

    def get_access_tier(self, item_id):
        with self._lock:
            key = self._get_key(item_id)
            self.initialize()
            try:
                props = self._properties(key)
                tier = props.get("blob_tier")
                tier = getattr(tier, "value", tier)
                if tier not in TIERS:
                    raise ValueError("Unsupported observed Azure access tier")
                return {"tier": tier, "archive_status": props.get("archive_status")}
            except self._errors as error:
                raise StoreReadError("Azure access-tier query failed") from error

    def set_access_tier(self, item_id, tier):
        if tier not in TIERS:
            raise ValueError("Unsupported Azure access tier")
        with self._lock:
            key = self._get_key(item_id)
            try:
                self._mutable()
                previous = self._index[item_id]["etag"]
                observed = []
                self._guard()
                self._container_client.get_blob_client(key).set_standard_blob_tier(
                    tier,
                    **self._condition(previous),
                    raw_response_hook=lambda response: observed.append(
                        response.http_response.status_code
                    ),
                )
                self._guard()
                if len(observed) != 1 or observed[0] not in {200, 202}:
                    raise RuntimeError("Azure tier change has no confirmed HTTP receipt")
                if self._head(key) != previous:
                    raise RuntimeError("Azure record changed during tier update")
                return True
            except self._errors as error:
                self._needs_reconcile = True
                raise StoreWriteError("Azure access-tier update is unconfirmed") from error

    def close(self):
        with self._lock:
            failures = []
            if self._client is not None:
                try:
                    self._client.close()
                except BaseException as error:
                    failures.append(error)
                else:
                    self._client = None
                    self._container_client = None
            if self._credential is not None:
                try:
                    self._credential.close()
                except BaseException as error:
                    failures.append(error)
                else:
                    self._credential = None
            self._cleanup_uncertain = bool(failures)
            self._initialized = False
            if failures:
                raise BaseExceptionGroup("Azure client cleanup failed", failures)
