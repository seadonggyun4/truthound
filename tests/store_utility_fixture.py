"""Installed local native codec/flow-control compositions, no live endpoints."""

from contextlib import asynccontextmanager


@asynccontextmanager
async def factory():
    from datetime import timedelta

    from truthound.mcp.store_utilities import (
        StoreCompressionStreamBinding,
        StoreEncryptionStreamBinding,
        StoreKeyBinding,
        StoreUtilityRuntime,
    )
    from truthound.stores.backpressure.strategies import TokenBucketBackpressure
    from truthound.stores.compression.adaptive import AdaptiveCompressor, AdaptiveConfig
    from truthound.stores.compression.base import CompressionAlgorithm
    from truthound.stores.compression.pipeline import CompressionPipeline
    from truthound.stores.compression.providers import get_compressor
    from truthound.stores.compression.streaming import (
        ChunkedCompressor,
        StreamingCompressor,
        StreamingDecompressor,
    )
    from truthound.stores.encryption.keys import FileKeyStore, KeyManager, KeyManagerConfig
    from truthound.stores.encryption.pipeline import EncryptionPipeline
    from truthound.stores.encryption.streaming import (
        ChunkedEncryptor,
        StreamingDecryptor,
        StreamingEncryptor,
    )

    from . import MARKER, note

    class Pressure(TokenBucketBackpressure):
        async def acquire(self):
            note("utility-acquire")
            if MARKER.with_name("utility-pressure-failed").exists():
                raise OSError("synthetic-private-flow-provider")
            return await super().acquire()

    class CompressionStream(StreamingCompressor):
        def __exit__(self, *args):
            note("utility-stream-close")
            super().__exit__(*args)
            if MARKER.with_name("utility-stream-exit-failed").exists():
                raise OSError("synthetic-private-stream-exit")

    compression = (
        CompressionPipeline("synthetic-private-compressor")
        .add_compression(get_compressor("gzip"))
        .add_checksum()
    )
    # Public synthetic test key; no user credentials are read or installed.
    encryption = EncryptionPipeline().add_compression("gzip").add_encryption(key=bytes(range(32)))

    def audit(event, detail):
        note("utility-key-" + event)
        if event == "key_created" and MARKER.with_name("utility-key-receipt-failed").exists():
            raise OSError("synthetic-private-key-audit")

    manager = KeyManager(
        FileKeyStore(MARKER.with_name("operator-keys"), master_key=bytes(range(32))),
        config=KeyManagerConfig(
            default_ttl=timedelta(0) if MARKER.with_name("utility-key-expired").exists() else None
        ),
        audit_hook=audit,
    )
    note("utility-open")
    try:
        yield StoreUtilityRuntime(
            compression={"local": compression},
            encryption={"local": encryption},
            adaptive={
                "local": AdaptiveCompressor(
                    AdaptiveConfig(allowed_algorithms=[CompressionAlgorithm.GZIP])
                )
            },
            chunked_compression={"local": ChunkedCompressor(chunk_size=64)},
            chunked_encryption={"local": ChunkedEncryptor(bytes(range(32)), chunk_size=64)},
            backpressure={"local": Pressure()},
            payloads={"seed": b"synthetic-storage-payload" * 100},
            slots={"local": MARKER.with_name("operator-payload.bin")},
            keys={"local": StoreKeyBinding(manager, "synthetic-private-key-id")},
            compression_streams={
                "local": StoreCompressionStreamBinding(
                    lambda: CompressionStream(flush_threshold=10000), StreamingDecompressor
                )
            },
            encryption_streams={
                "local": StoreEncryptionStreamBinding(
                    lambda output: StreamingEncryptor(bytes(range(32)), output, chunk_size=64),
                    lambda source: StreamingDecryptor(bytes(range(32)), source),
                )
            },
        )
    finally:
        note("utility-close")
