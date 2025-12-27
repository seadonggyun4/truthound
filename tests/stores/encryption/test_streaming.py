"""Tests for streaming encryption module."""

import io
import tempfile
from pathlib import Path

import pytest

from truthound.stores.encryption.base import (
    DecryptionError,
    EncryptionAlgorithm,
    IntegrityError,
    generate_key,
)
from truthound.stores.encryption.streaming import (
    ChunkIndex,
    ChunkMetadata,
    ChunkedDecryptor,
    ChunkedEncryptor,
    StreamingDecryptor,
    StreamingEncryptor,
    StreamingHeader,
    StreamingMetrics,
    derive_chunk_nonce,
)


# Check if cryptography is available
try:
    import cryptography

    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False


class TestDeriveChunkNonce:
    """Tests for chunk nonce derivation."""

    def test_deterministic(self):
        """Test that nonce derivation is deterministic."""
        base_nonce = b"x" * 12

        nonce1 = derive_chunk_nonce(base_nonce, 0)
        nonce2 = derive_chunk_nonce(base_nonce, 0)
        assert nonce1 == nonce2

    def test_unique_per_chunk(self):
        """Test that each chunk gets a unique nonce."""
        base_nonce = b"x" * 12

        nonces = set()
        for i in range(100):
            nonce = derive_chunk_nonce(base_nonce, i)
            nonces.add(nonce)

        assert len(nonces) == 100

    def test_custom_size(self):
        """Test nonce derivation with custom size."""
        base_nonce = b"x" * 24
        nonce = derive_chunk_nonce(base_nonce, 5, nonce_size=24)
        assert len(nonce) == 24


class TestStreamingMetrics:
    """Tests for StreamingMetrics."""

    def test_overhead_calculation(self):
        """Test overhead bytes calculation."""
        metrics = StreamingMetrics(
            total_plaintext_bytes=1000,
            total_ciphertext_bytes=1100,
        )
        assert metrics.overhead_bytes == 100

    def test_throughput_calculation(self):
        """Test throughput calculation."""
        metrics = StreamingMetrics(
            total_plaintext_bytes=1024 * 1024,  # 1 MB
            total_time_ms=100,  # 100 ms
        )
        assert metrics.throughput_mbps == 10.0

    def test_to_dict(self):
        """Test metrics serialization."""
        metrics = StreamingMetrics(
            total_chunks=10,
            total_plaintext_bytes=1000,
            algorithm=EncryptionAlgorithm.AES_256_GCM,
        )
        d = metrics.to_dict()
        assert d["total_chunks"] == 10
        assert d["algorithm"] == "aes-256-gcm"


class TestStreamingHeader:
    """Tests for StreamingHeader."""

    def test_serialization_round_trip(self):
        """Test header serialization and deserialization."""
        header = StreamingHeader(
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            chunk_size=1024 * 64,
            base_nonce=b"x" * 12,
        )

        serialized = header.to_bytes()
        restored = StreamingHeader.from_bytes(serialized)

        assert restored.algorithm == header.algorithm
        assert restored.chunk_size == header.chunk_size
        assert restored.base_nonce == header.base_nonce

    def test_invalid_magic(self):
        """Test rejection of invalid magic bytes."""
        with pytest.raises(DecryptionError):
            StreamingHeader.from_bytes(b"XXXX" + b"\x00" * 50)

    def test_checksum_verification(self):
        """Test checksum verification."""
        header = StreamingHeader()
        serialized = bytearray(header.to_bytes())

        # Corrupt data before checksum
        serialized[10] ^= 0xFF

        with pytest.raises(IntegrityError):
            StreamingHeader.from_bytes(bytes(serialized))


@pytest.mark.skipif(not HAS_CRYPTOGRAPHY, reason="cryptography not installed")
class TestStreamingEncryptor:
    """Tests for StreamingEncryptor."""

    @pytest.fixture
    def key(self):
        """Generate test key."""
        return generate_key(EncryptionAlgorithm.AES_256_GCM)

    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        return b"x" * 1000

    def test_basic_encryption(self, key, sample_data):
        """Test basic streaming encryption to BytesIO."""
        output = io.BytesIO()

        with StreamingEncryptor(key, output, chunk_size=256) as enc:
            enc.write(sample_data)

        encrypted = output.getvalue()
        assert len(encrypted) > len(sample_data)  # Includes overhead

    def test_encryption_to_file(self, key, sample_data):
        """Test streaming encryption to file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            output_path = f.name

        try:
            with StreamingEncryptor(key, output_path, chunk_size=256) as enc:
                enc.write(sample_data)

            # File should exist and have content
            assert Path(output_path).exists()
            assert Path(output_path).stat().st_size > 0
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_multiple_writes(self, key):
        """Test multiple write calls."""
        output = io.BytesIO()

        with StreamingEncryptor(key, output, chunk_size=100) as enc:
            for _ in range(10):
                enc.write(b"chunk_data_" * 10)

        assert output.getvalue()

    def test_metrics(self, key, sample_data):
        """Test metrics collection."""
        output = io.BytesIO()

        with StreamingEncryptor(key, output, chunk_size=256) as enc:
            enc.write(sample_data)
            metrics = enc.finalize()

        assert metrics.total_plaintext_bytes == len(sample_data)
        assert metrics.total_chunks > 0
        assert metrics.algorithm == EncryptionAlgorithm.AES_256_GCM

    def test_empty_write(self, key):
        """Test encryption of empty data."""
        output = io.BytesIO()

        with StreamingEncryptor(key, output, chunk_size=256) as enc:
            enc.write(b"")

        # Should still have header and end marker
        assert len(output.getvalue()) > 0


@pytest.mark.skipif(not HAS_CRYPTOGRAPHY, reason="cryptography not installed")
class TestStreamingDecryptor:
    """Tests for StreamingDecryptor."""

    @pytest.fixture
    def key(self):
        """Generate test key."""
        return generate_key(EncryptionAlgorithm.AES_256_GCM)

    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        return b"hello world! " * 100

    def test_basic_decryption(self, key, sample_data):
        """Test basic streaming decryption."""
        # Encrypt
        encrypted = io.BytesIO()
        with StreamingEncryptor(key, encrypted, chunk_size=256) as enc:
            enc.write(sample_data)

        # Decrypt
        encrypted.seek(0)
        with StreamingDecryptor(key, encrypted) as dec:
            decrypted = dec.read_all()

        assert decrypted == sample_data

    def test_chunk_iteration(self, key, sample_data):
        """Test iterating over decrypted chunks."""
        # Encrypt
        encrypted = io.BytesIO()
        with StreamingEncryptor(key, encrypted, chunk_size=256) as enc:
            enc.write(sample_data)

        # Decrypt via iteration
        encrypted.seek(0)
        chunks = []
        with StreamingDecryptor(key, encrypted) as dec:
            for chunk in dec:
                chunks.append(chunk)

        decrypted = b"".join(chunks)
        assert decrypted == sample_data

    def test_file_round_trip(self, key, sample_data):
        """Test encryption/decryption to/from file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            output_path = f.name

        try:
            # Encrypt to file
            with StreamingEncryptor(key, output_path, chunk_size=256) as enc:
                enc.write(sample_data)

            # Decrypt from file
            with StreamingDecryptor(key, output_path) as dec:
                decrypted = dec.read_all()

            assert decrypted == sample_data
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_wrong_key(self, key, sample_data):
        """Test decryption with wrong key."""
        wrong_key = generate_key(EncryptionAlgorithm.AES_256_GCM)

        # Encrypt
        encrypted = io.BytesIO()
        with StreamingEncryptor(key, encrypted, chunk_size=256) as enc:
            enc.write(sample_data)

        # Decrypt with wrong key
        encrypted.seek(0)
        with pytest.raises(IntegrityError):
            with StreamingDecryptor(wrong_key, encrypted) as dec:
                dec.read_all()

    def test_tampered_data(self, key, sample_data):
        """Test detection of tampered ciphertext."""
        # Encrypt
        encrypted = io.BytesIO()
        with StreamingEncryptor(key, encrypted, chunk_size=256) as enc:
            enc.write(sample_data)

        # Tamper with encrypted data
        data = bytearray(encrypted.getvalue())
        data[100] ^= 0xFF  # Flip a bit in ciphertext
        tampered = io.BytesIO(bytes(data))

        # Should detect tampering
        with pytest.raises((IntegrityError, DecryptionError)):
            with StreamingDecryptor(key, tampered) as dec:
                dec.read_all()

    def test_large_data(self, key):
        """Test with larger data."""
        large_data = b"x" * (1024 * 1024)  # 1 MB

        encrypted = io.BytesIO()
        with StreamingEncryptor(key, encrypted, chunk_size=64 * 1024) as enc:
            enc.write(large_data)

        encrypted.seek(0)
        with StreamingDecryptor(key, encrypted) as dec:
            decrypted = dec.read_all()

        assert decrypted == large_data

    def test_with_aad(self, key, sample_data):
        """Test streaming with additional authenticated data."""
        aad = b"authenticated_context"

        # Encrypt with AAD
        encrypted = io.BytesIO()
        with StreamingEncryptor(key, encrypted, chunk_size=256, aad=aad) as enc:
            enc.write(sample_data)

        # Decrypt with same AAD
        encrypted.seek(0)
        with StreamingDecryptor(key, encrypted, aad=aad) as dec:
            decrypted = dec.read_all()

        assert decrypted == sample_data

        # Decrypt with wrong AAD should fail
        encrypted.seek(0)
        with pytest.raises(IntegrityError):
            with StreamingDecryptor(key, encrypted, aad=b"wrong_aad") as dec:
                dec.read_all()


@pytest.mark.skipif(not HAS_CRYPTOGRAPHY, reason="cryptography not installed")
class TestChunkedEncryptor:
    """Tests for ChunkedEncryptor (random access support)."""

    @pytest.fixture
    def key(self):
        """Generate test key."""
        return generate_key(EncryptionAlgorithm.AES_256_GCM)

    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        return b"0123456789" * 100  # 1000 bytes

    def test_basic_encryption(self, key, sample_data):
        """Test basic chunked encryption."""
        enc = ChunkedEncryptor(key, chunk_size=256)
        ciphertext, index = enc.encrypt(sample_data)

        assert len(ciphertext) > len(sample_data)
        assert len(index.chunks) > 0
        assert index.total_plaintext_size == len(sample_data)

    def test_chunk_index_serialization(self, key, sample_data):
        """Test ChunkIndex serialization."""
        enc = ChunkedEncryptor(key, chunk_size=256)
        _, index = enc.encrypt(sample_data)

        serialized = index.to_bytes()
        restored = ChunkIndex.from_bytes(serialized)

        assert restored.algorithm == index.algorithm
        assert restored.chunk_size == index.chunk_size
        assert len(restored.chunks) == len(index.chunks)


@pytest.mark.skipif(not HAS_CRYPTOGRAPHY, reason="cryptography not installed")
class TestChunkedDecryptor:
    """Tests for ChunkedDecryptor (random access support)."""

    @pytest.fixture
    def key(self):
        """Generate test key."""
        return generate_key(EncryptionAlgorithm.AES_256_GCM)

    @pytest.fixture
    def sample_data(self):
        """Generate sample data with known pattern."""
        # Create data where each chunk has identifiable content
        return b"".join(f"CHUNK{i:04d}__".encode() * 25 for i in range(10))

    def test_decrypt_all(self, key, sample_data):
        """Test decrypting all chunks."""
        enc = ChunkedEncryptor(key, chunk_size=256)
        ciphertext, index = enc.encrypt(sample_data)

        dec = ChunkedDecryptor(key, index)
        decrypted = dec.decrypt_all(ciphertext)

        assert decrypted == sample_data

    def test_decrypt_single_chunk(self, key, sample_data):
        """Test decrypting a single chunk."""
        enc = ChunkedEncryptor(key, chunk_size=256)
        ciphertext, index = enc.encrypt(sample_data)

        dec = ChunkedDecryptor(key, index)

        # Decrypt specific chunk
        chunk_0 = dec.decrypt_chunk(ciphertext, 0)
        assert len(chunk_0) <= 256

        chunk_1 = dec.decrypt_chunk(ciphertext, 1)
        assert chunk_1 != chunk_0

    def test_decrypt_range(self, key, sample_data):
        """Test decrypting a byte range."""
        enc = ChunkedEncryptor(key, chunk_size=256)
        ciphertext, index = enc.encrypt(sample_data)

        dec = ChunkedDecryptor(key, index)

        # Decrypt middle range
        start, end = 100, 500
        range_data = dec.decrypt_range(ciphertext, start, end)

        assert range_data == sample_data[start:end]

    def test_file_based_decryption(self, key, sample_data):
        """Test decryption from file-like object."""
        enc = ChunkedEncryptor(key, chunk_size=256)
        ciphertext, index = enc.encrypt(sample_data)

        # Use BytesIO as file-like object
        ciphertext_file = io.BytesIO(ciphertext)

        dec = ChunkedDecryptor(key, index)
        chunk = dec.decrypt_chunk(ciphertext_file, 2)

        assert chunk  # Should get some data

    def test_invalid_chunk_index(self, key, sample_data):
        """Test handling of invalid chunk index."""
        enc = ChunkedEncryptor(key, chunk_size=256)
        ciphertext, index = enc.encrypt(sample_data)

        dec = ChunkedDecryptor(key, index)

        with pytest.raises(DecryptionError):
            dec.decrypt_chunk(ciphertext, 999)  # Way out of range

    def test_encrypt_to_file(self, key, sample_data):
        """Test encrypting to file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            output_path = f.name
        index_path = output_path + ".idx"

        try:
            enc = ChunkedEncryptor(key, chunk_size=256)
            index = enc.encrypt_to_file(sample_data, output_path, index_path)

            # Verify files exist
            assert Path(output_path).exists()
            assert Path(index_path).exists()

            # Decrypt
            with open(output_path, "rb") as f:
                ciphertext = f.read()

            dec = ChunkedDecryptor(key, index)
            decrypted = dec.decrypt_all(ciphertext)
            assert decrypted == sample_data
        finally:
            Path(output_path).unlink(missing_ok=True)
            Path(index_path).unlink(missing_ok=True)


class TestChunkMetadata:
    """Tests for ChunkMetadata."""

    def test_creation(self):
        """Test metadata creation."""
        meta = ChunkMetadata(
            index=0,
            nonce=b"x" * 12,
            plaintext_size=256,
            ciphertext_size=284,
            offset=0,
        )

        assert meta.index == 0
        assert meta.ciphertext_size > meta.plaintext_size


class TestChunkIndex:
    """Tests for ChunkIndex."""

    def test_get_chunk_for_offset(self):
        """Test offset to chunk index mapping."""
        index = ChunkIndex(chunk_size=256)

        assert index.get_chunk_for_offset(0) == 0
        assert index.get_chunk_for_offset(255) == 0
        assert index.get_chunk_for_offset(256) == 1
        assert index.get_chunk_for_offset(512) == 2

    def test_serialization(self):
        """Test index serialization."""
        index = ChunkIndex(
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            base_nonce=b"x" * 12,
            chunk_size=256,
            total_plaintext_size=1000,
            chunks=[
                ChunkMetadata(
                    index=0,
                    nonce=b"n" * 12,
                    plaintext_size=256,
                    ciphertext_size=284,
                    offset=0,
                ),
            ],
        )

        serialized = index.to_bytes()
        restored = ChunkIndex.from_bytes(serialized)

        assert restored.algorithm == index.algorithm
        assert restored.base_nonce == index.base_nonce
        assert len(restored.chunks) == 1
