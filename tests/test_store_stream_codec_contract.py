"""Real native AEAD framing rejects incomplete streams and untrusted metadata."""

import io
import struct
import zlib

import pytest

from truthound.stores.encryption.base import DecryptionError, EncryptionError
from truthound.stores.encryption.streaming import (
    StreamingDecryptor,
    StreamingEncryptor,
    StreamingHeader,
)

pytestmark = pytest.mark.contract
KEY = bytes(range(32))


def encrypted(data=b"abcdefghijkl"):
    output = io.BytesIO()
    with StreamingEncryptor(KEY, output, chunk_size=4) as writer:
        writer.write(data)
    return output.getvalue()


@pytest.mark.parametrize("mutation", ["missing_end", "remove_suffix", "header_edit"])
def test_incomplete_or_edited_stream_never_reports_success(mutation):
    data = encrypted()
    header = StreamingHeader.from_bytes(data)
    if mutation == "missing_end":
        altered = data[:-4]
    elif mutation == "remove_suffix":
        altered = data[: header.size] + struct.pack(">I", 0)
    else:
        updated = bytearray(data[: header.size])
        updated[6:10] = struct.pack(">I", 8)
        updated[-4:] = zlib.crc32(updated[:-4]).to_bytes(4, "big")
        altered = bytes(updated) + data[header.size :]
    with pytest.raises(DecryptionError), StreamingDecryptor(KEY, io.BytesIO(altered)) as reader:
        reader.read_all()


def test_empty_stream_still_authenticates_key_and_aad():
    with (
        pytest.raises(DecryptionError),
        StreamingDecryptor(bytes(reversed(KEY)), io.BytesIO(encrypted(b""))) as reader,
    ):
        reader.read_all()


def test_short_nonseekable_reads_and_short_writes_roundtrip():
    class ShortOutput(io.BytesIO):
        def write(self, data):
            return super().write(data[:3])

    class ShortInput(io.BytesIO):
        def read(self, size=-1):
            return super().read(min(size, 3) if size >= 0 else 3)

        def seek(self, *args):
            raise io.UnsupportedOperation("not seekable")

    output = ShortOutput()
    with StreamingEncryptor(KEY, output, chunk_size=4) as writer:
        writer.write(b"abcdefghijkl")
    with StreamingDecryptor(KEY, ShortInput(output.getvalue())) as reader:
        assert reader.read_all() == b"abcdefghijkl"


def test_invalid_configuration_cannot_truncate_existing_output(tmp_path):
    path = tmp_path / "owned.enc"
    path.write_bytes(b"original")
    with pytest.raises((ValueError, EncryptionError)):
        StreamingEncryptor(b"", path, chunk_size=0)
    assert path.read_bytes() == b"original"


def test_failed_body_never_commits_an_authenticated_partial_stream():
    output = io.BytesIO()
    with (
        pytest.raises(RuntimeError, match="body failed"),
        StreamingEncryptor(KEY, output, chunk_size=4) as writer,
    ):
        writer.write(b"abcdefgh")
        raise RuntimeError("body failed")
    with (
        pytest.raises(DecryptionError),
        StreamingDecryptor(KEY, io.BytesIO(output.getvalue())) as reader,
    ):
        reader.read_all()


def test_legacy_v1_requires_explicit_opt_in_and_never_claims_complete_authentication():
    from truthound.stores.encryption.base import EncryptionAlgorithm
    from truthound.stores.encryption.providers import get_encryptor

    nonce = bytes(range(12))
    header = StreamingHeader(version=1, base_nonce=nonce, chunk_size=4)
    record = get_encryptor(EncryptionAlgorithm.AES_256_GCM).encrypt_with_metrics(
        b"old", KEY, nonce=nonce
    )
    chunk = record.ciphertext + record.tag
    old = header.to_bytes() + struct.pack(">I", len(chunk)) + chunk + struct.pack(">I", 0)
    with pytest.raises(DecryptionError, match="Legacy"):
        StreamingDecryptor(KEY, io.BytesIO(old))
    with StreamingDecryptor(KEY, io.BytesIO(old), allow_legacy=True) as reader:
        assert reader.read_all() == b"old"
        assert reader.authenticated_complete is False


def test_guard_revocation_after_native_encrypt_cannot_write_or_finalize(monkeypatch):
    output = io.BytesIO()
    allowed = [True]
    writer = StreamingEncryptor(KEY, output, chunk_size=4)

    def guard():
        if not allowed[0]:
            raise PermissionError("revoked")

    native = writer._encryptor.encrypt_with_metrics

    def revoking(*args, **kwargs):
        result = native(*args, **kwargs)
        allowed[0] = False
        return result

    writer.set_execution_guard(guard)
    monkeypatch.setattr(writer._encryptor, "encrypt_with_metrics", revoking)
    with pytest.raises(PermissionError, match="revoked"):
        writer.write(b"abcd")
    writer.close()
    assert len(output.getvalue()) == StreamingHeader.from_bytes(output.getvalue()).size
    with (
        pytest.raises(DecryptionError),
        StreamingDecryptor(KEY, io.BytesIO(output.getvalue())) as reader,
    ):
        reader.read_all()


@pytest.mark.parametrize("mutation", ["drop_suffix", "replace_nonce", "alter_total"])
def test_chunked_index_metadata_is_authenticated_before_any_plaintext(mutation):
    from copy import deepcopy

    from truthound.stores.encryption.streaming import ChunkedDecryptor, ChunkedEncryptor

    ciphertext, original = ChunkedEncryptor(KEY, chunk_size=4).encrypt(b"abcdefghijkl")
    index = deepcopy(original)
    if mutation == "drop_suffix":
        index.chunks.pop()
        index.total_plaintext_size -= 4
    elif mutation == "replace_nonce":
        index.chunks[1].nonce = index.chunks[0].nonce
    else:
        index.total_plaintext_size += 1
    with pytest.raises(DecryptionError):
        ChunkedDecryptor(KEY, index).decrypt_all(ciphertext)


def test_chunked_negative_index_and_invalid_ranges_never_select_another_chunk():
    from truthound.stores.encryption.streaming import ChunkedDecryptor, ChunkedEncryptor

    ciphertext, index = ChunkedEncryptor(KEY, chunk_size=4).encrypt(b"abcdefghijkl")
    reader = ChunkedDecryptor(KEY, index)
    with pytest.raises(DecryptionError):
        reader.decrypt_chunk(ciphertext, -1)
    for start, end in [(-1, 4), (0, 13), (5, 4)]:
        with pytest.raises(DecryptionError):
            reader.decrypt_range(ciphertext, start, end)
    assert reader.decrypt_range(ciphertext, 4, 4) == b""


def test_empty_chunked_index_authenticates_key_and_full_length_is_checked():
    from truthound.stores.encryption.streaming import ChunkedDecryptor, ChunkedEncryptor

    ciphertext, index = ChunkedEncryptor(KEY).encrypt(b"")
    with pytest.raises(DecryptionError):
        ChunkedDecryptor(bytes(reversed(KEY)), index).decrypt_all(ciphertext)
    ciphertext, index = ChunkedEncryptor(KEY, chunk_size=4).encrypt(b"abcdefghijkl")
    with pytest.raises(DecryptionError):
        ChunkedDecryptor(KEY, index).decrypt_all(ciphertext + b"trailing")
