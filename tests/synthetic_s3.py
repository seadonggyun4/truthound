"""Synthetic object protocol for tests; never connects to a cloud provider."""

class MemoryS3:
    """Synthetic S3 protocol only; no remote availability claim."""

    def __init__(self):
        from types import SimpleNamespace

        self.objects = {}
        self.exceptions = SimpleNamespace(NoSuchKey=KeyError)
        self.closed = False

    def head_bucket(self, **kwargs):
        return {}

    def head_object(self, *, Key, IfMatch=None, **kwargs):
        etag = self.etag(Key)
        if IfMatch is not None and IfMatch != etag:
            raise RuntimeError("synthetic-precondition-failed")
        return {"ETag": etag, "ContentLength": len(self.objects[Key])}

    def put_object(self, *, Key, Body, **kwargs):
        if kwargs.get("IfNoneMatch") == "*" and Key in self.objects:
            raise RuntimeError("synthetic-precondition-failed")
        if "IfMatch" in kwargs and kwargs["IfMatch"] != self.etag(Key):
            raise RuntimeError("synthetic-precondition-failed")
        self.objects[Key] = bytes(Body)
        return {"ETag": self.etag(Key)}

    def etag(self, key):
        import hashlib

        return '"' + hashlib.sha256(self.objects[key]).hexdigest() + '"'

    def get_object(self, *, Key, **kwargs):
        from io import BytesIO

        if "IfMatch" in kwargs and kwargs["IfMatch"] != self.etag(Key):
            raise RuntimeError("synthetic-precondition-failed")
        return {"Body": BytesIO(self.objects[Key]), "ETag": self.etag(Key)}

    def delete_object(self, *, Key, IfMatch=None, **kwargs):
        if IfMatch is not None and IfMatch != self.etag(Key):
            raise RuntimeError("synthetic-precondition-failed")
        del self.objects[Key]
        return {"ResponseMetadata": {"HTTPStatusCode": 204}}

    def get_paginator(self, operation):
        client = self

        class Paginator:
            def paginate(self, *, Prefix, Delimiter=None, **kwargs):
                keys = sorted(key for key in client.objects if key.startswith(Prefix))
                if Delimiter is not None:
                    yield {"CommonPrefixes": [{"Prefix": value} for value in sorted({Prefix + key[len(Prefix):].split("/", 1)[0] + "/" for key in keys})]}
                else:
                    yield {"Contents": [{"Key": key, "Size": len(client.objects[key])} for key in keys]}

        return Paginator()

    def close(self):
        self.closed = True


class PersistentMemoryS3(MemoryS3):
    """Private synthetic objects survive the installed test worker's death."""

    def __init__(self, path):
        from pathlib import Path

        super().__init__()
        self.path = Path(path)
        self._load()

    def _load(self):
        import base64
        import json

        if self.path.exists():
            self.objects = {key: base64.b64decode(value) for key, value in json.loads(self.path.read_text()).items()}

    def _save(self):
        import base64
        import json
        import os
        import tempfile

        descriptor, name = tempfile.mkstemp(prefix=".synthetic-s3-", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump({key: base64.b64encode(value).decode() for key, value in self.objects.items()}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def put_object(self, **arguments):
        self._load()
        result = super().put_object(**arguments)
        self._save()
        return result

    def delete_object(self, **arguments):
        self._load()
        result = super().delete_object(**arguments)
        self._save()
        return result

    def get_object(self, **arguments):
        self._load()
        return super().get_object(**arguments)

    def head_object(self, **arguments):
        self._load()
        return super().head_object(**arguments)

    def get_paginator(self, operation):
        self._load()
        return super().get_paginator(operation)


class OrdinaryMemoryS3(PersistentMemoryS3):
    """Use native botocore absence errors for ordinary S3Store workflows."""

    def get_object(self, **arguments):
        from botocore.exceptions import ClientError

        try:
            return super().get_object(**arguments)
        except KeyError:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject") from None

    def head_object(self, **arguments):
        from botocore.exceptions import ClientError

        try:
            return super().head_object(**arguments)
        except KeyError:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject") from None


class PersistentMultipartS3(PersistentMemoryS3):
    """Synthetic multipart receipts survive an actual owned worker exit."""

    def __init__(self, path, crash_at=None):
        import threading

        super().__init__(path)
        self.crash_at = crash_at
        self.upload_path = self.path.with_suffix('.uploads.json')
        self.upload_lock = threading.RLock()
        self.exceptions.NoSuchUpload = KeyError
        self.pending = {}
        self._load_uploads()

    def _load_uploads(self):
        import base64
        import json

        if self.upload_path.exists():
            raw = json.loads(self.upload_path.read_text())
            self.pending = {identity: (item['key'], {int(number): base64.b64decode(body) for number, body in item['parts'].items()}) for identity, item in raw.items()}

    def _save_uploads(self):
        import base64
        import json
        import os
        import tempfile

        descriptor, name = tempfile.mkstemp(prefix='.synthetic-multipart-', dir=self.path.parent)
        try:
            with os.fdopen(descriptor, 'w') as stream:
                json.dump({identity: {'key': key, 'parts': {str(number): base64.b64encode(body).decode() for number, body in parts.items()}} for identity, (key, parts) in self.pending.items()}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.upload_path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def create_multipart_upload(self, *, Key, **arguments):
        import os
        import uuid

        with self.upload_lock:
            self._load_uploads()
            identity = 'synthetic-upload-' + uuid.uuid4().hex
            self.pending[identity] = (Key, {})
            self._save_uploads()
        if self.crash_at == 'create':
            os._exit(74)
        return {'UploadId': identity}

    def upload_part(self, *, Key, UploadId, PartNumber, Body, **arguments):
        import hashlib
        import os

        with self.upload_lock:
            self._load_uploads()
            key, parts = self.pending[UploadId]
            if key != Key:
                raise RuntimeError('Synthetic upload key mismatch')
            parts[PartNumber] = bytes(Body)
            self._save_uploads()
        if self.crash_at == 'part':
            os._exit(74)
        return {'ETag': '"' + hashlib.sha256(Body).hexdigest() + '"'}

    def complete_multipart_upload(self, *, Key, UploadId, IfNoneMatch=None, **arguments):
        with self.upload_lock:
            self._load_uploads()
            key, parts = self.pending[UploadId]
            if key != Key:
                raise RuntimeError('Synthetic upload key mismatch')
            result = self.put_object(Key=Key, Body=b''.join(parts[number] for number in sorted(parts)), IfNoneMatch=IfNoneMatch)
            del self.pending[UploadId]
            self._save_uploads()
            return result

    def abort_multipart_upload(self, *, Key, UploadId, **arguments):
        with self.upload_lock:
            self._load_uploads()
            if self.pending[UploadId][0] != Key:
                raise RuntimeError('Synthetic upload key mismatch')
            del self.pending[UploadId]
            self._save_uploads()
            return {'ResponseMetadata': {'HTTPStatusCode': 204}}
