"""Real cloud SDKs over owned HTTP responses; no remote requests or credentials."""

import base64
import hashlib
import io
import json
from email.parser import BytesParser
from email.policy import default
from urllib.parse import parse_qs, unquote, urlsplit

import requests
from azure.core.pipeline.transport import HttpTransport
from azure.core.rest._requests_basic import RestRequestsTransportResponse
from urllib3.response import HTTPResponse


class GoogleObjectHTTP(requests.Session):
    """Exercise Google's actual JSON/media serialization and gzip decoder."""

    def __init__(self, path=None):
        super().__init__()
        self.objects = {}
        self.requests = []
        self.generation = 0
        self.closed = False
        self.is_mtls = False
        self.denied = set()
        self.before_media = None
        self.omit_upload_generation = False
        self.path = path
        self._load()

    def _load(self):
        from pathlib import Path

        if self.path is not None and Path(self.path).exists():
            document = json.loads(Path(self.path).read_text())
            self.generation = document["generation"]
            self.objects = {
                key: (base64.b64decode(row["data"]), row["metadata"])
                for key, row in document["objects"].items()
            }

    def _save(self):
        import os
        import tempfile
        from pathlib import Path

        if self.path is None:
            return
        path = Path(self.path)
        descriptor, temporary = tempfile.mkstemp(prefix=".native-google-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(
                    {
                        "generation": self.generation,
                        "objects": {
                            key: {"data": base64.b64encode(data).decode(), "metadata": meta}
                            for key, (data, meta) in self.objects.items()
                        },
                    },
                    stream,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def seed(self, key, content, **extra):
        self.generation += 1
        self.objects[key] = (content, self.metadata(key, content, self.generation, extra))
        self._save()

    @staticmethod
    def metadata(name, data, generation, extra=None):
        return {
            "name": name,
            "bucket": "synthetic",
            "generation": str(generation),
            "metageneration": "1",
            "size": str(len(data)),
            "etag": '"g' + str(generation) + '"',
            "md5Hash": base64.b64encode(hashlib.md5(data).digest()).decode(),
            **(extra or {}),
        }

    @staticmethod
    def response(method, url, status, body, headers=None):
        response = requests.Response()
        response.status_code = status
        response.url = url
        response.request = requests.Request(method, url).prepare()
        response.headers.update({"content-length": str(len(body)), **(headers or {})})
        response.raw = HTTPResponse(
            body=io.BytesIO(body),
            headers=response.headers,
            preload_content=False,
            request_method=method,
        )
        return response

    def request(self, method, url, data=None, headers=None, **kwargs):
        self._load()
        parsed = urlsplit(url)
        query = parse_qs(parsed.query)
        method = method.upper()
        self.requests.append((method, url, dict(headers or {})))
        path = unquote(parsed.path)
        key = path.split("/o/", 1)[1] if "/o/" in path else None

        def error(code, message):
            return self.response(
                method,
                url,
                code,
                json.dumps({"error": {"code": code, "message": message}}).encode(),
                {"content-type": "application/json"},
            )

        if key in self.denied:
            return error(403, "synthetic denied")
        if (
            key is not None
            and method == "GET"
            and query.get("alt") == ["media"]
            and self.before_media is not None
        ):
            callback, self.before_media = self.before_media, None
            callback(key)
        if method == "POST" and "/upload/" in path:
            content_type = next(
                value for key, value in headers.items() if key.lower() == "content-type"
            )
            if isinstance(content_type, bytes):
                content_type = content_type.decode()
            message = BytesParser(policy=default).parsebytes(
                b"Content-Type: " + content_type.encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + data
            )
            parts = list(message.iter_parts())
            meta = json.loads(parts[0].get_payload(decode=True))
            key = meta["name"]
            previous = self.objects.get(key)
            expected = query.get("ifGenerationMatch", [None])[0]
            if expected is not None and (
                expected == "0"
                and previous is not None
                or expected != "0"
                and (previous is None or previous[1]["generation"] != expected)
            ):
                return error(412, "synthetic precondition failed")
            self.generation += 1
            body = parts[1].get_payload(decode=True)
            metadata = self.metadata(key, body, self.generation, meta)
            self.objects[key] = (body, metadata)
            self._save()
            receipt = dict(metadata)
            if self.omit_upload_generation:
                del receipt["generation"]
            return self.response(
                method, url, 200, json.dumps(receipt).encode(), {"content-type": "application/json"}
            )
        if key is not None:
            if key not in self.objects:
                return error(404, "synthetic missing")
            body, metadata = self.objects[key]
            expected = query.get("ifGenerationMatch", [None])[0]
            if expected is not None and expected != metadata["generation"]:
                return error(412, "synthetic precondition failed")
            if method == "DELETE":
                del self.objects[key]
                self._save()
                return self.response(method, url, 204, b"")
            if method != "GET":
                raise AssertionError("Unexpected native GCS object method")
            if query.get("alt") == ["media"]:
                properties = {
                    "etag": metadata["etag"],
                    "x-goog-generation": metadata["generation"],
                    "x-goog-hash": "md5=" + metadata["md5Hash"],
                }
                if metadata.get("contentEncoding"):
                    properties["content-encoding"] = metadata["contentEncoding"]
                return self.response(method, url, 200, body, properties)
            return self.response(
                method,
                url,
                200,
                json.dumps(metadata).encode(),
                {"content-type": "application/json"},
            )
        if method == "GET" and path.endswith("/o"):
            prefix = query.get("prefix", [""])[0]
            return self.response(
                method,
                url,
                200,
                json.dumps(
                    {
                        "items": [
                            meta
                            for name, (_, meta) in self.objects.items()
                            if name.startswith(prefix)
                        ]
                    }
                ).encode(),
                {"content-type": "application/json"},
            )
        if method == "GET" and path.endswith("/b/synthetic"):
            return self.response(
                method, url, 200, b'{"name":"synthetic"}', {"content-type": "application/json"}
            )
        raise AssertionError("Unexpected native GCS HTTP request")

    def close(self):
        self.closed = True
        super().close()


def google_client(transport):
    from google.auth.credentials import AnonymousCredentials
    from google.cloud.storage import Client

    return Client(project="synthetic", credentials=AnonymousCredentials(), _http=transport)


class AzureObjectResponse(RestRequestsTransportResponse):
    def __init__(self, request, status, body=b"", headers=None):
        response = GoogleObjectHTTP.response(request.method, request.url, status, body, headers)
        response.reason = {
            200: "OK",
            201: "Created",
            202: "Accepted",
            206: "Partial Content",
            403: "Forbidden",
            404: "Not Found",
            412: "Precondition Failed",
        }.get(status, "Synthetic")
        super().__init__(request=request, internal_response=response)


class AzureObjectHTTP(HttpTransport):
    """Actual Azure pipeline/serializers with a wholly offline object service."""

    def __init__(self, path=None):
        self.objects = {}
        self.requests = []
        self.counter = 0
        self.closed = False
        self.denied = set()
        self.before_download = None
        self.omit_upload_etag = False
        self.reject_tier = False
        self.path = path
        self._load()

    def _load(self):
        from pathlib import Path

        if self.path is not None and Path(self.path).exists():
            state = json.loads(Path(self.path).read_text())
            self.counter = state["counter"]
            self.objects = {
                key: {**row, "data": base64.b64decode(row["data"])}
                for key, row in state["objects"].items()
            }

    def _save(self):
        import os
        import tempfile
        from pathlib import Path

        if self.path is None:
            return
        path = Path(self.path)
        descriptor, temporary = tempfile.mkstemp(prefix=".native-azure-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(
                    {
                        "counter": self.counter,
                        "objects": {
                            key: {**row, "data": base64.b64encode(row["data"]).decode()}
                            for key, row in self.objects.items()
                        },
                    },
                    stream,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def open(self):
        pass

    def close(self):
        self.closed = True

    def __exit__(self, *args):
        self.close()

    def send(self, request, **kwargs):
        import xml.etree.ElementTree as ET

        self._load()
        method, parsed = request.method.upper(), urlsplit(request.url)
        query = parse_qs(parsed.query)
        parts = unquote(parsed.path).lstrip("/").split("/", 1)
        key = parts[1] if len(parts) > 1 else None
        headers = requests.structures.CaseInsensitiveDict(request.headers)
        self.requests.append((method, request.url, dict(headers)))
        common = {
            "Last-Modified": "Mon, 26 Oct 2026 00:00:00 GMT",
            "x-ms-request-id": "synthetic-request",
            "x-ms-version": "2025-11-05",
        }

        def reply(status, body=b"", extra=None):
            return AzureObjectResponse(request, status, body, {**common, **(extra or {})})

        def error(status, code):
            return reply(
                status,
                b""
                if method == "HEAD"
                else f"<Error><Code>{code}</Code><Message>synthetic provider error</Message></Error>".encode(),
                {"Content-Type": "application/xml", "x-ms-error-code": code},
            )

        if key in self.denied:
            return error(403, "AuthorizationFailure")
        if self.reject_tier and method == "PUT" and headers.get("x-ms-access-tier"):
            return error(409, "InvalidBlobTier")
        if key is None:
            if (
                method == "HEAD"
                or method == "GET"
                and query.get("restype") == ["container"]
                and not query.get("comp")
            ):
                return reply(200, extra={"ETag": '"container"'})
            if method == "GET" and query.get("comp") == ["list"]:
                root = ET.Element(
                    "EnumerationResults",
                    ServiceEndpoint="https://synthetic.blob.core.windows.net/",
                    ContainerName="synthetic",
                )
                blobs = ET.SubElement(root, "Blobs")
                prefix = query.get("prefix", [""])[0]
                for name, row in sorted(self.objects.items()):
                    if not name.startswith(prefix):
                        continue
                    blob = ET.SubElement(blobs, "Blob")
                    ET.SubElement(blob, "Name").text = name
                    props = ET.SubElement(blob, "Properties")
                    for label, value in [
                        ("Last-Modified", common["Last-Modified"]),
                        ("Etag", row["etag"]),
                        ("Content-Length", str(len(row["data"]))),
                        ("BlobType", "BlockBlob"),
                        ("AccessTier", row["tier"]),
                    ]:
                        ET.SubElement(props, label).text = value
                ET.SubElement(root, "NextMarker")
                return reply(200, ET.tostring(root), {"Content-Type": "application/xml"})
            raise AssertionError("Unexpected native Azure container request")
        if method == "GET" and key is not None and self.before_download is not None:
            callback, self.before_download = self.before_download, None
            callback(key)
        previous = self.objects.get(key)
        if (
            headers.get("If-None-Match") == "*"
            and previous is not None
            or headers.get("If-Match") is not None
            and (previous is None or previous["etag"] != headers["If-Match"])
        ):
            return error(412, "ConditionNotMet")
        if method == "PUT" and query.get("comp") == ["tier"]:
            if previous is None:
                return error(404, "BlobNotFound")
            previous["tier"] = headers["x-ms-access-tier"]
            self._save()
            return reply(200)
        if method == "PUT" and not query.get("comp"):
            body = request.body
            if hasattr(body, "read"):
                body = body.read()
            self.counter += 1
            row = {
                "data": bytes(body),
                "etag": '"az' + str(self.counter) + '"',
                "tier": headers.get("x-ms-access-tier", "Hot"),
                "type": headers.get("x-ms-blob-content-type", "application/octet-stream"),
                "encoding": headers.get("x-ms-blob-content-encoding"),
                "metadata": {
                    name[10:]: value
                    for name, value in headers.items()
                    if name.lower().startswith("x-ms-meta-")
                },
            }
            self.objects[key] = row
            self._save()
            return reply(201, extra={} if self.omit_upload_etag else {"ETag": row["etag"]})
        if previous is None:
            return error(404, "BlobNotFound")
        properties = {
            "ETag": previous["etag"],
            "Content-Length": str(len(previous["data"])),
            "x-ms-blob-type": "BlockBlob",
            "x-ms-access-tier": previous["tier"],
            "Content-Type": previous["type"],
        }
        if previous["encoding"]:
            properties["Content-Encoding"] = previous["encoding"]
        if method == "HEAD":
            return reply(200, extra=properties)
        if method == "DELETE":
            del self.objects[key]
            self._save()
            return reply(202)
        if method == "GET":
            data = previous["data"]
            range_header = headers.get("x-ms-range", headers.get("Range"))
            status = 200
            if range_header:
                start, end = range_header.removeprefix("bytes=").split("-", 1)
                start, end = int(start), min(int(end), len(data) - 1)
                properties["Content-Range"] = f"bytes {start}-{end}/{len(data)}"
                data = data[start : end + 1]
                status = 206
                properties["Content-Length"] = str(len(data))
            return reply(status, data, properties)
        raise AssertionError("Unexpected native Azure object request")


def azure_client(transport):
    from azure.storage.blob import BlobServiceClient

    return BlobServiceClient(
        "https://synthetic.blob.core.windows.net",
        credential=None,
        transport=transport,
        retry_total=0,
    )
