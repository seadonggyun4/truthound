"""Tests for versioning storage module."""

import pytest
import tempfile
from pathlib import Path
from truthound.datadocs.versioning.storage import (
    InMemoryVersionStorage,
    FileVersionStorage,
)


class TestInMemoryVersionStorage:
    """Tests for InMemoryVersionStorage class."""

    def test_save_first_version(self):
        """Test saving first version."""
        storage = InMemoryVersionStorage()
        version = storage.save(
            report_id="test_report",
            content="<html>Report v1</html>",
            format="html",
            message="Initial version",
        )
        assert version.version == 1
        assert version.content == "<html>Report v1</html>"
        assert version.info.message == "Initial version"

    def test_save_multiple_versions(self):
        """Test saving multiple versions."""
        storage = InMemoryVersionStorage()
        v1 = storage.save("report", "Content v1")
        v2 = storage.save("report", "Content v2")
        v3 = storage.save("report", "Content v3")

        assert v1.version == 1
        assert v2.version == 2
        assert v3.version == 3

    def test_load_latest(self):
        """Test loading latest version."""
        storage = InMemoryVersionStorage()
        storage.save("report", "v1")
        storage.save("report", "v2")
        storage.save("report", "v3")

        loaded = storage.load("report")
        assert loaded is not None
        assert loaded.version == 3
        assert loaded.content == "v3"

    def test_load_specific_version(self):
        """Test loading specific version."""
        storage = InMemoryVersionStorage()
        storage.save("report", "v1")
        storage.save("report", "v2")
        storage.save("report", "v3")

        loaded = storage.load("report", version=2)
        assert loaded is not None
        assert loaded.version == 2
        assert loaded.content == "v2"

    def test_load_nonexistent(self):
        """Test loading nonexistent report."""
        storage = InMemoryVersionStorage()
        loaded = storage.load("nonexistent")
        assert loaded is None

    def test_list_versions(self):
        """Test listing versions."""
        storage = InMemoryVersionStorage()
        storage.save("report", "v1")
        storage.save("report", "v2")
        storage.save("report", "v3")

        versions = storage.list_versions("report")
        assert len(versions) == 3
        # Should be newest first
        assert versions[0].version == 3
        assert versions[2].version == 1

    def test_list_versions_with_limit(self):
        """Test listing with limit."""
        storage = InMemoryVersionStorage()
        for i in range(5):
            storage.save("report", f"v{i+1}")

        versions = storage.list_versions("report", limit=2)
        assert len(versions) == 2
        assert versions[0].version == 5
        assert versions[1].version == 4

    def test_list_versions_with_offset(self):
        """Test listing with offset."""
        storage = InMemoryVersionStorage()
        for i in range(5):
            storage.save("report", f"v{i+1}")

        versions = storage.list_versions("report", offset=2)
        assert len(versions) == 3
        assert versions[0].version == 3

    def test_get_latest_version(self):
        """Test getting latest version number."""
        storage = InMemoryVersionStorage()
        assert storage.get_latest_version("report") is None

        storage.save("report", "v1")
        assert storage.get_latest_version("report") == 1

        storage.save("report", "v2")
        assert storage.get_latest_version("report") == 2

    def test_delete_version(self):
        """Test deleting a version."""
        storage = InMemoryVersionStorage()
        storage.save("report", "v1")
        storage.save("report", "v2")

        assert storage.delete_version("report", 1) is True
        assert storage.count_versions("report") == 1
        assert storage.load("report", version=1) is None

    def test_delete_nonexistent(self):
        """Test deleting nonexistent version."""
        storage = InMemoryVersionStorage()
        assert storage.delete_version("report", 1) is False

    def test_count_versions(self):
        """Test counting versions."""
        storage = InMemoryVersionStorage()
        assert storage.count_versions("report") == 0

        storage.save("report", "v1")
        assert storage.count_versions("report") == 1

        storage.save("report", "v2")
        assert storage.count_versions("report") == 2

    def test_clear(self):
        """Test clearing all versions."""
        storage = InMemoryVersionStorage()
        storage.save("report1", "content")
        storage.save("report2", "content")

        storage.clear()

        assert storage.count_versions("report1") == 0
        assert storage.count_versions("report2") == 0

    def test_save_with_bytes_content(self):
        """Test saving bytes content."""
        storage = InMemoryVersionStorage()
        version = storage.save(
            report_id="pdf_report",
            content=b"PDF content bytes",
            format="pdf",
        )
        assert isinstance(version.content, bytes)
        assert version.info.size_bytes == len(b"PDF content bytes")

    def test_save_with_metadata(self):
        """Test saving with metadata."""
        storage = InMemoryVersionStorage()
        version = storage.save(
            report_id="report",
            content="content",
            metadata={"author": "test", "tags": ["v1"]},
        )
        assert version.info.metadata["author"] == "test"


class TestFileVersionStorage:
    """Tests for FileVersionStorage class."""

    def test_save_and_load(self):
        """Test saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileVersionStorage(tmpdir)

            version = storage.save(
                report_id="test_report",
                content="<html>Report</html>",
                format="html",
            )

            loaded = storage.load("test_report")
            assert loaded is not None
            assert loaded.content == "<html>Report</html>"

    def test_multiple_versions(self):
        """Test multiple versions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileVersionStorage(tmpdir)

            storage.save("report", "v1")
            storage.save("report", "v2")
            storage.save("report", "v3")

            assert storage.get_latest_version("report") == 3
            assert storage.load("report", version=2).content == "v2"

    def test_list_versions(self):
        """Test listing versions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileVersionStorage(tmpdir)

            storage.save("report", "v1", message="First")
            storage.save("report", "v2", message="Second")

            versions = storage.list_versions("report")
            assert len(versions) == 2
            assert versions[0].version == 2

    def test_delete_version(self):
        """Test deleting version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileVersionStorage(tmpdir)

            storage.save("report", "v1")
            storage.save("report", "v2")

            assert storage.delete_version("report", 1) is True
            assert storage.count_versions("report") == 1

    def test_binary_content(self):
        """Test binary content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileVersionStorage(tmpdir)

            content = b"Binary PDF content"
            storage.save("pdf_report", content, format="pdf")

            loaded = storage.load("pdf_report")
            assert loaded.content == content

    def test_special_characters_in_id(self):
        """Test report ID with special characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileVersionStorage(tmpdir)

            # IDs with slashes should be sanitized
            storage.save("path/to/report", "content")
            loaded = storage.load("path/to/report")
            assert loaded is not None

    def test_count_versions(self):
        """Test counting versions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileVersionStorage(tmpdir)

            assert storage.count_versions("report") == 0

            storage.save("report", "v1")
            storage.save("report", "v2")

            assert storage.count_versions("report") == 2
