from __future__ import annotations

import json

import pytest

from truthound.datadocs.i18n.catalog import ReportCatalog
from truthound.datadocs.versioning.storage import FileVersionStorage
from truthound.datadocs.versioning.version import SemanticStrategy


pytestmark = pytest.mark.fault


def test_report_catalog_rejects_corrupted_json_payload(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        ReportCatalog.from_json(catalog_path)


def test_report_catalog_keeps_template_when_params_are_incomplete():
    catalog = ReportCatalog(locale="en", messages={"summary": "Rows: {rows}, Issues: {issues}"})

    message = catalog.get("summary", rows=10)

    assert message == "Rows: {rows}, Issues: {issues}"


def test_file_version_storage_returns_none_when_content_file_is_missing(tmp_path):
    storage = FileVersionStorage(tmp_path)
    saved = storage.save("nightly-report", "<html>ok</html>", format="html")

    content_path = tmp_path / "nightly-report" / f"v{saved.version}.html"
    content_path.unlink()

    assert storage.load("nightly-report", saved.version) is None


def test_semantic_strategy_rejects_non_numeric_version_strings():
    strategy = SemanticStrategy()

    with pytest.raises(ValueError):
        strategy.parse_version("one.two.three")
