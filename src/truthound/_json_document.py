"""Shared JSON document loading contract."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl


def read_json_document(path: str | Path) -> pl.LazyFrame:
    """Read one complete JSON document into a bounded LazyFrame."""
    document_path = Path(path)
    with document_path.open("r", encoding="utf-8") as file:
        document = json.load(file)

    if isinstance(document, list):
        if not document:
            return pl.DataFrame().lazy()
        if all(isinstance(item, dict) for item in document):
            return pl.from_dicts(document).lazy()
        return pl.DataFrame({"value": document}).lazy()

    if isinstance(document, dict):
        return pl.from_dicts([document]).lazy()

    return pl.DataFrame({"value": [document]}).lazy()
