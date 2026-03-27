"""Assertion helpers and base test case for reporter SDK tests."""

from __future__ import annotations

import csv
import json
import re
from io import StringIO
from typing import Any
from xml.etree import ElementTree

from truthound.reporters.sdk._testing_models import (
    MockValidationResult,
    create_mock_result,
    create_mock_results,
)


class ReporterTestCase:
    """Base class for reporter test cases."""

    reporter_class: type | None = None
    reporter_kwargs: dict[str, Any] = {}

    def setUp(self) -> None:
        if self.reporter_class is not None:
            self.reporter = self.reporter_class(**self.reporter_kwargs)
        else:
            self.reporter = None

    def tearDown(self) -> None:
        self.reporter = None

    def create_result(self, **kwargs: Any) -> MockValidationResult:
        return create_mock_result(**kwargs)

    def create_results(self, count: int = 5, **kwargs: Any) -> list[MockValidationResult]:
        return create_mock_results(count=count, **kwargs)

    def assert_valid_output(self, output: Any, format: str | None = None) -> None:
        assert_valid_output(output, format=format)

    def assert_json_structure(
        self,
        output: Any,
        required_keys: list[str] | None = None,
    ) -> None:
        data = json.loads(output) if isinstance(output, str) else output
        if required_keys:
            for key in required_keys:
                assert key in data, f"Missing required key: {key}"

    def assert_contains_all(self, output: str, substrings: list[str]) -> None:
        for substring in substrings:
            assert substring in output, f"Missing expected content: {substring}"


def assert_valid_output(output: Any, format: str | None = None) -> None:
    """Assert that output is valid for an expected format."""
    assert output is not None, "Output is None"

    if format is None:
        return

    normalized = format.lower()
    if normalized == "json":
        assert_json_valid(output)
    elif normalized == "xml":
        assert_xml_valid(output)
    elif normalized == "csv":
        assert_csv_valid(output)
    elif normalized == "text":
        assert isinstance(output, str), f"Expected string, got {type(output)}"


def assert_json_valid(output: Any) -> dict[str, Any]:
    """Assert that output is valid JSON."""
    if isinstance(output, dict):
        return output

    assert isinstance(output, str), f"Expected string or dict, got {type(output)}"

    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"Invalid JSON: {exc}") from exc


def assert_xml_valid(output: Any) -> ElementTree.Element:
    """Assert that output is valid XML."""
    if isinstance(output, ElementTree.Element):
        return output

    assert isinstance(output, (str, bytes)), (
        f"Expected string, bytes, or Element, got {type(output)}"
    )

    try:
        return ElementTree.fromstring(output)
    except ElementTree.ParseError as exc:
        raise AssertionError(f"Invalid XML: {exc}") from exc


def assert_csv_valid(output: Any, has_header: bool = True) -> list[list[str]]:
    """Assert that output is valid CSV."""
    assert isinstance(output, str), f"Expected string, got {type(output)}"

    try:
        rows = list(csv.reader(StringIO(output)))
        assert rows, "CSV is empty"
        if has_header:
            assert rows[0], "CSV header is empty"
        return rows
    except Exception as exc:
        raise AssertionError(f"Invalid CSV: {exc}") from exc


def assert_contains_patterns(
    output: str,
    patterns: list[str],
    regex: bool = False,
) -> None:
    """Assert that output contains all specified patterns."""
    for pattern in patterns:
        if regex:
            match = re.search(pattern, output)
            assert match is not None, f"Pattern not found: {pattern}"
        else:
            assert pattern in output, f"Pattern not found: {pattern}"


__all__ = [
    "ReporterTestCase",
    "assert_contains_patterns",
    "assert_csv_valid",
    "assert_json_valid",
    "assert_valid_output",
    "assert_xml_valid",
]
