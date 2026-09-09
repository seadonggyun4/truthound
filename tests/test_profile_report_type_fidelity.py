"""Type presentation must retain canonical and legacy Profile labels verbatim."""

import json
import re
from collections import Counter
from copy import deepcopy
from html import escape, unescape
from html.parser import HTMLParser

import pytest

from truthound.datadocs import HTMLReportBuilder
from truthound.datadocs.base import SectionType
from truthound.datadocs.builder import ProfileDataConverter
from truthound.datadocs.report_renderers import ReportDocumentRenderer

TYPE_CASES = [
    ({"inferred_type": "integer", "physical_type": "Int64", "dtype": "legacy"}, "integer"),
    ({"physical_type": "Int64", "dtype": "legacy"}, "Int64"),
    ({"dtype": "String"}, "String"),
    ({"inferred_type": None, "physical_type": "Float64", "dtype": "legacy"}, "Float64"),
    ({"inferred_type": "", "physical_type": "Boolean", "dtype": "legacy"}, "Boolean"),
    ({"inferred_type": "  ", "physical_type": None, "dtype": "Date"}, "Date"),
    ({"inferred_type": None, "physical_type": "", "dtype": "Datetime"}, "Datetime"),
    ({"physical_type": "  ", "dtype": "Decimal"}, "Decimal"),
    ({"inferred_type": "unknown", "physical_type": "Int64"}, "unknown"),
    ({"inferred_type": None, "physical_type": "", "dtype": " "}, "unknown"),
    ({}, "unknown"),
]


def _profile(columns):
    return {
        "source": "Synthetic",
        "row_count": 10,
        "column_count": len(columns),
        "duplicate_row_count": 2,
        "duplicate_row_ratio": 0.2,
        "columns": [
            {"name": f"column-{index}", "null_count": 2, "null_ratio": 0.2,
             "unique_ratio": 0.5, "distinct_count": 5, **column}
            for index, column in enumerate(columns)
        ],
    }


class _ReportLabels(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.badges = []
        self._row = None
        self._cell = None
        self._badge = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag == "td":
            self._cell = []
        elif tag == "span" and "column-type" in dict(attrs).get("class", "").split():
            self._badge = []

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)
        if self._badge is not None:
            self._badge.append(data)

    def handle_endtag(self, tag):
        if tag == "td" and self._cell is not None:
            if self._row is not None:
                self._row.append("".join(self._cell))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None
        elif tag == "span" and self._badge is not None:
            self.badges.append("".join(self._badge))
            self._badge = None


@pytest.mark.parametrize("column,label", TYPE_CASES)
def test_profile_type_distribution_retains_nonempty_type_precedence(column, label):
    chart = ProfileDataConverter(_profile([column])).get_type_distribution()
    assert chart.labels == [label]
    assert chart.values == [1]


@pytest.mark.parametrize("column,label", TYPE_CASES)
def test_profile_summary_and_appendix_use_same_type_label(column, label):
    profile = _profile([column])
    section = HTMLReportBuilder()._build_section(SectionType.COLUMNS, ProfileDataConverter(profile))
    assert section.tables[0]["rows"][0][1] == label
    row = ReportDocumentRenderer._render_profile_row(profile["columns"][0])
    cells = [unescape(cell) for cell in re.findall(r"<td>(.*?)</td>", row)]
    assert cells[1] == label


@pytest.mark.parametrize("svg", [True, False])
@pytest.mark.parametrize("for_pdf", [True, False])
@pytest.mark.parametrize("language", ["en", "ko"])
@pytest.mark.parametrize("theme", ["light", "dark", "minimal"])
def test_profile_html_type_labels_agree_without_mutating_input_or_quality(svg, for_pdf, language, theme):
    profile = _profile([column for column, _ in TYPE_CASES])
    original = deepcopy(profile)
    before_metrics = ProfileDataConverter(profile).get_overview_metrics()
    expected = [label for _, label in TYPE_CASES]
    builder = HTMLReportBuilder(theme=theme, language=language, _use_svg=svg)
    report = (builder.build_for_pdf if for_pdf else builder.build)(profile)
    parsed = _ReportLabels()
    parsed.feed(report)
    column_rows = [row for row in parsed.rows if len(row) == 5 and row[0].startswith("column-")]
    assert [row[1] for row in column_rows] == expected + expected
    assert parsed.badges == expected
    counts = Counter(expected)
    if svg:
        for label, count in counts.items():
            assert f"<title>{escape(label)}: {count} (" in report
    else:
        options = [json.loads(value) for value in re.findall(r"var options = (\{.*?\});", report, re.S)]
        donut = next(value for value in options if value["chart"]["type"] == "donut")
        assert donut["labels"] == list(counts)
        assert donut["series"] == list(counts.values())
    assert ProfileDataConverter(profile).get_overview_metrics() == before_metrics
    assert profile == original


@pytest.mark.parametrize("field", ["inferred_type", "physical_type", "dtype"])
@pytest.mark.parametrize("svg", [True, False])
def test_profile_type_labels_remain_text_not_markup(field, svg):
    label = '</script><img src=x onerror="alert(1)">&'
    profile = _profile([{field: label}])
    original = deepcopy(profile)
    report = HTMLReportBuilder(_use_svg=svg).build(profile)
    assert label not in report
    assert escape(label) in report
    parsed = _ReportLabels()
    parsed.feed(report)
    assert parsed.badges == [label]
    assert [row[1] for row in parsed.rows if len(row) == 5 and row[0] == "column-0"] == [label, label]
    assert profile == original
