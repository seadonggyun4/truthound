import polars as pl

from truthound.datadocs import HTMLReportBuilder
from truthound.profiler import DataProfiler


def test_dark_profile_svg_text_uses_theme_foreground():
    profile = DataProfiler().profile(pl.DataFrame({"measure": [1, 2, None, 4]}).lazy()).to_dict()
    html = HTMLReportBuilder(theme="dark", _use_svg=True).build(profile)
    assert 'fill: #1a1a2e' not in html
    assert 'fill: #374151' not in html
    assert 'class="percentage" fill="#f8fafc"' in html


def test_profile_frequency_chart_displays_counts_not_percentages():
    profile = DataProfiler().profile(pl.DataFrame({"measure": ["synthetic-A", "synthetic-A", "synthetic-B"]}).lazy()).to_dict()
    html = HTMLReportBuilder(_use_svg=True).build(profile)
    assert '<title>synthetic-A: 2</title>' in html
    assert '<title>synthetic-A: 2.0%</title>' not in html


def test_print_components_keep_theme_foreground():
    from truthound.datadocs.styles import COMPONENTS_CSS, REPORT_DOCUMENT_CSS

    recommendation_rule = COMPONENTS_CSS.split('.recommendation-item {', 1)[1].split('}', 1)[0]
    assert 'color: var(--color-text-primary)' in recommendation_rule
    header_rule = REPORT_DOCUMENT_CSS.split('.data-table th {', 1)[1].split('}', 1)[0]
    assert 'color: var(--color-background)' in header_rule
