from html import escape

import polars as pl
import pytest

from truthound.datadocs import HTMLReportBuilder
from truthound.profiler import DataProfiler


@pytest.mark.parametrize("svg", [True, False])
@pytest.mark.parametrize("theme", ["light", "dark", "minimal"])
def test_profile_keeps_user_strings_as_text_not_markup(svg, theme):
    column = '<img src=x>'
    value = '<script>x()</script>'
    profile = DataProfiler().profile(pl.DataFrame({column: [value, value, 'safe', None]}).lazy()).to_dict()
    html = HTMLReportBuilder(theme=theme, _use_svg=svg).build(profile)
    assert column not in html
    assert value not in html
    assert escape(column) in html
    assert profile['columns'][0]['name'] == column
