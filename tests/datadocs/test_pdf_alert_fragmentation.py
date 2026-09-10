"""Short alert cards must remain intact at printed page boundaries."""
from __future__ import annotations

import os

import pytest

from truthound.datadocs import HTMLReportBuilder
from truthound.datadocs.styles import get_complete_stylesheet


def _pdf_dependencies():
    try:
        from weasyprint import HTML
        from weasyprint.formatting_structure.boxes import TextBox
    except ImportError:
        if os.environ.get("TRUTHOUND_DATADOCS_REQUIRE_PDF_SMOKE", "").lower() in {"1", "true", "yes", "on"}:
            pytest.fail("PDF fragmentation regression requires WeasyPrint")
        pytest.skip("PDF fragmentation regression requires WeasyPrint")
    return HTML, TextBox


@pytest.mark.parametrize("theme", ["light", "dark", "minimal"])
@pytest.mark.parametrize("language", ["en", "ko"])
def test_short_alert_cards_keep_text_inside_printed_card(theme, language):
    HTML, TextBox = _pdf_dependencies()

    builder = HTMLReportBuilder(theme=theme, language=language)
    css = get_complete_stylesheet(builder._theme_config.to_css_vars(), is_dark=theme == "dark")
    css += builder._get_pdf_professional_css()
    title = "Synthetic warning" if language == "en" else "합성 경고"
    cards = "".join(
        f'<div class="alert alert-info" data-card="{index}">'
        f'<div class="alert-header"><span class="alert-title">{title} {index}</span></div>'
        f'<div class="alert-message">DetailMarker{index}</div>'
        f'<div class="alert-suggestion">SuggestionMarker{index}</div></div>'
        for index in range(8)
    )
    document = HTML(string=f'<html><head><style>{css}</style></head><body class="pdf-document">'
        f'<div style="height:200mm"></div><div class="alerts-container">{cards}</div>'
        '</body></html>').render()
    seen = []
    for page in document.pages:
        for box in page._page_box.descendants():
            element = box.element
            if element is None or element.get("data-card") is None or box.element_tag != "div":
                continue
            # Only the outer card owns direct children with their own elements.
            if not any(child.element is not element for child in getattr(box, "children", ())):
                continue
            index = element.get("data-card")
            texts = [child for child in box.descendants() if isinstance(child, TextBox)]
            text = "".join(child.text for child in texts)
            assert f"DetailMarker{index}" in text
            assert f"SuggestionMarker{index}" in text
            assert all(child.position_y + child.height <= box.border_box_y() + box.border_height() + 0.1
                       for child in texts), "alert text extends beyond its printed card"
            assert all(child.position_y + child.height <= page._page_box.content_box_y() + page._page_box.height + 0.1
                       for child in texts), "alert text extends beyond the printable page"
            seen.append(index)
    assert sorted(seen) == [str(index) for index in range(8)]


@pytest.mark.parametrize("theme", ["light", "dark", "minimal"])
@pytest.mark.parametrize("language", ["en", "ko"])
def test_full_profile_alert_cards_do_not_overflow_page(theme, language):
    import polars as pl

    from truthound.profiler import DataProfiler

    HTML, TextBox = _pdf_dependencies()
    profile = DataProfiler(sample_size=None).profile(pl.DataFrame({
        f"synthetic_{index}": [1, 1] if index < 5 else [1, 2]
        for index in range(11)
    }).lazy()).to_dict()
    content = HTMLReportBuilder(theme=theme, language=language, _use_svg=True).build_for_pdf(profile)
    document = HTML(string=content).render()
    seen = 0
    for page in document.pages:
        for box in page._page_box.descendants():
            element = box.element
            if element is None or box.element_tag != "div" or "alert" not in element.get("class", "").split():
                continue
            if not any(child.element is not element for child in getattr(box, "children", ())):
                continue
            texts = [child for child in box.descendants() if isinstance(child, TextBox)]
            assert texts
            assert all(child.position_y + child.height <= box.border_box_y() + box.border_height() + 0.1
                       for child in texts), "alert text extends beyond its printed card"
            assert all(child.position_y + child.height <= page._page_box.content_box_y() + page._page_box.height + 0.1
                       for child in texts), "alert text extends beyond the printable page"
            seen += 1
    assert seen >= 10
