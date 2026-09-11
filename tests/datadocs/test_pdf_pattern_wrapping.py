"""Long pattern examples must wrap without losing content in printed reports."""
from __future__ import annotations

import html
import json
import os
from copy import deepcopy

import pytest

from truthound.datadocs import HTMLReportBuilder
from truthound.datadocs.styles import get_complete_stylesheet

from .fixtures import sample_a4_report_profile


@pytest.mark.parametrize("theme", ["light", "dark", "minimal"])
@pytest.mark.parametrize("language", ["en", "ko"])
@pytest.mark.parametrize("length", [240, 5000])
def test_printed_pattern_samples_wrap_and_preserve_every_character(theme, language, length):
    if not os.environ.get("TRUTHOUND_DATADOCS_REQUIRE_PDF_SMOKE"):
        pytest.importorskip("weasyprint")
    from weasyprint import HTML
    from weasyprint.formatting_structure.boxes import TextBox
    builder = HTMLReportBuilder(theme=theme, language=language)
    css = get_complete_stylesheet(builder._theme_config.to_css_vars(), is_dark=theme == "dark")
    css += builder._get_pdf_professional_css()
    label = "Synthetic sample" if language == "en" else "합성 예시"
    sample = json.dumps({"value": "x" * length, "label": label}, ensure_ascii=False)
    content = (
        f'<html><head><style>{css}</style></head><body class="pdf-document">'
        '<div class="patterns-list"><div class="pattern-item">'
        '<div class="pattern-header"><span class="pattern-column">synthetic_json</span></div>'
        f'<div class="pattern-samples">{label}: <code data-example="true">'
        f'{html.escape(sample)}</code></div></div></div></body></html>'
    )
    document = HTML(string=content).render()
    fragments = []
    for page in document.pages:
        page_box = page._page_box
        for box in page_box.descendants():
            if not isinstance(box, TextBox):
                continue
            assert box.position_x >= page_box.content_box_x() - 0.1
            assert box.position_x + box.width <= page_box.content_box_x() + page_box.width + 0.1, (
                "pattern text exceeds the printable page width"
            )
            assert box.position_y + box.height <= page_box.content_box_y() + page_box.height + 0.1
            if box.element is not None and box.element.get("data-example") == "true":
                fragments.append(box.text)
    assert "".join("".join(fragments).split()) == "".join(sample.split())


@pytest.mark.parametrize("theme", ["light", "dark", "minimal"])
@pytest.mark.parametrize("language", ["en", "ko"])
def test_full_profile_preserves_long_json_pattern_examples(theme, language):
    if not os.environ.get("TRUTHOUND_DATADOCS_REQUIRE_PDF_SMOKE"):
        pytest.importorskip("weasyprint")
    from weasyprint import HTML
    from weasyprint.formatting_structure.boxes import TextBox

    sample = json.dumps({"synthetic": "x" * 600, "end": "CompleteExampleMarker"})
    profile = sample_a4_report_profile()
    profile["columns"][3]["detected_patterns"] = [
        {"pattern": "json", "match_ratio": 1.0, "sample_matches": [sample]}
    ]
    before = deepcopy(profile)
    content = HTMLReportBuilder(theme=theme, language=language, _use_svg=True).build_for_pdf(profile)
    document = HTML(string=content).render()
    fragments = []
    for page in document.pages:
        page_box = page._page_box
        for box in page_box.descendants():
            if not isinstance(box, TextBox) or box.element_tag != "code":
                continue
            assert box.position_x >= page_box.content_box_x() - 0.1
            assert box.position_x + box.width <= page_box.content_box_x() + page_box.width + 0.1
            fragments.append(box.text)
    assert "".join(sample.split()) in "".join("".join(fragments).split())
    assert profile == before
